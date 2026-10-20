# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the pinned native aggregate evaluator outside the solver sandbox."""
import json
import math
import threading
import sys
from pathlib import Path
from benchmarks.scivisagentbench.native import validate_source
from benchmarks.scivisagentbench.judge import JudgeClient, score_validity
from benchmarks.scivisagentbench.transport import MODEL

async def _evaluate(source: Path, source_case: dict, case_dir: Path, data_dir: Path,
                   agent_mode: str, sender, output: Path):
    provenance = validate_source(source)
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(source/'benchmark'))
    sys.path.insert(0, str(source/'benchmark/evaluation_helpers'))
    from evaluation_framework.evaluation_manager import EvaluationManager
    import llm_evaluator
    original_client = llm_evaluator.OpenAI
    original_text = llm_evaluator.LLMEvaluator.evaluate_text
    original_vision = llm_evaluator.LLMEvaluator.evaluate_visualization
    original_result_only = llm_evaluator.LLMEvaluator.evaluate_visualization_result_only
    previous_price = llm_evaluator.LLMEvaluator.MODEL_PRICING.get(MODEL)
    assertions = source_case['assert']
    if any(a['type'] != 'llm-rubric' or a['subtype'] not in ('vision','text') for a in assertions):
        raise ValueError('unsupported native assertion; do not silently omit')
    rubrics, files = {}, {}
    for a in assertions:
        rubrics.setdefault(a['subtype'], []).append(a['value'])
        files.setdefault(a['subtype'], []).append({k.replace('-','_'):a[k] for k in ('gs-file','rs-file') if k in a})
    parsed = []
    def wrap(method, modality):
        def call(self, *args, **kwargs):
            value = method(self, *args, **kwargs)
            index = sum(r['modality'] == modality for r in parsed)
            goals = len([line for line in rubrics[modality][index].splitlines() if line.strip()])
            parsed.append({'modality':modality,'index':index,'native_result':value,
                           'validity':score_validity(value, modality=modality, goals=goals)})
            return value
        return call
    try:
        llm_evaluator.OpenAI = lambda **kwargs: JudgeClient(sender, output/'judge-provider.jsonl')
        llm_evaluator.LLMEvaluator.MODEL_PRICING[MODEL] = {'input':0,'cached_input':0,'output':0,'provider':'openai'}
        llm_evaluator.LLMEvaluator.evaluate_text = wrap(original_text,'text')
        llm_evaluator.LLMEvaluator.evaluate_visualization = wrap(original_vision,'vision')
        llm_evaluator.LLMEvaluator.evaluate_visualization_result_only = wrap(original_result_only,'vision')
        manager = EvaluationManager(eval_mode='mcp',openai_api_key='host-only-placeholder',eval_model=MODEL,
                                    static_screenshot=True,agent_mode=agent_mode,openai_base_url='http://unused-injected-client')
        result = await manager.run_evaluation(str(case_dir),case_dir.name,list(rubrics),rubrics,files,str(data_dir))
        aggregate = aggregate_validity(result, parsed, assertions)
        valid = len(parsed)==len(assertions) and all(r['validity']['valid'] for r in parsed) and aggregate['valid']
        record = {'native_result':result,'aggregate_validity':aggregate,'parsed_judgements':parsed,'expected_judgements':len(assertions),
                  'valid_for_interpretation':valid,'failure_kind':None if valid else 'incomplete_or_invalid_native_judgement',
                  'source_provenance':provenance,'scope':'Native aggregate and component scores preserved; separate validity guard.'}
        (output/'evaluation.json').write_text(json.dumps(record,indent=2))
        return record
    finally:
        llm_evaluator.OpenAI = original_client
        llm_evaluator.LLMEvaluator.evaluate_text = original_text
        llm_evaluator.LLMEvaluator.evaluate_visualization = original_vision
        llm_evaluator.LLMEvaluator.evaluate_visualization_result_only = original_result_only
        if previous_price is None:llm_evaluator.LLMEvaluator.MODEL_PRICING.pop(MODEL,None)
        else:llm_evaluator.LLMEvaluator.MODEL_PRICING[MODEL]=previous_price


_EVALUATOR_LOCK = threading.Lock()
async def evaluate(*args, **kwargs):
    # Fail closed instead of allowing process-global creator factory patches to overlap.
    if not _EVALUATOR_LOCK.acquire(blocking=False):
        raise RuntimeError('native_evaluator_already_active_in_process')
    try:
        return await _evaluate(*args, **kwargs)
    finally:
        _EVALUATOR_LOCK.release()


def aggregate_validity(result, parsed, assertions):
    """Check native component arithmetic without changing its result."""
    try:
        assert result['status']=='completed'
        expected = {}
        for a in assertions:expected[a['subtype']]=expected.get(a['subtype'],0)+1
        assert set(result['subtype_results'])==set(expected)
        total=maximum=0
        def number(v):
            assert type(v) in (int,float) and math.isfinite(v)
            return v
        def same(a,b):assert abs(number(a)-number(b))<1e-6
        for modality,count in expected.items():
            sub=result['subtype_results'][modality];assert sub['status']=='completed'
            parts=sub['individual_results'];assert len(parts)==count
            judgements=[p for p in parsed if p['modality']==modality];assert len(judgements)==count
            subtotal=submax=0
            for part,j in zip(parts,judgements):
                assert part['status']=='completed' and j['validity']['valid']
                scores=part['scores'];actual=j['validity']['score'];possible=j['validity']['max_score']
                if modality=='vision' and count==1:
                    for key,cap in [('output_generation',5),('efficiency',10)]:
                        value=number(scores[key]);assert 0<=value<=cap;actual+=value;possible+=cap
                same(scores['total_score'],actual);same(scores['max_possible_score'],possible)
                subtotal+=actual;submax+=possible
            if modality=='vision' and count>1:
                # Native adds output/efficiency once for multiple vision rubrics.
                details=sub['detailed_scores']
                for key,cap in [('output_generation',5),('efficiency',10)]:
                    value=number(details[key]['score']);assert 0<=value<=cap;subtotal+=value;submax+=cap
            same(sub['scores']['total_score'],subtotal);same(sub['scores']['max_possible_score'],submax)
            same(sub['scores']['percentage'],100*subtotal/submax)
            total+=subtotal;maximum+=submax
        same(result['scores']['total_score'],total);same(result['scores']['max_possible_score'],maximum)
        same(result['scores']['percentage'],100*total/maximum)
        return {'valid':True,'recomputed_total':total,'recomputed_maximum':maximum}
    except (AssertionError,KeyError,TypeError,ZeroDivisionError):
        return {'valid':False,'failure':'native_aggregate_status_coverage_or_arithmetic'}
