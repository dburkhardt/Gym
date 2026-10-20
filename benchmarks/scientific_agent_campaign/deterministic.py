# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned creator scorers; annotations stay in verifier metadata, never model input."""
import argparse
import ast
from types import SimpleNamespace
import importlib.util
import json
import os
import re
import random
import math
import subprocess
import sys
from pathlib import Path
try:
    from .prepare import stratified
except ImportError:
    from prepare import stratified

PINS = {'scienceqa':'2cbf8318e07b9ece895bb2ae605e71e38d623264','scibench':'e14e0ca3d3db0f02928a18d435b9c3d15656a7c5','fgbench':'fe67aded66bb08534da180ad6073cef42f422666','moleculariq':'425ecaaa8faf65aa43aa60ec0f584b7b7f060063','sino_us_drugqa':'cd026a27a5e039863de9080b8d7d95acaf1f5a39','chemiq':'63826f64990802e87f1b11b9192606a20838d522',
        'qcbench':'195b4788375d1701b405f1addbf84903a41b5dc1',
        'genehop':'0a2a327a507be49e660597769cc53431abd6613a'}


def module(path: Path):
    spec = importlib.util.spec_from_file_location('creator_' + path.stem, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


def validate_repo(kind: str, repo: Path) -> None:
    pin = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if pin != PINS[kind]:
        raise ValueError(f'Expected {PINS[kind]}, got {pin}')


def prepare(kind: str, repo: Path, output: Path) -> dict:
    validate_repo(kind, repo)
    rows = []
    if kind == 'scienceqa':
        problems=json.loads((repo/'data/scienceqa/problems.json').read_text()); ids=json.loads((repo/'data/scienceqa/pid_splits.json').read_text())['test'];captions=json.loads((repo/'data/captions.json').read_text())['captions']
        base=module(repo/'models/base_prompt.py')
        for id in ids:
            x=problems[id];x={**x,'caption':captions.get(id,'')}
            prompt=base.create_one_example('QCM-A',x['question'],base.get_context_text(x,True),base.get_choice_text(x,list('ABCDE')),'','','',test_example=True)
            rows.append({'uuid':id,'stratum':x['subject'],'responses_create_params':{'input':[{'role':'user','content':prompt}]},'verifier_metadata':x})
    elif kind == 'scibench':
        post=module(repo/'eval/post_process.py');prompts=module(repo/'eval/prompt/prompt_scai.py')
        for book in ['atkins','chemmc','thermo','matter']:
            for i,x in enumerate(json.loads((repo/f'dataset/original/{book}.json').read_text())):
                unit=post.remove_not(x['unit']) or x['unit']
                prompt='Q: '+x['problem_text']+' The unit of the answer is '+unit+'.\nA: The answer is'
                rows.append({'uuid':book+':'+str(i),'stratum':book,'responses_create_params':{'input':[{'role':'system','content':prompts.sys_cal_box2},{'role':'user','content':prompt}]},'verifier_metadata':x})
    elif kind == 'fgbench':
        data = [json.loads(x) for x in Path(os.environ['FGBENCH_JSONL']).read_text().splitlines()]
        for i,x in enumerate(data):
            rows.append({'uuid':str(i),'stratum':x['type'].rsplit('_',1)[0],
                         'responses_create_params':{'input':[{'role':'user','content':x['question']}]},
                         'verifier_metadata':x})
    elif kind == 'moleculariq':
        import pyarrow.parquet as pq
        source = Path(os.environ['MOLECULARIQ_PARQUET'])
        data = pq.read_table(source).to_pylist()
        tree = ast.parse((repo/'lm_eval/tasks/moleculariq/task_processor.py').read_text())
        prompt = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SYSTEM_PROMPT' for t in n.targets))
        for x in data:
            rows.append({'uuid':x['uid'],'stratum':x['task_type'],
                         'responses_create_params':{'input':[{'role':'system','content':prompt},{'role':'user','content':x['question']}]},
                         'verifier_metadata':x})
    elif kind == 'sino_us_drugqa':
        upstream = module(repo/'evaluation/common.py')
        data = upstream.load_release(str(repo/'data/release/all.jsonl'))
        for x in data:
            system = upstream.SYSTEM_PROMPT
            # Creator .format mistakes literal JSON braces for a field; replace only the two intended fields.
            prompt = upstream.USER_PROMPT_TEMPLATE.replace('{question}',x['question']).replace('{options}',upstream.format_options(x['choices']))
            rows.append({'uuid':str(x['id']),'stratum':x['task_type']+'/'+x['language'],
                         'responses_create_params':{'input':[{'role':'system','content':system},{'role':'user','content':prompt}]},
                         'verifier_metadata':x})
    elif kind == 'chemiq':
        data = [json.loads(x) for x in (repo/'questions/chemiq.jsonl').read_text().splitlines()]
        for x in data:
            rows.append({'uuid':x['uuid'],'stratum':x['question_category'],
                         'responses_create_params':{'input':[{'role':'user','content':x['prompt']}]},
                         'verifier_metadata':x})
    elif kind == 'qcbench':
        data = json.loads((repo/'QCBench.json').read_text())
        system = ('You are an expert chemist. Please read the following question and provide a step-by-step solution. '
                  'Your final answer must be presented as a readable LaTeX formula, enclosed in a \\boxed{} environment. '
                  'If the final answer is numerical, write only the numeric value inside \\boxed{}; place the unit immediately after the box (not inside), using the unit specified in the problem.')
        for x in data:
            prompt = x['question']
            if x.get('unit','').strip():
                prompt += f" The unit of the final answer is {x['unit'].strip()}. " + 'Do not put the unit inside the \\boxed{{}}; place it right after the box.'
            rows.append({'uuid':str(x['index']),'stratum':x['class'],
                         'responses_create_params':{'input':[{'role':'system','content':system},{'role':'user','content':prompt}]},
                         'verifier_metadata':x})
    else:
        data = json.loads((repo/'data/genehop.json').read_text())
        for category, pairs in data.items():
            for i,(q,a) in enumerate(pairs.items()):
                rows.append({'uuid':f'{category}:{i}','stratum':category,
                             'responses_create_params':{'input':[{'role':'user','content':q}]},
                             'verifier_metadata':{'question':q,'answer':a,'category':category}})
    assert len({x['uuid'] for x in rows}) == len(rows)
    selected = stratified(rows,'stratum',50)
    output.mkdir(parents=True,exist_ok=True)
    for n in (5,20,50):
        (output/f'{kind}-{n}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in selected[:n]))
    record = {'benchmark':kind,'pin':PINS[kind],'population':len(rows),
              'sample20':[x['uuid'] for x in selected[:20]],'heldout30':[x['uuid'] for x in selected[20:]],
              'qualification': 'GeneHop requires retrieval harness and only disease-location has automatic upstream scoring; prepared QA is NOT full GeneGPT protocol' if kind=='genehop' else 'Creator deterministic scoring; QCBench numeric metric differs from xVerify metric; ChemIQ OPSIN requires network'}
    (output/'manifest.json').write_text(json.dumps(record,indent=2))
    return record


def score(kind: str, repo: Path, task: dict, answer: str) -> dict:
    """Execute on grader host; ChemIQ SIGALRM requires main thread/subprocess."""
    validate_repo(kind, repo)
    meta = task['verifier_metadata']
    if kind == 'scienceqa':
        found=re.findall(r'The answer is ([A-Z]).',answer)
        letter=found[0] if len(found)==1 else 'FAILED'
        valid=letter in 'ABCDE'[:len(meta['choices'])]
        predicted='ABCDE'.index(letter) if valid else random.Random(task['uuid']).choice(range(len(meta['choices'])))
        return {'reward':float(predicted==meta['answer']),'strict_score':float(valid and predicted==meta['answer']),'fallback_randomized':not valid,'metric':'creator_gpt_caption_accuracy_seeded_fallback'}
    if kind == 'scibench':
        post=module(repo/'eval/post_process.py'); parsed=post.parse_math_answer(answer)
        if parsed is None:return {'reward':0.0,'parse_error':True}
        target=meta['answer_number'];unit=post.remove_not(meta['unit']) or meta['unit']
        if unit!=meta['unit']:
            parsed=post.cal_not(post.parse_not(parsed));target=post.cal_not((target,meta['unit']))
        fn=next(n for n in ast.parse((repo/'eval/eval_zero.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='equiv')
        ns={'math':math};exec(compile(ast.Module(body=[fn],type_ignores=[]),'creator_equiv','exec'),ns)
        return {'reward':float(ns['equiv'](str(parsed),target,unit)),'parsed':parsed,'metric':'creator_5percent_numeric'}
    if kind == 'fgbench':
        upstream = module(repo/'result_ananlysis/analysis.py')
        kind = upstream.task_name(meta)
        gt = upstream.scalar(str(meta['answer'])); parsed = upstream.extract_answer(answer,'standard')
        if kind.endswith('bool'):
            return {'reward':float(gt==parsed),'metric':'creator_boolean_accuracy','valid':parsed is not None,'parsed':parsed}
        valid = bool(parsed and gt and isinstance(parsed,(int,float)) and isinstance(gt,(int,float)))
        squared = (gt-parsed)**2 if valid else None
        # Gym transport requires scalar reward; aggregate ONLY raw squared_error over valid rows into creator RMSE.
        return {'reward':-squared if valid else 0.0,'metric':'negative_squared_error_transport_NOT_accuracy','creator_squared_error':squared,'creator_valid':valid,'parsed':parsed,'target':gt}
    if kind == 'moleculariq':
        from moleculariq_core import evaluate_answer
        extractor = module(repo/'lm_eval/tasks/moleculariq/extractors.py').extract_moleculariq_answer
        tree = ast.parse((repo/'lm_eval/tasks/moleculariq/task_processor.py').read_text())
        fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='moleculariq_bencheval')
        ns = {'Dict':dict,'Any':object,'evaluate_answer':evaluate_answer,'processor':SimpleNamespace(extraction_function=extractor)}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'pinned_creator_scorer','exec'),ns)
        return ns['moleculariq_bencheval'](meta,answer)
    if kind == 'sino_us_drugqa':
        upstream = module(repo/'evaluation/common.py')
        parsed = upstream.parse_selected_answer(answer)
        prediction = upstream.Prediction(id=str(meta['id']),predicted_answer=parsed,valid_output=bool(parsed))
        scored = upstream.score_prediction(meta,prediction)
        return {'reward':float(bool(scored.correct)),'parsed':parsed,'valid_output':scored.valid_output,'metric':'creator_accuracy'}
    if kind == 'qcbench':
        scorer = module(repo/'eval.py')
        parsed = scorer.extract_boxed(answer)
        return {'reward':scorer.is_numeric_equal_pro(meta['answer'],parsed),'parsed':parsed,'metric':'creator_numeric'}
    if kind == 'genehop':
        if meta['category'] != 'Disease gene location':
            return {'reward':None,'status':'upstream_manual_scoring_required'}
        parsed = module(repo/'evaluate.py').get_answer(answer,'Disease gene location')
        return {'reward':sum(a in parsed for a in meta['answer'])/len(meta['answer']),'metric':'creator_reference_recall'}
    parser = module(repo/'utils/parser.py').AnswerParser(repo/'questions/chemiq.jsonl')
    verifier = module(repo/'utils/answer_verifier.py').AnswerVerifier(repo/'questions/chemiq.jsonl')
    parsed = parser.parse(task['uuid'],answer)
    result = verifier.check_answer(task['uuid'],parsed)
    return {'reward':float(result['is_correct']),'parsed':parsed,'metric':'creator_check_answer'}


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=PINS);p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.kind,a.repo,a.output)))
