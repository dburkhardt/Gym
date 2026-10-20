# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned creator scorers; annotations stay in verifier metadata, never model input."""
import argparse
import importlib.util
import json
import subprocess
from pathlib import Path
try:
    from .prepare import stratified
except ImportError:
    from prepare import stratified

PINS = {'chemiq':'63826f64990802e87f1b11b9192606a20838d522',
        'qcbench':'195b4788375d1701b405f1addbf84903a41b5dc1',
        'genehop':'0a2a327a507be49e660597769cc53431abd6613a'}


def module(path: Path):
    spec = importlib.util.spec_from_file_location('creator_' + path.stem, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def validate_repo(kind: str, repo: Path) -> None:
    pin = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if pin != PINS[kind]:
        raise ValueError(f'Expected {PINS[kind]}, got {pin}')


def prepare(kind: str, repo: Path, output: Path) -> dict:
    validate_repo(kind, repo)
    rows = []
    if kind == 'chemiq':
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
