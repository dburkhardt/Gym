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

PINS = {'hiscibench':'022fd997164f7dcb9a82c1c1809adf540973950c','genomeqa':'7ccc230e734fc11a365fd1963b593b5fd11e834d','omebench':'48632ca037cbb7d6e09a22132257c55ee6008edb','smolinstruct':'43ab5fccd14514ddf756534d18a6917e7e11d0ae','lpm24':'8e747238e90734bf8774712ecc010ee24efcd346','molinstructions':'d08a15f88e0c1b5e6d1e08330f570983885ddee1','molpuzzle':'4b62e754ddb5c9b3a5c870a6221b55d71c70ebd2','s2tomg':'df70b90de6efb62e753345b886234b7acfeff874','suprabench':'bcb06eb6e1d283b70e943ed0c90aa9e6f97826e5','mollangbench':'76dcacb082e20871851cbca6395180503a8b6d22','fukuyama':'63bb79f912f0b2de593996b80ffeea894f6f1a59','scienceqa':'2cbf8318e07b9ece895bb2ae605e71e38d623264','scibench':'e14e0ca3d3db0f02928a18d435b9c3d15656a7c5','fgbench':'fe67aded66bb08534da180ad6073cef42f422666','moleculariq':'425ecaaa8faf65aa43aa60ec0f584b7b7f060063','sino_us_drugqa':'cd026a27a5e039863de9080b8d7d95acaf1f5a39','chemiq':'63826f64990802e87f1b11b9192606a20838d522',
        'qcbench':'195b4788375d1701b405f1addbf84903a41b5dc1',
        'genehop':'0a2a327a507be49e660597769cc53431abd6613a'}


def module(path: Path):
    spec = importlib.util.spec_from_file_location('creator_' + path.stem, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


def validate_repo(kind: str, repo: Path) -> None:
    if kind=='hiscibench':
        import hashlib
        assert hashlib.sha256((repo/'source-manifest.json').read_bytes()).hexdigest()=='5e78ad47da2156882be9b2c1d5e2ae563adf3f6ae851948afcbba4af462d8fe0'
        manifest=json.loads((repo/'source-manifest.json').read_text());assert manifest['revision']==PINS[kind]
        for name,digest in manifest['sha256'].items():assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest
        return
    pin = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if pin != PINS[kind]:
        raise ValueError(f'Expected {PINS[kind]}, got {pin}')


def prepare(kind: str, repo: Path, output: Path) -> dict:
    validate_repo(kind, repo)
    rows = []
    if kind in ['genomeqa','omebench','hiscibench']:
        from benchmarks.scientific_agent_campaign.recovered_releases import prepare_rows
        rows=prepare_rows(kind,repo)
    elif kind == 'smolinstruct':
        import zipfile
        z=zipfile.ZipFile(os.environ['SMOLINSTRUCT_ZIP'])
        for sub in ['name_conversion-i2f','name_conversion-s2f']:
            record=json.loads(z.read('sample/instruction_tuning/test/'+sub+'.json'));templates=json.loads(z.read('template/'+record['template_name']+'/'+sub+'.json'));data=list(map(json.loads,z.read('raw/test/'+sub+'.jsonl').decode().splitlines()));tags=json.loads(z.read('core_tag/'+sub+'.json'))
            for item in record['samples']:
                x=data[item['idx']];assert x['split']=='test';left,right=tags['input'];value=(left+' '+x['input']+' '+right) if left is not None else x['input'];prompt=templates[item['template_id']]['input'].replace('<INPUT>',value)
                rows.append({'uuid':sub+'_'+str(item['idx']),'stratum':sub,'responses_create_params':{'input':[{'role':'user','content':prompt}]},'verifier_metadata':{'formula':x['output'],'output_tags':tags['output']}})
    elif kind == 'lpm24':
        for i,line in enumerate((repo/'final_evaluation/molgen/eval-molgen.txt').read_text().splitlines()):
            parts=line.split('\t');smiles,description=parts[:2]
            rows.append({'uuid':str(i),'stratum':'molgen','responses_create_params':{'input':[{'role':'user','content':'Generate a molecule matching this description. Return only its SMILES string.\n\n'+description}]},'verifier_metadata':{'smiles':smiles}})
    elif kind == 'molinstructions':
        import zipfile
        z=zipfile.ZipFile(os.environ['MOLINSTRUCTIONS_BIOTEXT_ZIP']);templates=json.loads((repo/'demo/templates/alpaca.json').read_text())
        for sub in ['multi_choice_question','true_or_false_question']:
            for i,x in enumerate(json.loads(z.read('Biomolecular_Text_Instructions/'+sub+'.json'))):
                if x['metadata']['split']!='test':continue
                prompt=templates['prompt_input' if x['input'] else 'prompt_no_input'].format(instruction=x['instruction'],input=x['input'])
                rows.append({'uuid':sub+'_'+str(i),'stratum':sub,'responses_create_params':{'input':[{'role':'user','content':prompt}]},'verifier_metadata':{'output':x['output'],'subtask':sub}})
    elif kind == 'molpuzzle':
        import base64
        tree=ast.parse((repo/'src/molpuzzle/spectrum_analysis.py').read_text());prompts=next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SYS_PROMPTS' for t in n.targets))
        for i,x in enumerate(json.loads((repo/'Data/Stage2.json').read_text())):
            if x['spectrum_type']!='H-NMR':continue
            image=repo/'Data/Spectrum_images/nmr_H_complet'/(str(x['Molecule Index']).strip()+'.png')
            if not image.exists():continue
            content=[{'type':'input_text','text':prompts['H-NMR']+'\n'+x['Question']},{'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(image.read_bytes()).decode(),'detail':'auto'}]
            rows.append({'uuid':str(i),'stratum':x['Answer'],'responses_create_params':{'input':[{'role':'system','content':'You are an expert organic chemist.'},{'role':'user','content':content}]},'verifier_metadata':{'answer':x['Answer'],'spectrum_type':'H-NMR','molecule_id':str(x['Molecule Index']).strip()}})
    elif kind == 's2tomg':
        import csv
        tree=ast.parse((repo/'utils/dataset.py').read_text());system=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SYSTEM_HEAD_JSON' for t in n.targets))
        for i,x in enumerate(csv.DictReader(open(os.environ['S2TOMG_ATOMNUM_CSV']))):
            rows.append({'uuid':str(i),'stratum':'MolCustom_AtomNum','responses_create_params':{'input':[{'role':'system','content':system},{'role':'user','content':x['Instruction']}]},'verifier_metadata':{'counts':{k:int(v) for k,v in x.items() if k!='Instruction'}}})
    elif kind == 'suprabench':
        for x in map(json.loads,(repo/'data/tbs/base.jsonl').read_text().splitlines()):
            rows.append({'uuid':x['id'],'stratum':'tbs','responses_create_params':{'input':[{'role':'user','content':x['question']}]},'verifier_metadata':{'answer':str(x['answer']).strip().upper(),'host_name':x.get('host_name','')}})
    elif kind == 'mollangbench':
        import pyarrow.parquet as pq
        data=Path(os.environ['MOLLANG_DATA_DIR'])
        for sub in ['generation','edit']:
            template=(repo/'prompts'/('editing' if sub=='edit' else sub)/'prompt_smiles.txt').read_text().strip()
            table=pq.read_table(data/(sub+'-test-00000-of-00001.parquet'))
            for i,x in enumerate(table.to_pylist()):
                prompt=template.format(structure_description=x['structure_description']) if sub=='generation' else template.format(original_smiles=x['original_smiles'],edit_instructions=x['edit_instructions'])
                rows.append({'uuid':sub+'_'+str(i),'stratum':sub,'responses_create_params':{'input':[{'role':'user','content':prompt}]},'verifier_metadata':{'smiles':x['smiles'] if sub=='generation' else x['edited_smiles'],'subtask':sub}})
    elif kind == 'fukuyama':
        import yaml
        from typing import Dict,List,Optional
        tree=ast.parse((repo/'eval/run_infer_pathway.py').read_text());fns=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['load_pathway_data','format_user_prompt']]
        ns={'os':os,'json':json,'Dict':Dict,'List':List,'Optional':Optional};exec(compile(ast.Module(body=fns,type_ignores=[]),'creator_loader','exec'),ns)
        data=ns['load_pathway_data'](str(repo/'fukuyama_bench/mechanisms')); prompts=yaml.safe_load((repo/'eval/prompts/infer_pathway_prompts.yaml').read_text());grader=module(repo/'eval/eval_infer_pathway.py')
        for x in data:
            ckpt=grader.load_checkpoints(str(repo/'fukuyama_bench/mechanisms'),x['case_id'])
            rows.append({'uuid':x['case_id'],'stratum':x['case_id'][0],'responses_create_params':{'input':[{'role':'system','content':prompts['pathway_student_system_prompt']},{'role':'user','content':ns['format_user_prompt'](prompts['pathway_student_template_prompt'],x)}]},'verifier_metadata':{'pathway':x['ground_truth_pathway'],'checkpoints':ckpt}})
    elif kind == 'scienceqa':
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
    record['qualification'] = {'hiscibench':'L1 Biology200 borrowed SuperGPQA questions only; exact creator zero-shot prompt/parser; overlap, not novel coverage.','genomeqa':'All5200 public binary/mcq records, exact creator system/user prompts and option parser; original option permutations preserved.', 'omebench':'oMeGold196 only; exact default prompt and creator oMeS metrics, S_total primary reward not weighted composite.','fukuyama': 'Text pathway inference; creator lenient checkpoint exact-match;319 public mechanisms.', 'mollangbench': 'Generation/edit SMILES test subset400; recognition and image-input tasks excluded.', 'suprabench': 'TBS base2064 only; creator accuracy excludes unparsed outputs; report parsed-only plus all-attempt.', 's2tomg': 'MolCustom AtomNum5000 only; creator JSON correction and17-element constraints; disconnected structures can pass.', 'molpuzzle': 'Repaired image-transport variant; Stage2 H-NMR2592 eligible of2808;216 missing source images excluded; Yes/No balanced pilot.', 'molinstructions': 'Held-out biotext1001MCQ+100TF; original Alpaca template and accuracy; empty MCQ explicitly failed instead of upstream crash.', 'lpm24': 'Molgen21805; explicit chat-instruction variant; creator InChI exact-match only, not full shared-task metric suite.', 'smolinstruct': 'Official test formula conversion i2f+s2f2993 each; exact templates/core tags/extraction and element-match.'}.get(kind,record['qualification'])
    (output/'manifest.json').write_text(json.dumps(record,indent=2))
    return record


def score(kind: str, repo: Path, task: dict, answer: str) -> dict:
    """Execute on grader host; ChemIQ SIGALRM requires main thread/subprocess."""
    validate_repo(kind, repo)
    meta = task['verifier_metadata']
    if kind in ['genomeqa','omebench','hiscibench']:
        from benchmarks.scientific_agent_campaign.recovered_releases import score_release
        return score_release(kind,repo,meta,answer)
    if kind == 'smolinstruct':
        import numpy as np
        from collections import defaultdict
        funcs=[]
        for path,names in [(repo/'utils/metrics.py',['parse_molecule','count_element_match','calculate_formula_metrics']),(repo/'extract_prediction.py',['extract_answer_part'])]:funcs.extend(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names)
        ns={'re':re,'np':np,'defaultdict':defaultdict};exec(compile(ast.Module(body=funcs,type_ignores=[]),'creator_smol','exec'),ns);pred=ns['extract_answer_part']([answer],*meta['output_tags'],mode='tag');result=ns['calculate_formula_metrics']([pred],[[meta['formula']]])
        return {'reward':float(result['num_t1_ele_match']),'creator_metrics':result,'predicted':pred,'metric':'creator_formula_element_match'}
    if kind == 'lpm24':
        import numpy as np
        from nltk.translate.bleu_score import corpus_bleu
        from Levenshtein import distance
        from rdkit import Chem
        fn=next(n for n in ast.parse((repo/'final_evaluation/molgen/scorer.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='evaluate');ns={'np':np,'corpus_bleu':corpus_bleu,'lev':distance,'Chem':Chem};exec(compile(ast.Module(body=[fn],type_ignores=[]),'creator_lpm','exec'),ns)
        bleu,exact,lev,valid=ns['evaluate']([meta['smiles']],[answer.strip()])
        return {'reward':float(exact),'valid':bool(valid),'nonempty_validity':bool(valid) and bool(answer.strip()),'levenshtein':float(lev),'metric':'creator_InChI_exact_match_only; explicit_chat_prompt_variant'}
    if kind == 'molinstructions':
        if not answer:return {'reward':0.0,'valid':False,'metric':'empty_response_failed; creator_mcq_would_raise_IndexError'}
        fn=next(n for n in ast.parse((repo/'evaluation/biotext'/(meta['subtask']+'.py')).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='calculate_accuracy');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'creator_molins','exec'),ns)
        return {'reward':float(ns['calculate_accuracy']([{'output':meta['output'],'my_output':answer}])/100),'valid':True,'metric':'creator_'+meta['subtask']+'_accuracy'}
    if kind == 'molpuzzle':
        pred=1 if 'yes' in answer.lower() else 0 if 'no' in answer.lower() else -1
        ref=1 if str(meta['answer']).strip().lower()=='yes' else 0
        return {'reward':float(pred==ref),'valid':pred!=-1,'prediction':pred,'reference':ref,'metric':'creator_binary_accuracy; repaired_image_transport_variant'}
    if kind == 's2tomg':
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        functions=[]
        for path,names in [(repo/'utils/evaluation.py',['mol_prop']),(repo/'evaluate.py',['correct_text'])]:
            functions.extend(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names)
        ns={'Chem':Chem,'Descriptors':Descriptors,'re':re,'json':json};exec(compile(ast.Module(body=functions,type_ignores=[]),'creator_s2','exec'),ns);pred=ns['correct_text'](answer);valid=bool(ns['mol_prop'](pred,'validity'))
        passed=valid and all(ns['mol_prop'](pred,'num_'+k)==v for k,v in meta['counts'].items())
        return {'reward':float(passed),'valid':valid,'nonempty_validity':valid and bool(pred.strip()),'metric':'creator_MolCustom_AtomNum_correct_JSON','predicted_smiles':pred}
    if kind == 'suprabench':
        tree=ast.parse((repo/'src/eval/tbs.py').read_text()); fns=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['_after_think','_extract_answer_tag','parse_mcq']];ns={'re':re};exec(compile(ast.Module(body=fns,type_ignores=[]),'creator_parser','exec'),ns);pred=ns['parse_mcq'](answer)
        return {'reward':float(pred==meta['answer']),'valid':pred is not None,'predicted':pred,'metric':'creator_TBS_parse; aggregate_accuracy_uses_parsed_only'}
    if kind == 'mollangbench':
        upstream=module(repo/'scripts/evaluate_results.py'); predicted=upstream.canonicalize_smiles(upstream.parse_smiles_from_output(answer)); target=upstream.canonicalize_smiles(meta['smiles'])
        return {'reward':float(predicted is not None and predicted==target),'valid':predicted is not None,'nonempty_validity':predicted is not None and predicted!='','metric':'creator_canonical_smiles_accuracy'}
    if kind == 'fukuyama':
        upstream=module(repo/'eval/eval_infer_pathway.py');pred=upstream.extract_pathway_from_response(answer)
        if not pred:return {'reward':0.0,'valid':False,'parse_error':True}
        if meta['checkpoints']:
            result=upstream.compare_pathways_with_checkpoints(pred,meta['pathway'],meta['checkpoints'],lenient=True)
        else:result=upstream.compare_pathways(pred,meta['pathway'],lenient=True)
        return {'reward':float(result['exact_match']),'creator_metrics':result,'metric':'creator_lenient_checkpoint_exact_match'}
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
