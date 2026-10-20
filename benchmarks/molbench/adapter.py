# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned creator tool-free MolBench ms-3 baseline; not MolClaw agent."""
import hashlib,importlib.util,json,tempfile,random
from pathlib import Path

def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def modules(repo):
 repo=Path(repo)
 for relative,sha in json.loads(Path(__file__).with_name('source-sha256.json').read_text()).items():
  assert hashlib.sha256((repo/relative).read_bytes()).hexdigest()==sha,relative
 return (load(repo/'molclaw_run/data_loader/bench_loaders.py','molbench_creator_loader'),load(repo/'molbench/eval/eval_runner.py','molbench_creator_scoring'))

def score(repo,meta,answer):
 loaders,metrics=modules(repo)
 if meta.get('stratum')=='mo-edit':
  dependency=Path(repo)/'molbench/eval/ChemCoTBench/baseline_and_eval'
  expected=json.loads(Path(__file__).with_name('editing-source-sha256.json').read_text())
  for relative,sha in expected.items():
   assert hashlib.sha256((Path(repo)/relative).read_bytes()).hexdigest()==sha,relative
  loader=loaders.ChemCoTBenchMolEditLoader();subtask=meta['subtask'];parsed=loader.parse_agent_result({'action_history':[{'finish':answer}]},subtask);entry=loader.build_pred_entry(meta['sample'],parsed,answer,subtask)
  with tempfile.TemporaryDirectory(prefix='molbench-edit-') as temporary:
   directory=Path(temporary);predictions=directory/'preds';predictions.mkdir();path=predictions/(subtask+'.json');path.write_text(json.dumps([entry]))
   aggregate=metrics.ChemCoTBenchMolEditEval().run(str(predictions),str(directory/'results'),str(dependency))
   native=json.loads(path.read_text())[0]['metrics']
  return {'reward':native['correct_rate'],'metric':'creator_edit_correct_rate','creator_metrics':native,'creator_aggregate':aggregate,'parsed':parsed,'source_condition':'tool-free editing baseline; native chemical substructure checks'}
 if meta.get('stratum') in ('ms-1','ms-2'):
  kind=meta['stratum'];loader=loaders.RdkitBenchLoader() if kind=='ms-1' else loaders.ACNetCuratedLoader();subtask='rdkit_bench' if kind=='ms-1' else 'qa'
  parsed=loader.parse_agent_result({'action_history':[{'finish':answer}]},subtask)
  entry=loader.build_pred_entry(meta['sample'],parsed,answer,subtask)
  with tempfile.TemporaryDirectory(prefix='molbench-score-') as temporary:
   directory=Path(temporary);predictions=directory/'preds';predictions.mkdir();path=predictions/(subtask+'.json');path.write_text(json.dumps([entry]))
   evaluator=metrics.RdkitBenchEval() if kind=='ms-1' else metrics.ACNetCuratedEval()
   aggregate=evaluator.run(str(predictions),str(directory/'results'),str(repo))
   native=json.loads(path.read_text())[0]['metrics']
  return {'reward':native['acc' if kind=='ms-1' else 'score'],'metric':'creator_exact_match','creator_metrics':native,'creator_aggregate':aggregate,'parsed':parsed,'source_condition':'tool-free '+kind+' baseline; native exact-string scoring and validity preserved'}
 parsed=loaders.MolbenchVsLoader().parse_agent_result({'action_history':[{'finish':answer}]},'molbench_vs')
 entry=loaders.MolbenchVsLoader().build_pred_entry(meta['sample'],parsed,answer,'molbench_vs')
 result=metrics._compute_molbench_vs_metrics(entry)
 ranking=parsed.get('top3',[])
 return {'reward':result['top3_hit_rate'],'metric':'creator_top3_hit_rate','creator_metrics':result,
         'parsed':parsed,'duplicate_ranking':len(set(ranking))!=len(ranking),
         'parsed_count':len(ranking),'source_condition':'tool-free ms-3 baseline; native parser truncates to3; duplicates can receive repeated credit'}

def prepare_rows(repo,frozen):
 repo,frozen=Path(repo),Path(frozen);loaders,_=modules(repo)
 templates=load(repo/'molclaw_run/templates/template.py','molbench_creator_template')
 loader=loaders.MolbenchVsLoader();source={r['index']:r for r in loader.load_data(str(repo/'molbench/data/molbench-ms-3/molbench-ms-3.csv'),'molbench_vs')}
 public=json.loads((frozen/'public-prompts.json').read_text());private={r['id']:r['sample'] for r in json.loads((frozen/'evaluator-only.json').read_text())}
 result=[]
 for row in public:
  sample=source[row['id']];assert sample==private[row['id']]
  query=loader.get_query(sample,'molbench_vs');content=loader.get_dataset_content(sample,'molbench_vs')
  if content and '#candidates.smi' in query:query=query.replace('#candidates.smi','the following list (one SMILES per line):\n'+content)
  hint=loader.get_extra_answer_hint(sample,'molbench_vs') or templates.ANSWER_OUTPUT_HINT
  messages=[{'role':'system','content':templates.SYSTEM_PROMPT_MINIMAL},{'role':'user','content':query+'\n\n'+hint}]
  assert messages==row['messages']
  result.append({'uuid':row['id'],'stratum':'ms-3','responses_create_params':{'input':messages},'verifier_metadata':{'sample':sample}})
 return result


def prepare_screening_rows(repo,kind,seed=20261020):
 """Native released ms1/ms2 loaders and baseline prompt; one fixed permutation."""
 assert kind in ('ms-1','ms-2')
 repo=Path(repo);loaders,_=modules(repo);templates=load(repo/'molclaw_run/templates/template.py','molbench_creator_template')
 loader=loaders.RdkitBenchLoader() if kind=='ms-1' else loaders.ACNetCuratedLoader();subtask='rdkit_bench' if kind=='ms-1' else 'qa'
 samples=loader.load_data(str(repo/'molbench/data'/('molbench-'+kind)/('molbench-'+kind+'.csv')),subtask)
 rows=[]
 for i,sample in enumerate(samples):
  query=loader.get_query(sample,subtask)
  hint=templates.RDKIT_ANSWER_HINT if kind=='ms-1' else templates.ANSWER_OUTPUT_HINT
  messages=[{'role':'system','content':templates.SYSTEM_PROMPT_MINIMAL},{'role':'user','content':query+'\n\n'+hint}]
  rows.append({'uuid':kind+'-'+str(i),'stratum':kind,'responses_create_params':{'input':messages},'verifier_metadata':{'stratum':kind,'sample':sample}})
 random.Random(seed).shuffle(rows)
 return rows


def prepare_editing_rows(repo,seed=20261020):
 repo=Path(repo);loaders,_=modules(repo);templates=load(repo/'molclaw_run/templates/template.py','molbench_creator_template');loader=loaders.ChemCoTBenchMolEditLoader();rows=[]
 for subtask in loader.get_subtasks():
  for sample in loader.load_data(str(repo/'molbench/data/molbench-mo'),subtask):
   messages=[{'role':'system','content':templates.SYSTEM_PROMPT_MINIMAL},{'role':'user','content':loader.get_query(sample,subtask)+'\n\n'+templates.ANSWER_OUTPUT_HINT}]
   rows.append({'uuid':'mo-edit-'+sample['id'],'stratum':'mo-edit-'+subtask,'responses_create_params':{'input':messages},'verifier_metadata':{'stratum':'mo-edit','subtask':subtask,'sample':sample}})
 random.Random(seed).shuffle(rows)
 return rows
