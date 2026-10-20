# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned creator tool-free MolBench ms-3 baseline; not MolClaw agent."""
import hashlib,importlib.util,json
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
