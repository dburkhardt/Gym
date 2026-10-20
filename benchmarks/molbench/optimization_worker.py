# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolated native PyTDC scorer; stdout is JSON, source progress goes to stderr."""
import contextlib,hashlib,json,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from benchmarks.molbench.adapter import modules

def evaluate(repo,meta,answer):
 for relative,sha in json.loads(Path(__file__).with_name('optimization-source-sha256.json').read_text()).items():
  assert hashlib.sha256((repo/relative).read_bytes()).hexdigest()==sha,relative
 loaders,metrics=modules(repo);loader=loaders.ChemCoTBenchMolOptPhyschemLoader();subtask=meta['subtask'];parsed=loader.parse_agent_result({'action_history':[{'finish':answer}]},subtask);entry=loader.build_pred_entry(meta['sample'],parsed,answer,subtask)
 with tempfile.TemporaryDirectory(prefix='molbench-opt-') as temporary:
  directory=Path(temporary);predictions=directory/'preds';predictions.mkdir();path=predictions/(subtask+'.json');path.write_text(json.dumps([entry]));aggregate=metrics.ChemCoTBenchMolOptPhyschemEval().run(str(predictions),str(directory/'results'),str(repo/'molbench/eval/ChemCoTBench/baseline_and_eval'));native=json.loads(path.read_text())[0]['metrics']
 return {'reward':native['improvement'],'metric':'creator_property_improvement','creator_metrics':native,'creator_aggregate':aggregate,'parsed':parsed,'source_condition':'tool-free optimization; native penalized-logP/QED/ESOL and scaffold metrics; not binary accuracy'}

if __name__=='__main__':
 data=json.load(sys.stdin)
 with contextlib.redirect_stdout(sys.stderr):result=evaluate(Path(sys.argv[1]),data['meta'],data['answer'])
 print(json.dumps(result,allow_nan=False))
