"""Invoke creator metrics in its separate compatible Python environment."""
import os,sys,json,math,ast,subprocess
from pathlib import Path
repo=Path(os.environ['BIOPLANNER_UPSTREAM'])
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()=='80b69543f6d4c3abbeb5fcc9cb1bacef76bf974a'
r=json.load(sys.stdin);gold=r['gold_pseudocode'];functions=r['admissible_functions'];answer=r['answer']
reference_error=None
try: ast.parse(gold)
except SyntaxError as e: reference_error=str(e)
sys.path.insert(0,str(repo/'src'))
import torch
torch.set_num_threads(2)
import evaluate_metrics as m
# Creator metrics consume definitions plus generated calls. No AST repair or code execution.
prediction=functions+'\n'+answer
import contextlib
with contextlib.redirect_stdout(sys.stderr):
 metrics={};errors={}
 for names,call in [
  (('function_precision','function_recall'),lambda:m.get_PR_funcs(gold,prediction)),
  (('argument_precision','argument_recall'),lambda:m.get_PR_args(gold,prediction)),
  (('argument_similarity',),lambda:(m.get_args_similarity(gold,prediction),)),
  (('argument_bleu',),lambda:(m.get_args_bleu(gold,prediction),)),
  (('edit_distance',),lambda:(m.get_levenshtein_distance(functions,gold,prediction),))]:
  try:
   for key,value in zip(names,call()):metrics[key]=None if not math.isfinite(float(value)) else float(value)
  except Exception as e:
   for key in names:metrics[key]=None;errors[key]=type(e).__name__+': '+str(e)
print(json.dumps({'metrics':metrics,'metric_errors':errors,'reference_ast_error':reference_error}))
