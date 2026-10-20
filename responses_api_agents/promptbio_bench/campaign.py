"""Nemotron-only campaign CLI. No paid judge path is reachable."""
import argparse
import json
from pathlib import Path
from responses_api_agents.promptbio_bench.nemotron_proxy import MODEL, GATEWAY
from responses_api_agents.promptbio_bench.runtime import download_task, solve, grade

def main():
    p=argparse.ArgumentParser()
    for field in ['selection','key-file','cache','output','scorer','scorer-python']:
        p.add_argument('--'+field,required=True,type=Path)
    p.add_argument('--image',required=True)
    p.add_argument('--limit',type=int,default=5)
    a=p.parse_args()
    selection=json.loads(a.selection.read_text())
    for task_id in selection['task_ids'][:a.limit]:
        destination=a.output/task_id
        destination.mkdir(parents=True,exist_ok=True)
        try:
            task=download_task(task_id,a.cache)
            run=solve(task,a.output,model=MODEL,base_url=GATEWAY,api_key=a.key_file.read_text().strip(),image=a.image)
            scores=grade(task,[run],scorer_dir=a.scorer,python=str(a.scorer_python),output=a.output/task_id/'grading',judge_model=MODEL,judge_key_file=a.key_file,budget_ledger=a.output/task_id/'judge-receipts.jsonl',judge_base_url=GATEWAY)
            (a.output/task_id/'campaign-result.json').write_text(json.dumps({'task_id':task_id,'solver':MODEL,'judge':MODEL,'scores':scores,'run':str(run),'protocol':'minimal-shell-agent; creator artifact evaluator; Nemotron judge variant'},indent=2))
        except Exception as exc:
            (destination/'campaign-failure.json').write_text(json.dumps({'task_id':task_id,'failure':str(exc),'classification':'execution_or_grading_failure','score':None,'retain_in_attempted_denominator':True},indent=2))
if __name__=='__main__':main()
