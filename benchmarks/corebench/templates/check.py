import json,pathlib,sys
sys.path.insert(0,"/tests")
from evaluations import eval_result_json
p=pathlib.Path("/workspace/report.json")
r=json.loads(p.read_text()) if p.exists() else {}
e=eval_result_json(json.load(open("/tests/answers.json")),r)
n=e["total_written_questions"]+e["total_vision_questions"]
assert n>0
out={"reward":(e["correct_written_answers"]+e["correct_vision_answers"])/n,"components":e}
pathlib.Path("/logs/verifier").mkdir(parents=True,exist_ok=True)
pathlib.Path("/logs/verifier/components.json").write_text(json.dumps(e))
pathlib.Path("/logs/verifier/reward.json").write_text(json.dumps({"reward":out["reward"]}))
