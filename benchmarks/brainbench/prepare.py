import ast,csv,json,os,random,subprocess
from pathlib import Path
import numpy as np
PINS={'experiments':'9808e5d61ee88c51967e43b5a48ad89b3aeee421','testcases':'89869dab3be1ec096dc38931ea33e43268c65d30'}
def build(experiments,testcases):
    for name,p in [('experiments',experiments),('testcases',testcases)]:
        assert subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()==PINS[name]
    tree=ast.parse((experiments/'utils/data_utils.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['extract_abstract_pair','prepare_prompt_multiple_choice_harness']];ns={'re':__import__('re')};exec(compile(ast.Module(body=nodes,type_ignores=[]),'creator','exec'),ns)
    template=(experiments/'prompt_templates/gpt2.txt').read_text();rng=np.random.RandomState(42);out=[]
    for i,r in enumerate(csv.DictReader((testcases/'BrainBench_Human_v0.1.csv').open())):
        a,b=ns['extract_abstract_pair'](r['combined_abstract']);target=0
        if rng.rand()>0.5:a,b=b,a;target=1
        choices=ns['prepare_prompt_multiple_choice_harness'](a,b,template)
        out.append({'uuid':f'brainbench-human-{i}','responses_create_params':{'input':[{'role':'user','content':json.dumps({'choices':choices})}]},'verifier_metadata':{'target':target,'source_index':i}})
    random.Random(20261020).shuffle(out);return out

def prepare()->Path:
    sample=Path(os.environ['CAMPAIGN_SAMPLE_JSONL']);assert sample.is_file();return sample
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--experiments',type=Path,required=True);p.add_argument('--testcases',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();rows=build(a.experiments,a.testcases);a.out.mkdir(parents=True,exist_ok=True)
    for n in [5,20,50]:(a.out/f'sample-{n}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows[:n]))
    print({'population':len(rows)})
