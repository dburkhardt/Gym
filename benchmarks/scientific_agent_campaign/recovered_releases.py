"""Adapters for subsequently located creator public repositories."""
import ast,json,re
from pathlib import Path

def genome_source(repo):
    tree=ast.parse((repo/'run.py').read_text());prompt=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SIMPLE_SYSTEM_PROMPT' for t in n.targets));method=next(m for n in tree.body if isinstance(n,ast.ClassDef) for m in n.body if isinstance(m,ast.FunctionDef) and m.name=='extract_final_option');ns={'re':re};exec(compile(ast.Module(body=[method],type_ignores=[]),'creator_genomeqa','exec'),ns);return prompt,ns['extract_final_option']
def prepare_rows(kind,repo):
    rows=[]
    if kind=='hiscibench':
        import yaml
        template=yaml.safe_load((repo/'code/L1/config/prompt/zero-shot.yaml').read_text())['prompt_format'][0]
        for r in map(json.loads,(repo/'data/L1/ori_data/SuperGPQA-all.jsonl').read_text().splitlines()):
            if r['field']!='Biology':continue
            value=r['question']+'\n'+'\n'.join(f'{chr(65+i)}) {o}' for i,o in enumerate(r['options']))
            rows.append({'uuid':r['uuid'],'stratum':r['subfield'],'responses_create_params':{'input':[{'role':'user','content':template.format(value)}]},'verifier_metadata':{'options':r['options'],'answer':r['answer_letter']}})
    elif kind=='genomeqa':
        prompt,_=genome_source(repo)
        for path in sorted((repo/'benchmark').glob('*/*.jsonl')):
            for i,line in enumerate(path.read_text().splitlines()):
                r=json.loads(line);options='\n'.join(f'{k}. {v}' for k,v in sorted(r['options'].items()));user=f"Question:\n{r['question']}\n\nOptions:\n{options}\n\nPlease select the correct option."
                rows.append({'uuid':str(path.relative_to(repo))+'#'+str(i),'stratum':str(path.relative_to(repo).parent),'responses_create_params':{'input':[{'role':'system','content':prompt},{'role':'user','content':user}]},'verifier_metadata':{'answer':r['answer'],'options_keys':list(r['options'])}})
    else:
        template=(repo/'prompts/default.txt').read_text()
        for r in json.loads((repo/'data/oMe_Gold.json').read_text()):
            prompt=template
            for key in ['reactants_smiles','products_smiles','conditions']:prompt=prompt.replace('{ '+key+' }',json.dumps(r[key]))
            rows.append({'uuid':r['reaction_id'],'stratum':r.get('level','gold'),'responses_create_params':{'input':[{'role':'user','content':prompt}]},'verifier_metadata':{'mechanism':r['mechanism']}})
    return rows

def score_release(kind,repo,meta,answer):
    if kind=='hiscibench':
        import timeout_decorator
        nodes=[n for n in ast.parse((repo/'code/L1/eval/eval.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ['safe_regex_search','extract_option_labels','extract_option_content']]
        ns={'re':re,'timeout_decorator':timeout_decorator};exec(compile(ast.Module(body=nodes,type_ignores=[]),'creator_hisci','exec'),ns)
        pred=ns['extract_option_labels'](answer,'ABCDEFGHIJ')
        if pred is None:
            content=ns['extract_option_content'](answer,meta['options']);pred=chr(meta['options'].index(content)+65) if content else None
        return {'reward':float(pred==meta['answer']),'predicted':pred,'metric':'creator_zero_shot_accuracy'}
    if kind=='genomeqa':
        _,extract=genome_source(repo);pred=extract(None,answer,meta['options_keys']);return {'reward':float(pred==meta['answer']),'predicted':pred,'metric':'creator_option_accuracy'}
    from benchmarks.scientific_agent_campaign.deterministic import module
    source=module(repo/'scripts/utils_eval.py')
    raw=answer
    if '[ANSWER]' in raw:raw=raw.split('[ANSWER]')[1].split('[/ANSWER]')[0].strip()
    if raw.startswith('```'):raw=raw.strip('`').replace('json\n','').replace('JSON\n','').strip()
    if 'is :' in raw:raw=raw.split('is :')[1].strip()
    try:
        pred=json.loads(raw)
        if isinstance(pred,dict):pred=pred.get('mechanism',pred)
        if not isinstance(pred,list):raise ValueError('Not mechanism list')
        result=source.oMeS([(j['subtype'],j['intermediate_smiles'],j['step_weight']) for j in meta['mechanism']],[(j['subtype'],j['intermediate_smiles']) for j in pred])
        scores={'S_total':round(result.S_total,2),'S_partial':round(result.S_partial,2),'V':result.V,'L':result.L}
    except (ValueError,TypeError,KeyError):scores={'S_total':0.0,'S_partial':0.0,'V':0,'L':0}
    return {'reward':scores['S_total'],'creator_metrics':scores,'metric':'creator_S_total_not_composite'}
