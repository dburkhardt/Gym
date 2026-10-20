# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native evaluator parser controls with an injected no-network provider."""
import importlib.util
import json
import sys
from pathlib import Path
from benchmarks.scivisagentbench.judge import JudgeClient, score_validity
from benchmarks.scivisagentbench.transport import MODEL

source, output = [Path(p).resolve() for p in sys.argv[1:3]]
output.mkdir(parents=True, exist_ok=True)
spec=importlib.util.spec_from_file_location('creator_llm_evaluator',source/'benchmark/evaluation_helpers/llm_evaluator.py')
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
# Register the campaign model/provider; pricing is inapplicable to free internal inference.
native.LLMEvaluator.MODEL_PRICING[MODEL]={'input':0,'cached_input':0,'output':0,'provider':'openai'}
fixtures=[('correct', '{"score":10,"max_score":10,"explanation":"control"}', 'stop', MODEL, True),
 ('wrong', '{"score":0,"max_score":10,"explanation":"control"}', 'stop', MODEL, True),
 ('empty', '', 'stop', MODEL, False), ('malformed', '{bad', 'stop', MODEL, False),
 ('incomplete', '{"score":10}', 'length', MODEL, False),
 ('wrong-model','{"score":10}', 'stop','other',False),
 ('out-of-range','{"score":1000}', 'stop',MODEL,False),
 ('missing-score','{"explanation":"no score"}', 'stop',MODEL,False)]
records=[]
for name,text,finish,model,valid in fixtures:
    calls=[]
    def sender(request):
        calls.append(request)
        return {'id':'mock-'+name,'model':model,'choices':[{'finish_reason':finish,'message':{'role':'assistant','content':text}}]}
    native.OpenAI=lambda **kwargs: JudgeClient(sender,output/(name+'.jsonl'))
    evaluator=native.LLMEvaluator(api_key='mock',model=MODEL,base_url='http://invalid-no-network')
    parsed=evaluator.evaluate_text('Native parser control only; no scientific judgement.')
    annotation=score_validity(parsed,modality='text',goals=1)
    assert annotation['valid']==valid,(name,parsed,annotation)
    assert len(calls)==1
    assert calls[0]['max_tokens']==1000 and calls[0]['temperature']==.1
    records.append({'fixture':name,'native_result':parsed,'validity':annotation,'mock_calls':1})
# Exercise the creator vision parser with real image encoding and mocked provider text.
image=source.parent.parent.parent.parent/'scivisagentbench/worker-controls/render-synthetic.png'
if not image.exists():
    from PIL import Image
    image=output/'synthetic-parser-control.png'
    Image.new('RGB',(8,8),(0,128,0)).save(image)
for name,values,valid in [('vision-valid',{'goal_1_score':10,'goal_2_score':10,'total_score':20},True),('vision-wrong',{'goal_1_score':0,'goal_2_score':0,'total_score':0},True),('vision-incomplete-goals',{'total_score':20},False),('vision-inconsistent-total',{'goal_1_score':10,'goal_2_score':0,'total_score':20},False)]:
    calls=[]
    def sender(request):
        calls.append(request)
        return {'id':'mock-'+name,'model':MODEL,'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':json.dumps(values)}}]}
    native.OpenAI=lambda **kwargs: JudgeClient(sender,output/(name+'.jsonl'))
    evaluator=native.LLMEvaluator(api_key='mock',model=MODEL,base_url='http://invalid-no-network')
    parsed=evaluator.evaluate_visualization([str(image)],[str(image)],'Parser control: two visual goals, no scientific judgement.')
    annotation=score_validity(parsed,modality='vision',goals=2)
    assert annotation['valid']==valid,(name,parsed,annotation)
    assert len(calls)==1
    assert len([x for x in calls[0]['messages'][0]['content'] if x['type']=='image_url'])==2
    records.append({'fixture':name,'native_result':parsed,'validity':annotation,'mock_calls':1})
(output/'controls.json').write_text(json.dumps({'llm_calls':0,'source':str(source),'controls':records,'scope':'Native parser/transport controls; not scientific VLM calibration.'},indent=2))
print('12 native parser/transport controls passed')
