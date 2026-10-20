"""Override every upstream LangChain LLM role; leave creator grading logic intact.

The source has nested gpt-5.4 defaults not propagated from --model. Capture and
replace only model initialization, explicitly recording requested original role.
"""
import json
import os
import runpy
import sys
from pathlib import Path
import langchain.chat_models

MODEL='nvidia/nvidia/nemotron-3.5-super-vl'
original=langchain.chat_models.init_chat_model

def nemotron_init(model=None, **kwargs):
    print(json.dumps({'campaign_model_override':{'upstream_requested':model,'actual':MODEL}}),flush=True)
    kwargs.update(model_provider='openai',base_url=os.environ['OPENAI_BASE_URL'],api_key='campaign-local-proxy',max_tokens=16384)
    return original(MODEL,**kwargs)

langchain.chat_models.init_chat_model=nemotron_init
script=Path(sys.argv[1]).resolve()
sys.argv=sys.argv[1:]
sys.path.insert(0,str(script.parent))
runpy.run_path(str(script),run_name='__main__')
