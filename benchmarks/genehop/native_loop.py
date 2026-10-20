"""GeneGPT chat loop over GeneHop, preserving native URL/answer protocol.

Only API transport/model configuration and bounded HTTP handling differ from
pinned main_turbo.py. No model-generated code is executed.
"""
import ast
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

MODEL = 'nvidia/nvidia/nemotron-3.5-super-vl'
ALLOWED = {'eutils.ncbi.nlm.nih.gov': '/entrez/eutils/', 'blast.ncbi.nlm.nih.gov': '/blast/Blast.cgi'}

def validate_url(url):
    p = urllib.parse.urlsplit(url)
    if p.scheme != 'https' or p.username or p.password or p.port not in (None,443) or p.hostname not in ALLOWED or not p.path.startswith(ALLOWED[p.hostname]):
        raise ValueError('Only the published NCBI HTTPS tools are allowed')
    return url.replace(' ', '+')

class NativeLoop:
    def __init__(self, upstream, artifacts, api_key, base_url):
        self.upstream=Path(upstream); self.artifacts=Path(artifacts)
        self.artifacts.mkdir(parents=True,exist_ok=True)
        self.api_key=api_key;self.base_url=base_url.rstrip('/');self.tool_count=0

    def call_api(self,url):
        url=validate_url(url);time.sleep(1); self.tool_count+=1
        request=urllib.request.Request(url)
        with urllib.request.urlopen(request,timeout=120) as response:
            final_url=validate_url(response.url);body=response.read(8*1024*1024+1)
        if len(body)>8*1024*1024:raise RuntimeError('Tool response exceeds8MiB capture limit')
        stem=self.artifacts/f'tool-{self.tool_count:03d}'
        stem.with_suffix('.body').write_bytes(body)
        stem.with_suffix('.json').write_text(json.dumps({'url':url,'final_url':final_url,'sha256':hashlib.sha256(body).hexdigest(),'bytes':len(body),'time':time.time()},indent=2))
        return body

    def header(self):
        path=getattr(self,'header_path',self.artifacts/'prompt-header.txt')
        if path.exists():return path.read_text()
        tree=ast.parse((self.upstream/'main_turbo.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_prompt_header')
        namespace={'call_api':self.call_api,'re':re,'time':time}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'creator-get-prompt-header','exec'),namespace)
        text=namespace['get_prompt_header']([True]*6);path.write_text(text);return text

    def run(self,question):
        prompt=self.header()+f'Question: {question}\n';calls=[];answer=None
        for turn in range(10):
            if len(prompt)>36000:prompt=prompt[-36000:]
            payload={'model':MODEL,'messages':[{'role':'user','content':prompt}],'temperature':0.0,'max_tokens':16384,'stop':['->','\n\nQuestion']}
            started=time.time();req=urllib.request.Request(self.base_url+'/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+self.api_key})
            # No automatic retry: uncertain inference must be reconciled before reissue.
            try:
                with urllib.request.urlopen(req,timeout=600) as response:raw=json.load(response)
            except urllib.error.HTTPError as exc:
                receipt={'turn':turn,'request':payload,'http_status':exc.code,'body':exc.read().decode('utf-8','replace'),'elapsed_seconds':time.time()-started}
                (self.artifacts/f'model-{turn:02d}-http-error.json').write_text(json.dumps(receipt,indent=2))
                raise
            record={'turn':turn,'request':payload,'response':raw,'elapsed_seconds':time.time()-started}
            (self.artifacts/f'model-{turn:02d}.json').write_text(json.dumps(record,indent=2));calls.append(record)
            if raw.get('model')!=MODEL:raise RuntimeError('Unexpected provider model')
            choice=raw['choices'][0];text=choice['message'].get('content') or ''
            if choice.get('finish_reason')=='length':return {'status':'model_budget','answer':text,'calls':calls}
            if choice.get('finish_reason')!='stop':return {'status':'model_failure','answer':text,'calls':calls,'error':'Unexpected finish reason: '+str(choice.get('finish_reason'))}
            urls=re.findall(r'\[(https?://[^\[\]]+)\]',text)
            if not urls:answer=text;break
            url=urls[0]
            if 'blast' in url and 'Get' in url:time.sleep(30)
            tool=self.call_api(url)
            if 'blast' in url and 'Put' in url:
                match=re.search('RID = (.*)\n',tool.decode());
                if not match:raise RuntimeError('BLAST submission has no RID')
                tool=match.group(1)
            if len(tool)>20000:tool=tool[:20000]
            prompt=f'{prompt}{text}->[{tool}]\n'
        return {'status':'completed' if answer is not None else 'step_budget','answer':answer if answer is not None else 'numError','calls':calls}
