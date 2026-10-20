# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned creator entry point inside the network-disabled scientific container."""

import asyncio
import hashlib
import json
import os
from pathlib import Path

from native_phase import phase


def validate_creator(root: Path, manifest: dict):
    for name, expected in manifest['files'].items():
        file = root / name
        if file.is_symlink() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Corral creator source mismatch: {name}')


async def run():
    root = Path('/output')
    request = json.loads(Path('/controller/request.json').read_text())
    manifest = json.loads(Path('/controller/frozen_sources.json').read_text())
    validate_creator(Path('/opt/corral'), manifest)
    os.environ['RETRO_DB_HOST']='/db-socket'
    os.environ['RETRO_DB_PORT']='5432'
    os.environ['RETRO_DB_USER']='corral_task'
    os.environ['RETRO_DB_NAME']='reactions_production_db'
    os.environ['RETRO_DB_PASSWORD']=''
    if request['phase']=='scorer':
        saved=json.loads((root/'solver-result.json').read_text())
        result=await phase(task_id=request['task_id'], execution_id=saved['execution_id'], root=root, saved_state=saved, benchmark_run_id=request.get('benchmark_run_id'))
    elif request['phase']=='solver':
        import httpx
        from openai import AsyncOpenAI
        from guarded_agent import guarded_agent
        async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds='/model-socket/model.sock')) as http:
            async with AsyncOpenAI(base_url='http://gym-internal',api_key=os.environ['CORRAL_BRIDGE_TOKEN'],http_client=http,max_retries=0) as client:
                with guarded_agent(request['endpoint'],request['lineage'],root/'calls',client=client,dispatch_digest=request['dispatch_digest']) as agent:
                    result=await phase(task_id=request['task_id'],execution_id=request['execution_id'],root=root,agent=agent)
    else:
        raise ValueError('Unknown Corral native phase')
    (root/(request['phase']+'-result.json')).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    asyncio.run(run())
