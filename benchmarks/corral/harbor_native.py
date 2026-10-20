# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Creator Corral phases under Gym's existing Harbor and sandbox ownership."""

import hashlib
import json
import math
import secrets
import shutil
import uuid
from contextlib import AsyncExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

from harbor.agents.base import BaseAgent
from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier

from benchmarks.corral.database import database_clone
from benchmarks.corral.guarded_agent import MODEL
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.provider_dispatch import configuration_digest
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.lifecycle import _owned_sandbox, _publish
from nemo_gym.sandbox.model_bridge import bridge

IMAGE='sha256:24b49fd1a5fd0bd03192106d6c50b44564d5f413ff69bb6333a58ed0d79a51e3'
SNAPSHOT_SHA='a3fd9f2dcc919e29412cd819559de526d7bb606f0b427413842e21ed9d72cc42'


def copy_state(source: Path, destination: Path):
    if source.is_symlink() or any(p.is_symlink() for p in source.rglob('*')):
        raise ValueError('Corral saved state cannot contain symlinks')
    shutil.copytree(source,destination)


async def run_phase(*, task_id: str, phase: str, snapshot: Path, endpoint: str, output: Path, submitted: Path | None=None):
    manifest=json.loads(Path(__file__).with_name('frozen_sources.json').read_text())
    if task_id not in manifest['frozen5'] or phase not in ('solver','scorer'):
        raise ValueError('Corral task or phase outside frozen native selection')
    if phase=='solver' and (not get_global_config_dict().get('provider_dispatch_guard') or '/ng-rollout/' not in endpoint):
        raise ValueError('Corral requires native guarded rollout endpoint')
    if (phase=='scorer') != (submitted is not None):
        raise ValueError('Scoring requires saved state; solver must start fresh')
    output.mkdir(parents=True,exist_ok=False)
    state=output/'native'
    if submitted is None:
        state.mkdir()
    else:
        copy_state(submitted,state)
        (state / "scorer-result.json").unlink(missing_ok=True)
        saved = json.loads((state / "solver-result.json").read_text())
        if not saved.get("execution_id", "").startswith(f"corral:{task_id}:"):
            raise ValueError("Corral saved state belongs to a different task")
    controller=output/'controller'
    controller.mkdir()
    for name in ('container_phase.py','native_phase.py','guarded_agent.py','frozen_sources.json'):
        shutil.copy2(Path(__file__).with_name(name),controller/name)
    lineage=uuid.uuid4().hex
    digest=configuration_digest({'model':MODEL,'image':IMAGE,'db_snapshot':SNAPSHOT_SHA,'task_id':task_id,'temperature':1.0,'max_tokens':16384,'iterations':50,'phase':phase,'controller':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in controller.iterdir()}})
    request={'task_id':task_id,'phase':phase,'lineage':lineage,'dispatch_digest':digest,'execution_id':f'corral:{task_id}:{lineage}','endpoint':endpoint}
    _publish(controller/'request.json',request)
    token=secrets.token_urlsafe(32)
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(database_clone(snapshot=snapshot,expected_sha256=SNAPSHOT_SHA,socket_dir=output/'db-socket',receipt=output/'db-lifecycle.json'))
        volumes=[f'{controller.resolve()}:/controller:ro',f'{state.resolve()}:/output',f'{(output/"db-socket").resolve()}:/db-socket:ro']
        if phase=='solver':
            socket_dir=Path(stack.enter_context(TemporaryDirectory(prefix='corral-model-',dir='/tmp')))
            await stack.enter_async_context(bridge(socket_dir/'model.sock',endpoint=endpoint,token=token,lineage=lineage,digest=digest,namespace='corral',allow_stream=True,expected_model=MODEL))
            volumes.append(f'{socket_dir.resolve()}:/model-socket:ro')
        sandbox=AsyncSandbox({'docker':{'create':{'network':'none'}}},SandboxSpec(image=IMAGE,ttl_s=3720,resources=SandboxResources(cpu=2,memory_mib=4096),metadata={'workload':'gym-corral-'+phase,'lineage':lineage},env={'PYTHONPATH':'/controller:/opt/corral/src:/opt/corral/tasks/retrosynthesis','CORRAL_BRIDGE_TOKEN':token,'LITELLM_LOCAL_MODEL_COST_MAP':'True'},provider_options={'volumes':volumes}))
        async with _owned_sandbox(sandbox,output/'runtime-lifecycle.json'):
            result=await sandbox.exec('/opt/corral-runtime/bin/python /controller/container_phase.py',timeout_s=3600)
            (output/'stdout.log').write_text(result.stdout or '')
            (output/'stderr.log').write_text(result.stderr or '')
            if result.return_code!=0:
                raise RuntimeError('Corral native phase failed; preserved artifacts are not a scientific zero')
    result_path=state/(phase+'-result.json')
    if not result_path.is_file():
        raise RuntimeError('Native Corral phase result missing')
    return state


class CorralNativeAgent(BaseAgent):
    def __init__(self, *, database_snapshot: str, api_base: str, **kwargs):
        super().__init__(**kwargs)
        if self.model_name != MODEL:
            raise ValueError('Corral requires exact campaign model')
        self.snapshot,self.endpoint=Path(database_snapshot),api_base

    @staticmethod
    def name():
        return 'corral-native'

    def version(self):
        return 'f5a6963a9ccf6e3c0058c776fe4111b639add648'

    async def setup(self, environment):
        result=await environment.exec('mkdir -p /workspace',timeout_sec=30)
        if result.return_code!=0:
            raise RuntimeError('Corral artifact workspace setup failed')

    async def run(self, instruction, environment, context):
        state=await run_phase(task_id=instruction.strip(),phase='solver',snapshot=self.snapshot,endpoint=self.endpoint,output=self.logs_dir/'native-solver')
        await environment.upload_dir(state,'/workspace/native')


class CorralNativeVerifier(BaseVerifier):
    def __init__(self, *, database_snapshot: str, gym_model_api_base: str, gym_model_name: str, **kwargs):
        super().__init__(**kwargs)
        if gym_model_name!=MODEL:
            raise ValueError('Corral requires exact campaign model')
        self.snapshot=Path(database_snapshot)
        self.endpoint=gym_model_api_base

    async def verify(self):
        submitted=self.trial_paths.verifier_dir/'submitted'
        await self.environment.download_dir('/workspace/native',submitted)
        state=await run_phase(task_id=self.task.paths.task_dir.name,phase='scorer',snapshot=self.snapshot,endpoint=self.endpoint,output=self.trial_paths.verifier_dir/'native-scorer',submitted=submitted)
        result=json.loads((state/'scorer-result.json').read_text())
        score=result.get('score')
        if not isinstance(score,(int,float)) or not math.isfinite(score):
            raise RuntimeError('Corral creator score missing or nonfinite')
        return VerifierResult(rewards={'score':float(score)})
