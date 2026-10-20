# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serial Gym entry point for the unmodified creator Napari/TinyAgent harness.

Inference remains fail-closed until the independently reviewed sandbox/evaluator
packet enables it. Operational solver completion is never a scientific reward.
"""
import asyncio
import json
import time
import uuid
from pathlib import Path

from fastapi import Body, Request
from pydantic import ConfigDict, Field, PrivateAttr
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import request as http_request, get_response_json, raise_for_status
from benchmarks.scivisagentbench.native import run_native
from benchmarks.scivisagentbench.transport import MODEL


class Config(BaseResponsesAPIAgentConfig):
    source_root: str
    task_manifest: str
    native_config: str
    output_root: str
    endpoint: str
    max_tokens: int = 16384
    timeout_s: int = 7200
    inference_enabled: bool = False
    judge_command: list[str] = Field(default_factory=list)


class RunRequest(BaseRunRequest):
    model_config = ConfigDict(extra='allow')
    verifier_metadata: dict


class RunResponse(BaseVerifyResponse):
    artifact_dir: str
    native_solver_result: dict = Field(default_factory=dict)
    scientific_score: float | None = None


class Agent(SimpleResponsesAPIAgent):
    config: Config
    _semaphore: asyncio.Semaphore = PrivateAttr(default_factory=lambda: asyncio.Semaphore(1))

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()):
        raise NotImplementedError('Use /run with an allowlisted frozen case ID')

    async def run(self, body: RunRequest = Body(), request: Request = None) -> RunResponse:
        async with self._semaphore:
            if not self.config.inference_enabled:
                raise ValueError('SciVis inference gate is closed')
            tasks = json.loads(Path(self.config.task_manifest).read_text())
            task_id = body.verifier_metadata.get('task_id')
            if task_id not in tasks:
                raise ValueError('unknown frozen task ID')
            task = tasks[task_id]
            output = Path(self.config.output_root)/uuid.uuid4().hex
            async def sender(payload):
                reply = await http_request('POST', self.config.endpoint.rstrip('/')+'/chat/completions',
                    json=payload, headers={'Authorization': 'Bearer placeholder',
                    'X-Campaign-Benchmark': 'SciVisAgentBench'}, _max_num_tries=1,
                    _max_connection_retries=0)
                await raise_for_status(reply)
                return await get_response_json(reply)
            result = {}
            evaluation = {}
            failure = 'native_evaluator_pending'
            try:
                async with asyncio.timeout(self.config.timeout_s):
                    result = await run_native(Path(self.config.source_root),
                        json.loads(Path(self.config.native_config).read_text()), task, output,
                        sender, self.config.max_tokens)
                if not result.get('success'):
                    failure = 'native_solver_failed: '+str(result.get('error'))
                elif self.config.judge_command:
                    task_path=output/'judge-task.json'
                    task_path.write_text(json.dumps(task))
                    with (output/'native-judge.log').open('xb') as log:
                        process=await asyncio.create_subprocess_exec(*self.config.judge_command,
                            self.config.source_root,str(task_path),str(output/'evaluation'),self.config.endpoint,
                            stdout=log,stderr=asyncio.subprocess.STDOUT)
                        try:
                            code=await asyncio.wait_for(process.wait(),timeout=self.config.timeout_s)
                        except BaseException:
                            process.kill()
                            await process.wait()
                            raise
                    if code:raise RuntimeError(f'native judge exited {code}; raw log preserved')
                    evaluation=json.loads((output/'evaluation/evaluation.json').read_text())
                    failure=None if evaluation['valid_for_interpretation'] else evaluation['failure_kind']
            except Exception as exc:
                failure = type(exc).__name__+': '+str(exc)
            response = NeMoGymResponse(id=output.name, created_at=time.time(), model=MODEL,
                object='response', output=[], parallel_tool_calls=False, tools=[], tool_choice='auto')
            # Native percentage includes operational components; preserve component breakdown.
            score=evaluation['native_result']['scores']['percentage']/100 if evaluation.get('valid_for_interpretation') else None
            record = RunResponse(**body.model_dump(), response=response, reward=score if score is not None else 0,
                mask_sample=score is None, failure_kind='scivis:unscored' if score is None else None, failure_reason=failure,
                scientific_score=score, native_solver_result=result, artifact_dir=str(output))
            output.mkdir(parents=True, exist_ok=True)
            (output/'gym-result.json').write_text(record.model_dump_json(indent=2))
            return record

if __name__ == '__main__':
    Agent.run_webserver()
