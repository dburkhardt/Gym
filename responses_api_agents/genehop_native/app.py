# SPDX-License-Identifier: Apache-2.0
"""Native GeneGPT retrieval loop, with manual-only GeneHop scores explicitly masked."""

import ast
import asyncio
import json
import os
import re
import threading
import time
from pathlib import Path
from uuid import uuid4

from fastapi import Request, Response
from pydantic import ConfigDict, PrivateAttr

from benchmarks.genehop.native_loop import MODEL, NativeLoop
from benchmarks.genehop.source_identity import validate_source
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.provider_dispatch import configuration_digest
from nemo_gym.server_utils import raise_for_status, rollout_path_prefix


class Config(BaseResponsesAPIAgentConfig):
    upstream_repo: Path
    output_root: Path
    base_url: str = "http://127.0.0.1:49180/v1"
    api_key_env: str = "GENEHOP_API_KEY"
    model_server: ModelServerRef | None = None


class RunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    uuid: str
    verifier_metadata: dict


class VerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: dict
    uuid: str
    artifact_dir: str
    native_status: str
    raw_creator_score: float | None = None
    reward_metric: str = "creator_disease_location_recall_only"


class GeneHopAgent(SimpleResponsesAPIAgent):
    config: Config
    _lock: asyncio.Semaphore = PrivateAttr(default_factory=lambda: asyncio.Semaphore(1))

    async def responses(self, request: Request, response: Response, body: NeMoGymResponseCreateParamsNonStreaming):
        raise NotImplementedError("Use native /run")

    async def guarded_model_call(self, body: RunRequest, payload: dict, turn: int) -> dict:
        """Use the existing captured chat route without changing native request fields."""
        if self.config.model_server is None:
            raise ValueError("Shared model server is required")
        if not self.server_client.global_config_dict.get("provider_dispatch_guard"):
            raise ValueError("GeneHop shared transport requires provider_dispatch_guard")
        rollout_id = self.rollout_id_from_run(body)
        if rollout_id is None:
            raise ValueError("Guarded GeneHop requires native rollout identity")
        digest = configuration_digest({"harness": "genehop-native", "task": body.model_dump(), "payload": payload})
        async with asyncio.timeout(600):
            response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path=rollout_path_prefix(rollout_id) + "/v1/chat/completions",
                json=payload,
                headers={
                    "x-nemo-gym-logical-call-id": f"genehop:{body.uuid}:{turn}",
                    "x-nemo-gym-call-config-sha256": digest,
                },
            )
            await raise_for_status(response)
            return await response.json()

    def execute(self, body, model_call=None, stop_event=None):
        if not re.fullmatch(r"[A-Za-z0-9_: -]+", body.uuid):
            raise ValueError("Unsafe task UUID")
        taskid = body.uuid.replace(":", "-").replace(" ", "_")
        out = self.config.output_root / taskid
        out.mkdir(parents=True, exist_ok=False)
        try:
            loop = NativeLoop(
                self.config.upstream_repo,
                out,
                "" if model_call else os.environ[self.config.api_key_env],
                self.config.base_url,
                model_call=model_call,
                stop_event=stop_event,
            )
            loop.header_path = self.config.output_root / "prompt-header.txt"
            result = loop.run(body.verifier_metadata["question"])
        except Exception as exc:
            result = {
                "status": "infrastructure_failure",
                "answer": None,
                "error": type(exc).__name__ + ": " + str(exc),
                "calls": [],
            }
            (out / "failure.json").write_text(json.dumps(result, indent=2))
        (out / "native-result.json").write_text(json.dumps(result, indent=2))
        return out, result

    async def run(self, request: Request, body: RunRequest):
        question = body.verifier_metadata["question"]
        inp = body.responses_create_params.model_dump()["input"]
        if len(inp) != 1 or inp[0]["role"] != "user" or inp[0]["content"] != question:
            raise ValueError("Question mismatch")
        model_call = None
        stop_event = threading.Event()
        dispatch_lock = threading.Lock()
        if self.config.model_server is not None:
            validate_source(self.config.upstream_repo)
            event_loop = asyncio.get_running_loop()

            def model_call(payload, turn):
                with dispatch_lock:
                    if stop_event.is_set():
                        raise RuntimeError("GeneHop run cancelled before dispatch")
                    future = asyncio.run_coroutine_threadsafe(self.guarded_model_call(body, payload, turn), event_loop)
                return future.result()
        elif self.server_client.global_config_dict.get("provider_dispatch_guard") is True:
            raise ValueError("Guarded GeneHop cannot use legacy direct transport")
        async with self._lock:
            worker = asyncio.create_task(asyncio.to_thread(self.execute, body, model_call, stop_event))
            try:
                out, result = await asyncio.shield(worker)
            except asyncio.CancelledError:
                with dispatch_lock:
                    stop_event.set()
                # Let an already dispatched bounded request finish and retain its receipt.
                # Cancelling the bridge Future does not prove its HTTP coroutine settled.
                while not worker.done():
                    try:
                        await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                # Consume worker errors without replacing the original cancellation.
                if not worker.cancelled():
                    worker.exception()
                raise
        score = None
        failure = None
        if result["status"] != "completed":
            failure = "genehop_native:" + result["status"]
        scorable_status = result["status"] in {"completed", "model_budget", "step_budget"}
        if body.verifier_metadata["category"] != "Disease gene location":
            failure = failure or "genehop_native:manual_grading_required"
        elif scorable_status:
            tree = ast.parse((self.config.upstream_repo / "evaluate.py").read_text())
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_answer")
            ns = {}
            exec(compile(ast.Module(body=[node], type_ignores=[]), "creator-answer-parser", "exec"), ns)
            answer = ns["get_answer"](result["answer"] or "", "Disease gene location")
            gold = body.verifier_metadata["answer"]
            score = sum(x in answer for x in gold) / len(gold)
        response = NeMoGymResponse.model_validate(
            {
                "id": "resp_" + uuid4().hex,
                "object": "response",
                "created_at": time.time(),
                "model": MODEL,
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_" + uuid4().hex,
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": result["answer"] or "", "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            }
        )
        return VerifyResponse(
            **body.model_dump(by_alias=True),
            response=response,
            reward=score if score is not None else 0.0,
            raw_creator_score=score,
            mask_sample=score is None,
            failure_kind=failure,
            failure_reason=(
                "Unscored; inspect native status and creator manual-grading coverage"
                if score is None
                else "Benchmark budget exhausted; available answer scored with creator parser"
            )
            if failure
            else None,
            artifact_dir=str(out),
            native_status=result["status"],
        )


if __name__ == "__main__":
    GeneHopAgent.run_webserver()
