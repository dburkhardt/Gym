# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Creator scoring with criterion-bound Gym transport, never an alternative rubric."""

import asyncio
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

from openai.types.chat import ChatCompletion

from nemo_gym.provider_dispatch import configuration_digest
from nemo_gym.server_utils import request


MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"
SOURCE_COMMIT = "23e0a589321e449727c34aef107c49adcf977457"
SOURCE_HASHES = {
    "evaluation/score.py": "57320c35099b6089305913ad3d72bf246ee7cb2814170a84d363853366b87d51",
    "evaluation/config.py": "f45781b6009ddfb3073683f5d029b40a7b413757c63ec91e8eff0d8898d9ed65",
    "evaluation/utils.py": "199771f00e0635ccf0c812d9c89a7665312d95462b125e39dfabd9e7c9534271",
    "evaluation/__init__.py": "f6c6945b884630d0ad85fc3608e9d248f16cdbb81ff51507ad2b83c07f3a3c71",
}


class CriterionTransport:
    """One creator criterion; parser retries are distinct, transport retries are forbidden."""

    def __init__(self, send, loop, lineage, index, digest):
        self.send, self.loop = send, loop
        self.lineage, self.index, self.digest = lineage, index, digest
        self.attempt = 0
        self.lock = threading.Lock()
        self.pending = False
        self.failed = None
        self.requests = []

    def create(self, **payload):
        with self.lock:
            if self.pending or self.failed:
                raise RuntimeError("Criterion transport unresolved; creator retry cannot resend")
            if payload.get("model") != MODEL:
                raise ValueError("ResearchClaw judge requires authorized model")
            self.pending = True
            ordinal = self.attempt
            self.attempt += 1
        headers = {
            "x-nemo-gym-logical-call-id": f"researchclaw:{self.lineage}:criterion:{self.index}:try:{ordinal}",
            "x-nemo-gym-call-config-sha256": self.digest,
        }
        try:
            response = asyncio.run_coroutine_threadsafe(self.send(payload, headers), self.loop).result()
            if response.model != MODEL or not response.choices or response.choices[0].finish_reason != "stop":
                raise ValueError("Incomplete or wrong-model ResearchClaw judge response")
            with self.lock:
                self.requests.append(
                    {"logical_call_id": headers["x-nemo-gym-logical-call-id"], "response_id": response.id}
                )
            return response
        except BaseException as error:
            with self.lock:
                self.failed = type(error).__name__ + ": " + str(error)
            raise
        finally:
            with self.lock:
                self.pending = False


async def score(source: Path, workspace: Path, endpoint: str, lineage: str, expected_task: str) -> dict:
    if "/ng-rollout/" not in endpoint or not endpoint.endswith("/v1"):
        raise ValueError("ResearchClaw requires guarded Gym model capture route")
    for name, digest in SOURCE_HASHES.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Pinned creator source mismatch: {name}")
    if importlib.metadata.version("structai") != "0.1.23":
        raise ValueError("Creator parser requires structai0.1.23")
    meta = json.loads((workspace / "_meta.json").read_text())
    if meta.get("task_id") != expected_task:
        raise ValueError("Submitted workspace task identity mismatch")
    checklist = json.loads((source / "tasks" / expected_task / "target_study/checklist.json").read_text())
    if not checklist:
        raise ValueError("Empty checklist")
    os.environ.update(JUDGE_MODEL_NAME=MODEL, JUDGE_API_KEY="gym-internal", JUDGE_API_BASE=endpoint)
    sys.path.insert(0, str(source.resolve()))
    creator = importlib.import_module("evaluation.score")
    if Path(creator.__file__).resolve() != (source / "evaluation/score.py").resolve():
        raise ValueError("Unexpected creator module already imported")
    original_agent, original_threads = creator.LLMAgent, creator.multi_thread
    context = threading.local()
    transports, parsed = {}, {}
    loop = asyncio.get_running_loop()
    digest = configuration_digest(
        {
            "creator": SOURCE_HASHES,
            "structai": "0.1.23",
            "checklist": checklist,
            "model": MODEL,
            "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )

    async def send(payload, headers):
        response = await request(
            "POST",
            endpoint + "/chat/completions",
            json=payload,
            headers=headers,
            _max_num_tries=1,
            _max_connection_retries=0,
            _resend_after_disconnect=False,
        )
        try:
            response.raise_for_status()
            return ChatCompletion.model_validate(await response.json())
        finally:
            response.release()

    class BoundAgent:
        def __init__(self, **settings):
            self.settings = settings

        def __call__(self, *args, **kwargs):
            index = context.index
            transport = CriterionTransport(send, loop, lineage, index, digest)
            transports[index] = transport
            # Keep creator prompt construction, image encoding, timeout and parsing exactly in structai.
            agent = original_agent(**self.settings)
            agent.client.close()
            agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=transport.create)))
            result = agent(*args, **kwargs)
            parsed[index] = result
            if transport.pending or transport.failed:
                raise RuntimeError("Criterion provider outcome unresolved: " + str(transport.failed))
            return result

    def bound_threads(inputs, function, **kwargs):
        def bound(index, item_data):
            context.index = index
            return function(index=index, item_data=item_data)

        return original_threads(inputs, bound, **kwargs)

    creator.LLMAgent, creator.multi_thread = BoundAgent, bound_threads
    try:
        native = await asyncio.to_thread(creator.score_workspace, workspace)
    finally:
        creator.LLMAgent, creator.multi_thread = original_agent, original_threads
    failures = {}
    for index in range(len(checklist)):
        value = parsed.get(index)
        transport = transports.get(index)
        if not transport or transport.pending or transport.failed:
            failures[index] = "missing or unresolved provider outcome"
        elif (
            not isinstance(value, dict)
            or isinstance(value.get("score"), bool)
            or not isinstance(value.get("score"), (int, float))
            or not math.isfinite(value["score"])
            or not 0 <= value["score"] <= 100
        ):
            failures[index] = "invalid parsed native judge score"
    if native.get("error") or len(native.get("items", [])) != len(checklist):
        failures["native"] = native.get("error", "incomplete checklist coverage")
    return {
        "native": native,
        "failures": failures,
        "reward": None if failures else native["total_score"] / 100,
        "criterion_calls": {index: t.requests for index, t in transports.items()},
        "configuration_sha256": digest,
    }


if __name__ == "__main__":
    result = asyncio.run(score(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]))
    Path(sys.argv[6]).write_text(json.dumps(result, indent=2) + "\n")
    if result["failures"]:
        raise SystemExit(2)
