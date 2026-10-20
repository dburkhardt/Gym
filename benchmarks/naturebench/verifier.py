# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native separate verifier with a bounded authenticated Gym judge bridge."""

import asyncio
import hashlib
import json
import secrets
import sys
import uuid
from pathlib import Path

import aiohttp
from aiohttp import web
from harbor.verifier.verifier import Verifier

from nemo_gym.provider_dispatch import configuration_digest
from responses_api_agents.harbor_agent.custom_verifiers.rewardkit_agent.verifier import _publish, _settle


MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"


class NatureBenchVerifier(Verifier):
    def __init__(self, *, gym_model_api_base, gym_model_name, bridge_host, **kwargs):
        super().__init__(**kwargs)
        if gym_model_name != MODEL or "/ng-rollout/" not in gym_model_api_base:
            raise ValueError("NatureBench judge requires exact model and guarded rollout route")
        if self.task.config.verifier.environment_mode != "separate":
            raise ValueError("NatureBench judge must remain in separate verifier environment")
        self.endpoint = gym_model_api_base.rstrip("/") + "/responses"
        self.bridge_host = bridge_host
        self.lock = asyncio.Lock()

    async def verify(self):
        tests = self.task.paths.tests_dir
        if any(p.is_symlink() for p in tests.rglob("*")):
            raise ValueError("Verifier source symlinks are not supported")
        judge_source = tests / "judge_readme.py"
        if (
            not judge_source.is_file()
            or "Campaign judge model/provider is not allowed" not in judge_source.read_text()
        ):
            raise ValueError("Prepare the explicit Nemotron judge variant before verification")
        identity = configuration_digest(
            {
                "task": self.task.config.model_dump(mode="json"),
                "tests": {
                    str(p.relative_to(tests)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in tests.rglob("*")
                    if p.is_file()
                },
                "bridge": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
        )
        out = self.trial_paths.verifier_dir / "gym-judge"
        out.mkdir(parents=True, exist_ok=False)  # Fresh regrade is a new scorer record.
        lineage = uuid.uuid4().hex
        _publish(
            out / "identity.json",
            {
                "lineage": lineage,
                "configuration_sha256": identity,
                "recovery": "existing verifier attempt fails closed",
            },
        )
        token = secrets.token_urlsafe(32)
        recorded_request = None

        async def judge(request):
            nonlocal recorded_request
            if request.headers.get("Authorization") != "Bearer " + token:
                raise web.HTTPUnauthorized()
            payload = await request.json()
            if payload.get("model") != MODEL or payload.get("stream"):
                raise web.HTTPBadRequest(text="Exact model, nonstream judge only")
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            async with self.lock:
                if recorded_request is not None and recorded_request != digest:
                    raise web.HTTPConflict(text="Creator protocol permits one judge request per verification")
                recorded_request = digest
                _publish(out / "intent.json", {"request_sha256": digest, "logical_call_id": lineage + ":judge:0"})
                headers = {
                    "x-nemo-gym-logical-call-id": lineage + ":judge:0",
                    "x-nemo-gym-call-config-sha256": identity,
                }
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200)) as session:
                    async with session.post(self.endpoint, json=payload, headers=headers) as response:
                        body = await response.read()
                        if response.status == 200:
                            _publish(out / "response.json", json.loads(body))
                        return web.Response(body=body, status=response.status, content_type="application/json")

        app = web.Application(client_max_size=32 * 1024 * 1024)
        app.router.add_post("/v1/responses", judge)
        runner = web.AppRunner(app, shutdown_timeout=1205)
        await runner.setup()
        try:
            site = web.TCPSite(runner, self.bridge_host, 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            base = f"http://{self.bridge_host}:{port}/v1"
            self.override_env.update(
                JUDGE_MODEL=MODEL, JUDGE_API_KEY=token, JUDGE_BASE_URL=base, CAMPAIGN_JUDGE_BASE_URL=base
            )
            return await super().verify()
        finally:
            original = sys.exc_info()[1]
            try:
                await _settle(asyncio.create_task(runner.cleanup()))
            except BaseException:
                if original is None:
                    raise
