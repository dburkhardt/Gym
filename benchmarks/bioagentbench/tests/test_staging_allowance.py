# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from pydantic import ValidationError

from responses_api_agents.bioagent_native import app


def config(tmp_path, **extra):
    manifest = tmp_path / "assets.json"
    manifest.write_text("{}")
    return app.Config(
        host="127.0.0.1",
        port=1,
        entrypoint="app.py",
        name="test",
        native_python=Path(sys.executable),
        creator_root=tmp_path,
        metadata_path=tmp_path / "metadata",
        asset_root=tmp_path,
        asset_manifest=manifest,
        image="sha256:fixture",
        output_root=tmp_path / "out",
        **extra,
    )


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_invalid_operational_allowance_rejected(tmp_path, value):
    with pytest.raises(ValidationError):
        config(tmp_path, staging_allowance_seconds=value)


@pytest.mark.asyncio
@pytest.mark.parametrize("allowance, expected", [(0, 25980), (3600, 29580)])
async def test_actual_dispatch_timeout_changes_without_science_changes(tmp_path, monkeypatch, allowance, expected):
    cfg = config(tmp_path, staging_allowance_seconds=allowance)
    body = app.RunRequest.model_validate(
        {"responses_create_params": {"input": []}, "verifier_metadata": {"id": "deseq"}}
    )
    monkeypatch.setattr(
        app,
        "rows",
        lambda _: [
            {
                "task_id": "deseq",
                "verifier_metadata": body.verifier_metadata,
                "responses_create_params": body.responses_create_params.model_dump(),
            }
        ],
    )
    monkeypatch.setattr(
        app, "get_global_config_dict", lambda: {"provider_dispatch_guard": True, "policy_model_name": app.MODEL}
    )
    monkeypatch.setattr(app, "get_first_server_config_dict", lambda *_: {"host": "127.0.0.1", "port": 9})
    monkeypatch.setattr(app, "current_rollout_id", lambda: "test")
    from benchmarks.bioagentbench import scorer_environment

    monkeypatch.setattr(scorer_environment, "binding", lambda *_: None)
    packets = []
    monkeypatch.setattr(app, "publish", lambda _, packet: packets.append(packet))

    @asynccontextmanager
    async def workspace(_):
        yield object()

    monkeypatch.setattr(app, "scoring_workspace", workspace)

    class DispatchReached(Exception):
        pass

    async def execute(self, handle, command, **kwargs):
        assert kwargs["timeout_s"] == expected
        assert kwargs["terminate_background_children"] is True
        raise DispatchReached

    monkeypatch.setattr(app.LocalProvider, "exec", execute)
    agent = app.Agent.model_construct(config=cfg)
    with pytest.raises(DispatchReached):
        await agent.run(None, body)
    packet = packets[0]
    assert packet["operational_worker_timeout_seconds"] == expected
    assert packet["native_config"]["timeout"] == {"setup": 3600, "rollout": 14400, "finalize": 3600, "scoring": 3600}
    assert packet["native_config"]["harness"]["runtime"]["cpu"] == 2
    assert packet["native_config"]["sampling"]["max_tokens"] == 32768
