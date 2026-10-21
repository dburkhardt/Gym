# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from unittest.mock import MagicMock

import pytest

from nemo_gym.server_utils import ServerClient
from responses_api_agents.genehop_native.app import Config, GeneHopAgent, NativeLoop, RunRequest


def test_repeat_and_attempt_artifacts_are_distinct_and_immutable(tmp_path, monkeypatch):
    agent = GeneHopAgent(
        config=Config(
            host="127.0.0.1",
            port=46001,
            entrypoint="app.py",
            name="genehop",
            upstream_repo=tmp_path,
            output_root=tmp_path / "out",
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )
    calls = []

    def offline_run(self, question):
        calls.append(question)
        return {"status": "completed", "answer": "fixture", "calls": []}

    monkeypatch.setattr(NativeLoop, "run", offline_run)
    base = {
        "uuid": "same-task",
        "responses_create_params": {"input": [{"role": "user", "content": "Q"}]},
        "verifier_metadata": {"question": "Q"},
        "_ng_task_index": 4,
    }
    bodies = [
        RunRequest.model_validate(base | {"_ng_rollout_index": repeat, "_ng_attempt_index": attempt})
        for repeat, attempt in [(0, 0), (1, 0), (0, 1)]
    ]
    outputs = [agent.execute(body, model_call=lambda *_: pytest.fail("no model calls"))[0] for body in bodies]
    assert len(set(outputs)) == 3 and calls == ["Q"] * 3
    identities = [json.loads((out / "execution-identity.json").read_text()) for out in outputs]
    assert [item["capture_key"] for item in identities] == ["4-0", "4-1", "4-0-a1"]
    before = {str(p): p.read_bytes() for out in outputs for p in out.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):
        agent.execute(bodies[0], model_call=lambda *_: pytest.fail("no retry"))
    assert calls == ["Q"] * 3
    assert before == {str(p): p.read_bytes() for out in outputs for p in out.iterdir() if p.is_file()}
    for changes in [{"uuid": "../escape"}, {"_ng_rollout_id": "../../escape"}]:
        with pytest.raises(ValueError):
            agent.execute(
                RunRequest.model_validate(base | {"_ng_rollout_index": 0} | changes), model_call=lambda *_: None
            )
    assert len(list((tmp_path / "out").iterdir())) == 3


def test_exact_task_identity_and_context_only_guard_refusal(tmp_path, monkeypatch):
    from nemo_gym.rollout_correlation import rollout_context

    agent = GeneHopAgent(
        config=Config(
            host="127.0.0.1",
            port=46001,
            entrypoint="app.py",
            name="genehop",
            upstream_repo=tmp_path,
            output_root=tmp_path / "out",
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )
    monkeypatch.setattr(
        NativeLoop, "run", lambda self, question: {"status": "completed", "answer": "fixture", "calls": []}
    )
    base = {
        "responses_create_params": {"input": [{"role": "user", "content": "Q"}]},
        "verifier_metadata": {"question": "Q"},
        "_ng_rollout_id": "same-episode",
    }
    paths = [
        agent.execute(RunRequest.model_validate(base | {"uuid": task}), model_call=lambda *_: None)[0]
        for task in ["a:b", "a-b"]
    ]
    assert paths[0] != paths[1]
    agent.server_client.global_config_dict = {"provider_dispatch_guard": True, "observability_enabled": True}
    body = RunRequest.model_validate(
        {k: v for k, v in (base | {"uuid": "context-only"}).items() if k != "_ng_rollout_id"}
    )
    with rollout_context("context-episode"), pytest.raises(ValueError, match="same native identity"):
        agent.execute(body, model_call=lambda *_: pytest.fail("no provider call"))
    assert len(list((tmp_path / "out").iterdir())) == 2
