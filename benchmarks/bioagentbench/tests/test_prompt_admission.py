# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from responses_api_agents.bioagent_native import app


def frozen_rows():
    path = Path(os.environ.get("BIOAGENT_TEST_SAMPLE", Path(__file__).parents[1] / "data/benchmark.jsonl"))
    if not path.is_file():
        pytest.skip("Exact prepared BioAgent ten-workflow fixture not present")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 10 and len({row["task_id"] for row in rows}) == 10
    return rows


class Admitted(Exception):
    pass


@pytest.mark.asyncio
async def test_all_frozen_prompts_normalize_without_changing_scientific_fields(monkeypatch):
    rows = frozen_rows()
    monkeypatch.setattr(app, "rows", lambda path: rows)

    def stop():
        raise Admitted("prompt checks passed; no runtime or model work")

    monkeypatch.setattr(app, "get_global_config_dict", stop)
    agent = SimpleNamespace(config=SimpleNamespace(metadata_path=Path("unused"), blocked_task_ids=[]))
    for row in rows:
        original = copy.deepcopy(row)
        body = app.RunRequest.model_validate(row)
        with pytest.raises(Admitted):
            await app.Agent.run(agent, None, body)
        assert row == original
        for change in ("content", "role", "extra", "image", "tool"):
            altered = copy.deepcopy(row)
            messages = altered["responses_create_params"]["input"]
            if change == "content":
                messages[0]["content"] += " changed"
            elif change == "role":
                messages[0]["role"] = "system"
            elif change == "extra":
                messages.append({"role": "user", "content": "extra"})
            elif change == "image":
                messages[0]["content"] = [
                    {"type": "input_image", "image_url": "https://example.invalid/image", "detail": "auto"}
                ]
            else:
                messages.append({"type": "function_call", "name": "tool", "arguments": "{}", "call_id": "call"})
            with pytest.raises(ValueError, match="Frozen task prompt"):
                await app.Agent.run(agent, None, app.RunRequest.model_validate(altered))


@pytest.mark.asyncio
async def test_conflicting_native_identity_refuses_before_artifacts(tmp_path, monkeypatch):
    rows = frozen_rows()
    monkeypatch.setattr(app, "rows", lambda path: rows)
    monkeypatch.setattr(
        app, "get_global_config_dict", lambda: {"provider_dispatch_guard": True, "policy_model_name": app.MODEL}
    )
    monkeypatch.setattr(app, "current_rollout_id", lambda: "context-episode")
    agent = SimpleNamespace(
        config=SimpleNamespace(metadata_path=Path("unused"), blocked_task_ids=[], output_root=tmp_path)
    )
    body = app.RunRequest.model_validate(rows[0] | {"_ng_rollout_id": "other-episode"})
    with pytest.raises(ValueError, match="identities disagree"):
        await app.Agent.run(agent, None, body)
    assert not list(tmp_path.iterdir())
