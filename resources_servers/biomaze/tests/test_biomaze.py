# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from resources_servers.biomaze.grading import grade


def test_controls_and_upstream_false_pass():
    assert grade("Yes", answer="Yes")["reward"] == 1
    assert grade("No", answer="Yes")["reward"] == 0
    assert grade("", answer="Yes")["reward"] == 0
    result = grade("", answer="No")
    assert result["reward"] == 1
    assert result["diagnostics"]["strict_valid"] is False
    assert grade("nonsense", answer="No")["diagnostics"]["strict_score"] == 0


def test_openended_parser():
    from resources_servers.biomaze.openended import parse_judge

    assert parse_judge("Comment: matches.\nJudge: Correct")
    assert not parse_judge("Judge: Incorrect")
    assert not parse_judge("")
    assert not parse_judge("Judge: Correct\nextra")


async def test_judge_model_is_configurable_and_campaign_allowlist_precedes_dispatch(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import pytest

    from resources_servers.biomaze.app import Verifier

    prompt = tmp_path / "prompt.txt"
    prompt.write_text("{question} {answer} {response}")
    verifier = Verifier()
    verifier.config = SimpleNamespace(
        judge_model_server=SimpleNamespace(name="judge"),
        judge_prompt_file=str(prompt),
        judge_model="offline-test-model",
        allowed_judge_models=["nvidia/nvidia/nemotron-3.5-super-vl"],
    )
    verifier.server_client = object()
    body = SimpleNamespace(
        verifier_metadata={"protocol": "openended", "question": "q", "answer": "a"},
        response=SimpleNamespace(output_text="a"),
        verification_id="persisted",
    )
    call = AsyncMock(side_effect=RuntimeError("mock reached"))
    monkeypatch.setattr("resources_servers.biomaze.app.call_judge", call)
    with pytest.raises(RuntimeError, match="campaign allowlist"):
        await verifier.verify(body)
    call.assert_not_awaited()
    verifier.config.allowed_judge_models = ["offline-test-model"]
    with pytest.raises(RuntimeError, match="mock reached"):
        await verifier.verify(body)
    assert call.call_args.kwargs["json"].model == "offline-test-model"
