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


async def test_judge_format_qualification_preserves_creator_raw_score(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from nemo_gym.openai_utils import NeMoGymResponse
    from resources_servers.biomaze.app import Request, Verifier
    from resources_servers.biomaze.openended import parse_judge

    def response(text):
        return NeMoGymResponse.model_validate(
            {
                "id": "r",
                "created_at": 0,
                "model": "offline",
                "object": "response",
                "status": "completed",
                "parallel_tool_calls": False,
                "tools": [],
                "tool_choice": "auto",
                "output": [
                    {
                        "id": "m",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    }
                ],
            }
        )

    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Return Comment and Judge: Correct or Incorrect")
    verifier = Verifier()
    verifier.config = SimpleNamespace(
        judge_model_server=SimpleNamespace(name="judge"),
        judge_prompt_file=str(prompt),
        judge_model="offline",
        allowed_judge_models=["offline"],
    )
    verifier.server_client = object()
    body = Request(
        responses_create_params={"input": []},
        response=response("candidate"),
        verifier_metadata={"protocol": "openended", "question": "q", "answer": "a"},
    )
    for text, valid in [
        ("Comment: yes\nJudge: Correct", True),
        ("Judge: Incorrect", True),
        ("The answer is correct", False),
        ("nonsense", False),
        ("Judge: Correct\nextra", False),
        ("Judge: Incorrect\nJudge: Correct", False),
    ]:
        monkeypatch.setattr("resources_servers.biomaze.app.call_judge", AsyncMock(return_value=response(text)))
        result = await verifier.verify(body)
        assert result.reward == float(parse_judge(text))
        assert result.diagnostics["creator_raw_reward"] == result.reward
        assert result.diagnostics["judge_format_valid"] is valid
        assert result.mask_sample is (not valid)
        assert result.failure_kind == (None if valid else "judge_unparseable")

    # Empty solver answers remain model outcomes if the judge returns a valid verdict.
    body.response = response("")
    monkeypatch.setattr(
        "resources_servers.biomaze.app.call_judge", AsyncMock(return_value=response("Judge: Incorrect"))
    )
    result = await verifier.verify(body)
    assert result.reward == 0 and not result.mask_sample
    for judged in [response(""), response("Judge: Correct").model_copy(update={"status": "incomplete"})]:
        monkeypatch.setattr("resources_servers.biomaze.app.call_judge", AsyncMock(return_value=judged))
        result = await verifier.verify(body)
        assert result.mask_sample and result.failure_kind == "judge_failed"
        assert result.diagnostics["reward_is_serialization_placeholder"]
