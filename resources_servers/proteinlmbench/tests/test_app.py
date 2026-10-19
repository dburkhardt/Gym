# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmarks.proteinlmbench.prepare import convert
from nemo_gym.verifier_fixture import exercise_verifier_fixture
from resources_servers.proteinlmbench.app import VERIFIER_FIXTURE, ProteinLMBenchRequest, ProteinLMBenchVerifier
from resources_servers.proteinlmbench.grading import grade


@pytest.mark.parametrize(
    ("text", "reward", "strict"),
    [
        ("The correct option is Option 2.", 1, 2),
        ("option 1", 0, 1),
        ("", 0, None),
        ("unsure", 0, None),
        ("option 20", 0, 20),
        ("Option 2 or 3", 1, None),
        ("<think>Option 1</think>Option 2", 1, 2),
        ("<think>Option 2", 0, None),
        ("reasoning 1</think>Option 2", 1, 2),
        ("2", 1, 2),
        ("There are 2 proteins. Option 1.", 1, None),
    ],
)
def test_grading(text, reward, strict):
    result = grade(text, expected_answer=2, option_count=6)
    assert result["reward"] == reward
    assert result["strict_answer"] == strict
    assert result["out_of_range"] == (text == "option 20")


def test_multidigit():
    assert grade("option 10", expected_answer=10, option_count=10)["reward"] == 1


def test_preparation_and_hidden_gold():
    source = [
        {
            "question": "Synthetic question",
            "options": ["option 1: a", "option 2: b"],
            "answer": "option 2",
            "explanation": "SECRET",
        }
    ]
    rows = convert(source)
    assert rows == convert(source)
    assert rows[0]["id"] == "proteinlmbench-0000"
    prompt = rows[0]["responses_create_params"]["input"][0]["content"]
    assert "SECRET" not in prompt
    assert "option 2: b" in prompt
    assert rows[0]["verifier_metadata"] == {"expected_answer": 2, "option_count": 2}
    with pytest.raises(ValueError):
        convert([{**source[0], "answer": "option 3"}])


@pytest.mark.asyncio
async def test_verify_and_missing_output():
    case = json.loads((Path(__file__).parent / "fixture_cases.jsonl").read_text().splitlines()[0])
    body = ProteinLMBenchRequest.model_validate(case["request"])
    server = ProteinLMBenchVerifier()
    assert (await server.verify(body)).reward == 1
    body.response.output = []
    assert (await server.verify(body)).reward == 0
    with pytest.raises(ValidationError):
        ProteinLMBenchRequest.model_validate({})


@pytest.mark.asyncio
async def test_fixture():
    await exercise_verifier_fixture(
        VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="unknown"
    )


def test_prepare_checksum_and_count(tmp_path, monkeypatch):
    import hashlib
    import io

    from benchmarks.proteinlmbench import prepare as module

    row = {"question": "synthetic", "options": ["option 1: a", "option 2: b"], "answer": "option 2"}
    payload = json.dumps([row] * 944).encode()
    monkeypatch.setattr(module, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    monkeypatch.setattr(module, "OUTPUT", tmp_path / "tasks.jsonl")
    monkeypatch.setattr(module, "SHA256", "wrong")
    with pytest.raises(ValueError, match="checksum"):
        module.prepare()
    monkeypatch.setattr(module, "SHA256", hashlib.sha256(payload).hexdigest())
    assert len(module.prepare().read_text().splitlines()) == 944
    payload = b"[]"
    monkeypatch.setattr(module, "SHA256", hashlib.sha256(payload).hexdigest())
    with pytest.raises(ValueError, match="944"):
        module.prepare()


def test_bad_metadata():
    from resources_servers.proteinlmbench.task_data import TaskData

    with pytest.raises(ValidationError):
        TaskData(expected_answer=4, option_count=2)
