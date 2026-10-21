# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body
from responses_api_agents.bioagent_native.app import RunRequest, VerifyResponse


def test_native_collection_identity_survives_request_and_result_schema():
    payload = {
        "responses_create_params": {"input": []},
        "verifier_metadata": {"id": "cystic-fibrosis"},
        "_ng_task_index": 7,
        "_ng_rollout_index": 2,
        "_ng_attempt_index": 3,
    }
    body = RunRequest.model_validate(payload)
    assert maybe_rollout_id_from_run_body(body) == "7-2-a3"
    assert body.model_dump()["_ng_attempt_index"] == 3
    result = VerifyResponse.model_validate(
        {
            **body.model_dump(),
            "reward": 0,
            "native_metrics": {},
            "artifact_dir": "fixture",
            "response": {
                "id": "fixture",
                "object": "response",
                "created_at": 0,
                "model": "fixture",
                "output": [],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            },
        }
    )
    assert maybe_rollout_id_from_run_body(result) == "7-2-a3"
    assert result.model_dump()["_ng_task_index"] == 7


def test_explicit_identity_retains_attempt_suffix():
    body = RunRequest.model_validate(
        {
            "responses_create_params": {"input": []},
            "verifier_metadata": {"id": "task"},
            "_ng_rollout_id": "owned-attempt",
            "_ng_attempt_index": 2,
        }
    )
    assert maybe_rollout_id_from_run_body(body) == "owned-attempt-a2"
