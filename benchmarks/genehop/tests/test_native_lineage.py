# SPDX-License-Identifier: Apache-2.0
from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body
from responses_api_agents.genehop_native.app import RunRequest, VerifyResponse


def test_native_request_and_result_retain_rollout_lineage():
    raw = {
        "responses_create_params": {"input": []},
        "uuid": "frozen-task",
        "verifier_metadata": {},
        "_ng_task_index": 4,
        "_ng_rollout_index": 2,
    }
    request = RunRequest.model_validate(raw)
    result = VerifyResponse.model_validate(
        {
            **request.model_dump(),
            "response": {
                "id": "control",
                "object": "response",
                "created_at": 0,
                "model": "control",
                "status": "completed",
                "output": [],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            },
            "reward": 0.0,
            "artifact_dir": "control",
            "native_status": "completed",
        }
    )
    for value in (request, result):
        assert value.model_dump()["_ng_task_index"] == 4
        assert value.model_dump()["_ng_rollout_index"] == 2
        assert maybe_rollout_id_from_run_body(value) == maybe_rollout_id_from_run_body(raw)
        assert maybe_rollout_id_from_run_body(value) is not None


import importlib

import pytest


@pytest.mark.parametrize(
    "module,request_name,result_name,extra",
    [
        ("genotex", "GenoTEXRunRequest", "GenoTEXResponse", {"elapsed_seconds": 0}),
        ("flyaoc_native", "RunRequest", "VerifyResponse", {"metrics": {}, "ng_trajectory": {}}),
        ("scivis_native", "RunRequest", "RunResponse", {}),
        (
            "bixbench3",
            "BixBench3Request",
            "BixBench3Result",
            {"task_id": "frozen", "settings_file": "control", "grade_path": "control", "inspect_log_dir": "control"},
        ),
        ("promptbio_bench", "PromptBioRequest", "PromptBioResult", {}),
    ],
)
def test_campaign_request_result_lineage_roundtrip(module, request_name, result_name, extra):
    app = importlib.import_module(f"responses_api_agents.{module}.app")
    raw = {
        "responses_create_params": {"input": []},
        "verifier_metadata": {},
        "_ng_task_index": 4,
        "_ng_rollout_index": 2,
        **extra,
    }
    request = getattr(app, request_name).model_validate(raw)
    response = {
        "id": "control",
        "object": "response",
        "created_at": 0,
        "model": "control",
        "status": "completed",
        "output": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }
    result = getattr(app, result_name).model_validate(
        {**request.model_dump(), "response": response, "reward": 0.0, "artifact_dir": "control", **extra}
    )
    restored = getattr(app, result_name).model_validate_json(result.model_dump_json())
    for value in (request, result, restored):
        assert value.model_dump()["_ng_task_index"] == 4
        assert value.model_dump()["_ng_rollout_index"] == 2
        assert maybe_rollout_id_from_run_body(value) == maybe_rollout_id_from_run_body(raw)
