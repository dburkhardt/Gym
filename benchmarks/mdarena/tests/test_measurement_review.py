# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from nemo_gym.reward_profile import select_measured
from responses_api_agents.harbor_agent.measurement_review import apply_measurement_review


HOOK = "benchmarks.mdarena.measurement_review:review"


def test_saved_native_xml_raw_scores_leave_quality_aggregate(tmp_path):
    archive = (
        Path(__file__).resolve().parents[5]
        / "outputs/broad-eval-campaign/foundation/mdarena-gym/xml-native-complete.tar.gz"
    )
    if not archive.exists():
        pytest.skip("Saved native XML artifact unavailable")
    before = hashlib.sha256(archive.read_bytes()).hexdigest()
    trial = None
    with tarfile.open(archive) as source:
        for item in source:
            if not item.isfile():
                continue
            if (
                "/02_conc_diff_sys_traj__" in item.name
                and item.name.endswith("/result.json")
                and "/verifier/" not in item.name
            ):
                trial = json.load(source.extractfile(item))
            if "/verifier/native-controller/" in item.name and item.name.endswith(
                ("reward-details.json", "-request.json", "identity.json")
            ):
                destination = tmp_path / item.name[item.name.index("/verifier/") + 1 :]
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.extractfile(item).read())
    assert trial is not None and trial["verifier_result"]["rewards"] == {"correctness": 1.0, "process": 1.0}
    original = copy.deepcopy(trial)
    annotated, kind, reason = apply_measurement_review(
        trial, trial_dir=tmp_path, instance_id="mdarena::02_conc_diff_sys_traj", import_path=HOOK
    )
    assert trial == original and annotated["verifier_result"] == original["verifier_result"]
    assert kind == "harbor:scientific_measurement_invalid"
    assert "irrelevant nonempty" in reason and "no recorded" in reason
    row = {"_ng_task_index": 3, "_ng_rollout_index": 0}
    saved = {**row, "reward": 1.0, "mask_sample": True, "metadata": annotated}
    measured_rows, measured, masked, coverage = select_measured([row], [saved])
    assert measured_rows == measured == [] and len(masked) == 1 and masked[0]["reward"] == 1.0
    assert coverage["coverage/measured_rollouts"] == 0 and coverage["coverage/masked_rollouts"] == 1
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == before
    identity_path = tmp_path / "verifier/native-controller/identity.json"
    identity_path.write_text(json.dumps({"configuration_sha256": "changed"}))
    for path in (tmp_path / "verifier/native-controller/controller/judge-receipts").glob("*/*-request.json"):
        request = json.loads(path.read_text())
        request["input"].append(
            {"role": "user", "content": "Actual workspace observations supplied inline in a changed protocol."}
        )
        path.write_text(json.dumps(request))
    changed, _, _ = apply_measurement_review(
        trial, trial_dir=tmp_path, instance_id="mdarena::02_conc_diff_sys_traj", import_path=HOOK
    )
    assert all(x["status"] == "unresolved" for x in changed["measurement_review"]["metrics"].values())


def test_other_tasks_and_unconfigured_model_zero_not_excluded(tmp_path):
    native = {
        "verifier_result": {"rewards": {"reward": 0.0}},
        "exception_info": {"exception_type": "AgentTimeoutError"},
    }
    assert apply_measurement_review(native, trial_dir=tmp_path, instance_id="other::task", import_path=None) == (
        native,
        None,
        None,
    )
    reviewed, kind, reason = apply_measurement_review(
        native, trial_dir=tmp_path, instance_id="mdarena::different-task", import_path=HOOK
    )
    assert kind is reason is None and reviewed["verifier_result"] == native["verifier_result"]


def test_missing_evidence_preserves_raw_score_and_fails_closed(tmp_path):
    native = {"verifier_result": {"rewards": {"correctness": 1.0, "process": 1.0}}}
    annotated, kind, reason = apply_measurement_review(
        native, trial_dir=tmp_path, instance_id="mdarena::02_conc_diff_sys_traj", import_path=HOOK
    )
    assert kind == "harbor:measurement_review_error" and reason
    assert annotated["verifier_result"] == native["verifier_result"]


@pytest.mark.asyncio
async def test_harbor_response_propagates_mask(monkeypatch):
    import responses_api_agents.harbor_agent.app as app
    from responses_api_agents.harbor_agent.tests.test_app import _harbor_run_mocks, _make_run_request, _make_server

    def rejected(trial, **kwargs):
        return (
            {**trial, "measurement_review": {"reason": "controlled invalid measurement"}},
            "harbor:scientific_measurement_invalid",
            "controlled invalid measurement",
        )

    monkeypatch.setattr(app, "apply_measurement_review", rejected)
    server = _make_server(harbor_measurement_review_import_path=HOOK)
    with _harbor_run_mocks():
        response = await server.run(_make_run_request())
    assert response.mask_sample and response.failure_kind == "harbor:scientific_measurement_invalid"
    assert response.metadata["verifier_result"]["rewards"]


def test_review_cannot_mutate_native_scores(tmp_path, monkeypatch):
    import benchmarks.mdarena.measurement_review as module

    def mutated(**kwargs):
        kwargs["trial_result"]["verifier_result"]["rewards"]["reward"] = 0
        return {"version": "control", "metrics": {}}

    monkeypatch.setattr(module, "review", mutated)
    native = {"verifier_result": {"rewards": {"reward": 1.0}}}
    annotated, _, _ = apply_measurement_review(native, trial_dir=tmp_path, instance_id="control", import_path=HOOK)
    assert native["verifier_result"]["rewards"]["reward"] == annotated["verifier_result"]["rewards"]["reward"] == 1.0
