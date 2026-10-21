# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from pathlib import Path

from benchmarks.biabench.measurement_review import review


def test_partial_process_coverage_retains_only_bound_outcome(tmp_path):
    manifest_bytes = (Path(__file__).parents[1] / "frozen_sources.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    path = tmp_path / "verifier/biabench/managed-process/score"
    path.mkdir(parents=True)
    record = {
        "status": "completed",
        "creator_import_sha256": manifest["creator_files"],
        "native_scores": {"result_score": 0.79, "checklist_score": 0.6},
        "measurement_validity": {"process": {"status": "unresolved", "reason": "Four references cannot be encoded."}},
    }
    (path / "result.json").write_text(json.dumps(record))
    (path / "identity.json").write_text(
        json.dumps({"source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()})
    )
    result = {"verifier_result": {"rewards": {"reward": 0.79, "outcome": 0.79, "process": 0.6}}}
    reviewed = review(trial_dir=tmp_path, instance_id="biabench::" + manifest["task_id"], trial_result=result)
    assert reviewed["metrics"]["outcome"]["status"] == "valid"
    assert reviewed["metrics"]["process"]["status"] == "unresolved"
    assert result["verifier_result"]["rewards"]["process"] == 0.6
    result["verifier_result"]["rewards"]["outcome"] = 0.8
    assert (
        review(trial_dir=tmp_path, instance_id="biabench::" + manifest["task_id"], trial_result=result)["metrics"][
            "outcome"
        ]["status"]
        == "unresolved"
    )
    del result["verifier_result"]["rewards"]["process"]
    assert (
        review(trial_dir=tmp_path, instance_id="biabench::" + manifest["task_id"], trial_result=result)["metrics"][
            "reward"
        ]["status"]
        == "unresolved"
    )
    assert (
        review(trial_dir=tmp_path, instance_id="wrong", trial_result=result)["metrics"]["outcome"]["status"]
        == "unresolved"
    )
