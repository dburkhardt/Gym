# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from unittest.mock import AsyncMock

import pytest

from responses_api_agents.scicode_agent.tests import scientific_recovery_control as control


def test_wrong_scientific_identity_refused(tmp_path):
    data = tmp_path / "data.h5"
    data.write_bytes(b"not the creator reference")
    with pytest.raises(ValueError, match="immutable"):
        control.validate_inputs(data, "arbitrary:latest")
    with pytest.raises(ValueError, match="checksum"):
        control.validate_inputs(data, control.IMAGE)


def receipt(tmp_path, *, run="a" * 32, sandbox=True):
    path = tmp_path / "baseline-execution" / run / "lifecycle.json"
    path.parent.mkdir(parents=True)
    proof = {"run": run, "image": control.IMAGE, "cleanup_confirmed": True, "program_sha256": "b" * 64}
    if sandbox:
        proof["sandbox"] = {"ownership_labels": {"nemo-gym.workload": "scicode-verifier", "nemo-gym.run": run}}
    path.write_text(json.dumps(proof))
    return path


@pytest.mark.asyncio
async def test_precreation_receipt_reconciled_only_after_processes_settle(tmp_path):
    receipt(tmp_path, sandbox=False)
    provider = AsyncMock()
    provider.reconcile_owned.return_value = {"all_removed": True}
    with pytest.raises(RuntimeError, match="not settled"):
        await control.reconcile_containers(tmp_path, {"cleanup_confirmed": False}, provider)
    provider.reconcile_owned.assert_not_awaited()
    result = await control.reconcile_containers(tmp_path, {"cleanup_confirmed": True, "error": None}, provider)
    assert result == [{"all_removed": True}]
    provider.reconcile_owned.assert_awaited_once_with(
        {"nemo-gym.workload": "scicode-verifier", "nemo-gym.run": "a" * 32}
    )


@pytest.mark.asyncio
async def test_mismatched_identity_cannot_cleanup_other_run(tmp_path):
    path = receipt(tmp_path)
    proof = json.loads(path.read_text())
    proof["sandbox"]["ownership_labels"]["nemo-gym.run"] = "b" * 32
    path.write_text(json.dumps(proof))
    provider = AsyncMock()
    with pytest.raises(ValueError, match="labels differ"):
        await control.reconcile_containers(tmp_path, {"cleanup_confirmed": True}, provider)
    provider.reconcile_owned.assert_not_awaited()


def test_full_scientific_outcome_and_program_comparison(tmp_path):
    a = receipt(tmp_path)
    b = tmp_path / "restore-execution" / ("c" * 32) / "lifecycle.json"
    b.parent.mkdir(parents=True)
    b.write_bytes(a.read_bytes())
    row = {
        "reward": 1,
        "step_results": [True],
        "measurement_validity": [],
        "num_steps_passed": 1,
        "num_steps_total": 1,
        "problem_accuracy": True,
        "subtask_accuracy": 1,
    }
    for phase, path in [("baseline", a), ("restore", b)]:
        (tmp_path / f"{phase}-scientific.json").write_text(
            json.dumps(
                {
                    **row,
                    "step_execution": [
                        {
                            "receipt_path": str(path),
                            "step_number": "20.1",
                            "passed": True,
                            "error": "",
                            "timed_out": False,
                            "return_code": 0,
                        }
                    ],
                }
            )
        )
    assert control.compare_science(tmp_path)["scientific_outcomes_equal"]
    changed = json.loads((tmp_path / "restore-scientific.json").read_text())
    changed["step_results"] = [False]
    (tmp_path / "restore-scientific.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="step_results"):
        control.compare_science(tmp_path)


def test_ordered_step_identity_cannot_be_relabelled(tmp_path):
    a = receipt(tmp_path)
    b = tmp_path / "restore-execution" / ("c" * 32) / "lifecycle.json"
    b.parent.mkdir(parents=True)
    b.write_bytes(a.read_bytes())
    row = {
        "reward": 1,
        "step_results": [True],
        "measurement_validity": [],
        "num_steps_passed": 1,
        "num_steps_total": 1,
        "problem_accuracy": True,
        "subtask_accuracy": 1,
    }
    for phase, path, step in [("baseline", a, "20.1"), ("restore", b, "20.2")]:
        execution = {
            "receipt_path": str(path),
            "step_number": step,
            "passed": True,
            "error": "",
            "timed_out": False,
            "return_code": 0,
        }
        (tmp_path / f"{phase}-scientific.json").write_text(json.dumps({**row, "step_execution": [execution]}))
    with pytest.raises(ValueError, match="labels/status"):
        control.compare_science(tmp_path)


def test_scientific_top_level_requires_supervised_entrypoint(tmp_path):
    import os
    import subprocess
    import sys

    script = control.Path(control.__file__).with_name("saved_task_process_control.py")
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(tmp_path / "unused.json"),
            "--captures",
            str(tmp_path / "unused.jsonl"),
            "--output",
            str(tmp_path / "output"),
            "--test-data",
            str(tmp_path / "unused.h5"),
            "--sandbox-image",
            control.IMAGE,
        ],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([os.environ.get("PYTHONPATH", ""), str(script.parent.parent)]),
        },
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 2, completed.stderr
    assert "Use scientific_recovery_control.py" in completed.stderr
    assert not (tmp_path / "output").exists()
