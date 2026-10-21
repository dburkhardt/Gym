# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in Linux/Docker saved-response recovery with the real pinned SciCode verifier.

This exercises native components/checkpoints, not full CLI or mutable tool restore.
Never downloads images/data or contacts a model. Run in an explicitly admitted runtime.
"""

import argparse
import asyncio
import hashlib
import json
import re
import signal
import subprocess
import sys
from pathlib import Path

from nemo_gym.agent_utils import process_supervisor
from nemo_gym.sandbox import create_provider


IMAGE = "sha256:2fae326d6d1d11e55b828f200bfcd349bf392e070fc69e7721a87d152db97918"
DATA_SHA256 = "48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890"


def validate_inputs(data, image):
    if image != IMAGE:
        raise ValueError("Scientific recovery requires the qualified immutable SciCode image")
    with data.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != DATA_SHA256:
            raise ValueError("Scientific recovery H5 checksum mismatch")


def compare_science(result):
    a = json.loads((result / "baseline-scientific.json").read_text())
    b = json.loads((result / "restore-scientific.json").read_text())
    keys = [
        "reward",
        "step_results",
        "measurement_validity",
        "num_steps_passed",
        "num_steps_total",
        "problem_accuracy",
        "subtask_accuracy",
    ]
    for key in keys:
        if a[key] != b[key]:
            raise ValueError(f"Restored scientific outcome differs: {key}")
    step_keys = ("step_number", "passed", "error", "timed_out", "return_code")

    def statuses(row):
        return [{key: item[key] for key in step_keys} for item in row["step_execution"]]

    if statuses(a) != statuses(b):
        raise ValueError("Ordered scientific step labels/status differ")
    programs = []
    for phase in ("baseline", "restore"):
        entries = a["step_execution"] if phase == "baseline" else b["step_execution"]
        hashes = []
        for item in entries:
            receipt = Path(item["receipt_path"])
            if receipt.resolve().parent.parent != (result / f"{phase}-execution").resolve():
                raise ValueError("Scientific receipt escaped its owned phase")
            proof = json.loads(receipt.read_text())
            if proof["image"] != IMAGE or not proof["cleanup_confirmed"]:
                raise ValueError("Scientific runtime/cleanup mismatch")
            hashes.append(proof["program_sha256"])
        programs.append(hashes)
    if not programs[0] or programs[0] != programs[1]:
        raise ValueError("Generated scientific programs differ or are absent")
    return {"scientific_outcomes_equal": True, "program_hashes": programs[0], "metrics": {key: a[key] for key in keys}}


async def reconcile_containers(result, supervisor, provider):
    # No container reconciliation while a creator process can still race publication.
    if supervisor.get("cleanup_confirmed") is not True or supervisor.get("error") is not None:
        raise RuntimeError("Owned processes are not settled; container cleanup refused")
    receipts = []
    for path in sorted(result.glob("*-execution/*/lifecycle.json")):
        if path.is_symlink() or path.parent.is_symlink() or path.parent.parent.is_symlink():
            raise ValueError("Symlink lifecycle receipt refused")
        proof = json.loads(path.read_text())
        run = proof.get("run")
        if (
            not isinstance(run, str)
            or not re.fullmatch(r"[a-f0-9]{32}", run)
            or path.parent.name != run
            or proof.get("image") != IMAGE
        ):
            raise ValueError("Unbound scientific ownership receipt")
        labels = {"nemo-gym.workload": "scicode-verifier", "nemo-gym.run": run}
        if "sandbox" in proof and proof["sandbox"].get("ownership_labels") != labels:
            raise ValueError("Scientific ownership labels differ")
        receipts.append(await provider.reconcile_owned(labels))
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "captures", "test-data", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--sandbox-image", required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("The shared owned-process supervisor requires Linux")
    validate_inputs(args.test_data, args.sandbox_image)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = out / "result"
    command = [
        sys.executable,
        str(Path(process_supervisor.__file__).resolve()),
        "--timeout",
        "480",
        "--cleanup-timeout",
        "10",
        "--receipt",
        str(out / "supervisor.json"),
        "--stop-file",
        str(out / "stop.request"),
        "--",
        sys.executable,
        str(Path(__file__).with_name("saved_task_process_control.py")),
        "--input",
        str(args.input.resolve()),
        "--captures",
        str(args.captures.resolve()),
        "--output",
        str(result),
        "--scientific-supervised",
        "--test-data",
        str(args.test_data.resolve()),
        "--sandbox-image",
        args.sandbox_image,
    ]

    # The shared supervisor owns the whole child process group, including baseline/
    # restore grandchildren if an inner parent-only timeout raises. Its bounded
    # cleanup runs before container reconciliation; never kill a name/prefix group.
    def request_stop(*_):
        # Cooperative stop lets the existing supervisor drain its owned tree;
        # the parent then reconciles only the durable scientific run labels.
        (out / "stop.request").touch()

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with (out / "supervisor.log").open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    supervisor = json.loads((out / "supervisor.json").read_text())

    async def cleanup():
        provider = create_provider({"docker": {"exec": {"default_timeout_s": 5}}})
        try:
            return await asyncio.wait_for(reconcile_containers(result, supervisor, provider), timeout=30)
        finally:
            await provider.aclose()

    cleanup_receipts = asyncio.run(cleanup())
    (out / "container-cleanup.json").write_text(json.dumps(cleanup_receipts, indent=2))
    if completed.returncode or supervisor.get("return_code") != 0 or supervisor.get("timed_out"):
        raise RuntimeError("Recovery control failed or exceeded its budget; original evidence retained")
    report = compare_science(result)
    report.update(
        {
            "external_inference_calls": 0,
            "transport": "exact saved-response fixture",
            "supported_cli_recovery": False,
            "mutable_state_restoration": False,
        }
    )
    (out / "scientific-acceptance.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
