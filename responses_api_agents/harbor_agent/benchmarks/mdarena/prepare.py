# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inspect pinned MDArena tasks and emit stable Gym rows; never launch inference."""

import argparse
import hashlib
import json
import subprocess
import tomllib
from pathlib import Path


SOURCE_URL = "https://github.com/weitse-hsu/MDArena"
SOURCE_COMMIT = "d1d789d92a9935e716182e31d99393edcd4ac7d6"  # pragma: allowlist secret
FROZEN_PILOT = (
    "24_ligand_parameterization_smirnoff_openfe",
    "44_traj_ana_atp8b_rmsd",
    "38_pept2_qm_cluster_orca",
    "02_conc_diff_sys_traj",
    "41_rexee_argon_hydration_setup",
)
MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"


def inspect_checkout(checkout: Path) -> dict:
    """Validate immutable source and explicit resources without changing task files."""
    head = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if head != SOURCE_COMMIT:
        raise ValueError(f"Expected MDArena {SOURCE_COMMIT}, found {head}")
    dirty = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if dirty:
        raise ValueError("MDArena checkout has tracked modifications; use an immutable pinned checkout")
    tasks = []
    for task_file in sorted((checkout / "tasks").glob("*/task.toml")):
        task = tomllib.loads(task_file.read_text())
        env = task["environment"]
        for key in ("cpus", "memory_mb"):
            if env.get(key) is None:
                raise ValueError(f"{task_file.parent.name}: explicit {key} is required")
        judges = []
        for judge_file in sorted(task_file.parent.glob("tests/**/judge.toml")):
            judge = tomllib.loads(judge_file.read_text())
            judges.append({"path": str(judge_file.relative_to(checkout)), "config": judge})
        hashes = {
            str(path.relative_to(checkout)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(task_file.parent.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
        }
        tasks.append(
            {
                "instance_id": f"mdarena::{task_file.parent.name}",
                "task_path": str(task_file.parent.resolve()),
                "source_task_config": task,
                "source_judges": judges,
                "source_sha256": hashes,
            }
        )
    names = {item["instance_id"].split("::", 1)[1] for item in tasks}
    if not set(FROZEN_PILOT).issubset(names):
        raise ValueError("Pinned checkout does not contain the complete frozen pilot")
    return {
        "source_url": SOURCE_URL,
        "source_commit": head,
        "required_model_all_llm_roles": MODEL,
        "task_count": len(tasks),
        "frozen_pilot": list(FROZEN_PILOT),
        "launch_ready": False,
        "blockers": [
            "Native RewardKit AgentCLI judge uses Claude; guarded Nemotron tool-judge bridge is not integrated",
            "Separate verifier must restore full submitted workspace and trajectory, not only declared outputs",
            "Solver and judge logical-call guard bindings require runtime acceptance before live execution",
        ],
        "harness_candidates": [
            {"harness": "Terminus2NemoGym", "parser_name": "json"},
            {"harness": "Terminus2NemoGym", "parser_name": "xml"},
        ],
        "tasks": tasks,
    }


def prepare(checkout: Path, output: Path) -> Path:
    """Write preflight evidence and frozen Gym input rows, without a runnable judge config."""
    report = inspect_checkout(checkout)
    output.mkdir(parents=True, exist_ok=True)
    (output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = [
        {
            "instance_id": f"mdarena::{task}",
            "responses_create_params": {"input": [], "temperature": 0.0},
            "agent_ref": {"name": "harbor_agent"},
        }
        for task in FROZEN_PILOT
    ]
    path = output / "pilot-5.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.checkout, args.output))
