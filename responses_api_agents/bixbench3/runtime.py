# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process boundary to the separately installed, pinned Edison BixBench3 harness.

The upstream CC-BY-SA source is not vendored into Gym. Gold files remain on the
host and are read only after Inspect exits. Dataset and licensed runtime setup
must be completed before using this adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import yaml


UPSTREAM_REVISION = "d0e0bbb41222335b1b8878f533a58466a3a782dc"


def run_fingerprint(settings_path: Path) -> str:
    """Bind reuse to settings and the actual staged input/reference bytes."""
    settings = json.loads(settings_path.read_text())
    digest = hashlib.sha256(settings_path.read_bytes())
    digest.update(UPSTREAM_REVISION.encode())
    roots = [
        Path(settings["run_root"]) / "dataset",
        Path(settings["data_dir"]),
        Path(settings["grading_specification"]).parent,
    ]
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("Staged input and gold trees must not contain symlinks")
            if path.is_file():
                digest.update(str(path.relative_to(root)).encode())
                digest.update(str(path.stat().st_size).encode())
                with path.open("rb") as stream:
                    while block := stream.read(1024 * 1024):
                        digest.update(block)
    return digest.hexdigest()


def check_revision(checkout: Path) -> None:
    """Reject an unpinned or locally modified upstream harness."""
    actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if actual != UPSTREAM_REVISION:
        raise ValueError(f"BixBench3 must be pinned to {UPSTREAM_REVISION}")
    dirty = subprocess.check_output(["git", "-C", str(checkout), "diff", "HEAD", "--"], text=True)
    if dirty:
        raise ValueError("Upstream tracked source has local modifications")


def check_gold_isolation(gold: Path, mounts: list[Path]) -> None:
    """Reject a gold directory overlapping any directory visible to a container."""
    gold = gold.resolve(strict=True)
    for mount in mounts:
        mount = mount.resolve(strict=True)
        if gold == mount or gold.is_relative_to(mount) or mount.is_relative_to(gold):
            raise ValueError(f"Gold overlaps a container mount: {mount}")


REQUESTED_MODELS = frozenset(
    {
        "nvidia/nvidia/nemotron-3.5-super-vl",
        "nvidia/nvidia/nemotron-3-ultra",
    }
)


def generation_settings(settings: dict) -> tuple[int, int]:
    """Require explicit context provenance instead of Inspect's unknown-model fallback."""
    if settings["model"] not in REQUESTED_MODELS:
        raise ValueError("Use one of the two requested Nemotron model IDs")
    context = settings.get("model_context_length")
    output = settings.get("max_output_tokens", 16384)
    if isinstance(context, bool) or not isinstance(context, int):
        raise ValueError("Set verified model_context_length and model_context_source")
    if not isinstance(settings.get("model_context_source"), str) or not settings["model_context_source"].strip():
        raise ValueError("Set verified model_context_length and model_context_source")
    if isinstance(output, bool) or not isinstance(output, int) or output <= 0 or context <= output:
        raise ValueError("Context length must exceed positive max_output_tokens")
    return context, output


def _artifact_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Artifact path must be a safe relative file path")
    parts = path.parts[1:] if path.parts[0] == "outputs" else path.parts
    if not parts:
        raise ValueError("Artifact path must name a file")
    return Path(*parts).as_posix()


def validate_grading_contract(*, case: dict, specification: Path, mounts: list[Path]) -> None:
    """Match the host grader to this case and keep every gold file host-only."""
    spec = json.loads(specification.read_text())
    if spec.get("paper_id") != case["task_id"]:
        raise ValueError("Grading specification paper_id does not match task_id")
    if case.get("doi") and spec.get("doi") != case["doi"]:
        raise ValueError("Grading specification DOI does not match case")
    expected = {item["artifact_id"]: _artifact_path(item["path"]) for item in case["expected_outputs"]}
    if len(expected) != len(case["expected_outputs"]) or not expected:
        raise ValueError("Case must declare distinct expected artifacts")
    actual = {}
    gold_root = specification.parent.resolve(strict=True)
    check_gold_isolation(gold_root, mounts)
    for artifact in spec["artifacts"]:
        name = artifact["artifact_id"]
        if name in actual:
            raise ValueError("Duplicate grading artifact ID")
        output = artifact.get("expected_output") or artifact.get("output_format")
        actual[name] = _artifact_path(output["path"])
        relative = Path(artifact["ground_truth"]["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Gold paths must stay inside the host-only specification tree")
        candidate = gold_root / relative
        if any(part.is_symlink() for part in [candidate, *candidate.parents] if part != gold_root.parent):
            raise ValueError("Gold paths must not traverse symlinks")
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(gold_root) or not resolved.is_file():
            raise ValueError("Gold reference must be a file inside its specification tree")
        check_gold_isolation(resolved, mounts)
    if actual != expected:
        raise ValueError("Grading artifact IDs or paths do not match expected outputs")


def read_artifact_score(path: Path) -> float:
    """Read the deterministic grader's score, never Inspect output completeness."""
    result = json.loads(path.read_text())
    if not isinstance(result, dict) or not isinstance(result.get("artifacts"), list):
        raise ValueError("Expected upstream artifact-grade result")
    score = result.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("Artifact score must be a number")
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Artifact score must be finite and between zero and one")
    if result.get("artifact_count") != len(result["artifacts"]) or not result["artifacts"]:
        raise ValueError("Artifact-grade result has inconsistent or empty artifacts")
    return float(score)


def grade_artifacts(*, python: Path, checkout: Path, specification: Path, artifacts: Path, output: Path) -> float:
    """Invoke the real upstream deterministic grader in its isolated environment."""
    check_revision(checkout)
    if any(path.is_symlink() for path in artifacts.rglob("*")):
        raise ValueError("Agent artifacts must not contain symlinks")
    output.parent.mkdir(parents=True, exist_ok=True)
    program = (
        "import json,sys; from pathlib import Path; from bixbench3.grading.grader import grade; "
        "r=grade(sys.argv[1],sys.argv[2],sys.argv[3]); "
        "Path(sys.argv[4]).write_text(json.dumps(r,indent=2)+'\\n')"
    )
    subprocess.run(
        [
            str(python),
            "-c",
            program,
            str(specification),
            str(artifacts),
            str(output.parent / "artifact_grades"),
            str(output),
        ],
        cwd=checkout,
        check=True,
        timeout=600,
    )
    return read_artifact_score(output)


def execute(*, settings_path: Path) -> float:
    """Run a previously staged case using the unmodified Inspect task and grader.

    Settings are operator-authored host configuration, not benchmark-supplied
    content. Secrets are read from files and passed only to the host process.
    The network judge endpoint must enforce the campaign's paid-call budget.
    """
    settings = json.loads(settings_path.read_text())
    checkout = Path(settings["upstream_checkout"]).resolve(strict=True)
    python = Path(settings["upstream_python"]).resolve(strict=True)
    root = Path(settings["run_root"]).resolve(strict=True)
    gold = Path(settings["grading_specification"]).resolve(strict=True)
    check_revision(checkout)
    data = Path(settings["data_dir"]).resolve(strict=True)
    for name in ["workspace", "outputs", "broker_logs", "broker_cache", "inspect_logs"]:
        (root / name).mkdir(exist_ok=True)
    mounts = [
        data,
        root / "workspace",
        root / "outputs",
        root / "dataset",
        root / "broker_logs",
        root / "broker_cache",
        root / "broker" / "policy.json",
    ]
    case = json.loads((root / "dataset" / "case.json").read_text())
    validate_grading_contract(case=case, specification=gold, mounts=mounts)
    context_length, max_output_tokens = generation_settings(settings)
    if (root / "grade.json").exists():
        raise FileExistsError("Existing graded run: use its result or choose a fresh run directory")
    if any((root / "outputs").iterdir()):
        raise FileExistsError("Outputs must be empty before a new rollout")
    receipt = root / "settings-sha256.txt"
    receipt.write_text(run_fingerprint(settings_path) + "\n")
    case = json.loads((root / "dataset" / "case.json").read_text())
    if case["task_id"] != settings["task_id"]:
        raise ValueError("Staged case does not match requested task")
    for field in ("sandbox_image", "gateway_image"):
        if "@sha256:" not in settings[field] and not settings[field].startswith("sha256:"):
            raise ValueError(f"{field} must use an immutable image digest")
    # Keep the upstream service topology and mounts. Only direct the paid judge
    # through the operator's budget proxy; do not bypass its network gateway.
    compose = yaml.safe_load((checkout / "sandbox" / "compose.yaml").read_text())
    label = {"gym.bixbench3.run": hashlib.sha256(str(root).encode()).hexdigest()[:24]}
    for section in ("services", "networks"):
        for name, entry in compose.get(section, {}).items():
            if entry is None:
                entry = compose[section][name] = {}
            labels = entry.get("labels", {})
            if isinstance(labels, list):
                labels = dict(item.split("=", 1) if "=" in item else (item, "") for item in labels)
            entry["labels"] = labels | label
    compose["services"]["network-gateway"]["environment"]["OPENAI_BASE_URL"] = settings["budget_proxy_url"]
    compose_path = root / "compose.yaml"
    compose_path.write_text(yaml.safe_dump(compose, sort_keys=False))
    env = os.environ.copy()
    env.update(
        {
            "BIXBENCH3_DATA_DIR": str(data),
            "BIXBENCH3_OUTPUTS_DIR": str(root / "outputs"),
            "BIXBENCH3_WORKSPACE_DIR": str(root / "workspace"),
            "BIXBENCH3_INSPECT_SANDBOX_COMPOSE": str(compose_path),
            "BIXBENCH3_SANDBOX_IMAGE": settings["sandbox_image"],
            "BIXBENCH3_NETWORK_GATEWAY_IMAGE": settings["gateway_image"],
            "BIXBENCH3_BROKER_POLICY": str(root / "broker" / "policy.json"),
            "BIXBENCH3_BROKER_LOG_DIR": str(root / "broker_logs"),
            "BIXBENCH3_BROKER_CACHE_DIR": str(root / "broker_cache"),
            "BIXBENCH3_BROKER_HOST_PORT": str(settings["broker_port"]),
            "BIXBENCH3_BROKER_ENDPOINT": f"http://127.0.0.1:{settings['broker_port']}",
            "BIXBENCH3_BROKER_TOKEN": Path(settings["broker_token_file"]).read_text().strip(),
            "OPENAI_API_KEY": Path(settings["budget_proxy_key_file"]).read_text().strip(),
            "OPENAI_BASE_URL": settings["budget_proxy_url"],
            "NEMOTRON_API_KEY": Path(settings["solver_key_file"]).read_text().strip(),
        }
    )
    if not (root / "broker" / "policy.json").is_file():
        raise FileNotFoundError("Stage the upstream network policy before execution")
    program = r"""
import json, os, sys
from pathlib import Path
from inspect_ai import eval
from inspect_ai.model import GenerateConfig, ModelInfo, get_model, set_model_info
from bixbench3.inspect_harness.tasks import bixbench3_cases
config = GenerateConfig(temperature=0, max_tokens=int(sys.argv[7]), max_retries=2)
model = get_model('openai/'+sys.argv[1], base_url=sys.argv[2],
                  api_key=os.environ['NEMOTRON_API_KEY'], responses_api=True, config=config)
set_model_info(model.canonical_name(), ModelInfo(organization='NVIDIA',
    model=sys.argv[1], context_length=int(sys.argv[6]), output_tokens=int(sys.argv[7]), reasoning=True))
logs = eval(bixbench3_cases(dataset_path=sys.argv[3], case_ids=[sys.argv[4]]),
            model=model, log_dir=sys.argv[5], max_tasks=1, max_samples=1,
            log_model_api=True, temperature=0, max_tokens=int(sys.argv[7]), max_retries=2)
summary = {'model': sys.argv[1], 'transport': 'responses', 'temperature': 0,
           'context_length': int(sys.argv[6]), 'max_output_tokens': int(sys.argv[7]),
           'logs': [{'status': log.status, 'location': str(log.location)} for log in logs],
           'measurement': 'Inspect scores are output completeness, not artifact accuracy'}
Path(sys.argv[5], 'execution-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
sys.exit(0 if len(logs)==1 and logs[0].status=='success' else 1)
"""
    command = [
        str(python),
        "-c",
        program,
        settings["model"],
        settings["solver_base_url"],
        str(root / "dataset" / "case.json"),
        settings["task_id"],
        str(root / "inspect_logs"),
        str(context_length),
        str(max_output_tokens),
    ]
    # Model credential wiring is verified by the adapter's real smoke test before
    # this execution path is treated as validated.
    with (root / "inspect-process.log").open("w") as log:
        subprocess.run(command, cwd=checkout, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=87000)
    return grade_artifacts(
        python=python, checkout=checkout, specification=gold, artifacts=root / "outputs", output=root / "grade.json"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps({"artifact_score": execute(settings_path=args.settings)}))
