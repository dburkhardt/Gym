# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted coordination of separate public solver and reference verifier sandboxes."""

import hashlib
import json
import secrets
import shutil
import time
from pathlib import Path

from benchmarks.moosechem.guarded import continuation_binding
from benchmarks.moosechem.isolated import pins, validate_public, verify_creator
from benchmarks.moosechem.isolated_runtime import MODULES, exact_tree, execute_phase, phase_spec, stage_modules
from benchmarks.moosechem.native import MODEL, append_durable
from benchmarks.moosechem.qualify_settled_tail import PublicSettledSequence, SettledSequence
from nemo_gym.sandbox.model_bridge import bridge
from nemo_gym.sandbox.providers.local.provider import LocalProvider
from nemo_gym.sandbox.scoring_workspace import scoring_workspace


def hashes(root):
    result = {}
    for p in Path(root).rglob("*"):
        if p.is_symlink():
            raise ValueError("Artifact symlink forbidden")
        if p.is_file():
            result[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


async def solver(config, task_id, out, endpoint, lineage, artifact_identity):
    """Trusted export precedes public-only execution; no gold file enters its mounts."""
    start = time.time()
    public = out / "public"
    task = validate_public(
        config.public_input_root,
        config.public_input_root / "task.json",
        expected_manifest_sha256=config.public_manifest_sha256,
    )
    if task["original_task_id"] != task_id:
        raise ValueError("Frozen public task identity changed")
    shutil.copytree(config.public_input_root, public)
    manifest_sha = config.public_manifest_sha256
    validate_public(public, public / "task.json", expected_manifest_sha256=manifest_sha)
    source = out / "modules"
    source_hashes = stage_modules(source)
    evidence, evidence_hashes, sequence = None, None, None
    if config.settled_recovery:
        r = config.settled_recovery
        SettledSequence(r.saved, r.evidence, r.gateway, config.upstream_repo)
        evidence = out / "settlement"
        evidence.mkdir()
        for name, src in (("saved.jsonl", r.saved), ("settlement.json", r.evidence), ("gateway.jsonl", r.gateway)):
            shutil.copyfile(src, evidence / name)
        evidence_hashes = hashes(evidence)
        sequence = PublicSettledSequence(
            evidence / "saved.jsonl",
            evidence / "settlement.json",
            evidence / "gateway.jsonl",
            public,
            manifest_sha256=manifest_sha,
            task_id=task_id,
        )
    identity = {
        "task": task_id,
        **artifact_identity,
        "lineage": lineage,
        "image": config.isolated_runtime_image,
        "public_task": task,
        "public_manifest_sha256": manifest_sha,
        "source_hashes": source_hashes,
        "evidence_hashes": evidence_hashes,
        "model": MODEL,
        "budget": config.output_token_budget,
        "native_timeout_seconds": config.native_timeout_seconds,
        "started": start,
        "deadline": start + config.native_timeout_seconds,
    }
    base_digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    digest = base_digest
    if sequence:
        identity["settled_continuation"], digest = continuation_binding(
            sequence, evidence / "settlement.json", base_digest
        )
    identity["base_digest"], identity["config_sha256"] = base_digest, digest
    append_durable(out / "execution-identity.jsonl", identity)
    (out / "phase-identity.json").write_text(json.dumps(identity, indent=2))
    phase_out = out / "solver"
    phase_out.mkdir()
    local = LocalProvider(concurrency=1)
    async with scoring_workspace(local) as handle:
        socket = handle.raw["workspace"] / "model.sock"
        token = secrets.token_urlsafe(32)
        async with bridge(
            socket,
            endpoint=endpoint,
            token=token,
            lineage=lineage,
            digest=digest,
            namespace="moosechem",
            expected_model=MODEL,
        ):
            spec = phase_spec(
                image=config.isolated_runtime_image,
                source=source,
                source_hashes=source_hashes,
                repo=public,
                repo_hashes=hashes(public),
                output=phase_out,
                socket=socket,
                timeout=identity["deadline"] - time.time(),
                lineage=lineage,
                phase="solver",
                evidence=evidence,
                evidence_hashes=evidence_hashes,
                public_manifest_sha256=manifest_sha,
            )
            args = [
                "solver",
                "--task-id",
                str(task_id),
                "--budget",
                str(config.output_token_budget),
                "--public-manifest-sha256",
                manifest_sha,
            ]
            if sequence:
                args += ["--continuation"]
            receipt = await execute_phase(
                spec,
                phase_out,
                args,
                {
                    "MOOSE_SOCKET_TOKEN": token,
                    "MOOSE_LINEAGE": lineage,
                    "MOOSE_CONFIG_SHA256": digest,
                    "MOOSE_BASE_CONFIG_SHA256": base_digest,
                    "MOOSE_MODULE_HASHES": json.dumps({n: sha(source / "benchmarks/moosechem" / n) for n in MODULES}),
                },
                deadline=identity["deadline"],
            )
    append_durable(out / "native-execution.jsonl", {**receipt, "phase": "solver"})
    if receipt["return_code"] or receipt["error_type"]:
        raise RuntimeError("Public solver failed; no scientific zero or automatic replay")
    submission = phase_out / "scientific"
    (out / "solver-output-hashes.json").write_text(json.dumps(hashes(submission), indent=2))
    return sha(out / "phase-identity.json")


async def verifier(config, out, identity_sha, endpoint, lineage):
    """A distinct trusted resource owns references and cannot rerun the solver."""
    root = config.output_root.resolve()
    if out.is_symlink() or not out.resolve().is_relative_to(root) or out.resolve() == root:
        raise ValueError("Verifier artifact path outside owned task root")
    for parent in (out, *out.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError("Verifier artifact path symlink")
    if sha(out / "phase-identity.json") != identity_sha:
        raise ValueError("Solver execution identity changed")
    identity = json.loads((out / "phase-identity.json").read_text())
    if (
        identity["lineage"] != lineage
        or identity["image"] != config.isolated_runtime_image
        or identity["budget"] != config.output_token_budget
    ):
        raise ValueError("Verifier runtime or rollout binding mismatch")
    if (
        identity["native_timeout_seconds"] != config.native_timeout_seconds
        or identity["deadline"] != identity["started"] + config.native_timeout_seconds
    ):
        raise ValueError("Verifier cannot reset the combined operational deadline")
    remaining = identity["deadline"] - time.time()
    if not 0 < remaining <= config.native_timeout_seconds:
        raise ValueError("No remaining native phase time")
    verify_creator(config.upstream_repo)
    source = out / "modules"
    exact_tree(source, identity["source_hashes"])
    submission = out / "solver/scientific"
    submission_hashes = json.loads((out / "solver-output-hashes.json").read_text())
    exact_tree(submission, submission_hashes)
    terminal = json.loads((out / "solver/terminal.jsonl").read_text().splitlines()[-1])
    ordinal = terminal["next_ordinal"]
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("Invalid solver dispatch ordinal")
    phase_out = out / "verifier"
    phase_out.mkdir(exist_ok=False)
    trusted = out / "trusted-verifier-inputs"
    trusted.mkdir(exist_ok=False)
    for name in pins():
        dest = trusted / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config.upstream_repo / name, dest)
    verify_creator(trusted)
    local = LocalProvider(concurrency=1)
    async with scoring_workspace(local) as handle:
        socket = handle.raw["workspace"] / "model.sock"
        token = secrets.token_urlsafe(32)
        async with bridge(
            socket,
            endpoint=endpoint,
            token=token,
            lineage=lineage,
            digest=identity["config_sha256"],
            namespace="moosechem",
            expected_model=MODEL,
        ):
            spec = phase_spec(
                image=config.isolated_runtime_image,
                source=source,
                source_hashes=identity["source_hashes"],
                repo=trusted,
                repo_hashes=hashes(trusted),
                output=phase_out,
                socket=socket,
                timeout=identity["deadline"] - time.time(),
                lineage=lineage,
                phase="verifier",
                submission=submission,
                submission_hashes=submission_hashes,
            )
            receipt = await execute_phase(
                spec,
                phase_out,
                [
                    "verifier",
                    "--task-id",
                    str(identity["task"]),
                    "--budget",
                    str(config.output_token_budget),
                    "--ordinal",
                    str(ordinal),
                ],
                {
                    "MOOSE_SOCKET_TOKEN": token,
                    "MOOSE_LINEAGE": lineage,
                    "MOOSE_CONFIG_SHA256": identity["config_sha256"],
                    "MOOSE_MODULE_HASHES": json.dumps({n: sha(source / "benchmarks/moosechem" / n) for n in MODULES}),
                },
                deadline=identity["deadline"],
            )
    append_durable(out / "native-execution.jsonl", {**receipt, "phase": "verifier"})
    if receipt["return_code"] or receipt["error_type"]:
        raise RuntimeError("Separate verifier failed; preserve unscored outcome")
    exact_tree(submission, submission_hashes)
    for name in ("result.json", "screening.json", "hypotheses.json", "evaluation.json"):
        shutil.copyfile(phase_out / "scientific" / name, out / name)
    with (out / "model-traces.jsonl").open("xb") as combined:
        for folder in (submission, phase_out / "scientific"):
            combined.write((folder / "model-traces.jsonl").read_bytes())
    return {"artifact_dir": str(out), "verifier_complete": True}
