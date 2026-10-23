# SPDX-License-Identifier: Apache-2.0
"""Prepare preserved CORE medium sources; publish only explicitly qualified images."""

import hashlib
import json
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.corebench.image_controls import qualification_digest, recipe_digest, source_digest
from nemo_gym.dataset_staging import stage_file


MANIFEST = Path(__file__).with_name("frozen20.json")
SHA = re.compile(r"sha256:[0-9a-f]{64}")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(name):
    p = Path(name)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError("Unsafe source path")
    return p


def verify_sources(row, medium_root, reference_root):
    root = medium_root / row["id"]
    expected = {f["path"]: f["sha256"] for f in row["medium_file_manifest"]}
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    if actual != set(expected):
        raise ValueError(f"Unrecorded/missing medium files: {row['id']}")
    for name, checksum in expected.items():
        p = root / relative(name)
        if p.is_symlink() or digest(p) != checksum:
            raise ValueError(f"Changed medium source: {name}")
    if (root / "results").exists() and any((root / "results").iterdir()):
        raise ValueError("Medium root results must be empty")
    refs = reference_root / row["id"]
    for name, key in [
        ("solver-input/instruction.txt", "instruction_sha256"),
        ("evaluator-only/answers.json", "reference_sha256"),
        ("evaluator-only/evaluations.py", "scorer_sha256"),
    ]:
        if digest(refs / name) != row[key]:
            raise ValueError(f"Changed creator prompt/reference/scorer: {name}")
    return root, refs


def validate_qualification(row, receipt):
    if receipt["capsule"] != row["id"] or receipt["creator_image"] != row["published_image"]:
        raise ValueError("Qualification binds a different creator image")
    if receipt.get("recipe_sha256") != recipe_digest(row) or receipt.get("source_sha256") != source_digest(row):
        raise ValueError("Qualification recipe/source differs")
    if receipt.get("qualification_sha256") != qualification_digest(row):
        from benchmarks.corebench.qualification_compatibility import reviewed_compatibility

        if reviewed_compatibility(row, receipt) is None:
            raise ValueError("Qualification probe/task/runtime binding missing or stale")
    if not SHA.fullmatch(receipt["final_image"]):
        raise ValueError("Final image must be immutable")
    if not receipt["scientific_before"] or receipt["scientific_before"] != receipt["scientific_after"]:
        raise ValueError("Scientific interpreter/package state changed")
    if not all(
        receipt.get(k) is True
        for k in [
            "harness_python_ok",
            "tmux_ok",
            "asciinema_ok",
            "reference_isolation_ok",
            "cleanup_ok",
            "image_source_verified",
        ]
    ):
        raise ValueError("Image-specific control incomplete")
    modes = receipt.get("image_source_modes", {})
    if set(modes) != {f["path"] for f in row["medium_file_manifest"]} or any(
        type(mode) is not int or not 0 <= mode <= 0o777 for mode in modes.values()
    ):
        raise ValueError("Image source modes missing or invalid")
    limits = receipt["resources"]
    if any(
        type(limits.get(k)) is not int or limits[k] <= 0
        for k in ["cpus", "memory_mb", "storage_mb", "agent_timeout_sec"]
    ):
        raise ValueError("Explicit positive resource limits required")
    if row["status"] == "requires_cuda_no_cpu_substitution" and not limits.get("gpus"):
        raise ValueError("Creator workflow requires CUDA")
    if row["status"] == "measured_oom_requires_resource_review" and not receipt.get("oom_resource_review"):
        raise ValueError("Observed OOM requires explicit resource review")
    return limits


def stage(
    medium_root,
    reference_root,
    qualifications,
    destination,
    *,
    manifest=MANIFEST,
    execution_resources=None,
    placement_evidence=None,
):
    rows = json.loads(manifest.read_text())["rows"]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate capsules")
    # Validate the complete retained source set and every supplied receipt before publication.
    sources = {r["id"]: verify_sources(r, medium_root, reference_root) for r in rows}
    unknown = set(qualifications) - set(sources)
    if unknown:
        raise ValueError("Qualification for an unknown capsule")
    compatibility = {}
    for r in rows:
        if r["id"] in qualifications:
            validate_qualification(r, qualifications[r["id"]])
            from benchmarks.corebench.qualification_compatibility import reviewed_compatibility

            review = reviewed_compatibility(r, qualifications[r["id"]])
            if review is not None:
                compatibility[r["id"]] = review
    if not qualifications:
        raise ValueError("No qualified images; source inventory alone is not runnable")
    execution_resources = {} if execution_resources is None else execution_resources
    if not isinstance(execution_resources, dict) or set(execution_resources) - set(qualifications):
        raise ValueError("Execution resources require a qualified capsule")
    placement_evidence = {} if placement_evidence is None else placement_evidence
    if not isinstance(placement_evidence, dict) or set(placement_evidence) - set(qualifications):
        raise ValueError("Placement evidence requires a qualified capsule")
    placement_reviews = {}
    resolved_resources = {}
    for cid, q in qualifications.items():
        limits = dict(q["resources"])
        override = execution_resources.get(cid, {})
        if not isinstance(override, dict) or set(override) - {"memory_mb", "gpus"}:
            raise ValueError("Only memory increases and evidence-bound GPU placement are supported")
        if "memory_mb" in override:
            memory = override["memory_mb"]
            if type(memory) is not int or memory < limits["memory_mb"]:
                raise ValueError("Execution memory must be an integer no smaller than qualification memory")
            limits["memory_mb"] = memory
        if "gpus" in override:
            gpu = override["gpus"]
            if type(gpu) is not int or gpu <= 0:
                raise ValueError("GPU count must be a positive integer")
            limits["gpus"] = gpu
        if limits.get("gpus"):
            from benchmarks.corebench.placement_evidence import validate_gpu_placement

            row = next(r for r in rows if r["id"] == cid)
            placement_reviews[cid] = validate_gpu_placement(row, q, limits, placement_evidence.get(cid))
        elif cid in placement_evidence:
            raise ValueError("Unused GPU placement evidence")
        resolved_resources[cid] = limits
    template = Path(__file__).parent / "templates"
    published = []
    with TemporaryDirectory() as temporary:
        tmp = Path(temporary)
        for row in rows:
            cid = row["id"]
            if cid not in qualifications:
                continue
            q = qualifications[cid]
            limits = resolved_resources[cid]
            source, refs = sources[cid]
            task = destination / "tasks" / cid
            stage_file(
                refs / "solver-input/instruction.txt",
                task / "instruction.md",
                expected_sha256=row["instruction_sha256"],
            )
            for f in row["medium_file_manifest"]:
                src = source / relative(f["path"])
                dst = task / "environment/capsule" / relative(f["path"])
                stage_file(src, dst, expected_sha256=f["sha256"])
                dst.chmod(src.stat().st_mode & 0o777)
            for name in ["answers.json", "evaluations.py"]:
                stage_file(refs / "evaluator-only" / name, task / "tests" / name)
            for name in ["check.py", "item_diagnostics.py", "capture_artifacts.py", "test.sh"]:
                stage_file(template / name, task / "tests" / name)
            config = (
                'version = "1.0"\nartifacts = ["/workspace"]\n[agent]\ntimeout_sec = %d\n[verifier]\ntimeout_sec = 120\n[environment]\ndocker_image = "%s"\ncpus = %d\nmemory_mb = %d\nstorage_mb = %d\nallow_internet = false\n'
                % (
                    limits["agent_timeout_sec"],
                    q["final_image"],
                    limits["cpus"],
                    limits["memory_mb"],
                    limits["storage_mb"],
                )
            )
            if limits.get("gpus"):
                config += "gpus = %d\n" % limits["gpus"]
            p = tmp / "task.toml"
            p.write_text(config)
            stage_file(p, task / "task.toml")
            p = tmp / "Dockerfile"
            p.write_text("FROM " + q["final_image"] + "\n")
            stage_file(p, task / "environment/Dockerfile")
            published.append(
                {
                    "instance_id": "corebench::" + cid,
                    "task_id": "corebench::" + cid,
                    "responses_create_params": {"input": [], "temperature": 0.0},
                    "agent_ref": {"name": "harbor_agent"},
                }
            )
        p = tmp / "benchmark.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in published))
        stage_file(p, destination / "benchmark.jsonl")
        p = tmp / "preparation.json"
        p.write_text(
            json.dumps(
                {
                    "manifest_sha256": digest(manifest),
                    "retained": len(rows),
                    "published": len(published),
                    "qualification_receipts": qualifications,
                    "qualification_compatibility_reviews": compatibility,
                    "execution_resources": resolved_resources,
                    "execution_resource_overrides": execution_resources,
                    "execution_placement_reviews": placement_reviews,
                    "blocked": [{"id": r["id"], "status": r["status"]} for r in rows if r["id"] not in qualifications],
                },
                indent=2,
            )
        )
        stage_file(p, destination / "preparation.json")
    return destination / "benchmark.jsonl"


def prepare():
    return stage(
        Path(os.environ["COREBENCH20_MEDIUM_ROOT"]),
        Path(os.environ["COREBENCH20_REFERENCE_ROOT"]),
        json.loads(Path(os.environ["COREBENCH20_QUALIFICATIONS"]).read_text()),
        Path(__file__).parent / "data20",
        execution_resources=json.loads(os.environ.get("COREBENCH20_EXECUTION_RESOURCES", "{}")),
        placement_evidence=json.loads(os.environ.get("COREBENCH20_PLACEMENT_EVIDENCE", "{}")),
    )


if __name__ == "__main__":
    print(prepare())
