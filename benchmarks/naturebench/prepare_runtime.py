"""Patch a copied NatureBench verifier for the explicit Nemotron judge variant.

Never run against the immutable downloaded task package. Scientific grading,
judge prompts and verdict parsing remain unchanged.
"""

from pathlib import Path


MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"


def patch_judge(source: str) -> str:
    old = '    return "openai" if model.lower().startswith("gpt-") else "anthropic"'
    new = f'''    if model != {MODEL!r}:
        raise ValueError("Campaign judge model is not allowed")
    return "openai"'''
    assert source.count(old) == 1, "Unexpected upstream provider dispatch"
    source = source.replace(old, new)
    old = "    return resolved_provider, resolved_model, resolved_base_url, resolved_api_key"
    new = f"""    expected_url = os.environ.get("CAMPAIGN_JUDGE_BASE_URL")
    if resolved_model != {MODEL!r} or resolved_provider != "openai":
        raise ValueError("Campaign judge model/provider is not allowed")
    if not expected_url or resolved_base_url.rstrip("/") != expected_url.rstrip("/"):
        raise ValueError("Campaign judge endpoint must be explicitly configured")
{old}"""
    assert source.count(old) == 1, "Unexpected upstream configuration"
    source = source.replace(old, new)
    old = '    output = payload.get("output", []) if isinstance(payload, dict) else []'
    new = f"""    if not isinstance(payload, dict) or payload.get("model") != {MODEL!r}:
        return False, "judge_response_model_mismatch"
    if payload.get("status") != "completed":
        return False, "judge_response_incomplete"
{old}"""
    assert source.count(old) == 1, "Unexpected upstream response parser"
    return source.replace(old, new)


def materialize_archive(
    archive: str | Path,
    destination: str | Path,
    *,
    expected_sha256: str,
    restore_molecular_evaluator_data: bool = False,
    snapshot_evaluation_inputs: bool = False,
    snapshot_capture_timeout_seconds: float | None = None,
) -> Path:
    """Extract a pinned release with judge routing and an optional public-input mount/snapshot variants."""
    import hashlib
    import json
    import tarfile
    from tempfile import TemporaryDirectory

    from nemo_gym.dataset_staging import stage_file

    def file_digest(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    if snapshot_evaluation_inputs and snapshot_capture_timeout_seconds is None:
        raise ValueError("Snapshot variant requires an explicit operational capture allowance")
    if not snapshot_evaluation_inputs and snapshot_capture_timeout_seconds is not None:
        raise ValueError("Capture allowance requires snapshot variant")
    archive, destination = Path(archive), Path(destination)
    if file_digest(archive) != expected_sha256:
        raise ValueError("NatureBench archive checksum mismatch")
    if destination.exists():
        raise FileExistsError("Refuse to merge an archive into existing task state")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix=".naturebench-") as temporary:
        root = Path(temporary)
        seen = set()
        with tarfile.open(archive) as tar:
            for member in tar:
                path = Path(member.name)
                if path.is_absolute() or ".." in path.parts or not (member.isdir() or member.isfile()):
                    raise ValueError("Unsafe NatureBench archive member")
                if member.name in seen:
                    raise ValueError("Duplicate NatureBench archive member")
                seen.add(member.name)
                tar.extract(member, root, filter="data")
        candidates = list(root.glob("*/task.toml"))
        if len(candidates) != 1:
            raise ValueError("Expected one complete Harbor task archive")
        task = candidates[0].parent
        if task.name != destination.name:
            raise ValueError("Archive task identity mismatch")
        hashes = {str(p.relative_to(task)): file_digest(p) for p in task.rglob("*") if p.is_file()}
        judge = task / "tests/judge_readme.py"
        judge.write_text(patch_judge(judge.read_text()))
        if restore_molecular_evaluator_data:
            from benchmarks.naturebench.packaging import restore_public_input_mount

            compose = task / "environment/docker-compose.yaml"
            compose.write_text(restore_public_input_mount(compose.read_text(), task_id=task.name))
        if snapshot_evaluation_inputs:
            from benchmarks.naturebench.snapshot_packaging import install_snapshot_variant

            install_snapshot_variant(task, capture_timeout_seconds=snapshot_capture_timeout_seconds)
        receipt = {
            "archive_sha256": expected_sha256,
            "source_files": hashes,
            "prepared_files": {str(p.relative_to(task)): file_digest(p) for p in task.rglob("*") if p.is_file()},
            "variant": "exact Nemotron judge; creator prompt/parser/evaluator retained",
            "molecular_evaluator_public_input_mount_restored": restore_molecular_evaluator_data,
            "evaluation_input_snapshot_variant": snapshot_evaluation_inputs,
            "snapshot_capture_timeout_seconds": snapshot_capture_timeout_seconds,
        }
        # Directory publication is a single same-filesystem rename; no partial task is visible.
        task.rename(destination)
        proof = root / "source.json"
        proof.write_text(json.dumps(receipt, indent=2) + "\n")
        stage_file(proof, destination.parent / (destination.name + ".source.json"))
    return destination
