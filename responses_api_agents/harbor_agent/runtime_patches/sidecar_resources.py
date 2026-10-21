# SPDX-License-Identifier: Apache-2.0
"""Explicit, checksum-gated resource variant of pinned Harbor's built-in sidecar."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from nemo_gym.dataset_staging import stage_file

COMMIT = "96ff9c6db90cc015f0fc0987d896670c9c5f1fad"
ORIGINAL_SHA256 = "6a7e63d3cdad6d7be919d110e536b8dbc52b871d0ee3a5195d65cbd0e5e36ca6"
SERVICE = b"  harbor-docker-egress-control-sidecar:\n"
LIMITS = b"    mem_limit: 128m\n    cpus: 0.25\n"


def apply(package: Path, provenance: dict, output: Path) -> dict:
    if provenance.get("vcs_info", {}).get("commit_id") != COMMIT:
        raise ValueError("Sidecar patch requires exact pinned Harbor source provenance")
    target = package / "environments/docker/docker-compose-egress-control.yaml"
    if target.is_symlink() or any(p.is_symlink() for p in target.parents):
        raise ValueError("Runtime patch target must not traverse symlinks")
    before = target.read_bytes()
    if hashlib.sha256(before).hexdigest() == ORIGINAL_SHA256:
        original = before
    else:
        original = before.replace(SERVICE + LIMITS, SERVICE, 1)
        if original == before or hashlib.sha256(original).hexdigest() != ORIGINAL_SHA256:
            raise ValueError("Sidecar runtime differs from the pinned original or this exact patch")
    patched = original.replace(SERVICE, SERVICE + LIMITS, 1)
    output.mkdir(parents=True, exist_ok=True)
    receipt = {
        "variant": "harbor-sidecar-resources-v1",
        "package_root": str(package.resolve()),
        "upstream_commit": COMMIT,
        "original_sha256": ORIGINAL_SHA256,
        "patched_sha256": hashlib.sha256(patched).hexdigest(),
        "memory_bytes": 134217728,
        "cpus": 0.25,
        "scope": "built-in sidecar for solver and separate verifier; no arbitrary compose inheritance",
    }
    old = output / "runtime-patch.json"
    if old.exists() and json.loads(old.read_text()) != receipt:
        raise ValueError("Refusing to overwrite a different runtime patch receipt")
    with TemporaryDirectory() as temp:
        temp = Path(temp)
        source = temp / "compose.yaml"
        source.write_bytes(original)
        stage_file(source, output / "original-compose.yaml", expected_sha256=ORIGINAL_SHA256)
        source.write_bytes(patched)
        stage_file(source, target, expected_sha256=receipt["patched_sha256"])
        source = temp / "receipt.json"
        source.write_text(json.dumps(receipt, indent=2) + "\n")
        stage_file(source, old)
    return receipt


def main():
    import harbor

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dist = importlib.metadata.distribution("harbor")
    package = Path(harbor.__file__).resolve().parent
    if dist.version != "0.24.0" or Path(dist.locate_file("harbor")).resolve() != package:
        raise ValueError("Imported Harbor package and pinned distribution metadata must agree")
    provenance = json.loads(dist.read_text("direct_url.json") or "{}")
    print(json.dumps(apply(package, provenance, args.output), indent=2))


if __name__ == "__main__":
    main()
