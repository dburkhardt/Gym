# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import socket
import tempfile
from pathlib import Path

import pytest

from benchmarks.moosechem import isolated, isolated_runtime
from benchmarks.moosechem.qualify_replay import ReplayRefused
from benchmarks.moosechem.qualify_settled_tail import PublicSettledSequence
from benchmarks.moosechem.test_qualify_settled_tail import fixture


def test_public_sequence_requires_trusted_manifest_and_task(tmp_path, monkeypatch):
    saved, gateway, proof, rows = fixture(tmp_path)
    for row in rows:
        row["stage"] = "hypothesis_generation"
    saved.write_text("\n".join(map(json.dumps, rows)))
    provenance = json.loads(proof.read_text())
    provenance["saved_sha256"] = hashlib.sha256(saved.read_bytes()).hexdigest()
    proof.write_text(json.dumps(provenance))
    (tmp_path / "public-manifest.json").write_text("{}")
    digest = hashlib.sha256(b"{}").hexdigest()
    seen = []

    def checked(repo, task_path, *, expected_manifest_sha256):
        seen.append(expected_manifest_sha256)
        return {"original_task_id": 26}

    monkeypatch.setattr(isolated, "validate_public", checked)
    seq = PublicSettledSequence(saved, proof, gateway, tmp_path, manifest_sha256=digest, task_id=26)
    assert seq.rows == rows and seen == [digest]
    with pytest.raises(ReplayRefused, match="binding"):
        PublicSettledSequence(saved, proof, gateway, tmp_path, manifest_sha256="0" * 64, task_id=26)
    with pytest.raises(ReplayRefused, match="task identity"):
        PublicSettledSequence(saved, proof, gateway, tmp_path, manifest_sha256=digest, task_id=27)


def test_exact_tree_rejects_extra_reference_and_symlink(tmp_path):
    (tmp_path / "public").write_text("public")
    pins = {"public": hashlib.sha256(b"public").hexdigest()}
    isolated_runtime.exact_tree(tmp_path, pins)
    (tmp_path / "gold.xlsx").write_text("gold")
    with pytest.raises(ValueError, match="contents"):
        isolated_runtime.exact_tree(tmp_path, pins)
    (tmp_path / "gold.xlsx").unlink()
    (tmp_path / "link").symlink_to(tmp_path / "public")
    with pytest.raises(ValueError, match="symlink"):
        isolated_runtime.exact_tree(tmp_path, pins)


@pytest.fixture
def short_socket():
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="muds-") as directory:
        path = Path(directory) / "model.sock"
        with socket.socket(socket.AF_UNIX) as listener:
            listener.bind(str(path))
            yield path


def test_phase_spec_public_mounts_only(tmp_path, monkeypatch, short_socket):
    source = tmp_path / "source"
    repo = tmp_path / "public"
    out = tmp_path / "owned"
    sock = short_socket
    for root in (source, repo, out):
        root.mkdir()
    (source / "worker.py").write_text("pass")
    (repo / "task.json").write_text("{}")
    sh = {"worker.py": hashlib.sha256(b"pass").hexdigest()}
    rh = {"task.json": hashlib.sha256(b"{}").hexdigest()}
    calls = []
    monkeypatch.setattr(isolated, "validate_public", lambda *a, **kw: calls.append(kw))
    spec = isolated_runtime.phase_spec(
        image="sha256:" + "a" * 64,
        source=source,
        source_hashes=sh,
        repo=repo,
        repo_hashes=rh,
        output=out,
        socket=sock,
        timeout=20,
        lineage="one",
        phase="solver",
        public_manifest_sha256="b" * 64,
    )
    assert calls == [{"expected_manifest_sha256": "b" * 64}]
    assert len(spec.provider_options["volumes"]) == 4
    assert all("/trusted" not in v and "/var/run/docker.sock" not in v for v in spec.provider_options["volumes"])
    assert spec.resources.cpu == 1 and spec.resources.memory_mib == 4096 and spec.ttl_s == 200
    with pytest.raises(ValueError, match="submission"):
        isolated_runtime.phase_spec(
            image="sha256:" + "a" * 64,
            source=source,
            source_hashes=sh,
            repo=repo,
            repo_hashes=rh,
            output=out,
            socket=sock,
            timeout=20,
            lineage="one",
            phase="verifier",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("now,expected_timeout", [(103, 7), (111, None)])
async def test_container_startup_consumes_deadline(tmp_path, monkeypatch, now, expected_timeout):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    calls = []
    cleaned = []

    class Box:
        async def serialize(self):
            return {"sandbox_id": "owned"}

        async def exec(self, command, **kw):
            calls.append(kw["timeout_s"])
            return SimpleNamespace(return_code=0, error_type=None, stdout="done", stderr="")

    @asynccontextmanager
    async def owned(box, receipt):
        try:
            yield box
        finally:
            cleaned.append(True)

    monkeypatch.setattr(isolated_runtime, "DockerProvider", lambda **kw: None)
    monkeypatch.setattr(isolated_runtime, "AsyncSandbox", lambda *a: Box())
    monkeypatch.setattr(isolated_runtime, "_owned_sandbox", owned)
    monkeypatch.setattr(isolated_runtime.time, "time", lambda: now)
    result = await isolated_runtime.execute_phase(SimpleNamespace(ttl_s=280), tmp_path, ["solver"], {}, deadline=110)
    assert cleaned == [True]
    assert calls == ([] if expected_timeout is None else [expected_timeout])
    assert result["error_type"] == ("deadline_before_exec" if expected_timeout is None else None)
