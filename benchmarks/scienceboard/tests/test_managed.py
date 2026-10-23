# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.scienceboard.managed_desktop import ManagedDesktop
from benchmarks.scienceboard.prepare import prepare


SOURCE = Path(
    os.environ.get(
        "SCIENCEBOARD_SOURCE_ROOT",
        str(
            Path(__file__).resolve().parents[5]
            / "outputs/broad-eval-campaign/harbor/agent-expansion/sources/223/OS-Copilot-ScienceBoard-21b5eb0"
        ),
    )
)


@pytest.mark.skipif(not SOURCE.is_dir(), reason="Pinned creator checkout absent")
def test_real_creator_twenty(tmp_path):
    rows = [json.loads(line) for line in prepare(SOURCE, tmp_path).read_text().splitlines()]
    assert len(rows) == len({r["task_id"] for r in rows}) == 20
    for row in rows:
        relative = row["verifier_metadata"]["creator_task_path"]
        original = SOURCE / relative
        assert (tmp_path / "creator_tasks" / relative).read_bytes() == original.read_bytes()
        assert row["responses_create_params"]["input"] == [
            {"role": "user", "content": json.loads(original.read_text())["instruction"]}
        ]
        assert "evaluate" not in row and "initialize" not in row


class FakeSandbox:
    specs = []
    fail = False
    cleanup_fail = False
    instances = []

    def __init__(self, provider, spec):
        self.specs.append(spec)
        self.instances.append(self)
        self.stopped = False

    def start(self):
        if self.fail:
            raise RuntimeError("creation failed")

    def exec(self, *args, **kwargs):
        return SimpleNamespace(return_code=0, stdout="172.18.0.2\n")

    def stop(self):
        self.stopped = True
        if self.cleanup_fail:
            raise OSError("cleanup failed")


class Session:
    def get(self, url, **kwargs):
        return SimpleNamespace(status_code=200, content=b"\x89PNGfixture")

    def close(self):
        self.closed = True


def make():
    return ManagedDesktop(
        "sha256:" + "a" * 64, controller_factory=lambda **kw: kw, sandbox_factory=FakeSandbox, session_factory=Session
    )


def test_reset_and_no_reference_mounts():
    desktop = make()
    first = desktop.sandbox
    spec = FakeSandbox.specs[-1]
    assert not spec.ports and not spec.provider_options.get("volumes")
    assert spec.resources.memory_mib == 24576
    assert spec.entrypoint == ["/usr/local/bin/scienceboard-entrypoint"]
    assert desktop.controller == {"vm_ip": "172.18.0.2", "server_port": 5000}
    desktop._revert_to_snapshot()
    assert first.stopped
    desktop._start_emulator()
    assert desktop.sandbox is not first
    desktop.close()
    desktop.close()


def test_creation_cleanup_error_preserves_original(monkeypatch):
    monkeypatch.setattr(FakeSandbox, "fail", True)
    monkeypatch.setattr(FakeSandbox, "cleanup_fail", True)
    with pytest.raises(RuntimeError, match="creation failed") as caught:
        make()
    assert FakeSandbox.instances[-1].stopped
    assert "cleanup also failed" in caught.value.__notes__[0]


def test_readiness_timeout_cleans(monkeypatch):
    times = iter([0, 999])
    monkeypatch.setattr("benchmarks.scienceboard.managed_desktop.time.monotonic", lambda: next(times))
    with pytest.raises(TimeoutError):
        make()
    assert FakeSandbox.instances[-1].stopped


@pytest.mark.parametrize("image", ["latest", "sha256:bad"])
def test_mutable_image_rejected_before_create(image):
    with pytest.raises(ValueError):
        ManagedDesktop(image, controller_factory=lambda **kw: kw)


def test_source_path_and_byte_changes_fail_closed(tmp_path):
    import hashlib

    from benchmarks.scienceboard.prepare import checked_source

    (tmp_path / "file").write_bytes(b"original")
    digest = hashlib.sha256(b"original").hexdigest()
    assert checked_source(tmp_path, "file", digest).read_bytes() == b"original"
    (tmp_path / "link").symlink_to(tmp_path / "file")
    for path in ("../outside", "link"):
        with pytest.raises(ValueError):
            checked_source(tmp_path, path, digest)
    (tmp_path / "file").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        checked_source(tmp_path, "file", digest)


def test_actual_pinned_vmanager_lifecycle():
    native = pytest.importorskip("sci.vm.vmanager")
    manager = native.VManager(version="0.1", provider_name="docker", docker_image="sha256:" + "a" * 64)
    manager._env_factory = make
    native.ENVS[manager.key] = make
    with manager:
        assert manager.entered
        first = manager.env.sandbox
        manager.env._revert_to_snapshot()
        manager.env._start_emulator()
        assert first.stopped
        second = manager.env.sandbox
    assert second.stopped and not manager.entered
    dockerfile = (SOURCE / "docker/Portable.Dockerfile").read_text()
    assert 'ENTRYPOINT ["/usr/local/bin/scienceboard-entrypoint"]' in dockerfile
