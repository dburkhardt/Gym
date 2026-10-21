import hashlib
import json

import pytest

from benchmarks.scienceboard.controller_dependencies import LOCK, offline_requirements, verify_artifacts


def test_lock_retains_cpu_versions_without_network_urls():
    lock = json.loads(LOCK.read_text())
    text = offline_requirements(lock["wheels"])
    assert len(text.splitlines()) == 225
    assert "torch==2.5.1+cpu --hash=sha256:" in text
    assert "torchvision==0.20.1+cpu --hash=sha256:" in text
    assert "https://" not in text
    assert len(lock["os_packages"]) == 25
    assert {p["name"] for p in lock["nondefault_sources"]} == {"torch", "torchvision"}


def test_real_bytes_tamper_missing_and_symlink_refusal(tmp_path):
    p = tmp_path / "artifact.whl"
    p.write_bytes(b"checked wheel bytes")
    expected = {"artifact.whl": hashlib.sha256(p.read_bytes()).hexdigest()}
    assert verify_artifacts(tmp_path, expected)["artifact.whl"] == p
    p.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        verify_artifacts(tmp_path, expected)
    p.unlink()
    with pytest.raises(ValueError, match="missing"):
        verify_artifacts(tmp_path, expected)
    other = tmp_path / "other"
    other.write_bytes(b"checked wheel bytes")
    p.symlink_to(other)
    with pytest.raises(ValueError, match="symlink"):
        verify_artifacts(tmp_path, expected)


def test_path_escape_refused(tmp_path):
    with pytest.raises(ValueError, match="Invalid"):
        verify_artifacts(tmp_path, {"../outside": "0" * 64})


def test_duplicate_distribution_refused():
    with pytest.raises(ValueError, match="Duplicate"):
        offline_requirements({"test-1-py3-none-any.whl": "0" * 64, "test-2-py3-none-any.whl": "1" * 64})


def test_preparation_validates_before_shared_publication(tmp_path, monkeypatch):
    from benchmarks.scienceboard import controller_dependencies as module

    source = tmp_path / "source"
    source.mkdir()
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    archives = tmp_path / "archives"
    archives.mkdir()
    (source / "requirements.txt").write_text("pinned")
    (wheels / "test-1-py3-none-any.whl").write_bytes(b"wheel")
    (archives / "lib.deb").write_bytes(b"deb")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = tmp_path / "creator.json"
    manifest.write_text(
        json.dumps({"source_revision": "pin", "files": {"requirements.txt": sha(source / "requirements.txt")}})
    )
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps(
            {
                "source_revision": "pin",
                "scope": "controller-only",
                "wheels": {"test-1-py3-none-any.whl": sha(wheels / "test-1-py3-none-any.whl")},
                "os_packages": [{"artifact_filename": "lib.deb", "SHA256": sha(archives / "lib.deb")}],
            }
        )
    )
    monkeypatch.setattr(module, "LOCK", lock)
    monkeypatch.setattr(module, "MANIFEST", manifest)
    output = tmp_path / "prepared"
    module.prepare(source, wheels, archives, output)
    assert (output / "requirements.txt").read_text().startswith("test==1 --hash=sha256:")
    with pytest.raises(FileExistsError):
        module.prepare(source, wheels, archives, output)
    (archives / "lib.deb").write_bytes(b"bad")
    with pytest.raises(ValueError):
        module.prepare(source, wheels, archives, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
