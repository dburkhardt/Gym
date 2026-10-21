# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from nemo_gym import dataset_staging
from nemo_gym.dataset_staging import stage_file


def test_fifo_destination_is_rejected_without_waiting_for_writer(tmp_path):
    source, destination = tmp_path / "cached", tmp_path / "final"
    source.write_bytes(b"dataset")
    os.mkfifo(destination)
    result = subprocess.run(
        [sys.executable, "-c", "from nemo_gym.dataset_staging import stage_file; import sys; stage_file(*sys.argv[1:])",
         str(source), str(destination)],
        capture_output=True, text=True, timeout=5,
    )
    assert result.returncode != 0
    assert "Staging path must be a regular file" in result.stderr
    assert destination.exists()


def test_interruption_preserves_final_and_resumes(tmp_path, monkeypatch):
    source, destination = tmp_path / "cached", tmp_path / "final"
    source.write_bytes(b"new dataset" * 1000)
    destination.write_bytes(b"old complete dataset")
    copy = dataset_staging.shutil.copyfileobj

    def interrupted(incoming, staged, length):
        staged.write(incoming.read(37))
        raise OSError("interrupted transfer")

    monkeypatch.setattr(dataset_staging.shutil, "copyfileobj", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        stage_file(source, destination)
    assert destination.read_bytes() == b"old complete dataset"
    assert next(tmp_path.glob("*.partial")).read_bytes() == source.read_bytes()[:37]
    offsets = []

    def resumed(incoming, staged, length):
        offsets.append(incoming.tell())
        copy(incoming, staged, length)

    monkeypatch.setattr(dataset_staging.shutil, "copyfileobj", resumed)
    stage_file(source, destination)
    assert offsets == [37]
    assert destination.read_bytes() == source.read_bytes()
    assert not list(tmp_path.glob("*.partial"))


def test_corrupt_partial_is_rebuilt(tmp_path):
    source, destination = tmp_path / "cached", tmp_path / "final"
    source.write_bytes(b"correct dataset")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (tmp_path / f".final.{digest}.partial").write_bytes(b"corrupt")
    stage_file(source, destination, expected_sha256=digest)
    assert destination.read_bytes() == source.read_bytes()


def test_wrong_digest_does_not_publish(tmp_path):
    source, destination = tmp_path / "cached", tmp_path / "final"
    source.write_bytes(b"bad")
    destination.write_bytes(b"previous")
    with pytest.raises(ValueError, match="checksum mismatch"):
        stage_file(source, destination, expected_sha256="0" * 64)
    assert destination.read_bytes() == b"previous"


def test_concurrent_writers_publish_once(tmp_path, monkeypatch):
    source, destination = tmp_path / "cached", tmp_path / "final"
    source.write_bytes(b"data" * 100000)
    copy = dataset_staging.shutil.copyfileobj
    calls = []

    def tracked(incoming, staged, length):
        calls.append(1)
        copy(incoming, staged, length)

    monkeypatch.setattr(dataset_staging.shutil, "copyfileobj", tracked)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: stage_file(source, destination), range(4)))
    assert results == [destination] * 4
    assert calls == [1]
    assert destination.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("target", ["destination", "partial", "lock"])
def test_destination_and_staging_symlinks_refused(tmp_path, target):
    source, destination, outside = tmp_path / "cached", tmp_path / "final", tmp_path / "outside"
    source.write_bytes(b"data")
    outside.write_bytes(b"untouched")
    digest = hashlib.sha256(b"data").hexdigest()
    paths = {
        "destination": destination,
        "partial": tmp_path / f".final.{digest}.partial",
        "lock": tmp_path / ".final.stage.lock",
    }
    paths[target].symlink_to(outside)
    with pytest.raises((ValueError, OSError)):
        stage_file(source, destination)
    assert outside.read_bytes() == b"untouched"


def test_cached_source_symlink_allowed(tmp_path):
    blob = tmp_path / "blob"
    blob.write_bytes(b"cached bytes")
    source = tmp_path / "snapshot"
    source.symlink_to(blob)
    destination = tmp_path / "final"
    stage_file(source, destination)
    assert destination.read_bytes() == blob.read_bytes()


def test_source_change_during_copy_is_not_published(tmp_path, monkeypatch):
    source, destination = tmp_path / "cached", tmp_path / "final"
    source.write_bytes(b"original")
    copy = dataset_staging.shutil.copyfileobj

    def changed(incoming, staged, length):
        source.write_bytes(b"modified")
        copy(incoming, staged, length)

    monkeypatch.setattr(dataset_staging.shutil, "copyfileobj", changed)
    with pytest.raises(ValueError, match="checksum mismatch"):
        stage_file(source, destination)
    assert not destination.exists()


def test_hf_raw_artifact_preserves_old_final_on_interruption(tmp_path, monkeypatch):
    from nemo_gym import hf_utils
    from nemo_gym.config_types import DownloadJsonlDatasetHuggingFaceConfig

    cached, destination = tmp_path / "cache.jsonl", tmp_path / "renamed.jsonl"
    cached.write_bytes(b'{"responses_create_params": {}}\n')
    destination.write_bytes(b"previous complete file")
    monkeypatch.setattr(hf_utils, "hf_hub_download", lambda **kwargs: str(cached))
    config = DownloadJsonlDatasetHuggingFaceConfig(
        repo_id="example/test", artifact_fpath="cache.jsonl", output_fpath=str(destination)
    )
    copy = dataset_staging.shutil.copyfileobj

    def interrupted(incoming, staged, length):
        staged.write(incoming.read(5))
        raise OSError("interrupted")

    monkeypatch.setattr(dataset_staging.shutil, "copyfileobj", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        hf_utils.download_hf_dataset_as_jsonl(config)
    assert destination.read_bytes() == b"previous complete file"
    monkeypatch.setattr(dataset_staging.shutil, "copyfileobj", copy)
    hf_utils.download_hf_dataset_as_jsonl(config)
    assert destination.read_bytes() == cached.read_bytes()


def test_verified_destination_needs_no_source(tmp_path):
    destination = tmp_path / "final"
    destination.write_bytes(b"verified")
    digest = hashlib.sha256(b"verified").hexdigest()
    assert stage_file(tmp_path / "absent", destination, expected_sha256=digest) == destination


def test_publication_fsyncs_parent(tmp_path, monkeypatch):
    import os
    import stat

    source, destination = tmp_path / "cached", tmp_path / "final"
    source.write_bytes(b"data")
    fsync = os.fsync
    kinds = []

    def tracked(fd):
        kinds.append(stat.S_IFMT(os.fstat(fd).st_mode))
        fsync(fd)

    monkeypatch.setattr(dataset_staging.os, "fsync", tracked)
    stage_file(source, destination)
    assert kinds == [stat.S_IFREG, stat.S_IFDIR]


def test_failed_directory_fsync_is_retried_on_cache_hit(tmp_path, monkeypatch):
    source, destination = tmp_path / "cached", tmp_path / "final"
    source.write_bytes(b"data")
    sync = dataset_staging._sync_parent
    calls = []

    def fail_once(path):
        calls.append(path)
        if len(calls) == 1:
            raise OSError("directory fsync failed")
        sync(path)

    monkeypatch.setattr(dataset_staging, "_sync_parent", fail_once)
    with pytest.raises(OSError, match="directory fsync failed"):
        stage_file(source, destination)
    assert destination.read_bytes() == b"data"
    stage_file(source, destination)
    assert calls == [destination, destination]


def _killed_copy_worker(source, destination, ready):
    import time

    def blocked(incoming, staged, length):
        staged.write(incoming.read(37))
        staged.flush()
        ready.set()
        while True:
            time.sleep(1)

    dataset_staging.shutil.copyfileobj = blocked
    stage_file(source, destination)


def test_sigkill_copy_resumes_and_preserves_final(tmp_path, monkeypatch):
    import multiprocessing

    source, destination = tmp_path / "cached", tmp_path / "final"
    source.write_bytes(b"new data" * 10000)
    destination.write_bytes(b"old complete data")
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    child = context.Process(target=_killed_copy_worker, args=(source, destination, ready))
    child.start()
    try:
        assert ready.wait(10), "child never reached a flushed partial copy"
        child.kill()
        child.join(10)
        assert not child.is_alive()
        assert child.exitcode == -9
        assert destination.read_bytes() == b"old complete data"
        offsets = []
        copy = dataset_staging.shutil.copyfileobj

        def resumed(incoming, staged, length):
            offsets.append(incoming.tell())
            copy(incoming, staged, length)

        monkeypatch.setattr(dataset_staging.shutil, "copyfileobj", resumed)
        stage_file(source, destination)
        assert offsets == [37]
        assert destination.read_bytes() == source.read_bytes()
    finally:
        if child.is_alive():
            child.kill()
            child.join(10)


def _concurrent_copy_worker(source, destination, start, calls):
    copy = dataset_staging.shutil.copyfileobj

    def tracked(incoming, staged, length):
        with calls.open("ab") as stream:
            stream.write(b"copy\n")
        copy(incoming, staged, length)

    dataset_staging.shutil.copyfileobj = tracked
    if not start.wait(10):
        raise TimeoutError("parent did not release concurrent writers")
    stage_file(source, destination)


def test_multiprocess_writers_publish_once(tmp_path):
    import multiprocessing

    source, destination, calls = tmp_path / "cached", tmp_path / "final", tmp_path / "calls"
    source.write_bytes(b"data" * 100000)
    context = multiprocessing.get_context("fork")
    start = context.Event()
    children = [
        context.Process(target=_concurrent_copy_worker, args=(source, destination, start, calls)) for _ in range(4)
    ]
    try:
        for child in children:
            child.start()
        start.set()
        for child in children:
            child.join(10)
            assert not child.is_alive()
            assert child.exitcode == 0
        assert calls.read_bytes() == b"copy\n"
        assert destination.read_bytes() == source.read_bytes()
    finally:
        for child in children:
            if child.is_alive():
                child.kill()
                child.join(10)
