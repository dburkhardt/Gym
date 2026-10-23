# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib

import pytest

from benchmarks.corebench import external_inputs, model_cache


def row():
    return {
        "id": "capsule-3249574",
        "published_image": model_cache.CREATOR,
        "external_input_variant": model_cache.VARIANT,
    }


def test_exact_opt_in_recipe():
    text = external_inputs.recipe(row())
    assert "HF_HUB_OFFLINE=1" in text
    assert all(sha in text for sha in model_cache.FILES.values())
    assert external_inputs.recipe({}) == ""
    with pytest.raises(ValueError):
        external_inputs.recipe(dict(row(), id="capsule-3272782"))


def test_staged_tree_verified_and_clean(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights").write_bytes(b"pinned")
    monkeypatch.setattr(model_cache, "FILES", {"weights": hashlib.sha256(b"pinned").hexdigest()})
    dest = tmp_path / "context"
    dest.mkdir()
    model_cache.stage(source, dest)
    assert (dest / "model-cache/weights").read_bytes() == b"pinned"
    assert list((dest / "model-cache").iterdir()) == [dest / "model-cache/weights"]
    (source / "weights").write_bytes(b"changed")
    with pytest.raises(Exception):
        model_cache.stage(source, dest)


def test_extra_and_symlink_inputs_refused(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "extra").write_text("x")
    monkeypatch.setattr(model_cache, "FILES", {})
    with pytest.raises(ValueError):
        model_cache.stage(source, tmp_path / "dest")
    (source / "extra").unlink()
    (source / "link").symlink_to("/etc/passwd")
    with pytest.raises(ValueError):
        model_cache.stage(source, tmp_path / "dest")


def test_existing_destination_and_late_bad_file_unchanged(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "first").write_bytes(b"good")
    (source / "second").write_bytes(b"bad")
    monkeypatch.setattr(
        model_cache,
        "FILES",
        {
            "first": hashlib.sha256(b"good").hexdigest(),
            "second": hashlib.sha256(b"expected").hexdigest(),
        },
    )
    dest = tmp_path / "context"
    dest.mkdir()
    existing = dest / "model-cache"
    existing.mkdir()
    (existing / "stale").write_bytes(b"preserve")
    with pytest.raises(ValueError, match="existing"):
        model_cache.stage(source, dest)
    assert (existing / "stale").read_bytes() == b"preserve"
    assert len(list(existing.iterdir())) == 1
    (existing / "stale").unlink()
    existing.rmdir()
    with pytest.raises(Exception):
        model_cache.stage(source, dest)
    assert not existing.exists()
