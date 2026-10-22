# SPDX-License-Identifier: Apache-2.0
import pytest

from benchmarks.corebench.external_inputs import CREATOR_IMAGE, VARIANT, recipe, stage_files


def row():
    return {
        "id": "capsule-8912293",
        "published_image": CREATOR_IMAGE,
        "filesystem_layout_variant": "8912293-relative-data-v1",
        "external_input_variant": VARIANT,
    }


def test_external_input_requires_explicit_identity_and_layout():
    assert recipe({}) == ""
    for field in row():
        with pytest.raises(ValueError, match="Unsupported"):
            recipe({**row(), field: "unknown"})


def test_external_input_missing_or_changed_bytes_never_published(tmp_path):
    destination = tmp_path / "context"
    destination.mkdir()
    with pytest.raises(ValueError, match="requires"):
        stage_files(row(), None, destination)
    source = tmp_path / "wrong.zip"
    source.write_bytes(b"wrong upstream bytes")
    with pytest.raises(ValueError):
        stage_files(row(), source, destination)
    assert list(destination.iterdir()) == []
    with pytest.raises(ValueError, match="without explicit"):
        stage_files({}, source, destination)


def test_default_context_does_not_add_profile(tmp_path):
    stage_files({}, None, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_complete_closure_recipe_and_all_or_nothing_staging(tmp_path, monkeypatch):
    import hashlib

    from benchmarks.corebench import external_inputs as module

    selected = {**row(), "external_input_variant": module.CLOSURE_VARIANT}
    script = recipe(selected)
    assert all(name in script for name in module.CLOSURE_INPUTS)
    assert "party_mun_zone_local" not in module.closure_profile()
    source = tmp_path / "sources"
    source.mkdir()
    destination = tmp_path / "context"
    destination.mkdir()
    monkeypatch.setattr(
        module,
        "CLOSURE_INPUTS",
        {"a.zip": hashlib.sha256(b"a").hexdigest(), "b.zip": hashlib.sha256(b"b").hexdigest()},
    )
    (source / "a.zip").write_bytes(b"a")
    (source / "b.zip").write_bytes(b"wrong")
    with pytest.raises(ValueError):
        stage_files(selected, source, destination)
    assert list(destination.iterdir()) == []
    (source / "b.zip").write_bytes(b"b")
    stage_files(selected, source, destination)
    assert (destination / "a.zip").read_bytes() == b"a"
    assert (destination / "b.zip").read_bytes() == b"b"
    assert "file.copy" in (destination / "external-input.Rprofile").read_text()
