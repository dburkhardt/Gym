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
