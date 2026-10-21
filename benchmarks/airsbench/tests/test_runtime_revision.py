import json
from pathlib import Path

import pytest

from benchmarks.airsbench.native_prepare import runtime_images

MANIFEST = Path(__file__).parents[1] / "frozen_sources.json"


def test_runtime_revision_is_explicit_and_retains_unmapped_original_images():
    original, historical = runtime_images(MANIFEST)
    selected, receipt = runtime_images(MANIFEST, "reconstructed-v1")
    assert historical["revision"] == "historical"
    assert receipt["revision"] == "reconstructed-v1"
    changed = {name for name in original if original[name] != selected[name]}
    assert changed == {"GraphRegressionZincMae", "CvMolecularPropertyPredictionQm9MeanAbsoluteError"}
    assert len(selected) == 5
    assert json.loads(MANIFEST.read_text())["tasks"][0]["image"] == original["GraphRegressionZincMae"]


@pytest.mark.parametrize("fault", ["manifest", "historical", "mutable"])
def test_revision_cannot_rebind_historical_science(tmp_path, fault):
    (tmp_path / "frozen_sources.json").write_bytes(MANIFEST.read_bytes())
    revisions = tmp_path / "runtime_revisions"
    revisions.mkdir()
    data = json.loads((MANIFEST.parent / "runtime_revisions/reconstructed-v1.json").read_text())
    binding = data["images"]["GraphRegressionZincMae"]
    if fault == "manifest":
        data["frozen_manifest_sha256"] = "wrong"
    elif fault == "historical":
        binding["historical_image"] = "sha256:" + "0" * 64
    else:
        binding["runtime_image"] = "mutable:latest"
    (revisions / "reconstructed-v1.json").write_text(json.dumps(data))
    with pytest.raises(ValueError):
        runtime_images(tmp_path / "frozen_sources.json", "reconstructed-v1")


def test_revision_cannot_load_arbitrary_path():
    with pytest.raises(ValueError):
        runtime_images(MANIFEST, "../../other")
