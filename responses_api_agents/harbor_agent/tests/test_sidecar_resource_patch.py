import hashlib
from pathlib import Path

import pytest
import yaml

from responses_api_agents.harbor_agent.runtime_patches.sidecar_resources import apply, COMMIT, ORIGINAL_SHA256

SOURCE = Path(
    "../../outputs/broad-eval-campaign/foundation/harbor-upgrade/venv/lib/python3.13/site-packages/harbor/environments/docker/docker-compose-egress-control.yaml"
)


def original_file(tmp_path):
    if not SOURCE.exists():
        pytest.skip("Pinned Harbor source fixture unavailable")
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ORIGINAL_SHA256
    target = tmp_path / "harbor/environments/docker/docker-compose-egress-control.yaml"
    target.parent.mkdir(parents=True)
    target.write_bytes(raw)
    return target


def test_exact_patch_preserves_original_and_is_idempotent(tmp_path):
    target = original_file(tmp_path)
    package = tmp_path / "harbor"
    out = tmp_path / "receipt"
    a = apply(package, {"vcs_info": {"commit_id": COMMIT}}, out)
    assert apply(package, {"vcs_info": {"commit_id": COMMIT}}, out) == a
    service = yaml.safe_load(target.read_text())["services"]["harbor-docker-egress-control-sidecar"]
    assert service["mem_limit"] == "128m" and service["cpus"] == 0.25
    assert hashlib.sha256((out / "original-compose.yaml").read_bytes()).hexdigest() == ORIGINAL_SHA256


def test_modified_runtime_or_wrong_provenance_fails_before_patch(tmp_path):
    target = original_file(tmp_path)
    original = target.read_bytes()
    with pytest.raises(ValueError):
        apply(tmp_path / "harbor", {}, tmp_path / "receipt")
    assert target.read_bytes() == original
    target.write_bytes(original + b"# local change\n")
    with pytest.raises(ValueError):
        apply(tmp_path / "harbor", {"vcs_info": {"commit_id": COMMIT}}, tmp_path / "receipt")
    assert not (tmp_path / "receipt").exists()
