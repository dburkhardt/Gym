# SPDX-License-Identifier: Apache-2.0
import sys

import pytest

from benchmarks.blade import harbor_native as native


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["solver", "scorer"])
@pytest.mark.parametrize("guard", [None, False, "true", 1])
async def test_explicit_guard_fails_closed_without_hydra(tmp_path, monkeypatch, phase, guard):
    monkeypatch.setattr(sys, "argv", ["ray::task", "--node-ip-address=127.0.0.1"])
    kwargs = {} if guard is None else {"gym_provider_dispatch_guard": guard}
    with pytest.raises(ValueError, match="guarded Gym"):
        await native.run_phase(
            root=tmp_path,
            dataset="crofoot",
            phase=phase,
            endpoint="http://localhost/ng-rollout/0-0/v1",
            output=tmp_path / "out",
            **kwargs,
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["solver", "scorer"])
async def test_explicit_guard_reaches_staging_without_hydra(tmp_path, monkeypatch, phase):
    monkeypatch.setattr(sys, "argv", ["ray::task", "--node-ip-address=127.0.0.1"])

    class ReachedStage(Exception):
        pass

    def stop(root):
        assert root == tmp_path
        raise ReachedStage

    monkeypatch.setattr(native, "validate_staged", stop)
    with pytest.raises(ReachedStage):
        await native.run_phase(
            root=tmp_path,
            dataset="crofoot",
            phase=phase,
            endpoint="http://localhost/ng-rollout/0-0/v1",
            output=tmp_path / "out",
            gym_provider_dispatch_guard=True,
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["http://localhost/v1", "http://localhost/ng-rollout/0-0/other"])
async def test_true_guard_does_not_bypass_endpoint_scope(tmp_path, endpoint):
    with pytest.raises(ValueError, match="guarded Gym"):
        await native.run_phase(
            root=tmp_path,
            dataset="crofoot",
            phase="solver",
            endpoint=endpoint,
            output=tmp_path / "out",
            gym_provider_dispatch_guard=True,
        )
    assert not (tmp_path / "out").exists()
