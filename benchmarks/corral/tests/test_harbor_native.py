# SPDX-License-Identifier: Apache-2.0
import pytest

from benchmarks.corral.harbor_native import copy_state, run_phase


def test_saved_state_rejects_links_before_copy(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "link").symlink_to(tmp_path / "private")
    with pytest.raises(ValueError, match="symlinks"):
        copy_state(source, tmp_path / "copy")
    assert not (tmp_path / "copy").exists()


@pytest.mark.asyncio
async def test_scorer_without_saved_state_fails_before_allocation(tmp_path):
    with pytest.raises(ValueError, match="saved state"):
        await run_phase(
            task_id="make_1_lvl3",
            phase="scorer",
            snapshot=tmp_path / "missing",
            endpoint="",
            output=tmp_path / "output",
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("guard", [None, False, "true", 1])
async def test_solver_explicit_guard_fails_closed_without_hydra(tmp_path, monkeypatch, guard):
    import sys

    monkeypatch.setattr(sys, "argv", ["ray::task", "--node-ip-address=127.0.0.1"])
    kwargs = {} if guard is None else {"gym_provider_dispatch_guard": guard}
    with pytest.raises(ValueError, match="guarded rollout"):
        await run_phase(
            task_id="make_1_lvl3",
            phase="solver",
            snapshot=tmp_path / "db",
            endpoint="http://localhost/ng-rollout/0-0/v1",
            output=tmp_path / "out",
            **kwargs,
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.asyncio
async def test_explicit_guard_reaches_bounded_stage_without_hydra(tmp_path, monkeypatch):
    import sys

    import benchmarks.corral.harbor_native as native

    monkeypatch.setattr(sys, "argv", ["ray::task", "--node-ip-address=127.0.0.1"])

    class ReachedStage(Exception):
        pass

    def stop(_):
        raise ReachedStage

    monkeypatch.setattr(native, "configuration_digest", stop)
    with pytest.raises(ReachedStage):
        await run_phase(
            task_id="make_1_lvl3",
            phase="solver",
            snapshot=tmp_path / "db",
            endpoint="http://localhost/ng-rollout/0-0/v1",
            output=tmp_path / "out",
            gym_provider_dispatch_guard=True,
        )


@pytest.mark.asyncio
async def test_true_guard_does_not_bypass_endpoint_scope(tmp_path):
    with pytest.raises(ValueError, match="guarded rollout"):
        await run_phase(
            task_id="make_1_lvl3",
            phase="solver",
            snapshot=tmp_path / "db",
            endpoint="http://localhost/v1",
            output=tmp_path / "out",
            gym_provider_dispatch_guard=True,
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.asyncio
async def test_deterministic_scorer_does_not_require_model_guard(tmp_path, monkeypatch):
    import json

    import benchmarks.corral.harbor_native as native

    submitted = tmp_path / "saved"
    submitted.mkdir()
    (submitted / "solver-result.json").write_text(
        json.dumps({"status": "submitted", "execution_id": "corral:make_1_lvl3:saved"})
    )

    class ReachedStage(Exception):
        pass

    def stop(_):
        raise ReachedStage

    monkeypatch.setattr(native, "configuration_digest", stop)
    with pytest.raises(ReachedStage):
        await run_phase(
            task_id="make_1_lvl3",
            phase="scorer",
            snapshot=tmp_path / "db",
            endpoint="",
            output=tmp_path / "out",
            submitted=submitted,
        )
