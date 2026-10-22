# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import nemo_gym.rollout_collection as collection
from tests.unit_tests.test_rollout_collection import TestDispatchBudget as BudgetFixtures


def config(tmp_path, **overrides):
    return collection.RolloutCollectionConfig(
        input_jsonl_fpath=str(tmp_path / "in.jsonl"),
        output_jsonl_fpath=str(tmp_path / "out.jsonl"),
        num_samples_in_parallel=1,
        **dict(worker_lease_deadline_unix_s=100, worker_lease_cleanup_reserve_s=10, drain_margin_s=20, **overrides),
    )


@pytest.mark.parametrize(
    "field",
    [
        "worker_lease_deadline_unix_s",
        "worker_lease_cleanup_reserve_s",
        "worker_lease_startup_reserve_s",
        "drain_margin_s",
    ],
)
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_bound_refused(tmp_path, field, value):
    kwargs = dict(worker_lease_deadline_unix_s=100, worker_lease_cleanup_reserve_s=10, drain_margin_s=20)
    kwargs[field] = value
    with pytest.raises(ValueError):
        collection.RolloutCollectionConfig(
            input_jsonl_fpath=str(tmp_path / "in"),
            output_jsonl_fpath=str(tmp_path / "out"),
            num_samples_in_parallel=1,
            **kwargs,
        )


@pytest.mark.parametrize("missing", ["drain_margin_s", "worker_lease_cleanup_reserve_s"])
def test_lease_requires_full_task_and_cleanup(tmp_path, missing):
    kwargs = dict(worker_lease_deadline_unix_s=100, worker_lease_cleanup_reserve_s=10, drain_margin_s=20)
    del kwargs[missing]
    with pytest.raises(ValueError, match="explicit positive finite"):
        collection.RolloutCollectionConfig(
            input_jsonl_fpath=str(tmp_path / "in"),
            output_jsonl_fpath=str(tmp_path / "out"),
            num_samples_in_parallel=1,
            **kwargs,
        )


@pytest.mark.parametrize("now,allowed", [(70, True), (71, False), (101, False)])
async def test_collector_refuses_before_any_post_or_allows_exact_boundary(monkeypatch, now, allowed):
    monkeypatch.setattr(collection, "time", SimpleNamespace(monotonic=lambda: now, time=lambda: now))
    monkeypatch.setattr(collection, "raise_for_status", AsyncMock())
    monkeypatch.setattr(collection, "get_response_json", AsyncMock(return_value={"reward": 1}))
    fixture = BudgetFixtures()
    posted = []

    def run():
        return fixture._helper(posted).run_examples(
            [fixture._row(0)],
            dispatch_budget_s=1000,
            drain_margin_s=20,
            worker_lease_deadline_unix_s=100,
            worker_lease_cleanup_reserve_s=10,
            max_resident_tasks=1,
        )

    if allowed:
        for future in run():
            await future
        assert posted == [0]
    else:
        with pytest.raises(ValueError, match="cannot fit"):
            run()
        assert posted == []


async def test_checkpoint_delay_expires_lease_without_post_or_history_change(monkeypatch, tmp_path):
    clock = [69]
    monkeypatch.setattr(collection, "time", SimpleNamespace(monotonic=lambda: clock[0], time=lambda: clock[0]))
    fixture = BudgetFixtures()
    posted = []
    helper = fixture._helper(posted)
    token = object()
    released = []

    async def before(row):
        clock[0] = 71
        return token

    helper._evaluation_checkpoint = SimpleNamespace(
        before_dispatch=before, after_not_dispatched=lambda row, reservation: released.append(reservation)
    )
    history = tmp_path / "prior.jsonl"
    history.write_bytes(b"unknown historical request\n")
    results = [
        await future
        for future in helper.run_examples(
            [fixture._row(0)],
            drain_margin_s=20,
            worker_lease_deadline_unix_s=100,
            worker_lease_cleanup_reserve_s=10,
            max_resident_tasks=1,
        )
    ]
    assert posted == [] and released == [token]
    assert results[0][1][collection.NG_NO_PERSIST_KEY]
    assert history.read_bytes() == b"unknown historical request\n"


def test_service_start_requires_and_accounts_for_startup(tmp_path, monkeypatch):
    monkeypatch.setattr(collection, "time", SimpleNamespace(monotonic=lambda: 69, time=lambda: 69))
    c = config(tmp_path)
    with pytest.raises(ValueError, match="startup_reserve"):
        c.check_worker_lease(before_service_start=True)
    c.worker_lease_startup_reserve_s = 2
    with pytest.raises(ValueError, match="cannot fit"):
        c.check_worker_lease(before_service_start=True)
    c.worker_lease_startup_reserve_s = 1
    c.check_worker_lease(before_service_start=True)
    # Collection does not charge an already-completed startup reserve twice.
    c.check_worker_lease()


def test_lease_alone_requires_bounded_dispatch(tmp_path):
    with pytest.raises(ValueError, match="requires num_samples_in_parallel"):
        collection.RolloutCollectionConfig(
            input_jsonl_fpath=str(tmp_path / "in"),
            output_jsonl_fpath=str(tmp_path / "out"),
            worker_lease_deadline_unix_s=100,
            worker_lease_cleanup_reserve_s=10,
            drain_margin_s=20,
        )
    fixture = BudgetFixtures()
    with pytest.raises(ValueError, match="bounded dispatch"):
        fixture._helper([]).run_examples(
            [fixture._row(0)], worker_lease_deadline_unix_s=100, worker_lease_cleanup_reserve_s=10, drain_margin_s=20
        )


def test_ether0_campaign_horizon_does_not_override_worker_lease(monkeypatch):
    from datetime import datetime

    admitted = datetime.fromisoformat("2026-10-21T15:15:59.454571+00:00").timestamp()
    shutdown = datetime.fromisoformat("2026-10-21T15:22:17.796313+00:00").timestamp()
    monkeypatch.setattr(collection, "time", SimpleNamespace(time=lambda: admitted, monotonic=lambda: 0))
    fixture = BudgetFixtures()
    posted = []
    with pytest.raises(ValueError, match="cannot fit"):
        fixture._helper(posted).run_examples(
            [fixture._row(0)],
            dispatch_budget_s=2160,
            drain_margin_s=1200,
            worker_lease_deadline_unix_s=shutdown,
            worker_lease_cleanup_reserve_s=120,
            max_resident_tasks=2,
        )
    assert not posted


async def test_semaphore_wait_expires_lease(monkeypatch):
    import asyncio

    clock = [69]
    monkeypatch.setattr(collection, "time", SimpleNamespace(time=lambda: clock[0], monotonic=lambda: clock[0]))
    fixture = BudgetFixtures()
    posted = []
    semaphore = asyncio.Semaphore(0)
    iterator = fixture._helper(posted).run_examples(
        [fixture._row(0)],
        semaphore=semaphore,
        drain_margin_s=20,
        worker_lease_deadline_unix_s=100,
        worker_lease_cleanup_reserve_s=10,
    )

    async def collect():
        return [await future for future in iterator]

    task = asyncio.create_task(collect())
    await asyncio.sleep(0)
    clock[0] = 101
    semaphore.release()
    results = await task
    assert not posted and results[0][1][collection.NG_NO_PERSIST_KEY]


@pytest.mark.parametrize(
    "plan_only,deadline,expect_start,custom_driver",
    [(True, 60, False, False), (False, 60, False, False), (False, 100, True, False), (False, 100, False, True)],
)
@pytest.mark.parametrize("interpolated", [False, True])
def test_runtime_yaml_reaches_e2e_service_admission(
    tmp_path, monkeypatch, plan_only, deadline, expect_start, custom_driver, interpolated
):
    import json

    from omegaconf import OmegaConf

    from nemo_gym.cli import admission_evidence
    from nemo_gym.cli import eval as eval_cli

    output = tmp_path / "result.jsonl"
    prepared = tmp_path / "result" / "preprocessed_datasets" / "benchmark.jsonl"
    prepared.parent.mkdir(parents=True)
    row = dict(BudgetFixtures._row(0), _ng_dataset_num_repeats=1)
    prepared.write_text(json.dumps(row) + "\n")
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text(f"""split: benchmark
output_jsonl_fpath: {output}
reuse_existing_data_preparation: true
num_samples_in_parallel: 1
plan_only: {str(plan_only).lower()}
worker_lease_deadline_unix_s: {deadline}
worker_lease_cleanup_reserve_s: 10
worker_lease_startup_reserve_s: 1
drain_margin_s: 20
""")
    if interpolated:
        monkeypatch.setenv("TEST_WORKER_DEADLINE", str(deadline))
        runtime.write_text(runtime.read_text().replace(
            f"worker_lease_deadline_unix_s: {deadline}",
            "worker_lease_deadline_unix_s: ${oc.env:TEST_WORKER_DEADLINE}",
        ))
    if custom_driver:
        runtime.write_text(runtime.read_text() + "rollout_collection_driver: forbidden.module:run\n")
    monkeypatch.setattr(collection, "time", SimpleNamespace(time=lambda: 69, monotonic=lambda: 69))
    monkeypatch.setattr(eval_cli, "get_global_config_dict", lambda: OmegaConf.load(runtime))
    monkeypatch.setattr(eval_cli, "validate_eval_compatibility", lambda _: None)
    monkeypatch.setattr(eval_cli, "_validate_split_datasets_declared", lambda *_: None)
    monkeypatch.setattr(eval_cli, "_ensure_guarded_batch_manifest", lambda *_: None)
    monkeypatch.setattr(collection.RolloutCollectionHelper, "_preprocess_rows_from_config", lambda *_: [row])
    monkeypatch.setattr(admission_evidence, "resolved_admission_evidence", lambda *_: {})
    starts = []

    def start(*_):
        starts.append(True)
        raise RuntimeError("intercepted service start")

    monkeypatch.setattr(eval_cli.RunHelper, "start", start)
    if plan_only:
        eval_cli.e2e_rollout_collection()
        assert output.with_stem("result_dispatch_plan").with_suffix(".json").exists()
    elif custom_driver:
        with pytest.raises(SystemExit):
            eval_cli.e2e_rollout_collection()
    else:
        with pytest.raises(
            (ValueError, RuntimeError), match="intercepted service start" if expect_start else "cannot fit"
        ):
            eval_cli.e2e_rollout_collection()
    assert bool(starts) is expect_start


def test_updated_top_level_and_collector_lease_preserve_restart_binding(tmp_path):
    from nemo_gym.batch_restart import bind_execution
    from tests.unit_tests.test_batch_restart import fixture

    tracker, global_config, inputs, _, _, kwargs = fixture(tmp_path)
    for deadline in [100, 200]:
        lease = dict(
            worker_lease_deadline_unix_s=deadline, worker_lease_cleanup_reserve_s=10, worker_lease_startup_reserve_s=5
        )
        global_config.update(lease)
        assert (
            bind_execution(
                tracker,
                global_config,
                inputs,
                resume=True,
                require_prior=True,
                collector_config=lease,
                capture_dirs=[tmp_path / "captures"],
            )
            == kwargs["binding_sha256"]
        )


@pytest.mark.parametrize("value", ["100", "not-a-number", "0", "nan", None])
def test_collector_lazy_env_resolution(tmp_path, monkeypatch, value):
    from omegaconf import OmegaConf

    monkeypatch.delenv("TEST_UNUSED_MODEL_KEY", raising=False)
    if value is None:
        monkeypatch.delenv("TEST_WORKER_DEADLINE", raising=False)
    else:
        monkeypatch.setenv("TEST_WORKER_DEADLINE", value)
    runtime = OmegaConf.create(dict(
        input_jsonl_fpath=str(tmp_path / "in"),
        output_jsonl_fpath=str(tmp_path / "out"),
        num_samples_in_parallel=1,
        worker_lease_deadline_unix_s="${oc.env:TEST_WORKER_DEADLINE}",
        worker_lease_cleanup_reserve_s=10,
        drain_margin_s=20,
        unrelated_server={"api_key": "${oc.env:TEST_UNUSED_MODEL_KEY}"},
    ))
    if value == "100":
        resolved = collection.RolloutCollectionConfig.model_validate(runtime)
        assert resolved.worker_lease_deadline_unix_s == 100
    else:
        with pytest.raises(ValueError):
            collection.RolloutCollectionConfig.model_validate(runtime)
