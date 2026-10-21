# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import nemo_gym.rollout_collection as collection
from nemo_gym._checkpoint.evaluation import EvaluationCheckpoint
from nemo_gym.agent_utils import process_supervisor as supervisor
from nemo_gym.config_types import ConfigError
from tests.unit_tests.test_rollout_collection import TestDispatchBudget as BudgetFixtures


@pytest.fixture
def bound(monkeypatch):
    monkeypatch.setattr(supervisor, "_kernel_boot_id", lambda: "boot")
    monkeypatch.delenv(supervisor.COLLECTOR_DEADLINE_ENV, raising=False)

    def set_bound(deadline=100, reserve=10):
        monkeypatch.setenv(
            supervisor.COLLECTOR_DEADLINE_ENV,
            json.dumps(
                {
                    "version": 1,
                    "boot_id": "boot",
                    "deadline": deadline,
                    "cleanup_reserve": reserve,
                }
            ),
        )

    return set_bound


@pytest.mark.parametrize(
    "raw",
    [
        "null",
        "{}",
        "[]",
        '{"version":1}',
        '{"version":1,"boot_id":"other","deadline":100,"cleanup_reserve":10}',
        '{"version":1,"boot_id":"boot","deadline":NaN,"cleanup_reserve":10}',
    ],
)
def test_malformed_or_foreign_boot_refused(bound, monkeypatch, raw):
    monkeypatch.setenv(supervisor.COLLECTOR_DEADLINE_ENV, raw)
    with pytest.raises(ValueError, match="Invalid supervisor"):
        supervisor.collector_deadline_from_environment()


def test_default_and_nested_bounds_never_extend(bound, monkeypatch):
    assert supervisor.collector_deadline_from_environment() is None
    bound(100, 10)
    env = supervisor.collector_environment(200, 20)
    monkeypatch.setenv(supervisor.COLLECTOR_DEADLINE_ENV, env[supervisor.COLLECTOR_DEADLINE_ENV])
    assert supervisor.collector_deadline_from_environment() == 90


@pytest.mark.parametrize("now,expected_posts", [(71, []), (70, [0]), (69, [0])])
async def test_actual_collector_late_start_and_exact_boundary(bound, monkeypatch, now, expected_posts):
    bound(100, 10)
    monkeypatch.setattr(collection, "time", SimpleNamespace(monotonic=lambda: now, time=lambda: 0))
    monkeypatch.setattr(collection, "raise_for_status", AsyncMock())
    monkeypatch.setattr(collection, "get_response_json", AsyncMock(return_value={"reward": 1}))
    fixture = BudgetFixtures()
    posted = []
    results = [
        await future
        for future in fixture._helper(posted).run_examples(
            [fixture._row(0)],
            dispatch_budget_s=1000,
            drain_margin_s=20,
        )
    ]
    assert posted == expected_posts
    if not posted:
        assert results[0][1][collection.NG_NO_PERSIST_KEY]
        assert results[0][1][collection.NG_DISPATCH_DRAINED_KEY]


@pytest.mark.parametrize("margin", [None, 0, -1, float("nan"), float("inf")])
def test_explicit_full_drain_required(bound, margin):
    bound()
    fixture = BudgetFixtures()
    with pytest.raises(ValueError, match="explicit positive finite"):
        fixture._helper([]).run_examples([fixture._row(0)], drain_margin_s=margin)


async def test_checkpoint_wait_rechecks_and_preserves_history(bound, monkeypatch, tmp_path):
    bound(100, 10)
    clock = [69]
    monkeypatch.setattr(collection, "time", SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 0))
    fixture = BudgetFixtures()
    posted = []
    helper = fixture._helper(posted)
    reservation = object()
    released = []
    history = tmp_path / "unknown.capture.jsonl"
    history.write_bytes(b"unknown external request\n")

    async def before(row):
        await asyncio.sleep(0)
        clock[0] = 71
        return reservation

    helper._evaluation_checkpoint = SimpleNamespace(
        before_dispatch=before, after_not_dispatched=lambda row, token: released.append(token)
    )
    result = [await future for future in helper.run_examples([fixture._row(0)], drain_margin_s=20)]
    assert posted == [] and released == [reservation]
    assert result[0][1][collection.NG_NO_PERSIST_KEY]
    assert history.read_bytes() == b"unknown external request\n"


def test_reservation_release_never_removes_persisted_or_replaced_episode():
    row = BudgetFixtures._row(0)
    key = "0-0"
    token = object()
    checkpoint = object.__new__(EvaluationCheckpoint)
    checkpoint.active = {key: token}
    checkpoint.active_agents = {key: "agent"}
    checkpoint.returned = {"completed"}
    checkpoint.persisted = {"completed"}
    checkpoint.after_not_dispatched(row, token)
    assert not checkpoint.active and checkpoint.persisted == {"completed"}
    with pytest.raises(ValueError, match="changed"):
        checkpoint.after_not_dispatched(row, None)
    checkpoint.active[key] = object()
    with pytest.raises(ValueError, match="changed"):
        checkpoint.after_not_dispatched(row, token)
    assert key in checkpoint.active
    checkpoint.active[key] = token
    checkpoint.persisted.add(key)
    with pytest.raises(ValueError, match="already returned"):
        checkpoint.after_not_dispatched(row, token)
    assert checkpoint.active[key] is token
    checkpoint.persisted.remove(key)
    checkpoint.returned.add(key)
    with pytest.raises(ValueError, match="already returned"):
        checkpoint.after_not_dispatched(row, token)
    assert checkpoint.active[key] is token


async def test_missing_checkpoint_graph_refuses_without_reservation(bound, monkeypatch):
    bound(100, 10)
    monkeypatch.setattr(collection, "time", SimpleNamespace(monotonic=lambda: 69, time=lambda: 0))
    fixture = BudgetFixtures()
    posted = []
    helper = fixture._helper(posted)
    checkpoint = object.__new__(EvaluationCheckpoint)
    checkpoint.admission = asyncio.Event()
    checkpoint.admission.set()
    checkpoint.error = None
    checkpoint.config = {}
    checkpoint.active = {}
    checkpoint.active_agents = {}
    helper._evaluation_checkpoint = checkpoint
    iterator = helper.run_examples([fixture._row(0)], drain_margin_s=20)
    try:
        with pytest.raises(ConfigError, match="complete inspected chains"):
            for future in iterator:
                await future
    finally:
        await iterator.aclose()
    assert not posted and not checkpoint.active


async def test_cancelled_checkpoint_wait_preserves_existing_history(bound, monkeypatch):
    bound(100, 10)
    monkeypatch.setattr(collection, "time", SimpleNamespace(monotonic=lambda: 69, time=lambda: 0))
    fixture = BudgetFixtures()
    posted = []
    helper = fixture._helper(posted)
    checkpoint = object.__new__(EvaluationCheckpoint)
    checkpoint.admission = asyncio.Event()
    checkpoint.error = None
    existing = object()
    checkpoint.active = {"old-unknown": existing}
    checkpoint.active_agents = {"old-unknown": "other"}
    checkpoint.persisted = {"completed"}
    checkpoint.returned = {"completed"}
    helper._evaluation_checkpoint = checkpoint
    iterator = helper.run_examples([fixture._row(0)], drain_margin_s=20)

    async def consume():
        for future in iterator:
            await future

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await iterator.aclose()
    assert posted == []
    assert checkpoint.active == {"old-unknown": existing}
    assert checkpoint.persisted == {"completed"}
