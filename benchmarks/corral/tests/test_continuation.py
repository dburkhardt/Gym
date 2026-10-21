# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import pytest

from benchmarks.corral.continuation import MAX_CALLS, NAMESPACE, _boundary, completed_calls, projection


def calls(tmp_path, count=2):
    for i in range(count):
        (tmp_path / f"{i:06d}.intent.json").write_text(
            json.dumps({"logical_call": f"corral:lineage:{i}", "config_digest": "p"})
        )
        (tmp_path / f"{i:06d}.complete.json").write_text('{"completed": true}')
    return tmp_path


def session():
    state = SimpleNamespace(
        agent_runs={"agent": SimpleNamespace(status="running")},
        submission=None,
        runtime=SimpleNamespace(status="running"),
        actions={},
        tool_invocations={},
        environment=SimpleNamespace(values={}),
        workspace=SimpleNamespace(files={}),
        usage=SimpleNamespace(llm_calls=13),
        task=SimpleNamespace(scaffold={"max_iterations": MAX_CALLS}),
    )
    return SimpleNamespace(state=state, actor=SimpleNamespace(run_id="agent"))


def test_guarded_ordinal_continuity(tmp_path):
    assert completed_calls(calls(tmp_path), "lineage", "p") == 2
    (tmp_path / "000002.intent.json").write_text("{}")
    with pytest.raises(ValueError, match="unresolved"):
        completed_calls(tmp_path, "lineage", "p")


@pytest.mark.parametrize("change", ["identity", "failure", "gap", "malformed"])
def test_guarded_history_refuses_change(tmp_path, change):
    calls(tmp_path)
    if change == "identity":
        (tmp_path / "000000.intent.json").write_text("{}")
    elif change == "failure":
        (tmp_path / "000001.failure.json").write_text("{}")
    elif change == "gap":
        (tmp_path / "000000.intent.json").unlink()
    else:
        (tmp_path / "000001.complete.json").write_text("{}")
    with pytest.raises(ValueError):
        completed_calls(tmp_path, "lineage", "p")


@pytest.mark.parametrize(
    "change",
    ["deadline", "terminal", "pending_action", "pending_tool", "workspace", "jobs", "budget", "max", "delegate"],
)
def test_boundary_refuses_unsafe_state(change):
    s = session()
    now = 10
    if change == "deadline":
        now = 20
    elif change == "terminal":
        s.state.runtime.status = "failed"
    elif change == "pending_action":
        s.state.actions["x"] = SimpleNamespace(status="pending")
    elif change == "pending_tool":
        s.state.tool_invocations["x"] = SimpleNamespace(status="running")
    elif change == "workspace":
        s.state.workspace.files["x"] = "hash"
    elif change == "jobs":
        s.state.environment.values["jobs"] = {"x": True}
    elif change == "budget":
        s.state.usage.llm_calls = 50
    elif change == "max":
        s.state.task.scaffold["max_iterations"] = 51
    else:
        s.state.agent_runs["other"] = SimpleNamespace(status="running")
    with pytest.raises(ValueError):
        _boundary(s, now=now, deadline=20)


def test_original_budget_has_37_remaining():
    assert MAX_CALLS - _boundary(session(), now=10, deadline=20) == 37


def test_projection_retains_science_but_ignores_only_snapshot():
    value = {
        "through_commit_hash": "h",
        "agent_runs": {"a": {"algorithm_state": {NAMESPACE: {"history": 1}, "other": {"x": 2}}}},
        "task": "exact",
    }
    s = SimpleNamespace(model_dump=lambda **kwargs: json.loads(json.dumps(value)))
    before = projection(s)
    value["agent_runs"]["a"]["algorithm_state"][NAMESPACE] = {"history": 2}
    assert projection(s) == before
    value["agent_runs"]["a"]["algorithm_state"]["other"]["x"] = 3
    assert projection(s) != before


def captured(tmp_path, count=2):
    from nemo_gym.base_responses_api_model import CaptureStore
    from nemo_gym.episode_types import EpisodeId

    store = CaptureStore(tmp_path, provider_dispatch_guard=True)
    store.identity()
    episode = EpisodeId.from_capture_key("0-0")
    for i in range(count):
        binding = {"logical_call": f"corral:lineage:{i}", "agent_config": "p", "logical_rollout": episode.rollout_id}
        store.write_dispatch(
            episode.capture_key,
            str(i),
            {"provider_attempts": [{"outcome": "response_headers", "status_code": 200}], "provider_binding": binding},
        )
        store.record(
            episode.capture_key,
            {"model_call_id": str(i), "response": {"id": str(i)}, "replay_response": {"body": str(i)}},
        )
        store.clear_pending(episode.capture_key, str(i))
    return store, episode


def test_actual_capture_store_binds_ordinals_and_refuses_late_unknown(tmp_path):
    from benchmarks.corral.continuation import capture_binding
    from nemo_gym._checkpoint.control import ControlError

    store, episode = captured(tmp_path)
    assert capture_binding(tmp_path, episode, "lineage", "p", 2)
    with pytest.raises(ValueError, match="binding"):
        capture_binding(tmp_path, episode, "different", "p", 2)
    store.write_dispatch("0-0-a1", "unknown", {"provider_attempts": [{"outcome": "unknown"}]})
    with pytest.raises(ControlError, match="unresolved"):
        capture_binding(tmp_path, episode, "lineage", "p", 2)


def test_missing_capture_store_not_created(tmp_path):
    from benchmarks.corral.continuation import capture_binding
    from nemo_gym.episode_types import EpisodeId

    missing = tmp_path / "absent"
    with pytest.raises(Exception):
        capture_binding(missing, EpisodeId.from_capture_key("0-0"), "lineage", "p", 2)
    assert not missing.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation", ["none", "missing", "history", "state", "implementation", "deadline", "later_dispatch"]
)
async def test_saved_boundary_binding_refuses_mutation(tmp_path, monkeypatch, mutation):
    import benchmarks.corral.continuation as module

    s = session()
    s.state.usage.llm_calls = 2
    s.payload = None
    body = {"through_commit_hash": "original", "agent_runs": {"agent": {"algorithm_state": {}}}, "task": "original"}
    s.state.model_dump = lambda **kwargs: json.loads(json.dumps(body))

    async def save(namespace, payload):
        s.payload = payload
        body["agent_runs"]["agent"]["algorithm_state"][namespace] = payload

    s.set_agent_state = save
    s.get_agent_state = lambda namespace: s.payload
    implementation = {"source": "unchanged"}
    monkeypatch.setattr(module, "implementation_binding", lambda: dict(implementation))
    calls_dir = tmp_path / "calls"
    calls_dir.mkdir()
    calls(calls_dir)
    store, episode = captured(tmp_path / "capture")
    kwargs = dict(
        calls=calls_dir,
        lineage="lineage",
        protocol="p",
        deadline=20,
        now=10,
        capture_dir=tmp_path / "capture",
        episode=episode,
    )
    await module.save_boundary(s, [{"role": "user", "content": "exact"}], **kwargs)
    if mutation == "none":
        assert module.validate_boundary(s, **kwargs)["next_ordinal"] == 2
        return
    if mutation == "missing":
        s.payload = None
    elif mutation == "history":
        s.payload["messages"][0]["content"] = "changed"
    elif mutation == "state":
        body["task"] = "different"
    elif mutation == "implementation":
        implementation["source"] = "different"
    elif mutation == "deadline":
        kwargs["deadline"] = 21
    else:
        store.write_dispatch("0-0-a1", "unknown", {"provider_attempts": [{"outcome": "unknown"}]})
    with pytest.raises(Exception):
        module.validate_boundary(s, **kwargs)


@pytest.mark.parametrize("key", ["0-1", "0-0-a1"])
def test_continuation_endpoint_cannot_change_capture_identity(tmp_path, monkeypatch, key):
    import sys
    import types

    from benchmarks.corral.guarded_agent import guarded_agent
    from nemo_gym.episode_types import EpisodeId

    base = types.ModuleType("corral.agents.base_agent")
    base.llm_call = object()
    agents = types.ModuleType("corral.agents")
    agents.base_agent = base
    cli = types.ModuleType("corral.cli")
    cli.create_agent = lambda **kwargs: pytest.fail("agent created before refusal")
    root = types.ModuleType("corral")
    root.agents = agents
    for name, module in [
        ("corral", root),
        ("corral.agents", agents),
        ("corral.agents.base_agent", base),
        ("corral.cli", cli),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    original = base.llm_call
    with pytest.raises(ValueError, match="endpoint differs"):
        with guarded_agent(
            f"http://model/ng-rollout/{key}/v1",
            "lineage",
            tmp_path / "calls",
            dispatch_digest="a" * 64,
            continuation_session=object(),
            capture_dir=tmp_path / "capture",
            episode=EpisodeId.from_capture_key("0-0"),
            deadline=20,
        ):
            pytest.fail("continuation admitted")
    assert base.llm_call is original and not list(tmp_path.iterdir())
