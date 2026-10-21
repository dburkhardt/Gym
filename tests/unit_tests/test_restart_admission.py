# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest

from nemo_gym import restart_admission as admission
from nemo_gym.base_responses_api_model import CaptureStore


def fixture(tmp_path, monkeypatch):
    monkeypatch.setenv(admission.KEY_ENV, "a" * 64)
    config = {
        "provider_dispatch_guard": True,
        "observability_enabled": True,
        "model_call_capture_dir": str(tmp_path),
        "agent": {"responses_api_agents": {"scicode_agent": {"entrypoint": "app.py", "num_workers": 1}}},
    }
    row = {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "_ng_attempt_index": 1,
        "problem_id": "20",
        "responses_create_params": {"input": [], "temperature": 0, "max_output_tokens": 16384},
    }
    store = CaptureStore(tmp_path, provider_dispatch_guard=True)
    store.identity()
    store.path_for("0-0").write_text(
        json.dumps(
            {
                "model_call_id": "call",
                "provider_attempts": [{"attempt_id": "a", "outcome": "rejected", "status_code": 429}],
            }
        )
        + "\n"
    )
    proof = admission.sign(
        row, agent="agent", prior="0-0", binding="b" * 64, receipt_sha256="c" * 64, capture_dirs=[tmp_path]
    )
    return config, row, store, proof


def test_valid_proof_consumed_once_and_prior_preserved(tmp_path, monkeypatch):
    config, row, store, proof = fixture(tmp_path, monkeypatch)
    before = store.path_for("0-0").read_bytes()
    used = set()
    assert admission.consume(proof, row, agent="agent", config=config, consumed=used)["prior"] == "0-0"
    with pytest.raises(ValueError, match="consumed"):
        admission.consume(proof, row, agent="agent", config=config, consumed=used)
    assert store.path_for("0-0").read_bytes() == before


@pytest.mark.parametrize(
    "change", ["forged", "task", "attempt", "model", "role", "store", "old_key", "missing_key", "workers"]
)
def test_changed_or_forged_admission_refused(tmp_path, monkeypatch, change):
    config, row, store, proof = fixture(tmp_path, monkeypatch)
    agent = "agent"
    if change == "forged":
        proof = proof[:-1] + ("0" if proof[-1] != "0" else "1")
    elif change == "task":
        row["problem_id"] = "79"
    elif change == "attempt":
        row["_ng_attempt_index"] = 2
    elif change == "model":
        row["responses_create_params"]["model"] = "other"
    elif change == "role":
        agent = "other"
    elif change == "store":
        config["model_call_capture_dir"] = str(tmp_path / "other")
    elif change == "old_key":
        admission.install_key()
    elif change == "missing_key":
        monkeypatch.delenv(admission.KEY_ENV)
    elif change == "workers":
        config["agent"]["responses_api_agents"]["scicode_agent"]["num_workers"] = 2
    with pytest.raises(ValueError):
        admission.consume(proof, row, agent=agent, config=config, consumed=set())


@pytest.mark.parametrize("kind", ["unknown", "occupied"])
def test_live_capture_evidence_refused_after_signing(tmp_path, monkeypatch, kind):
    config, row, store, proof = fixture(tmp_path, monkeypatch)
    if kind == "unknown":
        store.write_dispatch("0-0", "late", {"provider_attempts": [{"attempt_id": "late", "outcome": "unknown"}]})
    else:
        store.path_for("0-0-a1").write_text("{}\n")
    with pytest.raises(Exception, match="unresolved|prior provider"):
        admission.consume(proof, row, agent="agent", config=config, consumed=set())


def test_signer_rejects_wrong_prior_and_missing_provenance(tmp_path, monkeypatch):
    config, row, store, proof = fixture(tmp_path, monkeypatch)
    for prior, binding in [("1-0", "b" * 64), ("0-0", "")]:
        with pytest.raises(ValueError):
            admission.sign(
                row, agent="agent", prior=prior, binding=binding, receipt_sha256="c" * 64, capture_dirs=[tmp_path]
            )


def test_mixed_selected_restart_and_fresh_row(tmp_path, monkeypatch):
    config, row, store, proof = fixture(tmp_path, monkeypatch)
    selected = dict(prior={"0-0": "0-0"}, binding="b" * 64, receipt_sha256="c" * 64, capture_dirs=[tmp_path])
    assert admission.HEADER in admission.headers_for_request(selected, row, agent="agent")
    for attempt in (0, 1):
        other = {**row, "_ng_task_index": 2, "_ng_attempt_index": attempt}
        assert admission.headers_for_request(selected, other, agent="agent") == {}
        if attempt:
            with pytest.raises(ValueError, match="Missing"):
                admission.consume(None, other, agent="agent", config=config, consumed=set())


def test_replaced_capture_identity_refused(tmp_path, monkeypatch):
    config, row, store, proof = fixture(tmp_path, monkeypatch)
    (tmp_path / ".capture-store-identity").write_text("e" * 32)
    with pytest.raises(ValueError, match="binding mismatch"):
        admission.consume(proof, row, agent="agent", config=config, consumed=set())


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", [False, True])
async def test_actual_scicode_raw_http_admission_before_model(tmp_path, monkeypatch, tamper):
    from unittest.mock import AsyncMock

    from starlette.requests import Request

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "responses_api_agents/scicode_agent"))

    from responses_api_agents.scicode_agent.app import ScicodeAgentRunRequest
    from responses_api_agents.scicode_agent.tests.test_app import _agent, _run_request

    config, row, store, _ = fixture(tmp_path, monkeypatch)
    config["scicode_agent"] = config.pop("agent")
    agent = _agent(observability_enabled=True)
    agent.server_client.global_config_dict = config
    agent.server_client.post = AsyncMock(side_effect=RuntimeError("bounded model boundary"))
    raw = _run_request(n_steps=1, _ng_task_index=0, _ng_rollout_index=0, _ng_attempt_index=1).model_dump(
        mode="json", exclude_unset=True
    )
    # Type parsing adds defaults but raw transport proof binds exactly received fields.
    assert "tools" not in raw["responses_create_params"]
    proof = admission.sign(
        raw, agent="scicode_agent", prior="0-0", binding="b" * 64, receipt_sha256="c" * 64, capture_dirs=[tmp_path]
    )
    if tamper:
        raw["sub_steps"][0]["step_description_prompt"] = "changed"

    async def receive():
        return {"type": "http.request", "body": json.dumps(raw).encode(), "more_body": False}

    request = Request(
        {"type": "http", "method": "POST", "path": "/run", "headers": [(admission.HEADER.encode(), proof.encode())]},
        receive,
    )
    with pytest.raises((ValueError, RuntimeError), match="binding mismatch" if tamper else "bounded model boundary"):
        await agent.run(request, ScicodeAgentRunRequest.model_validate(raw))
    assert agent.server_client.post.await_count == (0 if tamper else 1)
    if not tamper:
        assert admission.HEADER not in agent.server_client.post.await_args.kwargs.get("headers", {})
        with pytest.raises(ValueError, match="consumed"):
            await agent.run(request, ScicodeAgentRunRequest.model_validate(raw))


@pytest.mark.asyncio
async def test_actual_collector_mixed_restart_headers(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from omegaconf import OmegaConf

    import nemo_gym.rollout_collection as collection

    config, row, store, proof = fixture(tmp_path, monkeypatch)
    config["relay"] = {"environment_servers": {"legacy_agent": {"agent_server": {"name": "agent"}}}}
    client = MagicMock()
    client.global_config_dict = OmegaConf.create(config)
    client.post = AsyncMock(return_value=MagicMock(status=200))
    monkeypatch.setattr(collection, "setup_server_client_utils", lambda *args, **kwargs: client)
    monkeypatch.setattr(collection, "raise_for_status", AsyncMock())
    monkeypatch.setattr(collection, "get_response_json", AsyncMock(return_value={"response": {}}))
    helper = collection.RolloutCollectionHelper()
    helper._explicit_restart_admission = dict(
        prior={"0-0": "0-0"}, binding="b" * 64, receipt_sha256="c" * 64, capture_dirs=[tmp_path]
    )
    rows = [
        {**row, "agent_ref": {"name": "agent"}},
        {**row, "agent_ref": {"name": "agent"}, "_ng_task_index": 1, "_ng_attempt_index": 0},
        {**row, "agent_ref": {"name": "agent"}, "_ng_task_index": 2, "_ng_attempt_index": 1},
    ]
    for future in helper._run_examples_with_metadata(rows):
        await future
    calls = {call.kwargs["json"]["_ng_task_index"]: call.kwargs for call in client.post.await_args_list}
    assert len(calls) == 3
    assert admission.HEADER in calls[0]["headers"]
    assert "headers" not in calls[1]
    assert "headers" not in calls[2]
    admission.consume(
        calls[0]["headers"][admission.HEADER], calls[0]["json"], agent="agent", config=config, consumed=set()
    )
