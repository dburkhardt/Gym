# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import multiprocessing
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym._checkpoint.control import ControlError
from nemo_gym._checkpoint.model import PolicyModelParticipant
from nemo_gym._checkpoint.model_workers import CoordinatedPolicyParticipant
from nemo_gym.base_responses_api_model import CaptureStore, ModelCallCaptureConfig
from nemo_gym.episode_types import EpisodeId
from nemo_gym.openai_utils import NeMoGymAsyncOpenAI
from nemo_gym.provider_dispatch import ProviderDispatch, current_provider_dispatch


def store_at(path: Path) -> CaptureStore:
    return CaptureStore(path, provider_dispatch_guard=True)


def dispatch_at(store: CaptureStore, key: str = "episode", call: str = "call") -> ProviderDispatch:
    store.mark_pending(key, call)
    return ProviderDispatch(lambda evidence: store.write_dispatch(key, call, evidence))


def test_guard_requires_capture():
    with pytest.raises(ValueError, match="observability_enabled"):
        ModelCallCaptureConfig(provider_dispatch_guard=True)


def test_physical_attempt_timing_survives_durable_dispatch(tmp_path, monkeypatch):
    import json

    ticks = iter([100.0, 101.0, 161.0, 162.0])
    monkeypatch.setattr("nemo_gym.provider_dispatch.time", lambda: next(ticks))
    store = store_at(tmp_path)
    dispatch = dispatch_at(store)
    dispatch.before_send()
    dispatch.response_headers(429)
    dispatch.attempts[-1]["outcome"] = "rejected"
    dispatch.persist(dispatch.evidence())
    dispatch.before_send()
    dispatch.response_headers(200)
    attempts = dispatch.evidence()["provider_attempts"]
    assert [(a["started_at"], a["response_headers_at"]) for a in attempts] == [(100.0, 101.0), (161.0, 162.0)]
    assert attempts[0]["attempt_id"] != attempts[1]["attempt_id"]
    # Evidence must reach disk, not just the in-memory dispatch object.
    persisted = json.loads(store.pending_path_for("episode", "call").read_text())
    assert persisted["provider_attempts"] == attempts
    # Timestamps do not make the second, accepted-but-uncaptured call retryable.
    with pytest.raises(RuntimeError, match="unresolved"):
        dispatch.before_send()


def test_queued_cancellation_is_not_sent(tmp_path):
    store = store_at(tmp_path)
    dispatch_at(store)
    store.assert_restore_safe(["episode"])
    store.clear_pending("episode", "call")
    assert not store.has_pending("episode")


def test_unknown_marker_is_not_removed(tmp_path):
    store = store_at(tmp_path)
    dispatch = dispatch_at(store)
    dispatch.before_send()
    store.clear_pending("episode", "call")
    assert store.has_pending("episode")
    with pytest.raises(ControlError, match="unresolved provider"):
        store.clear_all_pending("episode")


def test_failed_persistence_prevents_send_state(tmp_path, monkeypatch):
    store = store_at(tmp_path)
    dispatch = dispatch_at(store)
    monkeypatch.setattr(store, "_sync_directory", lambda: (_ for _ in ()).throw(OSError("disk failed")))
    with pytest.raises(OSError, match="disk failed"):
        dispatch.before_send()
    # Even if rename succeeded before fsync failed, recovery must not assume not_sent.
    with pytest.raises(ControlError, match="unresolved provider"):
        store.assert_restore_safe(["episode"])


@pytest.mark.asyncio
async def test_429_then_success_records_two_sends(tmp_path, monkeypatch):
    dispatch = dispatch_at(store_at(tmp_path))
    replies = [
        SimpleNamespace(status=429, content=SimpleNamespace(read=AsyncMock(return_value=b"busy"))),
        SimpleNamespace(status=200),
    ]
    request = AsyncMock(side_effect=replies)
    monkeypatch.setattr("nemo_gym.openai_utils.request", request)
    monkeypatch.setattr("nemo_gym.openai_utils.sleep", AsyncMock())
    token = current_provider_dispatch.set(dispatch)
    try:
        client = NeMoGymAsyncOpenAI(api_key="fake", base_url="http://localhost", max_http_attempts=2)
        assert await client._request_with_retry() is replies[-1]
    finally:
        current_provider_dispatch.reset(token)
    assert [row["outcome"] for row in dispatch.attempts] == ["rejected", "response_headers"]
    assert len({row["attempt_id"] for row in dispatch.attempts}) == 2
    assert all(call.kwargs["_max_num_tries"] == 1 for call in request.await_args_list)
    assert all(call.kwargs["_resend_after_disconnect"] is False for call in request.await_args_list)


@pytest.mark.asyncio
async def test_transport_timeout_is_not_retried(tmp_path, monkeypatch):
    dispatch = dispatch_at(store_at(tmp_path))
    request = AsyncMock(side_effect=TimeoutError("after send"))
    monkeypatch.setattr("nemo_gym.openai_utils.request", request)
    token = current_provider_dispatch.set(dispatch)
    try:
        client = NeMoGymAsyncOpenAI(api_key="fake", base_url="http://localhost", max_http_attempts=5)
        with pytest.raises(TimeoutError):
            await client._request_with_retry()
    finally:
        current_provider_dispatch.reset(token)
    assert request.await_count == 1
    assert dispatch.attempts[0]["outcome"] == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize("coordinated", [False, True])
async def test_old_checkpoint_cannot_erase_later_dispatch(tmp_path, coordinated):
    store = store_at(tmp_path)
    dispatch_at(store, "episode-a2").before_send()
    participant = (
        CoordinatedPolicyParticipant(None, expected_workers=2, provider_store=store)
        if coordinated
        else PolicyModelParticipant(provider_store=store)
    )
    with pytest.raises(ControlError, match="unresolved provider"):
        await participant.install([], [EpisodeId(rollout_id="episode")])
    assert not participant.restored_targets if coordinated else not participant.gate.restored_targets


def test_export_without_token_ledger_checks_provider_evidence(tmp_path):
    store = store_at(tmp_path)
    dispatch_at(store).before_send()
    participant = PolicyModelParticipant(provider_store=store)
    with pytest.raises(ControlError, match="unresolved provider"):
        participant.export_records([EpisodeId(rollout_id="episode")])


def test_completed_response_is_preserved_but_not_silently_reissued(tmp_path):
    store = store_at(tmp_path)
    dispatch = dispatch_at(store)
    dispatch.before_send()
    dispatch.response_headers(200)
    store.record("episode", {"model_call_id": "call", "response": {"id": "saved-response"}})
    store.clear_pending("episode", "call")
    assert not store.has_pending("episode")
    participant = PolicyModelParticipant(provider_store=store)
    records = participant.export_records([EpisodeId(rollout_id="episode")])
    assert records[0].provider_calls[0]["provider_attempts"][0]["outcome"] == "captured"
    assert store.read("episode")[0]["response"]["id"] == "saved-response"
    with pytest.raises(ControlError, match="saved-response replay"):
        store.assert_restore_safe(["episode"])


def test_legacy_marker_fails_closed(tmp_path):
    store = store_at(tmp_path)
    store.pending_path_for("episode", "old").touch()
    with pytest.raises(ControlError, match="unreadable provider"):
        store.assert_restore_safe(["episode"])


def _worker_mark(root: str, call: str) -> None:
    store = store_at(Path(root))
    dispatch_at(store, call=call).before_send()


def test_concurrent_process_writers_are_all_seen_on_restore(tmp_path):
    context = multiprocessing.get_context("spawn")
    workers = [context.Process(target=_worker_mark, args=(str(tmp_path), f"worker-{i}")) for i in range(3)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(30)
        assert worker.exitcode == 0
    store = store_at(tmp_path)
    assert len(store.dispatch_records("episode")) == 3

    async def restores():
        participants = [CoordinatedPolicyParticipant(None, expected_workers=3, provider_store=store) for _ in range(2)]
        outcomes = await asyncio.gather(
            *(p.install([], [EpisodeId(rollout_id="episode")]) for p in participants), return_exceptions=True
        )
        assert all(isinstance(outcome, ControlError) for outcome in outcomes)
        assert all(not p.restored_targets for p in participants)

    asyncio.run(restores())


def test_process_death_after_loopback_acceptance_blocks_replay(tmp_path):
    # Real HTTP transport boundary: the peer observes the POST, then the model worker
    # is killed without receiving headers/body. This is not a fabricated marker test.
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(30)
    port = listener.getsockname()[1]
    script = """
import asyncio, sys
root, url = sys.argv[1:]
sys.argv = sys.argv[:1]
from nemo_gym.base_responses_api_model import CaptureStore
from nemo_gym.provider_dispatch import ProviderDispatch, current_provider_dispatch
from nemo_gym.openai_utils import NeMoGymAsyncOpenAI
store = CaptureStore(root, provider_dispatch_guard=True)
store.mark_pending("episode", "killed")
current_provider_dispatch.set(ProviderDispatch(lambda evidence: store.write_dispatch("episode", "killed", evidence)))
async def run():
    client = NeMoGymAsyncOpenAI(api_key="fake", base_url="http://127.0.0.1", max_http_attempts=1)
    await client._request(method="POST", url=url, json={"input": "control"})
asyncio.run(run())
"""
    with (tmp_path / "child.log").open("wb") as log:
        child = subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path), f"http://127.0.0.1:{port}/v1/responses"],
            stdout=log,
            stderr=log,
        )
        try:
            connection, _ = listener.accept()
            with connection:
                assert b"POST /v1/responses" in connection.recv(65536)
                os.kill(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
            store = store_at(tmp_path)
            with pytest.raises(ControlError, match="unresolved provider"):
                asyncio.run(
                    PolicyModelParticipant(provider_store=store).install([], [EpisodeId(rollout_id="episode")])
                )
            listener.settimeout(0.1)
            with pytest.raises(TimeoutError):
                listener.accept()
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
            listener.close()


async def _invoke_guarded(
    store, logical_call, *, attempt="episode", model_digest="b" * 64, disconnect=False, counter=None
):
    import json

    from nemo_gym.base_responses_api_model import _CaptureMiddleware

    async def app(scope, receive, send):
        await receive()
        dispatch = current_provider_dispatch.get()
        dispatch.before_send()
        if counter is not None:
            counter.append(logical_call)
        await asyncio.sleep(0.01)
        dispatch.response_headers(200)
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": json.dumps({"id": "saved", "output": []}).encode()})

    middleware = _CaptureMiddleware(app, store=store, model_server_name="solver", model_config_digest=model_digest)
    scope = {
        "type": "http",
        "path": f"/ng-rollout/{attempt}/v1/responses",
        "method": "POST",
        "headers": [
            (b"x-nemo-gym-logical-call-id", logical_call.encode()),
            (b"x-nemo-gym-call-config-sha256", b"a" * 64),
        ],
    }
    messages = [{"type": "http.request", "body": b'{"input":"same prompt"}'}]
    sent = []

    async def receive():
        return messages.pop(0)

    async def send(message):
        if disconnect and message["type"] == "http.response.body":
            raise ConnectionError("downstream disconnected")
        sent.append(message)

    await middleware(scope, receive, send)
    return sent


@pytest.mark.asyncio
async def test_saved_response_replayed_after_downstream_disconnect(tmp_path):
    store = store_at(tmp_path)
    calls = []
    with pytest.raises(ConnectionError):
        await _invoke_guarded(store, "persisted-session:1", disconnect=True, counter=calls)
    records = PolicyModelParticipant(provider_store=store).export_records([EpisodeId(rollout_id="episode")])
    restarted = PolicyModelParticipant(provider_store=store_at(tmp_path))
    await restarted.install(records, [EpisodeId(rollout_id="episode")])
    sent = await _invoke_guarded(store_at(tmp_path), "persisted-session:1", attempt="episode-a1", counter=calls)
    assert len(calls) == 1
    assert b'"id": "saved"' in sent[-1]["body"]


@pytest.mark.asyncio
async def test_fresh_scientific_attempt_does_not_replay_identical_prompt(tmp_path):
    store = store_at(tmp_path)
    calls = []
    await _invoke_guarded(store, "first-session:1", counter=calls)
    await _invoke_guarded(store, "new-session:1", attempt="episode-a1", counter=calls)
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_concurrent_workers_reuse_only_same_logical_call(tmp_path):
    calls = []
    await asyncio.gather(*(_invoke_guarded(store_at(tmp_path), "session:1", counter=calls) for _ in range(4)))
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_reused_call_with_changed_configuration_is_rejected(tmp_path):
    calls = []
    await _invoke_guarded(store_at(tmp_path), "session:1", counter=calls)
    with pytest.raises(ControlError, match="changed request/configuration"):
        await _invoke_guarded(store_at(tmp_path), "session:1", model_digest="c" * 64, counter=calls)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_empty_store_cannot_restore_even_empty_guarded_checkpoint(tmp_path):
    original = PolicyModelParticipant(provider_store=store_at(tmp_path / "original"))
    records = original.export_records([EpisodeId(rollout_id="episode")])
    replacement = PolicyModelParticipant(provider_store=store_at(tmp_path / "empty"))
    with pytest.raises(ControlError, match="identity does not match"):
        await replacement.install(records, [EpisodeId(rollout_id="episode")])


@pytest.mark.asyncio
async def test_capture_failure_does_not_deliver_successful_body(tmp_path, monkeypatch):
    store = store_at(tmp_path)
    calls = []
    monkeypatch.setattr(store, "record", lambda *args: (_ for _ in ()).throw(OSError("capture disk failed")))
    with pytest.raises(OSError, match="capture disk failed"):
        await _invoke_guarded(store, "session:1", counter=calls)
    assert len(calls) == 1
    assert store.has_pending("episode")
    with pytest.raises(ControlError, match="incomplete"):
        store.assert_restore_safe(["episode"])


@pytest.mark.asyncio
async def test_outer_retry_cannot_resend_ambiguous_transport(tmp_path, monkeypatch):
    dispatch = dispatch_at(store_at(tmp_path))
    request = AsyncMock(side_effect=TimeoutError("after send"))
    monkeypatch.setattr("nemo_gym.openai_utils.request", request)
    token = current_provider_dispatch.set(dispatch)
    try:
        client = NeMoGymAsyncOpenAI(api_key="fake", base_url="http://localhost")
        with pytest.raises(TimeoutError):
            await client._request_with_retry()
        with pytest.raises(RuntimeError, match="automatic resend refused"):
            await client._request_with_retry()
    finally:
        current_provider_dispatch.reset(token)
    assert request.await_count == 1
    assert len(dispatch.attempts) == 1


@pytest.mark.asyncio
async def test_incomplete_capture_cannot_be_replayed(tmp_path):
    store = store_at(tmp_path)
    calls = []
    await _invoke_guarded(store, "session:1", counter=calls)
    store.mark_incomplete("episode")
    with pytest.raises(ControlError, match="incomplete"):
        await _invoke_guarded(store, "session:1", counter=calls)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_model_outer_retry_policy_cannot_duplicate_unknown_send(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from nemo_gym.server_utils import ServerClient
    from responses_api_models.openai_model.app import (
        SimpleModelServer,
        SimpleModelServerConfig,
        UpstreamRetriesExhaustedError,
    )

    config = SimpleModelServerConfig(
        host="127.0.0.1",
        port=8081,
        name="test",
        entrypoint="",
        openai_base_url="http://unused",
        openai_api_key="unused",
        openai_model="test",
        upstream_max_num_tries=1,
        upstream_retry_policy={"max_attempts": 3, "backoff_initial_seconds": 0},
    )
    model = SimpleModelServer(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))
    request = AsyncMock(side_effect=TimeoutError("accepted, then lost"))
    monkeypatch.setattr("nemo_gym.openai_utils.request", request)
    dispatch = dispatch_at(store_at(tmp_path))
    token = current_provider_dispatch.set(dispatch)
    try:
        client = NeMoGymAsyncOpenAI(api_key="fake", base_url="http://localhost")
        with pytest.raises(UpstreamRetriesExhaustedError) as error:
            await model._call_upstream(client._request_with_retry)
        assert "automatic resend refused" in str(error.value.__cause__)
    finally:
        current_provider_dispatch.reset(token)
    assert request.await_count == 1
    assert len(dispatch.attempts) == 1


def test_semantic_configuration_digest_ignores_deployment_and_credentials_only():
    from nemo_gym.provider_dispatch import configuration_digest

    original = {
        "host": "127.0.0.1",
        "port": 1000,
        "num_workers": 1,
        "openai_api_key": "first",
        "openai_base_url": "https://provider/v1",
        "openai_model": "model-a",
        "max_output_tokens": 100,
    }
    moved = {**original, "host": "other", "port": 2000, "num_workers": 4, "openai_api_key": "rotated"}
    assert configuration_digest(original) == configuration_digest(moved)
    for changed in [
        {"entrypoint": "different_model.py"},
        {"openai_model": "model-b"},
        {"max_output_tokens": 200},
        {"openai_base_url": "https://other/v1"},
    ]:
        assert configuration_digest(original) != configuration_digest({**original, **changed})


@pytest.mark.asyncio
async def test_unresolved_headers_with_replay_bytes_are_not_replayed(tmp_path):
    store = store_at(tmp_path)
    binding = {"logical_rollout": "episode", "logical_call": "session:1"}
    store.write_dispatch(
        "episode",
        "call",
        {
            "provider_binding": binding,
            "provider_attempts": [{"attempt_id": "attempt", "outcome": "response_headers", "status_code": 200}],
        },
    )
    store.record(
        "episode",
        {"model_call_id": "call", "response": None, "replay_response": {"status": 200, "headers": [], "body_b64": ""}},
    )
    with pytest.raises(ControlError, match="unresolved"):
        async with store.bind_logical_call(binding):
            pytest.fail("unresolved response must not reach replay")


def test_replayed_capture_preserves_raw_usage_without_counting_new_inference():
    from nemo_gym.base_responses_api_model import build_model_call_record

    raw = {
        "model_call_id": "replay",
        "replayed_from_model_call_id": "original",
        "upstream_attempted": False,
        "response": {"id": "original-response", "usage": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25}},
    }
    record = build_model_call_record(raw, call_index=0)
    assert record.response["usage"]["total_tokens"] == 25
    assert record.tokens_total == record.tokens_in == record.tokens_out == 0


@pytest.mark.asyncio
async def test_judge_server_client_reverify_resume_replays(tmp_path, monkeypatch):
    import hashlib
    import json
    from urllib.parse import urlsplit

    from omegaconf import OmegaConf
    from pydantic import BaseModel

    from nemo_gym.base_responses_api_model import _CaptureMiddleware
    from nemo_gym.judge import call_judge
    from nemo_gym.rollout_correlation import rollout_context
    from nemo_gym.rollout_reverification import _verification_run_identity
    from nemo_gym.server_utils import ServerClient

    class Verdict(BaseModel):
        id: str
        output: list

    calls = []
    paths = []

    async def app(scope, receive, send):
        await receive()
        dispatch = current_provider_dispatch.get()
        dispatch.before_send()
        calls.append(1)
        dispatch.response_headers(200)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b'{"id":"verdict","output":[]}'})

    async def transport(*, url, **kwargs):
        paths.append(urlsplit(url).path)
        middleware = _CaptureMiddleware(
            app, store=store_at(tmp_path / "capture"), model_server_name="judge", model_config_digest="b" * 64
        )
        scope = {
            "type": "http",
            "method": "POST",
            "path": paths[-1],
            "headers": [(k.encode(), v.encode()) for k, v in kwargs["headers"].items()],
        }
        sent = []

        async def receive():
            return {"type": "http.request", "body": json.dumps(kwargs["json"]).encode()}

        async def send(message):
            sent.append(message)

        await middleware(scope, receive, send)
        return SimpleNamespace(
            status=200, ok=True, read=AsyncMock(return_value=sent[-1]["body"]), raise_for_status=lambda: None
        )

    monkeypatch.setattr("nemo_gym.server_utils.request", transport)
    client = ServerClient(
        head_server_config={"name": "head", "host": "localhost", "port": 10000},
        global_config_dict=OmegaConf.create(
            {
                "provider_dispatch_guard": True,
                "observability_enabled": True,
                "judge": {"responses_api_models": {}, "host": "localhost", "port": 10001},
            }
        ),
    )
    client._server_base_urls["judge"] = "http://localhost:10001"
    output = tmp_path / "out.jsonl"
    for resume in (False, True):
        run_id = _verification_run_identity(output, resume=resume, overwrite=False, append=False)
        verification_id = run_id + ":task-1"
        with rollout_context("reverify-" + hashlib.sha256(verification_id.encode()).hexdigest()):
            response = await call_judge(
                client,
                server_name="judge",
                url_path="/v1/responses",
                json={"input": "grade"},
                response_model=Verdict,
                verification_id=verification_id,
                call_id="conclusion",
                scorer_identity={"version": "1"},
            )
        assert response.id == "verdict"
    assert paths[0] == paths[1]
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_judge_http_process_restart_replays_saved_response(tmp_path, monkeypatch):
    import hashlib
    import textwrap

    from omegaconf import OmegaConf
    from pydantic import BaseModel

    from nemo_gym.judge import call_judge
    from nemo_gym.rollout_correlation import rollout_context
    from nemo_gym.rollout_reverification import _verification_run_identity
    from nemo_gym.server_utils import ServerClient

    class Verdict(BaseModel):
        id: str
        output: list

    script = tmp_path / "model_server.py"
    script.write_text(
        textwrap.dedent("""
        import sys
        from pathlib import Path
        import uvicorn
        from nemo_gym.base_responses_api_model import CaptureStore, _CaptureMiddleware
        from nemo_gym.provider_dispatch import current_provider_dispatch
        root, port = Path(sys.argv[1]), int(sys.argv[2])
        async def app(scope, receive, send):
            await receive()
            dispatch = current_provider_dispatch.get()
            dispatch.before_send()
            with (root / 'upstream-count').open('a') as handle:
                handle.write('sent\\n')
            dispatch.response_headers(200)
            await send({'type':'http.response.start','status':200,'headers':[(b'content-type',b'application/json')]})
            await send({'type':'http.response.body','body':b'{"id":"saved-judge","output":[]}'})
        middleware = _CaptureMiddleware(app, store=CaptureStore(root / 'capture', provider_dispatch_guard=True), model_server_name='judge', model_config_digest='b'*64)
        uvicorn.run(middleware,host='127.0.0.1',port=port,lifespan='off',log_level='error')
    """)
    )
    monkeypatch.setattr("nemo_gym.server_utils.get_global_config_dict", lambda **kwargs: {})
    output = tmp_path / "regrade.jsonl"
    paths = []
    for resume in (False, True):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        process = subprocess.Popen(
            [sys.executable, str(script), str(tmp_path), str(port)],
            env={**os.environ, "PYTHONPATH": str(Path.cwd())},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            for _ in range(100):
                if process.poll() is not None:
                    pytest.fail(process.stderr.read().decode())
                try:
                    reader, writer = await asyncio.open_connection("127.0.0.1", port)
                    writer.close()
                    await writer.wait_closed()
                    break
                except OSError:
                    await asyncio.sleep(0.05)
            else:
                pytest.fail("local model process never ready")
            client = ServerClient(
                head_server_config={"name": "head", "host": "localhost", "port": 10000},
                global_config_dict=OmegaConf.create(
                    {
                        "provider_dispatch_guard": True,
                        "observability_enabled": True,
                        "judge": {"responses_api_models": {}},
                    }
                ),
            )
            client._server_base_urls["judge"] = f"http://127.0.0.1:{port}"
            lineage = _verification_run_identity(output, resume=resume, overwrite=False, append=False) + ":task-1"
            path = "reverify-" + hashlib.sha256(lineage.encode()).hexdigest()
            paths.append(path)
            with rollout_context(path):
                result = await call_judge(
                    client,
                    server_name="judge",
                    url_path="/v1/responses",
                    json={"input": "grade"},
                    response_model=Verdict,
                    verification_id=lineage,
                    call_id="conclusion",
                    scorer_identity={"version": "1"},
                )
            assert result.id == "saved-judge"
        finally:
            process.terminate()
            process.wait(timeout=10)
            process.stderr.close()
    assert paths[0] == paths[1]
    assert (tmp_path / "upstream-count").read_text().splitlines() == ["sent"]


def test_completed_call_clears_while_sibling_is_unresolved(tmp_path):
    store = store_at(tmp_path)
    completed = dispatch_at(store, call="complete")
    sibling = dispatch_at(store, call="inflight")
    completed.before_send()
    sibling.before_send()
    completed.response_headers(200)
    store.record("episode", {"model_call_id": "complete", "response": {"id": "saved"}})
    store.clear_pending("episode", "complete")
    assert not store.pending_path_for("episode", "complete").exists()
    assert store.pending_path_for("episode", "inflight").exists()
    with pytest.raises(ControlError, match="unresolved provider"):
        store.assert_dispatch_reconciled(["episode"])
    store.clear_pending("episode", "inflight")
    assert store.pending_path_for("episode", "inflight").exists()
    sibling.response_headers(200)
    store.record("episode", {"model_call_id": "inflight", "response": {"id": "second"}})
    store.clear_pending("episode", "inflight")
    assert not store.has_pending("episode")
    with pytest.raises(ControlError, match="saved-response replay"):
        store.assert_restore_safe(["episode"])


def test_collector_guard_preserves_unknown_and_completed_evidence(tmp_path):
    from nemo_gym.base_responses_api_model import assert_guarded_rollout_dispatch_safe

    store = store_at(tmp_path)
    with pytest.raises(ControlError, match="lacks durable identity"):
        assert_guarded_rollout_dispatch_safe([{"missing": "identity"}], [tmp_path])
    row = {"_ng_task_index": 1, "_ng_rollout_index": 0}
    assert_guarded_rollout_dispatch_safe([row], [tmp_path])
    dispatch = dispatch_at(store, "1-0")
    dispatch.before_send()
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(ControlError, match="unresolved provider"):
        assert_guarded_rollout_dispatch_safe([row], [tmp_path])
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == before
    # A fresh retry suffix cannot launder an unknown original attempt.
    with pytest.raises(ControlError, match="unresolved provider"):
        assert_guarded_rollout_dispatch_safe([{**row, "_ng_attempt_index": 1}], [tmp_path])
    # Occupied successful capture without a collector result needs checkpoint recovery.
    store.path_for("2-0").write_text("")
    with pytest.raises(ControlError, match="prior provider evidence"):
        assert_guarded_rollout_dispatch_safe([{"_ng_task_index": 2, "_ng_rollout_index": 0}], [tmp_path])
    assert store.path_for("2-0").exists()


def test_collector_retry_suffix_cannot_duplicate_completed_capture(tmp_path):
    from nemo_gym.base_responses_api_model import assert_guarded_rollout_dispatch_safe

    store = store_at(tmp_path)
    path = store.path_for("1-0")
    path.write_text(
        '{"model_call_id":"saved","provider_attempts":[{"outcome":"captured"}],"provider_binding":{},"replay_response":{}}\n'
    )
    original = path.read_bytes()
    with pytest.raises(ControlError, match="prior provider evidence"):
        assert_guarded_rollout_dispatch_safe(
            [{"_ng_task_index": 1, "_ng_rollout_index": 0, "_ng_attempt_index": 1}], [tmp_path]
        )
    assert path.read_bytes() == original


def test_collector_retry_suffix_cannot_ignore_incomplete_only_capture(tmp_path):
    from nemo_gym.base_responses_api_model import assert_guarded_rollout_dispatch_safe

    store = store_at(tmp_path)
    store.mark_incomplete("1-0")
    with pytest.raises(ControlError, match="prior provider evidence"):
        assert_guarded_rollout_dispatch_safe(
            [{"_ng_task_index": 1, "_ng_rollout_index": 0, "_ng_attempt_index": 1}], [tmp_path]
        )
    assert store.is_incomplete("1-0")


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 413, 415, 422])
async def test_validation_rejection_is_terminal_without_automatic_retry(tmp_path, monkeypatch, status):
    store = store_at(tmp_path)
    dispatch = dispatch_at(store)
    reply = SimpleNamespace(status=status)
    request = AsyncMock(return_value=reply)
    monkeypatch.setattr("nemo_gym.openai_utils.request", request)
    token = current_provider_dispatch.set(dispatch)
    try:
        client = NeMoGymAsyncOpenAI(api_key="fake", base_url="http://localhost", max_http_attempts=5)
        assert await client._request_with_retry() is reply
    finally:
        current_provider_dispatch.reset(token)
    assert request.await_count == 1
    assert dispatch.attempts[0]["outcome"] == "rejected"
    assert dispatch.attempts[0]["status_code"] == status
    store.assert_restore_safe(["episode"])
    store.clear_pending("episode", "call")
    assert not store.has_pending("episode")
