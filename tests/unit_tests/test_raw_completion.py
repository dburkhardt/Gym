# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.base_responses_api_model import CaptureStore, read_model_call_records
from nemo_gym.health.checks import _response_has_content
from nemo_gym.openai_utils import NeMoGymCompletionCreateParamsNonStreaming
from nemo_gym.provider_dispatch import current_provider_dispatch
from nemo_gym.server_utils import ServerClient
from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig


REQUEST = {
    "model": "test",
    "prompt": "Question: literal\nAnswer:",
    "max_tokens": 512,
    "stop": ["\n"],
    "temperature": 0.0,
}
RESPONSE = {
    "id": "raw-1",
    "object": "text_completion",
    "created": 1,
    "model": "test",
    "choices": [{"index": 0, "text": "The answer is A.", "finish_reason": "stop"}],
}
HEADERS = {"x-nemo-gym-logical-call-id": "session:0", "x-nemo-gym-call-config-sha256": "a" * 64}


@pytest.mark.parametrize(
    "change",
    [
        {"prompt": [1, 2]},
        {"stream": True},
        {"n": 2},
        {"logprobs": 0},
        {"echo": True},
        {"suffix": "x"},
        {"best_of": 2},
        {"temperature": float("nan")},
        {"top_p": float("inf")},
        {"max_tokens": True},
    ],
)
def test_unsupported_request_rejected(change):
    with pytest.raises(ValidationError):
        NeMoGymCompletionCreateParamsNonStreaming.model_validate(REQUEST | change)


def server(tmp_path):
    config = SimpleModelServerConfig(
        host="127.0.0.1",
        port=8081,
        name="test",
        entrypoint="",
        openai_base_url="http://unused",
        openai_api_key="unused",
        openai_model="test",
        raw_completions_enabled=True,
        upstream_max_num_tries=1,
        upstream_retry_policy={"max_attempts": 1},
    )
    model = SimpleModelServer(
        config=config,
        server_client=MagicMock(
            spec=ServerClient,
            global_config_dict={
                "observability_enabled": True,
                "provider_dispatch_guard": True,
                "model_call_capture_dir": str(tmp_path),
            },
        ),
    )
    return model


def test_native_route_capture_replay_and_identity(tmp_path):
    model = server(tmp_path)
    calls = []

    async def complete(**body):
        dispatch = current_provider_dispatch.get()
        dispatch.before_send()
        calls.append(body)
        dispatch.response_headers(200)
        return RESPONSE

    model._client = MagicMock()
    model._client.create_completion = complete
    with TestClient(model.setup_webserver(), raise_server_exceptions=False) as client:
        first = client.post("/ng-rollout/raw/v1/completions", json=REQUEST, headers=HEADERS)
        assert first.status_code == 200, first.text
        second = client.post("/ng-rollout/raw/v1/completions", json=REQUEST, headers=HEADERS)
        assert second.json() == first.json()
        changed = client.post(
            "/ng-rollout/raw/v1/completions", json=REQUEST | {"prompt": "different"}, headers=HEADERS
        )
        assert changed.status_code >= 400
    assert calls == [REQUEST]
    record = read_model_call_records(CaptureStore(tmp_path), "raw")[0]
    assert record.dialect == "completion"
    assert record.request == REQUEST
    assert record.response["choices"][0]["text"] == RESPONSE["choices"][0]["text"]
    assert record.response["usage"] is None
    assert _response_has_content(record.response)
    assert not _response_has_content(RESPONSE | {"choices": [{"text": "", "finish_reason": "length"}]})


def test_native_ambiguous_request_cannot_resend(tmp_path):
    model = server(tmp_path)
    calls = []

    async def complete(**body):
        current_provider_dispatch.get().before_send()
        calls.append(body)
        raise TimeoutError("response lost after send")

    model._client = MagicMock()
    model._client.create_completion = complete
    with TestClient(model.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/ng-rollout/raw/v1/completions", json=REQUEST, headers=HEADERS).status_code >= 400
        assert client.post("/ng-rollout/raw/v1/completions", json=REQUEST, headers=HEADERS).status_code >= 400
    assert len(calls) == 1


def test_real_http_provider_literal_wire_and_capture(tmp_path, monkeypatch):
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    from omegaconf import OmegaConf

    import nemo_gym.global_config
    import nemo_gym.server_utils as server_utils
    from nemo_gym.openai_utils import NeMoGymAsyncOpenAI

    monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", OmegaConf.create({}))
    monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", None)
    received = []

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            payload = json.dumps(RESPONSE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        model = server(tmp_path)
        model._client = NeMoGymAsyncOpenAI(
            api_key="local-fixture", base_url=f"http://127.0.0.1:{upstream.server_port}/v1", max_http_attempts=1
        )
        with TestClient(model.setup_webserver()) as client:
            response = client.post("/ng-rollout/wire/v1/completions", json=REQUEST, headers=HEADERS)
            assert response.status_code == 200, response.text
            assert (
                client.post("/ng-rollout/wire/v1/completions", json=REQUEST, headers=HEADERS).json() == response.json()
            )
            client.portal.call(server_utils._GLOBAL_AIOHTTP_CLIENT.close)
        assert received == [("/v1/completions", REQUEST)]
        record = read_model_call_records(CaptureStore(tmp_path), "wire")[0]
        assert record.provider_attempts[0]["outcome"] == "captured"
        assert record.request["prompt"] == REQUEST["prompt"]
        assert record.response["model"] == "test"
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_config_extras_cannot_bypass_contract(tmp_path):
    model = server(tmp_path)
    model.config.extra_body = {"echo": True}
    model._client = MagicMock()
    with TestClient(model.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/ng-rollout/raw/v1/completions", json=REQUEST, headers=HEADERS).status_code >= 400
    model._client.create_completion.assert_not_called()


def test_response_identity_and_cardinality_rejected():
    from nemo_gym.openai_utils import NeMoGymCompletion

    for choices in ([], RESPONSE["choices"] * 2, [{"index": 0, "text": None, "finish_reason": "stop"}]):
        with pytest.raises(ValidationError):
            NeMoGymCompletion.model_validate(RESPONSE | {"choices": choices})


def test_startup_rejects_incompatible_raw_config(tmp_path):
    model = server(tmp_path)
    with pytest.raises(ValidationError):
        SimpleModelServerConfig.model_validate(model.config.model_dump() | {"extra_body": {"echo": True}})


def test_wrong_response_model_fails_with_raw_evidence(tmp_path):
    model = server(tmp_path)

    async def complete(**body):
        current_provider_dispatch.get().before_send()
        current_provider_dispatch.get().response_headers(200)
        return RESPONSE | {"model": "other"}

    model._client = MagicMock()
    model._client.create_completion = complete
    with TestClient(model.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/ng-rollout/wrong/v1/completions", json=REQUEST, headers=HEADERS).status_code == 500


def test_length_and_usage_are_retained(tmp_path):
    model = server(tmp_path)
    usage = {"prompt_tokens": 15, "completion_tokens": 512, "total_tokens": 527}

    async def complete(**body):
        current_provider_dispatch.get().before_send()
        current_provider_dispatch.get().response_headers(200)
        return RESPONSE | {"choices": [{"index": 0, "text": "", "finish_reason": "length"}], "usage": usage}

    model._client = MagicMock()
    model._client.create_completion = complete
    with TestClient(model.setup_webserver()) as client:
        assert client.post("/ng-rollout/length/v1/completions", json=REQUEST, headers=HEADERS).status_code == 200
    record = read_model_call_records(CaptureStore(tmp_path), "length")[0]
    assert record.response["usage"] == usage
    assert record.finish_reason == "length"
    assert not _response_has_content(record.response)
