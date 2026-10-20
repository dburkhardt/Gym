import json
from unittest.mock import patch

import pytest

from benchmarks.genehop.native_loop import MODEL, NativeLoop, validate_url


class HTTPResponse:
    def __init__(self, data):
        self.data = json.dumps(data).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, *args):
        return self.data


def reply(text, finish="stop"):
    return {"model": MODEL, "choices": [{"message": {"content": text}, "finish_reason": finish}]}


def test_native_tool_then_answer(tmp_path):
    loop = NativeLoop(tmp_path, tmp_path / "trial", "secret-test", "http://localhost/v1")
    values = [
        reply("[https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gene&term=X]"),
        reply("Answer: 11q13.4"),
    ]
    requests = []

    def send(req, **kwargs):
        requests.append(json.loads(req.data))
        return HTTPResponse(values.pop(0))

    with (
        patch.object(loop, "header", return_value="Native header\n"),
        patch.object(loop, "call_api", return_value=b'{"idlist":["1"]}') as tool,
        patch("urllib.request.urlopen", side_effect=send),
    ):
        result = loop.run("Question?")
    assert result["status"] == "completed" and result["answer"] == "Answer: 11q13.4"
    assert tool.call_count == 1 and len(requests) == 2
    assert "->[" in requests[1]["messages"][0]["content"]
    assert all(x["model"] == MODEL for x in requests)
    assert all("secret-test" not in p.read_text() for p in (tmp_path / "trial").glob("model-*.json"))


def test_budget_no_retry(tmp_path):
    loop = NativeLoop(tmp_path, tmp_path / "trial", "key", "http://localhost/v1")
    with (
        patch.object(loop, "header", return_value="header"),
        patch("urllib.request.urlopen", return_value=HTTPResponse(reply("", finish="length"))) as send,
    ):
        result = loop.run("Question?")
    assert result["status"] == "model_budget" and send.call_count == 1


@pytest.mark.parametrize(
    "url",
    [
        "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/x",
        "https://evil.example/",
        "https://eutils.ncbi.nlm.nih.gov.evil.test/entrez/eutils/x",
        "https://u:p@eutils.ncbi.nlm.nih.gov/entrez/eutils/x",
    ],
)
def test_tool_allowlist(url):
    with pytest.raises(ValueError):
        validate_url(url)


def test_wrong_model_receipt(tmp_path):
    loop = NativeLoop(tmp_path, tmp_path / "trial", "key", "http://localhost/v1")
    raw = reply("bad")
    raw["model"] = "wrong"
    with (
        patch.object(loop, "header", return_value="header"),
        patch("urllib.request.urlopen", return_value=HTTPResponse(raw)),
    ):
        with pytest.raises(RuntimeError):
            loop.run("Q")
    assert json.loads((tmp_path / "trial/model-00.json").read_text())["response"] == raw


def test_filtered_failure(tmp_path):
    loop = NativeLoop(tmp_path, tmp_path / "trial", "key", "http://localhost/v1")
    with (
        patch.object(loop, "header", return_value="header"),
        patch("urllib.request.urlopen", return_value=HTTPResponse(reply("", finish="content_filter"))),
    ):
        assert loop.run("Q")["status"] == "model_failure"


def test_native_post_failure_and_uuid(tmp_path):
    from unittest.mock import MagicMock

    from fastapi.testclient import TestClient

    from nemo_gym.server_utils import ServerClient
    from responses_api_agents.genehop_native.app import Config, GeneHopAgent, RunRequest

    config = Config(
        host="127.0.0.1",
        port=46001,
        entrypoint="app.py",
        name="genehop_agent",
        upstream_repo=tmp_path,
        output_root=tmp_path / "out",
    )
    agent = GeneHopAgent(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))
    body = {
        "uuid": "safe-1",
        "responses_create_params": {"input": [{"role": "user", "content": "Q"}]},
        "verifier_metadata": {"question": "Q", "category": "Disease gene location", "answer": ["x"]},
    }
    with (
        patch.dict("os.environ", {"GENEHOP_API_KEY": "test"}),
        patch.object(NativeLoop, "run", side_effect=OSError("tool unavailable")),
        TestClient(agent.setup_webserver()) as client,
    ):
        response = client.post("/run", json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["mask_sample"] and result["raw_creator_score"] is None
    assert result["native_status"] == "infrastructure_failure"
    assert (tmp_path / "out/safe-1/failure.json").exists()
    body["uuid"] = "../escape"
    with pytest.raises(ValueError, match="Unsafe"):
        agent.execute(RunRequest.model_validate(body))


def test_http_error_receipt(tmp_path):
    import io
    import urllib.error

    loop = NativeLoop(tmp_path, tmp_path / "trial", "secret-key", "http://localhost/v1")
    error = urllib.error.HTTPError(
        "http://localhost/v1/chat/completions", 503, "unavailable", {}, io.BytesIO(b'{"error":"busy"}')
    )
    with patch.object(loop, "header", return_value="header"), patch("urllib.request.urlopen", side_effect=error):
        with pytest.raises(urllib.error.HTTPError):
            loop.run("Q")
    saved = json.loads((tmp_path / "trial/model-00-http-error.json").read_text())
    assert saved["http_status"] == 503 and saved["body"] == '{"error":"busy"}'
    assert "secret-key" not in json.dumps(saved)


@pytest.mark.parametrize(
    "status,answer,expected",
    [("model_budget", "Answer: x", 1.0), ("model_budget", "", 0.0), ("step_budget", "numError", 0.0)],
)
def test_budget_outputs_remain_in_automatic_score_denominator(tmp_path, status, answer, expected):
    from unittest.mock import MagicMock

    from fastapi.testclient import TestClient

    from nemo_gym.server_utils import ServerClient
    from responses_api_agents.genehop_native.app import Config, GeneHopAgent

    (tmp_path / "evaluate.py").write_text(
        "def get_answer(answer, task):\n    return answer.strip().replace('Answer: ', '').split(', ')\n"
    )
    config = Config(
        host="127.0.0.1",
        port=46001,
        entrypoint="app.py",
        name="genehop_agent",
        upstream_repo=tmp_path,
        output_root=tmp_path / "out",
    )
    agent = GeneHopAgent(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))
    body = {
        "uuid": "budget-1",
        "responses_create_params": {"input": [{"role": "user", "content": "Q"}]},
        "verifier_metadata": {"question": "Q", "category": "Disease gene location", "answer": ["x"]},
    }
    with (
        patch.dict("os.environ", {"GENEHOP_API_KEY": "test"}),
        patch.object(NativeLoop, "run", return_value={"status": status, "answer": answer, "calls": []}),
        TestClient(agent.setup_webserver()) as client,
    ):
        response = client.post("/run", json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["raw_creator_score"] == result["reward"] == expected
    assert not result["mask_sample"]
    assert result["failure_kind"] == "genehop_native:" + status


def test_injected_model_transport_preserves_native_protocol(tmp_path):
    from unittest.mock import Mock

    transport = Mock(return_value=reply("Answer: x"))
    loop = NativeLoop(tmp_path, tmp_path / "trial", "", "", model_call=transport)
    with patch.object(loop, "header", return_value="Native header\n"), patch("urllib.request.urlopen") as direct:
        result = loop.run("Question?")
    assert result["answer"] == "Answer: x"
    direct.assert_not_called()
    payload, turn = transport.call_args.args
    assert turn == 0
    assert payload == {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Native header\nQuestion: Question?\n"}],
        "temperature": 0.0,
        "max_tokens": 16384,
        "stop": ["->", "\n\nQuestion"],
    }
    assert json.loads((tmp_path / "trial/model-00.json").read_text())["request"] == payload


@pytest.mark.asyncio
async def test_shared_chat_transport_uses_guarded_identity(tmp_path):
    from unittest.mock import AsyncMock, MagicMock

    from nemo_gym.config_types import ModelServerRef
    from nemo_gym.server_utils import ServerClient
    from responses_api_agents.genehop_native.app import Config, GeneHopAgent, RunRequest

    config = Config(
        host="127.0.0.1",
        port=46001,
        entrypoint="app.py",
        name="genehop_agent",
        upstream_repo=tmp_path,
        output_root=tmp_path / "out",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
    )
    server = MagicMock(
        spec=ServerClient, global_config_dict={"provider_dispatch_guard": True, "observability_enabled": True}
    )
    response = MagicMock()
    response.status = 200
    response.json = AsyncMock(return_value=reply("Answer: x"))
    server.post = AsyncMock(return_value=response)
    agent = GeneHopAgent(config=config, server_client=server)
    body = RunRequest.model_validate(
        {
            "_ng_rollout_id": "frozen-task-0",
            "uuid": "task-1",
            "responses_create_params": {"input": [{"role": "user", "content": "Q"}]},
            "verifier_metadata": {
                "question": "Q",
                "answer": ["SECRET_REFERENCE"],
                "category": "Disease gene location",
            },
        }
    )
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Q"}],
        "max_tokens": 16384,
        "stop": ["->", "\n\nQuestion"],
    }
    assert (await agent.guarded_model_call(body, payload, 0))["model"] == MODEL
    kwargs = server.post.call_args.kwargs
    assert kwargs["url_path"] == "/ng-rollout/frozen-task-0/v1/chat/completions"
    assert kwargs["json"] == payload
    assert "SECRET_REFERENCE" not in json.dumps(kwargs)
    assert kwargs["headers"]["x-nemo-gym-logical-call-id"] == "genehop:task-1:0"
    server.global_config_dict["provider_dispatch_guard"] = False
    with pytest.raises(ValueError, match="provider_dispatch_guard"):
        await agent.guarded_model_call(body, payload, 1)
    assert server.post.await_count == 1
