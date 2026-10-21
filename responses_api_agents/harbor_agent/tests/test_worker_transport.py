"""Actual shared HTTP requests from a reused Ray worker; no external provider."""
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import AsyncMock, patch

import pytest
import ray

from nemo_gym import server_utils
from responses_api_agents.harbor_agent import app


@pytest.mark.asyncio
async def test_close_attempts_both_pools_and_retains_failed_handle(monkeypatch):
    bad = AsyncMock()
    bad.closed = False
    bad.close.side_effect = RuntimeError("close failed")
    good = AsyncMock()
    good.closed = True
    monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", bad)
    monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CONTROL_CLIENT", good)
    with pytest.raises(RuntimeError, match="close failed"):
        await server_utils.global_aiohttp_client_exit_async()
    good.close.assert_awaited_once()
    assert server_utils._GLOBAL_AIOHTTP_CLIENT is bad
    assert server_utils._GLOBAL_AIOHTTP_CONTROL_CLIENT is None
    with pytest.raises(AssertionError):
        server_utils.set_global_aiohttp_client(server_utils.GlobalAIOHTTPAsyncClientConfig())


@pytest.mark.asyncio
@pytest.mark.parametrize("job_fails", [False, True])
async def test_primary_failure_preserved_on_cleanup_error(job_fails):
    job = AsyncMock(side_effect=ValueError("job failed") if job_fails else None, return_value="trial")
    with (
        patch.object(app, "set_global_aiohttp_client"),
        patch.object(app, "run_harbor_job", job),
        patch.object(app, "global_aiohttp_client_exit_async", AsyncMock(side_effect=RuntimeError("cleanup"))),
    ):
        with pytest.raises(ValueError if job_fails else RuntimeError) as error:
            await app._run_harbor_job_with_transport({}, {})
    if job_fails:
        assert "Worker HTTP cleanup failed: RuntimeError" in error.value.__notes__


def test_actual_ray_reused_worker_requests_and_failure_cleanup(tmp_path):
    requests = []

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(self.path)
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(429 if self.path == "/reject" else 200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def log_message(self, *args):
            pass

    class Worker:
        def run(self, url, fail):
            import sys
            from unittest.mock import patch
            from nemo_gym import server_utils as transport
            from responses_api_agents.harbor_agent import app as worker_app

            sessions = []

            async def job(_):
                sessions.append(transport.get_global_aiohttp_client())
                assert transport._GLOBAL_AIOHTTP_CLIENT_CONFIG.global_aiohttp_connector_limit == 7
                response = await transport.request(
                    "POST", url, json={"fixture": True}, _max_num_tries=1,
                    _max_connection_retries=0, _resend_after_disconnect=False,
                )
                async with response:
                    await response.read()
                    response.raise_for_status()
                transport.get_global_aiohttp_control_client()
                sessions.append(transport._GLOBAL_AIOHTTP_CONTROL_CLIENT)
                if fail:
                    raise ValueError("after request")
                return "trial"

            error = None
            with (
                patch.object(worker_app, "run_harbor_job", job),
                patch.object(transport, "get_global_config_dict", side_effect=AssertionError("Hydra forbidden")),
            ):
                try:
                    worker_app._run_harbor_job_sync({}, {"global_aiohttp_connector_limit": 7})
                except Exception as exc:
                    error = type(exc).__name__
            return {
                "pid": os.getpid(), "error": error, "closed": all(s.closed for s in sessions),
                "data_reset": transport._GLOBAL_AIOHTTP_CLIENT is None,
                "control_reset": transport._GLOBAL_AIOHTTP_CONTROL_CLIENT is None,
                "config_reset": transport._GLOBAL_AIOHTTP_CLIENT_CONFIG is None,
                "argv": sys.argv,
            }

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        ray.init(num_cpus=1, include_dashboard=False, logging_level="ERROR")
        worker = ray.remote(Worker).remote()
        base = f"http://127.0.0.1:{server.server_port}"
        results = [ray.get(worker.run.remote(base + path, fail), timeout=60)
                   for path, fail in [("/ok", False), ("/ok", True), ("/reject", False), ("/ok", False)]]
        assert len({r["pid"] for r in results}) == 1
        assert all(r["pid"] != os.getpid() for r in results)
        assert [r["error"] for r in results] == [None, "ValueError", "ClientResponseError", None]
        assert all(r["closed"] and r["data_reset"] and r["control_reset"] and r["config_reset"] for r in results)
        assert requests == ["/ok", "/ok", "/reject", "/ok"]
    finally:
        ray.shutdown()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.asyncio
async def test_parent_forwards_only_typed_transport_settings():
    from responses_api_agents.harbor_agent.tests.test_app import (
        _GLOBAL_CONFIG, _harbor_run_mocks, _make_run_request, _make_server,
    )
    body = _make_run_request().model_copy(update={"_ng_task_index": 0, "_ng_rollout_index": 0})
    with _harbor_run_mocks() as runner:
        with patch.object(app, "get_global_config_dict", return_value={
            **_GLOBAL_CONFIG, "global_aiohttp_connector_limit": 7,
            "credential_sentinel": "must-not-cross-worker-boundary",
        }):
            await _make_server().run(body)
        params = runner.remote.call_args.args[1]
    settings = params["transport_config"]
    assert settings["global_aiohttp_connector_limit"] == 7
    assert set(settings) == set(server_utils.GlobalAIOHTTPAsyncClientConfig.model_fields)
    assert "credential_sentinel" not in settings
    assert "must-not-cross-worker-boundary" not in repr(params)
