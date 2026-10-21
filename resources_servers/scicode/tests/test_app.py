# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import hashlib
import tempfile
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import resources_servers.scicode.app as app
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.scicode.app import (
    ScicodeResourcesServer,
    ScicodeResourcesServerConfig,
    ScicodeVerifyRequest,
)
from resources_servers.scicode.scicode_integration.runner import build_test_program, run_substep, sanitize_test


def _server(test_data_fpath=None):
    config = ScicodeResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        test_data_fpath=test_data_fpath,
        test_data_sha256=hashlib.sha256(b"").hexdigest(),
        sandbox_image="sha256:" + "a" * 64,
    )
    return ScicodeResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _response() -> NeMoGymResponse:
    return NeMoGymResponse(
        id="r",
        created_at=0.0,
        model="d",
        object="response",
        output=[
            {
                "id": "m",
                "content": [{"annotations": [], "text": "", "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


def _request(solutions, n_steps=2):
    sub_steps = [{"step_number": f"1.{i + 1}", "test_cases": ["assert True"]} for i in range(n_steps)]
    return ScicodeVerifyRequest(
        responses_create_params={"input": []},
        response=_response(),
        problem_id="1",
        sub_steps=sub_steps,
        solutions=solutions,
    )


@contextmanager
def _mock_substep(passed: bool):
    """Stub sub-step execution so verify() runs without executing code or reading test_data.h5."""
    with patch.object(app, "execute_substep", AsyncMock(return_value={"passed": passed, "error": ""})):
        yield


# ----------------------------
# runner helpers
# ----------------------------
def test_sanitize_strips_scicode_imports():
    src = "from scicode.compare.cmp import cmp_tuple_or_list\nimport scicode\nassert f(1) == target"
    assert sanitize_test(src) == "assert f(1) == target"


def test_build_test_program_injects_path_and_targets():
    program = build_test_program("def f(x):\n    return x", "/data/test_data.h5", "1.1", ["assert f(1) == target"])
    assert 'H5PY_FILE = "/data/test_data.h5"' in program
    assert "process_hdf5_to_tuple('1.1', 1)" in program
    assert "target = targets[0]" in program
    assert "def cmp_tuple_or_list" in program


def test_run_substep_pass():
    assert run_substep("assert 1 == 1", timeout_secs=10.0)["passed"] is True


def test_run_substep_fail_returns_stderr():
    result = run_substep("raise ValueError('boom')", timeout_secs=10.0)
    assert result["passed"] is False
    assert "ValueError" in result["error"]


def test_run_substep_timeout():
    result = run_substep("import time\ntime.sleep(5)", timeout_secs=0.5)
    assert result == {"passed": False, "error": "timeout"}


def test_run_substep_null_byte_fails_instead_of_raising():
    result = run_substep("x = 1\n\0\nassert x == 1", timeout_secs=10.0)
    assert result["passed"] is False
    assert "null bytes" in result["error"]


def test_run_substep_program_larger_than_arg_max():
    program = f"payload = {'a' * 4_000_000!r}\nassert len(payload) == 4_000_000"
    assert run_substep(program, timeout_secs=30.0)["passed"] is True


# ----------------------------
# server
# ----------------------------
class TestApp:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("has_solution", [False, True])
    async def test_verify_preserves_step_usage(self, has_solution):
        request_json = _request(solutions={"1.1": "a"} if has_solution else None).model_dump()
        request_json["token_usage_version"] = 1
        request_json["step_usage"] = [
            {
                "step_number": "1.1",
                "status": "generated" if has_solution else "skipped",
                "usage": None,
            }
        ]
        request = ScicodeVerifyRequest.model_validate(request_json)
        with tempfile.NamedTemporaryFile(suffix=".h5") as h5, _mock_substep(passed=True):
            result = (await _server(h5.name).verify(request)).model_dump()
        assert result["token_usage_version"] == 1
        assert result["step_usage"] == request_json["step_usage"]
        assert result["reward"] == float(has_solution)

    def test_sanity(self):
        _server()

    def test_config_defaults(self):
        config = _server().config
        assert config.num_processes == 2
        assert config.timeout_secs == 30.0
        assert config.test_data_fpath is None

    @pytest.mark.asyncio
    async def test_verify_no_solutions_returns_zero(self):
        result = await _server().verify(_request(solutions=None))
        assert result.reward == 0.0
        assert result.num_steps_total == 0
        assert result.num_steps_passed == 0
        assert result.problem_accuracy is False

    @pytest.mark.asyncio
    async def test_verify_excludes_steps_absent_from_solutions(self):
        # Step 1.2 has no solution entry (prefilled) -> excluded from the denominator entirely.
        with tempfile.NamedTemporaryFile(suffix=".h5") as h5, _mock_substep(passed=True):
            result = await _server(h5.name).verify(_request(solutions={"1.1": "a"}, n_steps=2))
        assert result.num_steps_total == 1
        assert result.num_steps_passed == 1
        assert result.reward == 1.0
        assert result.problem_accuracy is True

    @pytest.mark.asyncio
    async def test_verify_unconfigured_test_data_raises(self):
        with pytest.raises(RuntimeError, match="not configured"):
            await _server().verify(_request(solutions={"1.1": "x = 1", "1.2": "y = 2"}))

    @pytest.mark.asyncio
    async def test_verify_missing_test_data_raises(self):
        server = _server(test_data_fpath="/nonexistent/test_data.h5")
        with pytest.raises(RuntimeError, match="not found"):
            await server.verify(_request(solutions={"1.1": "x = 1", "1.2": "y = 2"}))

    @pytest.mark.asyncio
    async def test_verify_relative_test_data_resolved_under_gym_root(self):
        from nemo_gym import PARENT_DIR

        server = _server(test_data_fpath="nonexistent/test_data.h5")
        with pytest.raises(RuntimeError, match=str(PARENT_DIR)):
            await server.verify(_request(solutions={"1.1": "x = 1", "1.2": "y = 2"}))

    @pytest.mark.asyncio
    async def test_verify_all_pass(self):
        with tempfile.NamedTemporaryFile(suffix=".h5") as h5, _mock_substep(passed=True):
            result = await _server(h5.name).verify(_request(solutions={"1.1": "a", "1.2": "b"}))
        assert result.reward == 1.0
        assert result.step_results == [True, True]
        assert result.num_steps_passed == 2
        assert result.problem_accuracy is True
        assert result.subtask_accuracy == 1.0
        assert result.problem_id == "1"  # preserved into the rollout output

    @pytest.mark.asyncio
    async def test_verify_all_fail(self):
        with tempfile.NamedTemporaryFile(suffix=".h5") as h5, _mock_substep(passed=False):
            result = await _server(h5.name).verify(_request(solutions={"1.1": "a", "1.2": "b"}))
        assert result.reward == 0.0
        assert result.num_steps_passed == 0
        assert result.problem_accuracy is False
        assert result.subtask_accuracy == 0.0

    @pytest.mark.asyncio
    async def test_verify_out_of_context_step_fails(self):
        # First sub-step runs (mocked pass); second is the out-of-context sentinel -> fails unrun.
        with tempfile.NamedTemporaryFile(suffix=".h5") as h5, _mock_substep(passed=True):
            result = await _server(h5.name).verify(_request(solutions={"1.1": "a", "1.2": "_ran_out_of_context_"}))
        assert result.step_results == [True, False]
        assert result.num_steps_passed == 1
        assert result.reward == 0.0
        assert result.subtask_accuracy == 0.5  # per-rollout sub-step pass fraction


def test_h5_checksum_mismatch_fails_closed():
    with tempfile.NamedTemporaryFile(suffix=".h5") as h5:
        h5.write(b"not the canonical data")
        h5.flush()
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            _server(h5.name)._resolve_test_data()


@pytest.mark.asyncio
async def test_infrastructure_failure_settles_sibling_cleanup():
    entered = asyncio.Event()
    cleaned = asyncio.Event()
    calls = 0

    async def execute(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            await entered.wait()
            raise RuntimeError("original infrastructure failure")
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0.01)
            cleaned.set()
            raise ValueError("sibling cleanup failure")

    with tempfile.NamedTemporaryFile(suffix=".h5") as h5, patch.object(app, "execute_substep", execute):
        with pytest.raises(RuntimeError, match="original infrastructure failure"):
            await _server(h5.name).verify(_request(solutions={"1.1": "a", "1.2": "b"}))
    assert cleaned.is_set()


@pytest.mark.parametrize("candidate", ["pass", "", "_ran_out_of_context_"])
async def test_known_shape_conflict_is_masked_without_execution(tmp_path, candidate):
    import json
    from pathlib import Path

    step = json.loads((Path(__file__).parent / "fixtures/step79_4.json").read_text())
    body = _request({"79.1": "pass", "79.4": candidate})
    body.problem_id = "79"
    body.sub_steps = [{"step_number": f"79.{i}", "test_cases": ["assert True"]} for i in range(1, 4)] + [step]
    server = _server()
    server.config.test_data_sha256 = "48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890"
    with (
        patch.object(app.ScicodeResourcesServer, "_resolve_test_data", return_value=str(tmp_path / "verified.h5")),
        patch.object(app, "execute_substep", AsyncMock(return_value={"passed": True, "error": ""})) as execute,
    ):
        result = await server.verify(body)
    assert execute.await_count == 1
    if candidate != "pass":
        assert result.step_results == [True, False]
        assert not result.mask_sample
        assert result.reward == 0.0 and result.subtask_accuracy == 0.5
        assert result.measurement_validity == []
        assert result.num_steps_total == 2
        return
    assert result.step_results == [True, None]
    assert [x["step_number"] for x in result.step_execution] == ["79.1", "79.4"]
    assert result.mask_sample and result.failure_kind == "scicode:reference_shape_conflict"
    assert result.problem_accuracy is None and result.subtask_accuracy is None
    assert result.num_steps_total == 2 and result.num_steps_passed == 1
    assert result.measurement_validity[0]["raw_score"] is None
    assert not result.measurement_validity[0]["executed"]
