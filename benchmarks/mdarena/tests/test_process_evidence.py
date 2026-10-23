# SPDX-License-Identifier: Apache-2.0
"""Offline evidence-route qualification, not model judgment or Docker acceptance."""

import hashlib
import importlib
import io
import json
import sys
import tomllib
from pathlib import Path

import pytest

from responses_api_agents.harbor_agent.custom_verifiers.rewardkit_agent import docker_executor, tool_loop


ROOT = Path(__file__).resolve().parents[5]
UPSTREAM = ROOT / "outputs/broad-eval-campaign/harbor/upstream/mdarena/tasks/41_rexee_argon_hydration_setup"
ORACLE = ROOT / (
    "outputs/broad-eval-campaign/foundation/mdarena-history-reconciliation-20261022/"
    "reconstruction/task41-preparation/target-evidence/oracle-job/"
    "41_rexee_argon_hydration_setup__N9Xk5pM/artifacts/app"
)


@pytest.mark.parametrize("case, verdict", [("saved", "yes"), ("missing", "no"), ("irrelevant", "no")])
def test_task41_saved_workspace_observations_reach_judge(tmp_path, monkeypatch, capsys, case, verdict):
    if not ORACLE.exists() or not UPSTREAM.exists():
        pytest.skip("Pinned local task41 evidence unavailable")
    judge_path = UPSTREAM / "tests/process/judge.toml"
    original = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [judge_path, ORACLE / "final_answer.txt", ORACLE / "outputs/run_log.txt"]
    }
    contract = tomllib.loads(judge_path.read_text())
    assert len(contract["criterion"]) == 1
    assert contract["judge"]["timeout"] == 600
    assert "answer 'no'" in contract["criterion"][0]["description"]
    files = {
        "/app/final_answer.txt": (ORACLE / "final_answer.txt").read_text(),
        "/app/outputs/run_log.txt": (ORACLE / "outputs/run_log.txt").read_text(),
    }
    if case == "missing":
        files = {}
    if case == "irrelevant":
        files["/app/final_answer.txt"] = "The capital of France is Paris."
    monkeypatch.setitem(sys.modules, "tool_loop", tool_loop)
    monkeypatch.setitem(sys.modules, "docker_executor", docker_executor)
    scientific = importlib.import_module(
        "responses_api_agents.harbor_agent.custom_verifiers.rewardkit_agent.scientific_tool_loop"
    )
    image = "sha256:" + "a" * 64
    monkeypatch.setenv("GYM_RK_JUDGE_SANDBOX", "fixture-task41")
    monkeypatch.setenv("GYM_RK_JUDGE_IMAGE", image)
    monkeypatch.setenv("GYM_RK_JUDGE_RUNTIME_APPROVED", "1")
    monkeypatch.setenv("GYM_RK_JUDGE_ENDPOINT", "http://fixture.invalid/ng-rollout/task41/v1")
    monkeypatch.setenv("GYM_RK_TRIAL_LINEAGE", "offline-task41")
    monkeypatch.setenv("GYM_RK_CALL_CONFIG_SHA256", "b" * 64)
    monkeypatch.setattr(tool_loop, "run_shell", tool_loop.run_shell)
    scientific.configure()
    commands = []
    observations = []

    def docker(argv, **kwargs):
        if argv[:2] == ["docker", "inspect"]:
            return json.dumps(
                [
                    {
                        "Image": image,
                        "Config": {
                            "Labels": {"campaign.role": "gym-rewardkit-judge-scientific"},
                            "User": "65534:65534",
                            "Env": [],
                        },
                        "HostConfig": {"NetworkMode": "none", "ReadonlyRootfs": True},
                        "Mounts": [{"Destination": p, "RW": False} for p in ["/app", "/logs/agent/trajectory.json"]],
                    }
                ]
            )
        assert argv[:3] == ["docker", "exec", "fixture-task41"]
        command = json.loads(argv[-1])["command"]
        commands.append(command)
        path = command.removeprefix("cat ")
        assert path in {"/app/final_answer.txt", "/app/outputs/run_log.txt"}
        result = {
            "stdout": files.get(path, ""),
            "stderr": "" if path in files else "No such file",
            "returncode": 0 if path in files else 1,
        }
        observations.append(result)
        return json.dumps(result)

    monkeypatch.setattr(docker_executor.subprocess, "check_output", docker)
    requests = []

    def http(request, timeout):
        body = json.loads(request.data)
        requests.append(body)
        if len(requests) == 1:
            output = [
                {
                    "type": "function_call",
                    "name": "shell",
                    "call_id": str(i),
                    "arguments": json.dumps({"command": "cat " + p}),
                }
                for i, p in enumerate(["/app/final_answer.txt", "/app/outputs/run_log.txt"])
            ]
        else:
            actual = [json.loads(x["output"]) for x in body["input"] if x.get("type") == "function_call_output"]
            assert actual == observations and len(actual) == 2
            # Scripted fixture verdict; no claim that a model recognizes inconsistency.
            output = [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps({"score": verdict, "reasoning": "deterministic transport fixture"}),
                        }
                    ],
                }
            ]
        return io.BytesIO(
            json.dumps(
                {
                    "model": tool_loop.MODEL,
                    "status": "completed",
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                    "output": output,
                }
            ).encode()
        )

    monkeypatch.setattr(tool_loop.urllib.request, "urlopen", http)
    schema = tmp_path / "schema.json"
    schema.write_text(
        json.dumps(
            {
                "type": "object",
                "properties": {"score": {"enum": ["yes", "no"]}, "reasoning": {"type": "string"}},
                "required": ["score", "reasoning"],
                "additionalProperties": False,
            }
        )
    )
    receipts = tmp_path / "receipts"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "loop",
            "--prompt",
            contract["criterion"][0]["description"],
            "--schema",
            str(schema),
            "--model",
            tool_loop.MODEL,
            "--timeout",
            "600",
            "--receipts",
            str(receipts),
        ],
    )
    tool_loop.main()
    assert json.loads(capsys.readouterr().out)["score"] == verdict
    assert len(requests) == 2 and len(commands) == 2
    assert json.loads((receipts / "001-request.json").read_text()) == requests[1]
    for observed, expected in zip(
        observations, [files.get("/app/final_answer.txt", ""), files.get("/app/outputs/run_log.txt", "")]
    ):
        assert hashlib.sha256(observed["stdout"].encode()).digest() == hashlib.sha256(expected.encode()).digest()
    if case == "missing":
        assert all(x["returncode"] == 1 and x["stderr"] == "No such file" for x in observations)
    for i, observation in enumerate(observations):
        assert json.loads((receipts / f"000-tool-{i}.json").read_text()) == observation
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in original.items())
