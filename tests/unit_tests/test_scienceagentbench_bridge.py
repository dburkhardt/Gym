import json
from pathlib import Path

import httpx
import pytest

from benchmarks.scienceagentbench.creator_bridge import CreatorEngine, one_pass_agent
from benchmarks.scienceagentbench.prepare import format_task, load_frozen


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / "scienceagentbench-upstream"
PARQUET = ROOT.parents[1] / "outputs/broad-eval-campaign/scientific-agents/scienceagentbench-verified.parquet"
MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"


def engine(tmp_path, handler):
    if not SOURCE.is_dir():
        pytest.skip("Retained pinned source required")
    return CreatorEngine(
        SOURCE,
        model=MODEL,
        socket_path="/unused-fixture.sock",
        token="local-only",
        lineage="fixture-0",
        configuration_sha256="a" * 64,
        output=tmp_path / "calls",
        transport=httpx.MockTransport(handler),
    )


def test_actual_one_pass_creator_loop_guard_and_reference_free_request(tmp_path):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 1,
                "model": MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "```python\nprint('fixture')\n```"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )

    bridge = engine(tmp_path, handler)
    try:
        row = load_frozen(SOURCE, PARQUET)[1][0][1]
        trim_calls = []

        def fixture_trim(messages, model, max_tokens):
            trim_calls.append((model, max_tokens))
            return messages

        agent = one_pass_agent(
            SOURCE,
            bridge,
            trim_messages=fixture_trim,
            tokenizer_identity="fixture-only-no-production-qualification",
            input_cost_per_token=0,
            output_cost_per_token=0,
        )
        result = agent.solve_task(format_task(row), str(tmp_path / "prediction.py"))
        assert (tmp_path / "prediction.py").read_text() == "print('fixture')"
        assert trim_calls == [(MODEL, 26000)]
        assert calls == [
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": agent.sys_msg}],
                "temperature": 0.2,
                "top_p": 0.95,
                "max_tokens": 2000,
            }
        ]
        assert row["gold_program_name"] not in json.dumps(calls)
        assert row["eval_script_name"] not in json.dumps(calls)
        assert result["history"][0]["content"] == agent.sys_msg
        assert len(list((tmp_path / "calls").glob("*.request.json"))) == 1
        with pytest.raises(RuntimeError, match="forbidden"):
            agent.install(str(tmp_path / "prediction.py"))
    finally:
        bridge.close()


def test_rejected_call_never_resends(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, json={"error": "fixture quota"})

    bridge = engine(tmp_path, handler)
    try:
        with pytest.raises(Exception):
            bridge.respond([{"role": "user", "content": "fixture"}], 0.2, 0.95)
        with pytest.raises(RuntimeError, match="no automatic resend"):
            bridge.respond([{"role": "user", "content": "fixture"}], 0.2, 0.95)
        assert len(calls) == 1
        assert len(list((tmp_path / "calls").glob("*.response.json"))) == 1
    finally:
        bridge.close()


def test_missing_tokenizer_and_self_debug_refuse_before_call(tmp_path):
    bridge = engine(tmp_path, lambda request: pytest.fail("No provider call allowed"))
    try:
        kwargs = dict(
            trim_messages=lambda *a, **k: [], tokenizer_identity="", input_cost_per_token=0, output_cost_per_token=0
        )
        with pytest.raises(ValueError, match="tokenizer"):
            one_pass_agent(SOURCE, bridge, **kwargs)
        with pytest.raises(ValueError, match="Self-debug"):
            one_pass_agent(SOURCE, bridge, **(kwargs | {"tokenizer_identity": "fixture", "use_self_debug": True}))
    finally:
        bridge.close()


def test_actual_visual_judge_three_samples_and_exact_parser(tmp_path):
    from benchmarks.scienceagentbench.creator_bridge import visual_judge

    calls = []
    texts = ["[FINAL SCORE]: 40", "unparseable", "[FINAL SCORE]: 80"]

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "visual-fixture",
                "object": "chat.completion",
                "created": 1,
                "model": MODEL,
                "choices": [
                    {"index": i, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
                    for i, text in enumerate(texts)
                ],
            },
        )

    bridge = engine(tmp_path, handler)
    try:
        assert visual_judge(SOURCE, bridge, "pred-fixture", "gold-fixture") == (texts, 40)
        assert len(calls) == 1
        assert calls[0]["n"] == 3
        assert calls[0]["model"] == MODEL
        assert (calls[0]["temperature"], calls[0]["top_p"], calls[0]["max_tokens"]) == (0.2, 0.95, 1000)
        content = calls[0]["messages"][0]["content"]
        assert content[1]["image_url"]["url"].endswith("pred-fixture")
        assert content[2]["image_url"]["url"].endswith("gold-fixture")
    finally:
        bridge.close()


def test_visual_judge_refuses_fewer_than_three_without_resend(tmp_path):
    from benchmarks.scienceagentbench.creator_bridge import visual_judge

    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "id": "short",
                "object": "chat.completion",
                "created": 1,
                "model": MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "[FINAL SCORE]: 100"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    bridge = engine(tmp_path, handler)
    try:
        with pytest.raises(ValueError, match="three captured textual choices"):
            visual_judge(SOURCE, bridge, "pred", "gold")
        with pytest.raises(RuntimeError, match="no automatic resend"):
            visual_judge(SOURCE, bridge, "pred", "gold")
        assert len(calls) == 1
    finally:
        bridge.close()
