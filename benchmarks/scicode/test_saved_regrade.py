import copy

import pytest

from benchmarks.scicode.saved_regrade import prepare
from nemo_gym.prompt import load_prompt_config


PROMPT = "benchmarks/scicode/prompts/background.yaml"


def fixture():
    step = {
        "step_number": "1.1",
        "step_description_prompt": "Compute one.",
        "step_background": "Background.",
        "function_header": "def answer():",
        "return_line": "return value",
        "test_cases": ["assert answer()==1"],
    }
    task = {
        "problem_id": "1",
        "sub_steps": [step],
        "required_dependencies": "import numpy as np",
        "responses_create_params": {"input": [], "temperature": 0, "max_output_tokens": 16384},
    }
    prompt = load_prompt_config(PROMPT).user.format(
        problem_steps_str="",
        next_step_str="Compute one.\nBackground.\n\ndef answer():\n\nreturn value",
        dependencies="import numpy as np",
    )
    response = {
        "id": "r",
        "created_at": 0,
        "model": "model",
        "object": "response",
        "output": [
            {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "", "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }
    capture = {
        "model_call_id": "call1",
        "request": {
            "input": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_output_tokens": 16384,
            "model": "policy_model",
            "tools": [],
        },
        "response": response,
        "status_code": 200,
        "error_category": None,
    }
    result = {
        "problem_id": "1",
        "response": response,
        "token_usage_version": 1,
        "step_usage": [{"step_number": "1.1", "status": "generated", "usage": None}],
        "ng_model_call_capture": {"calls": [{"model_call_id": "call1"}]},
    }
    return task, result, [capture]


def test_keyed_empty_output_and_optional_usage_preserved():
    t, r, c = fixture()
    out = prepare(t, r, c, prompt_fpath=PROMPT)
    assert out["solutions"] == {"1.1": "import numpy as np\n\n"}
    assert out["step_usage"] == r["step_usage"] and out["step_usage"][0]["usage"] is None


@pytest.mark.parametrize("change", ["missing", "duplicate", "prompt", "model", "temperature", "unknown", "identity"])
def test_ambiguous_or_changed_inputs_refused(change):
    t, r, c = fixture()
    if change == "missing":
        c = []
    elif change == "duplicate":
        c *= 2
    elif change == "prompt":
        c[0]["request"]["input"][0]["content"] = "changed"
    elif change == "model":
        c[0]["request"]["model"] = "other"
    elif change == "temperature":
        c[0]["request"]["temperature"] = 1
    elif change == "unknown":
        c[0]["status_code"] = None
    elif change == "identity":
        r["problem_id"] = "2"
    with pytest.raises(ValueError):
        prepare(t, r, c, prompt_fpath=PROMPT)


def test_all_prefilled_without_captures():
    t, r, c = fixture()
    t["problem_id"] = r["problem_id"] = "62"
    t["sub_steps"][0]["step_number"] = "62.1"
    r["step_usage"] = [{"step_number": "62.1", "status": "prefilled", "usage": None}]
    r["ng_model_call_capture"]["calls"] = []
    assert prepare(t, r, [], prompt_fpath=PROMPT)["solutions"] == {}


def test_context_error_requires_actual_terminal_then_preserves_skips():
    t, r, c = fixture()
    c[0]["status_code"] = 400
    c[0]["response"] = {"error": "Requested token count exceeds context"}
    r["step_usage"][0]["status"] = "context_window_exceeded"
    step = copy.deepcopy(t["sub_steps"][0])
    step["step_number"] = "1.2"
    t["sub_steps"].append(step)
    r["step_usage"].append({"step_number": "1.2", "status": "skipped", "usage": None})
    assert prepare(t, r, c, prompt_fpath=PROMPT)["solutions"] == {
        "1.1": "_ran_out_of_context_",
        "1.2": "_ran_out_of_context_",
    }
    c[0]["status_code"] = 200
    with pytest.raises(ValueError):
        prepare(t, r, c, prompt_fpath=PROMPT)
