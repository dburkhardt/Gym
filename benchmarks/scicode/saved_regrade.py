"""Prepare resource-only regrading from frozen tasks and exact saved model captures.

No provider client is constructed. The prompt and extraction helpers are the same
ones used by the native agent. Missing or mismatched evidence fails closed.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.prompt import load_prompt_config
from responses_api_agents.scicode_agent.app import ScicodeAgentRunRequest
from responses_api_agents.scicode_agent.step_utils import (
    OUT_OF_CONTEXT,
    PREFILLED_STEPS_CODE,
    extract_python_script,
    is_context_window_error,
    process_problem_steps,
)


def prepare(task, result, captures, *, prompt_fpath, with_background=True, model_server="policy_model"):
    body = ScicodeAgentRunRequest.model_validate(task)
    if result["problem_id"] != body.problem_id:
        raise ValueError("Frozen task/result identity mismatch")
    ids = [c["model_call_id"] for c in captures]
    expected = [c["model_call_id"] for c in result["ng_model_call_capture"]["calls"]]
    if len(set(ids)) != len(ids) or ids != expected:
        raise ValueError("Missing, duplicate, or reordered captures")
    statuses = result["step_usage"]
    if len(statuses) != len(body.sub_steps):
        raise ValueError("Missing step state")
    prompt = load_prompt_config(prompt_fpath)
    params = body.responses_create_params.model_dump(exclude_unset=True, exclude_none=True)
    params.pop("input", None)
    params["model"] = body.responses_create_params.model or model_server
    params["tools"] = body.responses_create_params.tools
    previous = [None] * len(body.sub_steps)
    solutions = {}
    cursor = 0
    out_of_context = False
    for i, step in enumerate(body.sub_steps):
        state = statuses[i]
        if state["step_number"] != step["step_number"]:
            raise ValueError("Step identity mismatch")
        key = f"{body.problem_id}.{i + 1}"
        if (body.problem_id, i) in PREFILLED_STEPS_CODE:
            if state["status"] != "prefilled":
                raise ValueError("Creator-prefilled state mismatch")
            previous[i] = PREFILLED_STEPS_CODE[(body.problem_id, i)]
            continue
        if out_of_context:
            if state["status"] != "skipped":
                raise ValueError("Context-window skipped state mismatch")
            solutions[key] = OUT_OF_CONTEXT
            continue
        if cursor >= len(captures):
            raise ValueError("Missing generated-step capture")
        capture = captures[cursor]
        cursor += 1
        problem_steps, next_step, previous_code = process_problem_steps(body.sub_steps, i, previous, with_background)
        request = params | {
            "input": [
                {
                    "role": "user",
                    "content": prompt.user.format(
                        problem_steps_str=problem_steps,
                        next_step_str=next_step,
                        dependencies=body.required_dependencies,
                    ),
                }
            ]
        }
        if capture["request"] != request:
            raise ValueError("Saved request differs from exact frozen prompt/generation parameters")
        if state["status"] == "context_window_exceeded":
            if capture["status_code"] == 200 or not is_context_window_error(
                Exception(json.dumps(capture.get("response")))
            ):
                raise ValueError("Context-window state lacks terminal evidence")
            out_of_context = True
            solutions[key] = OUT_OF_CONTEXT
            continue
        if state["status"] != "generated" or capture["status_code"] != 200 or capture.get("error_category"):
            raise ValueError("Unsettled or unsupported model outcome")
        response = NeMoGymResponse.model_validate(capture["response"])
        if state["usage"] != (response.usage.model_dump() if response.usage else None):
            raise ValueError("Saved optional usage mismatch")
        code = extract_python_script(response.output_text)
        previous[i] = code
        prefix = (
            f"{body.required_dependencies}\n{previous_code}\n" if previous_code else f"{body.required_dependencies}\n"
        )
        solutions[key] = prefix + "\n" + code
    if cursor != len(captures):
        raise ValueError("Extra captures outside task")
    return body.model_dump() | {
        "solutions": solutions,
        "response": result["response"],
        "step_usage": statuses,
        "token_usage_version": result["token_usage_version"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = prepare(
        json.loads(args.task.read_text()),
        json.loads(args.result.read_text()),
        [json.loads(line) for line in args.captures.read_text().splitlines()],
        prompt_fpath=args.prompt,
    )
    manifest = {
        "version": 1,
        "operation": "prepare_saved_scicode_verify_request",
        "model_calls": 0,
        "input_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.task, args.result, args.captures, Path(args.prompt))
        },
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in {
                Path(__file__),
                Path(inspect.getfile(ScicodeAgentRunRequest)),
                Path(inspect.getfile(process_problem_steps)),
                Path(inspect.getfile(load_prompt_config)),
            }
        },
        "problem_id": value["problem_id"],
    }
    with args.output.open("x") as stream:
        json.dump(value, stream, indent=2)
    manifest["output_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    with args.output.with_suffix(args.output.suffix + ".manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)


if __name__ == "__main__":
    main()
