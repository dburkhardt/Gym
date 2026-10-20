# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-local native reward classification; never modifies creator scores."""

import json
import math
from pathlib import Path


MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"


def classify(native_reward: dict, details: object, receipts: str, expected_calls: int) -> dict:
    root = Path(receipts).resolve()
    failures = []
    ids = []
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def walk(x):
        if isinstance(x, dict):
            if x.get("error"):
                failures.append("native_error:" + str(x["error"]))
            if x.get("warnings"):
                failures.append("native_warnings:" + str(x["warnings"]))
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(details)
    if (
        not isinstance(native_reward, dict)
        or not native_reward
        or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in native_reward.values())
    ):
        failures.append("invalid_native_reward")
    manifests = list(root.glob("*-manifest.json"))
    if len(manifests) != expected_calls:
        failures.append("native_invocation_count_mismatch")
    for path in manifests:
        try:
            manifest = json.loads(path.read_text())
            directory = Path(manifest["receipts"]).resolve()
            assert directory.parent == root and directory.name == manifest["invocation_id"]
            assert manifest["native_timeout"] == 600
            if list(directory.glob("*failure.json")) or list(directory.glob("*http-error-body.bin")):
                failures.append("judge_failure_receipt")
            requests = sorted(directory.glob("*-request.json"))
            responses = sorted(directory.glob("*-response.json"))
            assert requests and len(requests) == len(responses), "unresolved_request"
            for requestpath in requests:
                request = json.loads(requestpath.read_text())
                response = json.loads(
                    requestpath.with_name(requestpath.name.replace("-request.json", "-response.json")).read_text()
                )
                assert request["model"] == MODEL and request["max_output_tokens"] == 16384
                assert response["model"] == MODEL and response["status"] == "completed"
                assert isinstance(response.get("id"), str) and response["id"] and response["id"] not in ids
                u = response["usage"]
                assert all(type(u.get(k)) is int and u[k] >= 0 for k in usage)
                assert u["total_tokens"] == u["input_tokens"] + u["output_tokens"] and u["output_tokens"] <= 16384
                ids.append(response["id"])
                for k in usage:
                    usage[k] += u[k]
        except (AssertionError, ValueError, KeyError, OSError, TypeError) as e:
            failures.append(type(e).__name__ + ":" + str(e))
    return {
        "native_reward": native_reward,
        "mask_sample": bool(failures),
        "failure_reasons": failures,
        "response_ids": ids,
        "observed_valid_response_usage": usage,
        "usage_scope": "valid exact-model completed responses only; malformed or unresolved receipts excluded",
        "expected_native_judge_invocations": expected_calls,
        "actual_invocation_manifests": len(manifests),
    }
