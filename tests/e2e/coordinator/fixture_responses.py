# SPDX-License-Identifier: Apache-2.0
"""Validate loopback fixture bodies before starting native services."""

import json
from collections.abc import Mapping
from typing import Any

from nemo_gym.openai_utils import NeMoGymChatCompletion


def validated_chat_fixture(payload: Mapping[str, Any]) -> bytes:
    """Use the serving model's schema, retaining the exact supplied fields."""
    encoded = json.dumps(dict(payload)).encode()
    NeMoGymChatCompletion.model_validate_json(encoded)
    return encoded


def cancellation_action_fixture(model: str) -> bytes:
    action = {
        "analysis": "Deterministic cancellation fixture",
        "plan": "Preserve known bytes then wait",
        "commands": [
            {"keystrokes": "printf preserved-by-real-shell > /workspace/preservation.txt\n", "duration": 0.1}
        ],
        "task_complete": False,
    }
    return validated_chat_fixture(
        {
            "id": "fixture-response",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": json.dumps(action)}, "finish_reason": "stop"}
            ],
        }
    )
