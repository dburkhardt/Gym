# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native RewardKit extension with isolated scientific shell, exact model only."""

import hashlib
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

from rewardkit.agents import AgentCLI, register_agent
from tool_loop import durable_write


MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"


class ScientificSuperJudge(AgentCLI):
    name = "gym-super-scientific"
    cli_name = sys.executable

    def ensure_installed(self) -> None:
        assert Path(__file__).with_name("scientific_tool_loop.py").is_file()
        assert os.environ.get("GYM_RK_JUDGE_SANDBOX") and os.environ.get("GYM_RK_JUDGE_IMAGE")

    def build_command(self, prompt: str, schema: dict, allowed_tools: tuple[str, ...] = ()) -> list[str]:
        assert not allowed_tools, "No unreviewed MCP tool substitution"
        fd, self.schema_path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(schema, f)
        root = Path(os.environ["GYM_RK_JUDGE_RECEIPTS"])
        root.mkdir(parents=True, exist_ok=True)
        identity = uuid.uuid4().hex
        receipts = root / identity
        timeout = float(os.environ["GYM_RK_NATIVE_JUDGE_TIMEOUT"])
        durable_write(
            root / (identity + "-manifest.json"),
            json.dumps(
                {
                    "invocation_id": identity,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
                    "native_timeout": timeout,
                    "receipts": str(receipts),
                }
            ).encode(),
        )
        return [
            sys.executable,
            str(Path(__file__).with_name("scientific_tool_loop.py")),
            "--prompt",
            prompt,
            "--schema",
            self.schema_path,
            "--receipts",
            str(receipts),
            "--timeout",
            str(timeout),
        ]

    def model_args(self, model: str) -> list[str]:
        assert model == MODEL
        return ["--model", model]

    def cleanup(self) -> None:
        Path(self.schema_path).unlink(missing_ok=True)


register_agent(ScientificSuperJudge)
