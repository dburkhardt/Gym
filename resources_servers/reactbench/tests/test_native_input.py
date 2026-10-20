# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import runpy
from pathlib import Path

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming


def test_image_detail_is_required_and_prepared(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[3]
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"responses_create_params": {"input": [{"role": "user", "content": [
        {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
        {"type": "input_text", "text": "Read the reaction"},
    ]}]}, "verifier_metadata": {"answer": "A"}}) + "\n")
    monkeypatch.setenv("CAMPAIGN_SAMPLE_JSONL", str(source))
    # Run a copied script so its relative output stays inside the temporary tree.
    script = tmp_path / "benchmarks/reactbench/prepare.py"
    script.parent.mkdir(parents=True)
    script.write_text((repo / "benchmarks/reactbench/prepare.py").read_text())
    runpy.run_path(str(script), run_name="__main__")
    row = json.loads((tmp_path / "resources_servers/reactbench/data/benchmark.jsonl").read_text())
    NeMoGymResponseCreateParamsNonStreaming.model_validate(row["responses_create_params"])
    image = row["responses_create_params"]["input"][0]["content"][0]
    assert image == {"type": "input_image", "image_url": "data:image/png;base64,AA==", "detail": "auto"}
