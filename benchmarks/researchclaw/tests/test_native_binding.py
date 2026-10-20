# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.researchclaw import stage_frozen
from nemo_gym.benchmarks import BenchmarkConfig
from nemo_gym.eval_plan import PlanBenchmark, PlanGym, build_eval_plan


def test_frozen_harbor_rows_keep_native_protocol_and_plan(tmp_path, monkeypatch):
    rows = [
        {
            "instance_id": "researchclaw::" + name,
            "responses_create_params": {"input": [], "temperature": 0.0},
            "agent_ref": {"name": "harbor_agent"},
        }
        for name in ["Neuroscience_000", "Life_002"]
    ]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    original = source.read_bytes()
    monkeypatch.setenv("CAMPAIGN_SAMPLE_JSONL", str(source))
    monkeypatch.setenv("CAMPAIGN_SAMPLE_SHA256", hashlib.sha256(original).hexdigest())
    monkeypatch.setattr(stage_frozen, "__file__", str(tmp_path / "benchmarks/researchclaw/stage_frozen.py"))
    output = stage_frozen.prepare()
    assert output.read_bytes() == original == source.read_bytes()
    plan = build_eval_plan(
        rows,
        benchmark=PlanBenchmark(name="researchclaw", config_paths=[]),
        taskset="researchclaw",
        gym=PlanGym(version="test", commit=None),
    )
    assert len(plan.tasks) == 2
    monkeypatch.setenv("CAMPAIGN_SAMPLE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="hash mismatch"):
        stage_frozen.prepare()
    assert output.read_bytes() == original


def test_native_benchmark_config_selects_scientific_verifier(monkeypatch):
    for key in [
        "POLICY_BASE_URL",
        "POLICY_API_KEY",
        "SCIENTIFIC_HARBOR_TASKS_DIR",
        "SCIENTIFIC_HARBOR_JOBS_DIR",
        "RESEARCHCLAW_SOURCE",
        "RESEARCHCLAW_FROZEN_MANIFEST",
        "RESEARCHCLAW_FROZEN_MANIFEST_SHA256",
        "RESEARCHCLAW_CONTROLLER_PYTHON",
    ]:
        monkeypatch.setenv(key, "offline-control")
    config = Path(__file__).resolve().parents[1] / "config.yaml"
    parsed = BenchmarkConfig.from_config_path(config)
    assert parsed.name == "researchclaw" and parsed.agent_name == "harbor_agent"
    assert str(parsed.dataset.prepare_script).endswith("stage_frozen.py")
    text = config.read_text()
    assert "ResearchClawVerifier" in text
    assert "artifact_capture_complete" not in text
