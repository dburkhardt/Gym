# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.server_utils import ServerClient
from resources_servers.campaign_creator_scoring.app import Config, CreatorResourcesServer


@pytest.mark.parametrize(
    "benchmark",
    [
        "molbench",
        "moljson",
        "hiscibench",
        "omebench",
        "genomeqa",
        "qcbench",
        "chemiq",
        "sino_us_drugqa",
        "moleculariq",
        "fgbench",
        "scienceqa",
        "scibench",
        "fukuyama",
        "mollangbench",
        "suprabench",
        "s2tomg",
        "molpuzzle",
        "molinstructions",
        "lpm24",
        "smolinstruct",
    ],
)
def test_full_configuration_and_resource_startup(benchmark, monkeypatch):
    monkeypatch.setenv("CAMPAIGN_UPSTREAM_REPO", "/tmp/creator")
    monkeypatch.setenv("CAMPAIGN_SAMPLE_JSONL", "/tmp/sample.jsonl")
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {"config_paths": [f"benchmarks/{benchmark}/config.yaml"]},
    )
    parser = GlobalConfigDictParser()
    global_config = parser.parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial, skip_load_from_cli=True, skip_load_from_dotenv=True, offline=True
        )
    )
    instances = parser.filter_for_server_instance_configs(global_config)
    resource = next(x for x in instances if x.get_server_ref().name == benchmark + "_resources")
    config = Config.model_validate({**resource.get_inner_run_server_config_dict(), "name": benchmark + "_resources"})
    client = ServerClient(head_server_config={"host": "127.0.0.1", "port": 23001}, global_config_dict=global_config)
    server = CreatorResourcesServer(config=config, server_client=client)
    with TestClient(server.setup_webserver()) as web:
        response = web.get("/reverify_mode")
        assert response.status_code == 200
        assert "stateless" in response.text.lower()


def test_verify_endpoint_with_pinned_qcbench(monkeypatch):
    import json
    from pathlib import Path

    repo = Path("../qcbench-upstream").resolve()
    root = Path("../../outputs/broad-eval-campaign").resolve()
    sample = root / "scientific-agents/qcbench/qcbench-5.jsonl"
    trace = root / "proteinlmbench/pilot-20-v2.jsonl"
    if not (repo.exists() and sample.exists() and trace.exists()):
        pytest.skip("Campaign pinned source and saved response fixture unavailable")
    monkeypatch.setenv("CAMPAIGN_UPSTREAM_REPO", str(repo))
    monkeypatch.setenv("CAMPAIGN_SAMPLE_JSONL", str(sample))
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT, {"config_paths": ["benchmarks/qcbench/config.yaml"]}
    )
    parser = GlobalConfigDictParser()
    global_config = parser.parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial, skip_load_from_cli=True, skip_load_from_dotenv=True, offline=True
        )
    )
    resource = next(
        x
        for x in parser.filter_for_server_instance_configs(global_config)
        if x.get_server_ref().name == "qcbench_resources"
    )
    config = Config.model_validate({**resource.get_inner_run_server_config_dict(), "name": "qcbench_resources"})
    server = CreatorResourcesServer(
        config=config,
        server_client=ServerClient(
            head_server_config={"host": "127.0.0.1", "port": 23001}, global_config_dict=global_config
        ),
    )
    row = json.loads(sample.read_text().splitlines()[0])
    response = json.loads(trace.read_text().splitlines()[0])["response"]
    response["output"] = [
        {
            "id": "control",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": "\\boxed{" + row["verifier_metadata"]["answer"] + "}",
                    "annotations": [],
                }
            ],
        }
    ]
    with TestClient(server.setup_webserver()) as web:
        result = web.post("/verify", json={**row, "response": response})
        assert result.status_code == 200, result.text
        assert result.json()["reward"] == 1
        assert result.json()["scoring_details"]["metric"] == "creator_numeric"


@pytest.mark.parametrize(
    "benchmark",
    [
        "molbench",
        "moljson",
        "hiscibench",
        "omebench",
        "genomeqa",
        "qcbench",
        "chemiq",
        "sino_us_drugqa",
        "moleculariq",
        "fgbench",
        "scienceqa",
        "scibench",
        "fukuyama",
        "mollangbench",
        "suprabench",
        "s2tomg",
        "molpuzzle",
        "molinstructions",
        "lpm24",
        "smolinstruct",
    ],
)
def test_prepare_and_real_collation(benchmark, monkeypatch, tmp_path):
    import importlib
    import json
    from pathlib import Path

    from nemo_gym.train_data_utils import TrainDataProcessor

    sample = Path("../../outputs/broad-eval-campaign/scientific-agents") / benchmark / (benchmark + "-5.jsonl")
    if not sample.exists():
        pytest.skip("Pinned campaign sample unavailable")
    monkeypatch.setenv("CAMPAIGN_SAMPLE_JSONL", str(sample.resolve()))
    monkeypatch.setenv("CAMPAIGN_UPSTREAM_REPO", "/tmp/creator")
    staged = importlib.import_module("benchmarks." + benchmark + ".prepare").prepare()
    source_rows = [json.loads(line) for line in sample.read_text().splitlines()]
    staged_rows = [json.loads(line) for line in staged.read_text().splitlines()]
    assert len(staged_rows) == len(source_rows)
    for original, prepared in zip(source_rows, staged_rows, strict=True):
        assert {key: prepared[key] for key in original} == original
        assert prepared["task_id"]

    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {
            "config_paths": [f"benchmarks/{benchmark}/config.yaml"],
            "output_dirpath": str(tmp_path),
            "mode": "train_preparation",
            "should_download": False,
            "overwrite_metrics_conflicts": True,
        },
    )
    global_config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial, skip_load_from_cli=True, skip_load_from_dotenv=True, offline=True
        )
    )
    TrainDataProcessor().run(global_config)
    actual = [json.loads(x) for x in (tmp_path / "benchmark.jsonl").read_text().splitlines()]
    expected = [json.loads(x) for x in sample.read_text().splitlines()]
    assert len(actual) == len(expected) == 5
    assert {x["uuid"] for x in actual} == {x["uuid"] for x in expected}


def test_structured_output_response_alias_roundtrip():
    import json
    from pathlib import Path

    from resources_servers.campaign_creator_scoring.app import VerifyRequest, VerifyResponse

    sample = json.loads(
        Path("../../outputs/broad-eval-campaign/scientific-agents/moljson/moljson-5.jsonl").read_text().splitlines()[0]
    )
    trace = json.loads(
        Path("../../outputs/broad-eval-campaign/proteinlmbench/pilot-20-v2.jsonl").read_text().splitlines()[0]
    )
    response = trace["response"]
    response["text"] = sample["responses_create_params"]["text"]
    body = VerifyRequest.model_validate({**sample, "response": response})
    actual = VerifyResponse(**body.model_dump(by_alias=True), reward=1, scoring_details={})
    assert actual.response.text.format.schema_ == response["text"]["format"]["schema"]


@pytest.mark.asyncio
async def test_model_budget_accounting_keeps_score_and_denominator(monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from resources_servers.campaign_creator_scoring import app

    rows = list(
        map(
            json.loads,
            Path("../../outputs/broad-eval-campaign/moljson-schema-repaired/pilot-5-v2.jsonl")
            .read_text()
            .splitlines(),
        )
    )
    row = next(r for r in rows if r["response"]["status"] == "incomplete")
    body = app.VerifyRequest.model_validate(row)

    async def fake_score(**kwargs):
        return {"reward": 0, "metric": "fixture"}

    monkeypatch.setattr(app, "run_score", fake_score)
    server = SimpleNamespace(config=SimpleNamespace(benchmark="moljson", upstream_repo="/tmp", scoring_timeout_s=600))
    result = await app.CreatorResourcesServer.verify(server, body)
    assert result.failure_kind == "model_budget"
    assert result.reward == 0 and not result.mask_sample
    assert result.response == body.response


def test_verify_endpoint_with_pinned_molbench(monkeypatch):
    import json
    from pathlib import Path

    repo = Path(
        "../../outputs/broad-eval-campaign/harbor/agent-expansion/sources/216/InternScience-MolClaw-c16be3a"
    ).resolve()
    root = Path("../../outputs/broad-eval-campaign").resolve()
    sample = root / "scientific-agents/molbench/molbench-5.jsonl"
    trace = root / "proteinlmbench/pilot-20-v2.jsonl"
    if not (repo.exists() and sample.exists() and trace.exists()):
        pytest.skip("Campaign pinned source and saved response fixture unavailable")
    monkeypatch.setenv("CAMPAIGN_UPSTREAM_REPO", str(repo))
    monkeypatch.setenv("CAMPAIGN_SAMPLE_JSONL", str(sample))
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT, {"config_paths": ["benchmarks/molbench/config.yaml"]}
    )
    parser = GlobalConfigDictParser()
    global_config = parser.parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial, skip_load_from_cli=True, skip_load_from_dotenv=True, offline=True
        )
    )
    resource = next(
        x
        for x in parser.filter_for_server_instance_configs(global_config)
        if x.get_server_ref().name == "molbench_resources"
    )
    config = Config.model_validate({**resource.get_inner_run_server_config_dict(), "name": "molbench_resources"})
    server = CreatorResourcesServer(
        config=config,
        server_client=ServerClient(
            head_server_config={"host": "127.0.0.1", "port": 23001}, global_config_dict=global_config
        ),
    )
    row = json.loads(sample.read_text().splitlines()[0])
    response = json.loads(trace.read_text().splitlines()[0])["response"]
    response["output"] = [
        {
            "id": "control",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": "<answer>" + json.dumps(row["verifier_metadata"]["sample"]["answer"][:3]) + "</answer>",
                    "annotations": [],
                }
            ],
        }
    ]
    with TestClient(server.setup_webserver()) as web:
        result = web.post("/verify", json={**row, "response": response})
        assert result.status_code == 200, result.text
        assert result.json()["reward"] == 1
        assert result.json()["scoring_details"]["metric"] == "creator_top3_hit_rate"
