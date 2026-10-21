# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.chemx.protocol import ModelOutputError, creator_definitions, score, validate_source


SOURCE = Path("../chemx-upstream").resolve()


@pytest.mark.parametrize("article", ["d0na00820f.pdf", "nano12183159.pdf"])
def test_actual_creator_multiset_identity_and_empty(article):
    if not SOURCE.exists():
        pytest.skip("Pinned creator source absent")
    definitions = creator_definitions(SOURCE)
    gold = pd.read_csv(SOURCE / "datasets/Nanomag.csv")
    for col in definitions["NUMERIC_COLUMNS"]["magnetic"]:
        gold[col] = gold[col].apply(lambda x: str(x.replace(",", ".")) if isinstance(x, str) else str(x))
    gold = gold.fillna("NOT_DETECTED")
    rows = gold.loc[gold.pdf == article, definitions["EXTRACTED_COLUMNS"]["magnetic"]].to_dict(orient="records")
    exact = score(SOURCE, article_id=article, answer=json.dumps({"samples": rows}))
    assert all(v["fp"] == 0 for v in exact["creator_column_metrics"].values())
    # Creator deduplicates entire predicted rows; preserve any effect on duplicates.
    assert exact["predicted_rows_after_deduplication"] == len(pd.DataFrame(rows).drop_duplicates())
    empty = score(SOURCE, article_id=article, answer='{"samples":[]}')
    assert empty["reward"] == 0
    assert all(v["fn"] == len(rows) for v in empty["creator_column_metrics"].values())
    assert exact["reward"] > empty["reward"]


def test_malformed_is_not_silent_zero():
    with pytest.raises(ModelOutputError):
        score(SOURCE, article_id="d0na00820f.pdf", answer="not JSON")
    with pytest.raises(ValueError, match="explicit two-article"):
        score(SOURCE, article_id="other.pdf", answer='{"samples":[]}')


def test_changed_source_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_source(tmp_path)


def test_column_multisets_do_not_preserve_record_association():
    definitions = creator_definitions(SOURCE)
    gold = pd.read_csv(SOURCE / "datasets/Nanomag.csv")
    for col in definitions["NUMERIC_COLUMNS"]["magnetic"]:
        gold[col] = gold[col].apply(lambda x: str(x.replace(",", ".")) if isinstance(x, str) else str(x))
    rows = (
        gold.fillna("NOT_DETECTED")
        .loc[gold.pdf == "d0na00820f.pdf", definitions["EXTRACTED_COLUMNS"]["magnetic"]]
        .to_dict(orient="records")
    )
    original = score(SOURCE, article_id="d0na00820f.pdf", answer=json.dumps({"samples": rows}))
    values = [r["np_core"] for r in rows]
    assert len(set(values)) > 1
    for row, value in zip(rows, values[1:] + values[:1]):
        row["np_core"] = value
    shuffled = score(SOURCE, article_id="d0na00820f.pdf", answer=json.dumps({"samples": rows}))
    assert shuffled["creator_column_metrics"] == original["creator_column_metrics"]


@pytest.mark.asyncio
async def test_native_verifier_preserves_lineage_and_rejects_wrong_manifest():
    import hashlib
    from types import SimpleNamespace

    from benchmarks.chemx.protocol import MANIFEST
    from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
    from resources_servers.chemx.app import Request, Server

    body = Request.model_construct(
        uuid="d0na00820f.pdf",
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        verifier_metadata={
            "article_id": "d0na00820f.pdf",
            "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        },
        response=NeMoGymResponse.model_validate(
            {
                "id": "fixture",
                "object": "response",
                "created_at": 0,
                "model": "fixture",
                "status": "completed",
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "output": [
                    {
                        "id": "m",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": '{"samples":[]}', "annotations": []}],
                    }
                ],
            }
        ),
        _ng_task_index=3,
        _ng_rollout_index=1,
    )
    server = SimpleNamespace(config=SimpleNamespace(upstream_repo=str(SOURCE)))
    result = await Server.verify(server, body)
    assert result.reward == 0 and result.model_dump()["_ng_task_index"] == 3
    body.__pydantic_extra__.update(mask_sample=True, failure_kind="transport_timeout", failure_reason="existing")
    preserved = await Server.verify(server, body)
    assert preserved.mask_sample and preserved.failure_kind == "transport_timeout"
    assert preserved.failure_reason == "existing"
    body.__pydantic_extra__.clear()
    body.response.output[0].content[0].text = ""
    malformed = await Server.verify(server, body)
    assert malformed.mask_sample and malformed.failure_kind == "chemx:invalid_model_output"
    assert malformed.scoring_details["raw_reward"] is None
    from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming

    incomplete = body.response.model_dump()
    incomplete.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
    body.response = NeMoGymResponse.model_validate(incomplete)
    budget = await Server.verify(server, body)
    assert budget.mask_sample and budget.failure_kind == "chemx:model_budget"
    assert budget.scoring_details["model_outcome"] == "output_budget_exhausted"
    body.verifier_metadata["manifest_sha256"] = "wrong"
    with pytest.raises(ValueError, match="manifest"):
        await Server.verify(server, body)


@pytest.mark.asyncio
async def test_converted_strict_schema_survives_native_verifier():
    import hashlib
    from types import SimpleNamespace

    from benchmarks.chemx.protocol import MANIFEST
    from nemo_gym.openai_utils import NeMoGymChatCompletion, NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.responses_converter import ResponsesConverter
    from resources_servers.chemx.app import Request, Server

    schema = json.loads((SOURCE / "LLM/data/schemas/magnetic.json").read_text())
    params = NeMoGymResponseCreateParamsNonStreaming(
        input="fixture",
        max_output_tokens=10000,
        text={"format": {"type": "json_schema", "name": "magnetic", "schema": schema, "strict": True}},
    )
    response = ResponsesConverter(return_token_id_information=False).chat_completion_to_response(
        responses_create_params=params,
        chat_completion=NeMoGymChatCompletion.model_validate(
            {
                "id": "fixture",
                "object": "chat.completion",
                "created": 0,
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": '{"samples":[]}'},
                    }
                ],
            }
        ),
    )
    body = Request(
        uuid="nano12183159.pdf",
        responses_create_params=params,
        response=response,
        verifier_metadata={
            "article_id": "nano12183159.pdf",
            "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        },
        _ng_task_index=0,
        _ng_rollout_index=0,
    )
    result = await Server.verify(SimpleNamespace(config=SimpleNamespace(upstream_repo=str(SOURCE))), body)
    assert result.reward == 0 and not result.mask_sample
    assert result.response.model_dump(by_alias=True)["text"]["format"]["schema"] == schema
    assert result.response.text.format.strict is True
    assert result.model_dump()["_ng_rollout_index"] == 0
