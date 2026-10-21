# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report complete creator chunk groups without pooling chunk accuracy."""

from benchmarks.csmed_ft.creator_protocol import PROTOCOL, ChunkPrediction, merge_chunks, parse_label


def report(inputs: list[dict], outputs: list[dict]) -> dict:
    """Require exact declared task scope and one output per chunk; missing stays unscored."""
    import sklearn
    from sklearn.metrics import accuracy_score, f1_score, fbeta_score, precision_score, recall_score

    from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

    expected = {row["task_id"]: row["verifier_metadata"] for row in inputs}
    requests = {row["task_id"]: row["responses_create_params"] for row in inputs}
    if len(expected) != len(inputs) or not expected:
        raise ValueError("Empty or duplicate materialized chunk identities")
    bindings = {(meta.get("protocol"), meta.get("recipe_sha256")) for meta in expected.values()}
    if len(bindings) != 1 or next(iter(bindings))[0] != PROTOCOL or not next(iter(bindings))[1]:
        raise ValueError("Mixed or missing creator recipe identity")
    received = {}
    for row in outputs:
        key = row["task_id"]
        if key not in expected or key in received:
            raise ValueError("Unexpected or duplicate chunk output")
        if row.get("verifier_metadata") != expected[key]:
            raise ValueError("Output reference/protocol identity differs from materialized input")
        actual_request = row.get("responses_create_params")
        if not isinstance(actual_request, dict):
            raise ValueError("Missing actual generation request")
        expected_params = NeMoGymResponseCreateParamsNonStreaming.model_validate(requests[key])
        actual_params = NeMoGymResponseCreateParamsNonStreaming.model_validate(actual_request)
        if actual_params.model_dump(exclude={"model"}) != expected_params.model_dump(exclude={"model"}):
            raise ValueError("Actual prompt/settings differ from materialized request")
        protocol = expected[key].get("generation_protocol", {})
        if not all(field in protocol for field in ("model", "temperature", "max_output_tokens")):
            raise ValueError("Missing explicit generation model/protocol identity")
        response = row.get("response", {})
        if response.get("model") != protocol["model"] or any(
            actual_request.get(field) != protocol[field] for field in ("temperature", "max_output_tokens")
        ):
            raise ValueError("Actual generation model/settings differ from declared protocol")
        received[key] = row
    groups = {}
    for key, meta in expected.items():
        groups.setdefault((meta["paper_id"], meta["review_id"], meta["uuid"]), []).append((key, meta))
    papers = []
    for identity, members in groups.items():
        counts = {meta["chunk_count"] for _, meta in members}
        ordinals = sorted(meta["paper_chunk_id"] for _, meta in members)
        if counts != {len(members)} or ordinals != list(range(len(members))):
            raise ValueError("Materialized paper chunk set is incomplete or inconsistent")
        golds = {meta["answer"] for _, meta in members}
        if len(golds) != 1 or not golds <= {"included", "excluded"}:
            raise ValueError("Conflicting or invalid paper references")
        missing = [key for key, _ in members if key not in received]
        if missing:
            papers.append(
                {"identity": identity, "status": "unresolved", "reason": "missing_chunk_outputs", "missing": missing}
            )
            continue
        chunks = []
        for key, meta in members:
            if received[key].get("mask_sample") is True:
                break
            response = received[key].get("response")
            if not isinstance(response, dict) or not isinstance(response.get("output"), list):
                break
            if response.get("status") in {"failed", "cancelled", "in_progress", "queued"} or response.get("error"):
                break
            text = "".join(
                part["text"]
                for item in response["output"]
                if item.get("type") == "message"
                for part in item.get("content", [])
                if part.get("type") == "output_text"
            )
            chunks.append(ChunkPrediction(identity[2], parse_label(text), int(meta["answer"] == "included")))
        if len(chunks) != len(members):
            papers.append({"identity": identity, "status": "unresolved", "reason": "missing_or_unusable_response"})
            continue
        merged = merge_chunks(chunks)[0]
        papers.append({"identity": identity, "status": "scored_creator_protocol", **vars(merged)})
    scored = [row for row in papers if row["status"] == "scored_creator_protocol"]
    truth = [row["gold_label"] for row in scored]
    predictions = [row["prediction"] for row in scored]
    metrics = None
    if scored:
        metrics = {
            "acc": accuracy_score(truth, predictions),
            "prec": precision_score(truth, predictions, average="macro"),
            "recall": recall_score(truth, predictions, average="macro"),
            "f1": f1_score(truth, predictions, average="binary"),
            "f1-macro": f1_score(truth, predictions, average="macro"),
            "f1-micro": f1_score(truth, predictions, average="micro"),
            "f3": fbeta_score(truth, predictions, beta=3, average="macro"),
        }
    return {
        "protocol": PROTOCOL,
        "metric_runtime": {"scikit_learn": sklearn.__version__, "creator_requirement": "~=1.2.2"},
        "papers": papers,
        "expected_papers": len(groups),
        "scored_papers": len(scored),
        "unresolved_papers": len(groups) - len(scored),
        "expected_chunks": len(expected),
        "received_chunks": len(received),
        "metric_denominator": len(scored),
        "metrics": metrics,
        "limitation": "Creator merge treats all-invalid chunk groups as included; not a strict-validity score.",
    }
