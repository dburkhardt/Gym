# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
from contextlib import nullcontext
import json
import os
import time
import traceback
from pathlib import Path

from metasyn.evaluator import evaluate_record
from metasyn.judge import CriteriaSimilarity
from metasyn.query import build_protocol_query
from metasyn.rag import extract_included_articles, generate_report


BASE = Path(os.environ["METASYN_DATA"])
MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"
REV = "c8fa07d89c44093d623f9a213c6bf070f40ab960"
os.environ.update(
    OPENAI_API_KEY="ssh-transport-placeholder", OPENAI_BASE_URL="http://127.0.0.1:49380/v1", OPENAI_MODEL=MODEL
)
parser = argparse.ArgumentParser()
parser.add_argument("--review-id", type=int, required=True)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--reference-file", type=Path, required=True)
args = parser.parse_args()
rows = [json.loads(x) for x in args.reference_file.read_text().splitlines()]
if args.review_id is not None:
    rows = [x for x in rows if x["ID"] == args.review_id]
assert rows, "No selected review"
# Guarded native launches pass only a task-scoped local bridge credential.
context = nullcontext()
if os.environ.get("METASYN_BRIDGE_SOCKET"):
    import metasyn.rag as creator_rag
    import metasyn.llm_client as creator_llm_client
    import metasyn.evaluator as creator_evaluator
    import metasyn.judge as creator_judge
    from responses_api_agents.metasyn_native.identity import verify_child
    verify_child(args.output_root / str(args.review_id), [creator_rag, creator_llm_client, creator_evaluator, creator_judge])
    from responses_api_agents.metasyn_native.guarded_clients import guarded_clients
    context = guarded_clients(
        rag=creator_rag, llm_client=creator_llm_client,
        socket_path=os.environ["METASYN_BRIDGE_SOCKET"],
        token=os.environ["METASYN_BRIDGE_TOKEN"],
        lineage=os.environ["METASYN_BRIDGE_LINEAGE"],
        digest=os.environ["METASYN_BRIDGE_DIGEST"],
        output=args.output_root / str(args.review_id) / "provider-calls",
    )
with context:
    for review in rows:
        out = args.output_root / str(review["ID"])
        out.mkdir(parents=True, exist_ok=True)
        if (out / "evaluation.json").exists():
            raise FileExistsError("Refusing to overwrite an existing evaluation")
        source = BASE / "rag" / str(review["ID"])
        started = time.time()
        try:
            prompt = (source / "prompt.txt").read_text()
            candidates = json.loads((source / "candidates.json").read_text())
            report, usage = generate_report(prompt, MODEL)
            (out / "report.md").write_text(report)
            result = {
                "review_id": review["ID"],
                "mode": "actual",
                "retriever": "bm25",
                "query": build_protocol_query(review),
                "retrieved_article_ids": [x["corpus_id"] for x in candidates],
                "retrieved_articles": candidates,
                **extract_included_articles(report, candidates),
                "usage": usage,
            }
            (out / "results.json").write_text(json.dumps(result, indent=2))
            similarity = CriteriaSimilarity(json.loads((BASE / "similarity-model.json").read_text())["local_path"])
            evaluation = evaluate_record(
                result,
                review,
                report,
                selection_source="report",
                dataset_id="THUIR/MetaSyn",
                dataset_revision=REV,
                judge_model=MODEL,
                similarity=similarity,
                id_only=False,
            )
            (out / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
            print(review["ID"], evaluation["status"], evaluation["metrics"], time.time() - started, flush=True)
        except Exception as e:
            (out / "failure.json").write_text(
                json.dumps(
                    {
                        "exception": type(e).__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                        "seconds": time.time() - started,
                    },
                    indent=2,
                )
            )
            print(review["ID"], "FAILED", type(e).__name__, str(e), flush=True)
            raise
