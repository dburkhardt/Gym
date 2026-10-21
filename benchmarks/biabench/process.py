# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned creator process evaluation of an immutable saved submission.

Run in a trusted, separate verifier environment. This does not execute a solver
or qualify arbitrary replay of a partly completed judge phase.
"""

import hashlib
import importlib
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import httpx
from openai import OpenAI

from benchmarks.biabench.prepare import validate
from benchmarks.biabench.regrade_outcome import imported_creator_hashes, prediction_hashes
from nemo_gym.sandbox.guarded_chat_transport import GuardedSyncChatTransport, publish


MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"


@contextmanager
def guarded_evaluator(module, *, socket_path, token, lineage, digest, output, transport=None):
    """Replace only SDK transport and retry admission, retaining creator prompts/budgets."""
    guarded = GuardedSyncChatTransport(
        socket_path=socket_path,
        token=token,
        lineage=lineage,
        digest=digest,
        namespace="biabench_process",
        model=MODEL,
        output=output,
        transport=transport,
    )
    original = module.LLMEvaluator
    with httpx.Client(transport=guarded, timeout=600, follow_redirects=False) as client:

        class Evaluator(original):
            def __init__(self, *args, **kwargs):
                if args or kwargs.get("model") != MODEL or kwargs.get("max_tokens") != 2048:
                    raise ValueError("Unreviewed BIABench judge model or native token budget")
                kwargs.update(
                    api_key="task-scoped-placeholder",
                    provider_hint="openai",
                    base_url="http://gym-internal/v1",
                    max_retries=1,
                )
                super().__init__(**kwargs)
                self.client.close()
                self.client = OpenAI(
                    api_key="task-scoped-placeholder",
                    base_url="http://gym-internal/v1",
                    http_client=client,
                    max_retries=0,
                )

        # Credential discovery is transport setup, not scientific scoring. Never load source-tree secrets.
        with (
            patch.object(module, "LLMEvaluator", Evaluator),
            patch.object(module, "_load_local_env_files", lambda: None),
        ):
            yield


def process_identity(submission: Path, lineage: str) -> tuple[dict, str]:
    manifest_path = Path(__file__).with_name("frozen_sources.json")
    before = prediction_hashes(submission)
    identity = {
        "lineage": lineage,
        "model": MODEL,
        "max_tokens": 2048,
        "chunk_size": 2,
        "samples": 1,
        "confidence_threshold": 0.6,
        "include_images": "plots",
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "submission_sha256": before,
        "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity, digest


def run_process(
    *,
    source: Path,
    task: Path,
    submission: Path,
    output: Path,
    socket_path: str,
    token: str,
    lineage: str,
    transport=None,
) -> dict:
    """Preserve complete native checklist results; failed judge calls are not valid scores."""
    import sys

    source, task, submission = (p.resolve(strict=True) for p in (source, task, submission))
    output = output.absolute()
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in (source, task, submission)):
        raise ValueError("New process output must be separate from immutable inputs")
    manifest_path = Path(__file__).with_name("frozen_sources.json")
    manifest = json.loads(manifest_path.read_text())
    validate(source, manifest["creator_files"])
    validate(task, manifest["task_files"])
    if not (submission / "artifacts").is_dir() or not (submission / "logs").is_dir():
        raise ValueError("Creator submission requires artifacts and logs")
    metadata = json.loads((submission / "submission.json").read_text())
    if metadata.get("task_id") != manifest["task_id"] or not metadata.get("task_instruction"):
        raise ValueError("Creator submission task identity and exact instruction required")
    if (
        hashlib.sha256(metadata["task_instruction"].encode()).hexdigest()
        != "380a3ac2065c676e69ae005df1c59409e693d7e9bb37e664ca6bc85db7e759c5"
    ):
        raise ValueError("Saved submission instruction differs from frozen CPU protocol")
    before = prediction_hashes(submission)
    identity, digest = process_identity(submission, lineage)
    output.mkdir(parents=True, exist_ok=False)
    publish(output / "identity.json", identity | {"digest": digest})
    try:
        sys.path.insert(0, str(source))
        evaluators = importlib.import_module("bioimage_agent_bench.evaluators")
        judge = importlib.import_module("bioimage_agent_bench.evaluators.vlm_judge")
        native = importlib.import_module("bioimage_agent_bench.eval.submission")
        imports = imported_creator_hashes(source, manifest["creator_files"])
        evaluation = output / "evaluation"
        evaluation.mkdir()
        evaluators.evaluate_task(
            manifest["task_id"],
            submission / "artifacts",
            task / "evaluation",
            context={"task_dir": str(task)},
            eval_out_dir=evaluation,
        )
        original_judge = judge.judge_manual_items

        def audited_judge(*args, **kwargs):
            kwargs.update(write_raw_log=True, use_cache=False)
            return original_judge(*args, **kwargs)

        with (
            guarded_evaluator(
                judge,
                socket_path=socket_path,
                token=token,
                lineage=lineage,
                digest=digest,
                output=output / "calls",
                transport=transport,
            ),
            patch.object(judge, "judge_manual_items", audited_judge),
        ):
            scores = native._apply_vlm_judgments(
                submission,
                evaluation,
                MODEL,
                2,
                None,
                task_dir=task,
                vlm_samples=1,
                vlm_confidence_threshold=0.6,
                vlm_include_images="plots",
                use_cache=False,
            )
        raw = [
            json.loads(line)
            for line in (evaluation / "vlm_judgement_raw.jsonl").read_text().splitlines()
            if line.strip()
        ]
        verdict = json.loads((evaluation / "vlm_judgement.json").read_text())
        responses = [json.loads(p.read_text()) for p in sorted((output / "calls").glob("*.response.json"))]
        for response in responses:
            body = json.loads(response["body"])
            if response["status"] != 200 or any(
                c.get("finish_reason") != "stop" or not c.get("message", {}).get("content")
                for c in body.get("choices", [])
            ):
                raise RuntimeError("Judge incomplete generation; process score is not accepted")
        if not raw or any(not r.get("parsed_ok") or r.get("error_kind") for r in raw):
            raise RuntimeError("Native judge transport/parse failure; process score is not accepted")
        if verdict["total_judged"] != verdict["total_candidates"] or scores is None:
            raise RuntimeError("Incomplete native checklist; process score is not accepted")
        if prediction_hashes(submission) != before:
            raise ValueError("Creator process evaluator changed immutable submission")
        imports = imported_creator_hashes(source, manifest["creator_files"])
        skipped_images = sum(r.get("n_images_skipped", 0) for r in raw)
        record = {
            "status": "completed",
            "measurement_validity": {
                "outcome": {
                    "status": "valid",
                    "reason": "Pinned native outcome scorer completed independently of VLM image encoding; execution fidelity is not universal scientific correctness.",
                },
                "process": {
                    "status": "unresolved" if skipped_images else "valid",
                    "reason": "Creator-selected reference images could not be encoded; process evidence coverage is incomplete."
                    if skipped_images
                    else "No creator-selected image encodings were omitted.",
                    "skipped_image_encodings": skipped_images,
                    "coverage_scope": "Image encoding completeness only, not a general judge accuracy claim.",
                },
            },
            "native_scores": scores,
            "creator_import_sha256": imports,
            "total_candidates": verdict["total_candidates"],
            "total_judged": verdict["total_judged"],
            "unknown_items": sum(r.get("vlm_status") == "unknown" for r in verdict["results"]),
            "skipped_image_encodings": sum(r.get("n_images_skipped", 0) for r in raw),
            "qualification": "Super judge variant; outcome and process remain separate; unknown is not zero",
        }
        publish(output / "result.json", record)
        return record
    except BaseException as exc:
        publish(
            output / "failure.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "process_score": None,
                "partial_creator_artifacts_retained": True,
            },
        )
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for field in ("source", "task", "submission", "output"):
        request[field] = Path(request[field])
    run_process(**request)
