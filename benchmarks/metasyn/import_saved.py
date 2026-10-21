# SPDX-License-Identifier: Apache-2.0
"""Read-only historical import; recompute ID metrics, never invoke a judge/encoder."""

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

from benchmarks.metasyn.prepare import stage
from nemo_gym.result_disposition import summarize_dispositions


MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Inputs:
    def __init__(self):
        self.hashes = {}

    def read(self, path):
        path = Path(path).absolute()
        if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
            raise ValueError(f"Expected regular nonsymlink input: {path}")
        data = path.read_bytes()
        self.hashes[str(path)] = digest(data)
        return data

    def json(self, path):
        return json.loads(self.read(path))

    def rows(self, path):
        return [json.loads(line) for line in self.read(path).splitlines() if line.strip()]


def creator_functions(source, inputs):
    manifest = Path(__file__).parents[2] / "responses_api_agents/metasyn_native/creator-python-files.json"
    for name, expected in inputs.json(manifest).items():
        if digest(inputs.read(source / name)) != expected:
            raise ValueError(f"Creator source checksum mismatch: {name}")
    # Load only the two exact modules needed; package __init__ imports datasets/encoder.
    name = "_metasyn_saved_import"
    package = types.ModuleType(name)
    package.__path__ = [str(source)]
    sys.modules[name] = package
    for module in ("rag", "evaluation"):
        spec = importlib.util.spec_from_file_location(f"{name}.{module}", source / f"{module}.py")
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = loaded
        spec.loader.exec_module(loaded)
    return loaded.resolve_selection, loaded.id_metrics


def text_content(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list) or any(
        not isinstance(c, dict) or c.get("type") not in {"text", "input_text", "output_text"} for c in content
    ):
        raise ValueError("Unsupported historical text content")
    return "".join(c["text"] for c in content)


def reconcile(root: Path, source: Path, transport: Path):
    inputs = Inputs()
    resolve, score = creator_functions(source, inputs)
    with tempfile.TemporaryDirectory() as temporary:
        temporary_root = Path(temporary).resolve()
        frozen = inputs.rows(stage(root, temporary_root / "sample.jsonl"))
        # Canonical paths also cover macOS /var -> /private/var aliases.
        inputs.hashes = {
            p: h for p, h in inputs.hashes.items() if not Path(p).resolve().is_relative_to(temporary_root)
        }
    for name in ("gym-pilot5.jsonl", "gym-remaining15.jsonl", "rag-preparation.json"):
        inputs.read(root / name)
    refs = inputs.rows(root / "full-reference-rows-20.jsonl")
    references = {r["ID"]: r for r in refs}
    ids = [r["verifier_metadata"]["review_id"] for r in frozen]
    if len(references) != len(refs) or set(references) != set(ids):
        raise ValueError("Reference identity set differs from frozen20")
    projections, bindings, summaries = [], [], {}
    for cohort, file, expected in (
        ("pilot5", "gym-pilot5-results", frozen[:5]),
        ("remaining15-run1", "gym-remaining15-results", frozen[5:]),
    ):
        rows = inputs.rows(root / f"{file}.jsonl")
        failures = inputs.rows(root / f"{file}_failures.jsonl")
        if len(rows) != len(expected) or failures:
            raise ValueError("Historical cohort incomplete or has failure sidecar; manual reconciliation required")
        cohort_rows = []
        for row, task in zip(rows, expected, strict=True):
            rid = task["verifier_metadata"]["review_id"]
            folder = root / cohort / str(rid)  # Never follow saved artifact_dir.
            report = inputs.read(folder / "report.md").decode()
            record = inputs.json(folder / "results.json")
            evaluation = inputs.json(folder / "evaluation.json")
            prompt = inputs.read(root / "rag" / str(rid) / "prompt.txt").decode()
            messages = row["responses_create_params"]["input"]
            if len(messages) != 1 or messages[0]["role"] != "user" or text_content(messages[0]["content"]) != prompt:
                raise ValueError("Saved executed prompt mismatch")
            outputs = row["response"]["output"]
            if len(outputs) != 1 or text_content(outputs[0]["content"]) != report:
                raise ValueError("Saved report mismatch")
            if (
                record["review_id"] != rid
                or evaluation["review_id"] != rid
                or evaluation["report_sha256"] != digest(report.encode())
            ):
                raise ValueError("Saved review/report identity mismatch")
            if (
                evaluation["status"] != "succeeded"
                or row["evaluation_status"] != "succeeded"
                or row.get("mask_sample")
                or row.get("failure_kind")
            ):
                raise ValueError("Unsupported incomplete historical execution")
            protocol = evaluation["evaluator"]
            if (
                protocol["selection_source"] != "report"
                or protocol["id_only"]
                or protocol["judge_model"] != MODEL
                or protocol["dataset_revision"] != "c8fa07d89c44093d623f9a213c6bf070f40ab960"
            ):
                raise ValueError("Historical scientific protocol mismatch")
            candidates = inputs.json(root / "rag" / str(rid) / "candidates.json")
            if record["retrieved_articles"] != candidates:
                raise ValueError("Saved candidate pool mismatch")
            selection = resolve(record, report, "report")
            if selection != evaluation["selection"]:
                raise ValueError("Creator selection mismatch")
            recomputed = score(
                references[rid].get("matched_corpus_ids") or [],
                selection["included_article_ids"],
                selection["retrieved_article_ids"],
                selection["unmatched_included_article_ids"],
                selection["unmatched_included_entries"],
            )
            if (
                row["metrics"] != evaluation["metrics"]
                or row["reward"] != evaluation["metrics"]["inclusion_f1"]
                or any(evaluation["metrics"].get(k) != v for k, v in recomputed.items())
            ):
                raise ValueError("Historical score differs from pinned deterministic recomputation")
            projection = copy.deepcopy(row)
            if projection.get("metadata"):
                raise ValueError("Existing metadata requires explicit merge review")
            projection["metadata"] = {
                "source": "MetaSyn historical native BM25 top200 fixed-pool RAG / frozen20",
                "config": {
                    "evaluator": protocol,
                    "creator_evaluation_sha256": digest((source / "evaluation.py").read_bytes()),
                    "reference_sha256": digest((root / "full-reference-rows-20.jsonl").read_bytes()),
                },
                "verifier_result": {"rewards": copy.deepcopy(evaluation["metrics"])},
                "measurement_review": {
                    "version": "metasyn-saved-id-metrics-v1",
                    "implementation": {
                        "import_path": "benchmarks.metasyn.import_saved:reconcile",
                        "sha256": digest(Path(__file__).read_bytes()),
                    },
                    "metrics": {
                        k: {
                            "status": "valid" if k in recomputed else "unresolved",
                            "reason": "Exact pinned creator deterministic ID-metric parity; historical top200 variant"
                            if k in recomputed
                            else "Preserved historical judge/encoder observation; not independently re-evaluated",
                        }
                        for k in evaluation["metrics"]
                    },
                },
            }
            projections.append(projection)
            cohort_rows.append(projection)
            bindings.append(
                {
                    "review_id": rid,
                    "cohort": cohort,
                    "historical_task_index": row.get("_ng_task_index"),
                    "historical_rollout_index": row.get("_ng_rollout_index"),
                    "prompt_sha256": digest(prompt.encode()),
                    "report_sha256": digest(report.encode()),
                    "deterministic_metrics": recomputed,
                }
            )
        summaries[cohort] = summarize_dispositions(cohort_rows, expected_count=len(expected))
    raw_audit = inputs.json(root / "remaining15-raw-body-audit.json")
    receipts = inputs.rows(transport / "remaining15-run1/receipts.jsonl")
    indexed = {r["request_artifact_id"]: r for r in receipts}
    if (
        len(indexed) != 75
        or len(receipts) != 75
        or len(raw_audit["artifacts"]) != 75
        or len({a["artifact_id"] for a in raw_audit["artifacts"]}) != 75
    ):
        raise ValueError("Historical raw artifact coverage mismatch")
    artifact_directories = {p.name for p in (transport / "remaining15-run1").iterdir() if p.is_dir()}
    if artifact_directories != set(indexed):
        raise ValueError("Unreconciled historical artifact directories")
    generators = {}
    for artifact in raw_audit["artifacts"]:
        aid = artifact["artifact_id"]
        if not isinstance(aid, str) or Path(aid).name != aid or aid in {".", ".."}:
            raise ValueError("Unsafe artifact identity")
        receipt = indexed[aid]
        folder = transport / "remaining15-run1" / aid
        req_bytes, res_bytes = inputs.read(folder / "request.json"), inputs.read(folder / "response.json")
        if digest(req_bytes) != artifact["request_sha256"] or digest(res_bytes) != artifact["response_sha256"]:
            raise ValueError("Historical raw artifact checksum mismatch")
        req, res = json.loads(req_bytes), json.loads(res_bytes)
        if (
            receipt["status"] != 200
            or receipt["requested_model"] != MODEL
            or receipt["returned_model"] != MODEL
            or receipt["usage"] != res["usage"]
            or req["model"] != MODEL
            or res["model"] != MODEL
            or res["id"] != receipt["response_id"]
            or res["id"] != artifact["response_id"]
        ):
            raise ValueError("Historical provider receipt mismatch")
        if receipt["role"] == "task_model":
            matches = [
                b
                for b in bindings[5:]
                if req["messages"]
                == [{"role": "user", "content": (root / "rag" / str(b["review_id"]) / "prompt.txt").read_text()}]
            ]
            if (
                len(matches) != 1
                or matches[0]["review_id"] in generators
                or digest(res["choices"][0]["message"]["content"].strip().encode()) != matches[0]["report_sha256"]
            ):
                raise ValueError("Historical generator request/report binding mismatch")
            generators[matches[0]["review_id"]] = aid
    if len(generators) != 15 or len({r["response_id"] for r in receipts}) != 75:
        raise ValueError("Historical generator/response identity coverage mismatch")
    inputs.read(transport / "local_bridge.py")
    pilot_receipts = inputs.rows(transport / "receipts.jsonl")
    if (
        len(pilot_receipts) != 25
        or any(
            r.get("status") != 200 or r.get("requested_model") != MODEL or r.get("returned_model") != MODEL
            for r in pilot_receipts
        )
        or len({r["response_id"] for r in pilot_receipts}) != 25
    ):
        raise ValueError("Pilot historical receipts need reconciliation")
    # Do not merge cohort-local task indices into invented execution identities.
    report_rows = [
        {k: v for k, v in r.items() if k not in {"_ng_task_index", "_ng_rollout_index", "_ng_attempt_index"}}
        for r in projections
    ]
    return projections, {
        "version": 1,
        "mode": "saved-output-only",
        "new_provider_calls": 0,
        "task_ids": ids,
        "bindings": bindings,
        "source_hashes": inputs.hashes,
        "cohort_summaries": summaries,
        "combined_metric_summary": summarize_dispositions(report_rows, expected_count=20),
        "provider_evidence": {
            "pilot5_receipts": 25,
            "pilot5_raw_bodies": "unavailable",
            "remaining15_raw_pairs_verified": 75,
            "remaining15_generators_bound": generators,
            "remaining15_judge_raw_pairs": 60,
            "unpaired_raw_artifact_directories": 0,
            "judge_per_review_attribution": "not inferred from timestamp/order",
            "unknown_historical_requests": "unknown: successful-response ledger is not a pre-dispatch intent ledger",
            "modern_capture_or_episode_ids_created": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialized", type=Path, required=True)
    parser.add_argument("--creator-source", type=Path, required=True)
    parser.add_argument("--transport", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, receipt = reconcile(args.materialized, args.creator_source, args.transport)
    args.output.mkdir(parents=True, exist_ok=False)
    output = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (args.output / "imported-results.jsonl").write_text(output)
    receipt["imported_results_sha256"] = digest(output.encode())
    (args.output / "reconciliation.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
