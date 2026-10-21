# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-bound creator-library aggregates over explicitly scoped saved draws."""

import ast
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path


SOURCE_COMMIT = "f033a3dfea50c3be5782a979e73eeeb842a7bae1"
LOADER_SHA256 = "d361d3e89bed955aa709c2f5ed319c1bb56bc9cccef9d135cc1c8e3e6b0e87f3"
SOURCE_HASHES = {
    "top_k.py": "44a307ec9f8e7c5c65a5dbeb1b721d008d9b2decea961a10d286b00cf7daeffb",
    "diversity_aware_top_k.py": "2e83e3437b9cbfc4b31e9b479cecb79346af54f04efa3f629afe28dca5bfff6b",
    "fingeprints_utils.py": "64d26f559c23d7e548fb76b11ff8496e984a1b0d1c567264e0f7f3124c3a9e8d",
}


def creator_functions(source_root: Path) -> dict:
    """Execute only reviewed hash-bound metric modules, without package side effects."""
    scope = {}
    for name in ("fingeprints_utils.py", "top_k.py", "diversity_aware_top_k.py"):
        path = source_root / "molrgen/evaluation" / name
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != SOURCE_HASHES[name]:
            raise ValueError(f"Pinned creator source mismatch: {name}")
        tree = ast.parse(source)
        # Resolve this one import to the already hash-verified fingerprint module.
        tree.body = [
            node
            for node in tree.body
            if not (isinstance(node, ast.ImportFrom) and node.module == "molrgen.evaluation.fingeprints_utils")
        ]
        exec(compile(tree, str(path), "exec"), scope)
    loader = source_root / "notebooks/utils/molgen_utils.py"
    if hashlib.sha256(loader.read_bytes()).hexdigest() != LOADER_SHA256:
        raise ValueError("Pinned creator extraction source mismatch")
    return scope


def _finite(value: object) -> bool:
    return type(value) in {int, float} and math.isfinite(value)


def _candidate(row: dict) -> tuple[str, float]:
    """Creator notebook last-SMILES convention; missing molecules remain zero draws."""
    native = row["diagnostics"]["creator_response"]
    receipt = native["creator_response"]
    if len(receipt["verifier_metadatas"]) != 1 or len(receipt["rewards"]) != 1:
        raise ValueError("Expected singleton creator receipt")
    if native.get("reward") != row["reward"] or receipt["rewards"][0] != row["reward"]:
        raise ValueError("Saved reward does not match creator receipt")
    text = "".join(
        part["text"]
        for message in row["response"].get("output", [])
        if message.get("type") == "message"
        for part in message.get("content", [])
        if part.get("type") == "output_text"
    )
    request = native.get("request", {})
    if request.get("query") != [text] or request.get("metadata") != [row["verifier_metadata"]["metadata"]]:
        raise ValueError("Creator receipt query/metadata does not match saved completion")
    metadata = receipt["verifier_metadatas"][0]["generation_verifier_metadata"]
    smiles = metadata["all_smi"]
    if not isinstance(smiles, list) or not all(isinstance(smi, str) for smi in smiles):
        raise ValueError("Invalid creator molecule metadata")
    if not smiles:
        return "", 0.0
    if len(smiles) == 1:
        score = row["reward"]
    else:
        scores = metadata["all_smi_rewards"]
        if len(scores) != len(smiles):
            raise ValueError("Creator molecule/reward pairing mismatch")
        score = scores[-1] if scores[-1] is not None else 0.0
    if not _finite(score):
        raise ValueError("Non-finite creator molecular reward")
    return smiles[-1], float(score)


def report(
    rows: list[dict],
    *,
    expected_draws: dict[str, int],
    source_root: Path,
    k: int,
    similarity_threshold: float,
    fingerprint: str,
) -> dict:
    """Compute declared library metrics only for complete homogeneous multi-draw groups.

    expected_draws is an explicit saved-run scope, not an inferred published sample
    count. This report never certifies reproduction of the paper's sampling recipe.
    """
    if type(k) is not int or k < 1 or not 0 <= similarity_threshold <= 1:
        raise ValueError("Positive k and similarity threshold in [0,1] required")
    if not expected_draws or any(not key or type(n) is not int or n < 2 for key, n in expected_draws.items()):
        raise ValueError("Explicit multi-draw prompt scope required; singleton data is not grouped sampling")
    functions = creator_functions(source_root)
    functions["fp_name_to_fn"](fingerprint)
    groups = {key: [] for key in expected_draws}
    for row in rows:
        prompt_id = row["verifier_metadata"]["metadata"]["prompt_id"]
        if prompt_id not in groups:
            raise ValueError("Saved output outside declared prompt scope")
        groups[prompt_id].append(row)
    results = []
    for prompt_id, group in groups.items():
        result = {
            "prompt_id": prompt_id,
            "expected_draws": expected_draws[prompt_id],
            "observed_draws": len(group),
            "raw_scores": [row.get("reward") for row in group],
            "draws": [
                {
                    "draw_id": row.get("_ng_rollout_index"),
                    "task_index": row.get("_ng_task_index"),
                    "raw_reward": row.get("reward"),
                    "mask_sample": row.get("mask_sample"),
                    "failure_kind": row.get("failure_kind"),
                    "failure_reason": row.get("failure_reason"),
                    "response_status": row.get("response", {}).get("status"),
                    "incomplete_details": row.get("response", {}).get("incomplete_details"),
                }
                for row in group
            ],
            "top_k": None,
            "diversity_aware_top_k": None,
        }
        if len(group) != expected_draws[prompt_id]:
            result.update(status="unresolved", reason="insufficient_or_excess_draws")
            results.append(result)
            continue
        ids = [row.get("_ng_rollout_index") for row in group]
        if any(type(identity) is not int for identity in ids) or len(set(ids)) != len(ids):
            raise ValueError("Missing or duplicate draw identity; regrades/retries are not independent draws")
        group = sorted(group, key=lambda row: row["_ng_rollout_index"])
        if any(
            not row["response"].get("model")
            or not row.get("diagnostics", {}).get("creator_response", {}).get("config")
            or not row.get("diagnostics", {}).get("creator_response", {}).get("oracle_variant")
            for row in group
        ):
            raise ValueError("Missing model/oracle protocol identity")
        bindings = {
            json.dumps(
                {
                    "request": row["responses_create_params"],
                    "metadata": row["verifier_metadata"],
                    "model": row["response"].get("model"),
                    "oracle_config": row.get("diagnostics", {}).get("creator_response", {}).get("config"),
                    "oracle_variant": row.get("diagnostics", {}).get("creator_response", {}).get("oracle_variant"),
                },
                sort_keys=True,
            )
            for row in group
        }
        if len(bindings) != 1:
            raise ValueError("Mixed prompt/model/generation/oracle protocol within one group")
        if any(
            row.get("mask_sample")
            or not _finite(row.get("reward"))
            or row["response"].get("status") not in {"completed", "incomplete"}
            or row["response"].get("error")
            or (
                row["response"].get("status") == "incomplete"
                and (row["response"].get("incomplete_details") or {}).get("reason") != "max_output_tokens"
            )
            for row in group
        ):
            result.update(status="unresolved", reason="unusable_scoring_draw")
            results.append(result)
            continue
        try:
            candidates = [_candidate(row) for row in group]
            molecules, scores = map(list, zip(*candidates))
            top = functions["top_k"](molecules, scores, k, canonicalize=True)
            diverse = functions["diversity_aware_top_k"](
                molecules, scores, k, similarity_threshold, fingerprint_name=fingerprint
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            result.update(status="unresolved", reason="creator_metric_or_receipt_error", error=str(exc))
        else:
            result.update(
                status="scored_creator_library_protocol",
                top_k=float(top),
                diversity_aware_top_k=float(diverse),
                selected_molecular_scores=scores,
                empty_molecule_draws=sum(not smi for smi in molecules),
            )
        results.append(result)
    return {
        "protocol": "molrgen-creator-library-grouped",
        "creator_commit": SOURCE_COMMIT,
        "source_sha256": SOURCE_HASHES,
        "extraction_source_sha256": LOADER_SHA256,
        "k": k,
        "similarity_threshold": similarity_threshold,
        "fingerprint": fingerprint,
        "groups": results,
        "expected_groups": len(groups),
        "scored_groups": sum(row["status"] == "scored_creator_library_protocol" for row in results),
        "expected_draws": sum(expected_draws.values()),
        "observed_draws": len(rows),
        "runtime_versions": {name: version(name) for name in ("rdkit", "numpy", "scipy")},
        "published_sampling_reproduced": False,
        "limitation": "Explicit library metrics only; notebook sampling/dedup/fingerprint recipe differs.",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scope = json.loads(args.scope.read_text())
    value = report(
        [json.loads(line) for line in args.rows.read_text().splitlines() if line], source_root=args.source, **scope
    )
    value["saved_rows_sha256"] = hashlib.sha256(args.rows.read_bytes()).hexdigest()
    value["scope_sha256"] = hashlib.sha256(args.scope.read_bytes()).hexdigest()
    with args.output.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
