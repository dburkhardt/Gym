# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve creator scores while qualifying their scientific interpretation."""

import hashlib
from pathlib import Path


DESEQ_SCORER_SHA256 = "e8daebb41e8578a5666fba5496d14c3575c24d654a2fd51f9da7f76be9f12b64"


def result_fields(task_id: str, reward: float, creator_root: Path) -> dict:
    if task_id != "deseq":
        return {}
    source = creator_root / "tasksets/bioagent/bioagent_taskset/scoring.py"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    known = digest == DESEQ_SCORER_SHA256
    reason = (
        "Creator reward checks five overlapping gene IDs; missing or invalid numeric columns can pass. "
        "Differential-expression accuracy requires independent review of outputs and analysis."
        if known
        else "DESeq scorer differs from the audited source; scientific measurement review is unresolved."
    )
    return {
        "mask_sample": True,
        "failure_kind": "bioagent:scientific_measurement_unresolved",
        "failure_reason": reason,
        "metadata": {
            "verifier_result": {"rewards": {"reward": reward}},
            "measurement_review": {
                "version": "bioagent-deseq-gene-overlap-v1",
                "metrics": {"reward": {"status": "unresolved", "reason": reason}},
                "evidence": {"creator_scoring_sha256": digest, "audited_source_matches": known},
                "creator_metric": "five_gene_id_overlap" if known else "unreviewed_creator_reward",
                "raw_reward_preserved": True,
            },
        },
    }
