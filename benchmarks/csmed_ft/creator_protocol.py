# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Creator chunk-label semantics; deliberately separate from the whole-paper variant.

This is a parser/merge contract, not a canonical generation implementation.
"""

from dataclasses import dataclass


PROTOCOL = "csmed-ft-creator-chunk-labels-81cd354"
CREATOR_COMMIT = "81cd35498bdea8981e5310842c32398625a10e95"
SOURCE_SHA256 = {
    "full_text_prompting.py": "516f0fc1a11bbe96664199a2d235e191ee3dbfbc4ae033d05378b82c111c30a0",
    "analyse_results.py": "065127448ce41ed0e6caf8aa8ba24c4e1bb7ed43b6ad8e0db28741b97e896096",
}


@dataclass(frozen=True)
class ChunkPrediction:
    uuid: int
    prediction: int
    gold_label: int


@dataclass(frozen=True)
class PaperPrediction:
    uuid: int
    prediction: int
    gold_label: int
    chunk_count: int
    invalid_chunk_count: int
    conflicting_gold_labels: bool


def parse_label(answer: str) -> int:
    """Mirror creator lower-only parsing, including rejection of whitespace."""
    if answer.lower() in {"included", "include", "yes"}:
        return 1
    if answer.lower() in {"excluded", "exclude", "no"}:
        return 0
    return -1


def merge_chunks(chunks: list[ChunkPrediction]) -> list[PaperPrediction]:
    """Mirror creator integer-UUID grouping and first-gold merge without repair.

    All-invalid groups become included upstream. Additional diagnostics expose that
    behavior and inconsistent references without changing the raw prediction.
    """
    grouped: dict[int, list[ChunkPrediction]] = {}
    for chunk in chunks:
        if type(chunk.uuid) is not int:
            raise ValueError("Creator dataset UUID must be its integer row index")
        if type(chunk.prediction) is not int or chunk.prediction not in {-1, 0, 1}:
            raise ValueError("Chunk prediction must be -1, 0, or 1")
        if type(chunk.gold_label) is not int or chunk.gold_label not in {0, 1}:
            raise ValueError("Gold label must be 0 or 1")
        grouped.setdefault(chunk.uuid, []).append(chunk)
    return [
        PaperPrediction(
            uuid=uuid,
            prediction=0 if any(chunk.prediction == 0 for chunk in group) else 1,
            gold_label=group[0].gold_label,
            chunk_count=len(group),
            invalid_chunk_count=sum(chunk.prediction == -1 for chunk in group),
            conflicting_gold_labels=len({chunk.gold_label for chunk in group}) > 1,
        )
        for uuid, group in sorted(grouped.items())
    ]
