# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Binary screening decisions; macro metrics must be aggregated separately."""


def grade(text: str, *, answer: str, **metadata: object) -> dict:
    if metadata.get("input_valid") is False:
        raise ValueError("Missing full text or eligibility criteria; invalid benchmark input")
    predicted = text.strip().lower()
    valid = predicted in {"included", "excluded"}
    return {
        "reward": float(valid and predicted == answer),
        "diagnostics": {"predicted": predicted if valid else None, "gold": answer, "valid_label": valid},
    }


def macro_metrics(gold: list[str], predictions: list[str]) -> dict:
    if not gold or len(gold) != len(predictions):
        raise ValueError("Equal nonempty lists required")
    per_class = {}
    for label in ["included", "excluded"]:
        tp = sum(g == label and p == label for g, p in zip(gold, predictions))
        fp = sum(g != label and p == label for g, p in zip(gold, predictions))
        fn = sum(g == label and p != label for g, p in zip(gold, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }
    return {
        "per_class": per_class,
        **{
            "macro_" + metric: sum(c[metric] for c in per_class.values()) / 2
            for metric in ["precision", "recall", "f1"]
        },
        "invalid_predictions": sum(p not in {"included", "excluded"} for p in predictions),
    }
