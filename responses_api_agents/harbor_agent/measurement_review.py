# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply explicit benchmark measurement reviews without changing native rewards."""

import copy
import hashlib
import importlib
from pathlib import Path


def apply_measurement_review(
    trial: dict, *, trial_dir: Path, instance_id: str, import_path: str | None
) -> tuple[dict, str | None, str | None]:
    """A configured review annotates immutable native values; it never repairs a scorer."""
    if import_path is None:
        return trial, None, None
    try:
        module_name, function_name = import_path.rsplit(":", 1)
        module = importlib.import_module(module_name)
        review = getattr(module, function_name)(
            trial_dir=trial_dir, instance_id=instance_id, trial_result=copy.deepcopy(trial)
        )
        if not isinstance(review, dict) or not isinstance(review.get("metrics"), dict) or not review.get("version"):
            raise ValueError("Malformed scientific measurement review")
        native = (trial.get("verifier_result") or {}).get("rewards") or {}
        for name, metric in review["metrics"].items():
            if (
                name not in native
                or not isinstance(metric, dict)
                or metric.get("status") not in ("valid", "invalid", "unresolved")
                or not metric.get("reason")
            ):
                raise ValueError("Malformed metric validity disposition")
        review = {
            **review,
            "implementation": {
                "import_path": import_path,
                "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            },
        }
        revised = {**trial, "measurement_review": review}
        excluded = [
            f"{name}: {metric['reason']}" for name, metric in review["metrics"].items() if metric["status"] != "valid"
        ]
        return revised, "harbor:scientific_measurement_invalid" if excluded else None, "; ".join(excluded) or None
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
        return (
            {**trial, "measurement_review": {"import_path": import_path, "error": reason}},
            "harbor:measurement_review_error",
            reason,
        )
