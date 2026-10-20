# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay creator RPF once real GT and matched answer graphs exist.

This does not manufacture GT or infer matches. Point upstream_root to the pinned
SUPERChem checkout. Historical demo graphs must not be promoted to full release.
"""

import importlib.util
from pathlib import Path


def score_rpf(reference: dict, matched_answer: dict, *, upstream_root: str) -> dict:
    path = Path(upstream_root) / "superchem" / "metrics.py"
    spec = importlib.util.spec_from_file_location("creator_superchem_metrics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load creator metric from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.graph_metrics(reference, matched_answer)
