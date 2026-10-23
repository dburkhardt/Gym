# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit, source-bound repair of the molecular evaluator public-data mount."""

import hashlib


MOLECULAR_TASK = "s42256-024-00916-5"
CREATOR_COMPOSE_SHA256 = "4380d6ee13b61b6774fb42b099eb272b7ff84e1e76e13a7057f3e12c23803700"


def restore_public_input_mount(source: str, *, task_id: str) -> str:
    """Repair a copied pinned task; reject other tasks, drift, and repeated application.

    These public inputs are already mounted in the solver. Evaluator-private
    references and all scoring code remain unchanged.
    """
    if task_id != MOLECULAR_TASK:
        raise ValueError("Public-input repair is qualified only for the pinned molecular task")
    if hashlib.sha256(source.encode()).hexdigest() != CREATOR_COMPOSE_SHA256:
        raise ValueError("Unexpected creator Compose source; refuse unqualified mount repair")
    old = "      - naturebench-workspace:/workspace:ro\n"
    if source.count(old) != 1:
        raise ValueError("Expected exactly one evaluator workspace mount")
    return source.replace(old, old + "      - ./input/data:/problem/data:ro\n")
