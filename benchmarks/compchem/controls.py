# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build non-LLM native Harbor scorer controls from a saved job configuration."""

from copy import deepcopy
from typing import Any


def control_config(source: dict[str, Any], agent: str) -> dict[str, Any]:
    """Replace Harbor's plural agent collection; never retain the saved solver route."""
    if agent not in ("nop", "oracle"):
        raise ValueError("Only native nop and oracle scorer controls are permitted")
    result = deepcopy(source)
    result.pop("agent", None)
    result["agents"] = [{"name": agent}]
    result["source_jobs"] = []
    return result


def validate_control(config: Any, agent: str) -> None:
    """Check the resolved native JobConfig before any control can execute."""
    agents = config.model_dump(mode="json")["agents"]
    if len(agents) != 1 or agents[0].get("name") != agent:
        raise ValueError("Resolved control selected an unexpected native agent")
    selected = agents[0]
    if selected.get("import_path") or selected.get("model_name") or selected.get("kwargs"):
        raise ValueError("A scorer control must not retain model or solver configuration")
