# SPDX-License-Identifier: Apache-2.0
"""Optional feasibility check for a declared static Ray node pool."""

import math
from collections.abc import Mapping


class StaticRayResourceError(ValueError):
    """No currently advertised static node can ever schedule this trial."""


def static_resource_feasibility(nodes: list[dict], options: Mapping) -> dict:
    """Use individual ALIVE nodes' totals, never free capacity or cluster sums.

    This is a static-placement contract, not ownership evidence. Busy feasible
    nodes remain eligible for Ray's normal queue. Autoscaling pools must not opt
    in unless the caller intentionally constrains placement to existing nodes.
    """
    requested = {"CPU": options.get("num_cpus", 1)}
    if "memory" in options:
        requested["memory"] = options["memory"]
    for key, value in requested.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid requested Ray resource {key}")
    if not isinstance(nodes, list):
        raise ValueError("Ray node inventory must be a list")
    alive = []
    for node in nodes:
        if not isinstance(node, Mapping) or not isinstance(node.get("Alive"), bool):
            raise ValueError("Invalid Ray node liveness metadata")
        if not node["Alive"]:
            continue
        resources = node.get("Resources")
        if not isinstance(resources, Mapping):
            raise ValueError("Missing Ray node total resources")
        totals = {key: resources.get(key, 0) for key in requested}
        for key, value in totals.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid advertised Ray resource {key}")
        alive.append(
            {
                "node_id": node.get("NodeID"),
                "total": totals,
                "feasible": all(totals[key] >= value for key, value in requested.items()),
            }
        )
    return {
        "version": 1,
        "policy": "declared_static_node_pool",
        "requested": requested,
        "alive_nodes": alive,
        "feasible": any(node["feasible"] for node in alive),
    }
