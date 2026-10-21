# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compose inspected SciCode and stateless creator closures; never restore tools."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from nemo_gym.config_types import ConfigError
from nemo_gym.creator_restart import exact_creator_graphs, qualify_creator_dependencies
from nemo_gym.scicode_restart import exact_scicode_graph, qualify_scicode_dependencies


KINDS = {"responses_api_models", "responses_api_agents", "resources_servers", "environment_servers"}


def partition_routes(config: dict, rows: list[dict]):
    """Require every configured component and materialized row to belong to a closure."""
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ConfigError("Mixed restart materialized rows must be objects")
    servers, globals_ = {}, {}
    for name, value in config.items():
        if not isinstance(value, dict):
            globals_[name] = value
            continue
        kinds = KINDS.intersection(value)
        if any(k.endswith(("_servers", "_agents", "_models", "_providers")) and k not in KINDS for k in value):
            raise ConfigError("Mixed restart unknown execution component kind")
        if not kinds:
            globals_[name] = value
            continue
        if len(kinds) != 1:
            raise ConfigError("Mixed restart component must have one execution role")
        kind = next(iter(kinds))
        entries = value[kind]
        if not isinstance(entries, dict) or len(entries) != 1:
            raise ConfigError("Mixed restart malformed component")
        implementation, settings = next(iter(entries.items()))
        if not isinstance(settings, dict):
            raise ConfigError("Mixed restart malformed component settings")
        servers[name] = (kind, implementation, settings)

    def unique(kind, implementation):
        found = [n for n, (k, i, _) in servers.items() if (k, i) == (kind, implementation)]
        if len(found) != 1:
            raise ConfigError("Mixed restart requires one SciCode chain and shared model")
        return found[0]

    model = unique("responses_api_models", "openai_model")
    agent = unique("responses_api_agents", "scicode_agent")
    resource = unique("resources_servers", "scicode")
    relays = [
        n
        for n, (k, i, v) in servers.items()
        if (k, i) == ("environment_servers", "legacy_agent")
        and isinstance(v.get("agent_server"), dict)
        and v["agent_server"].get("name") == agent
    ]
    if len(relays) != 1:
        raise ConfigError("Mixed restart SciCode relay must be unique")
    selected = {model, agent, resource, relays[0]}
    sci_config = globals_ | {n: config[n] for n in selected}
    creator_config = globals_ | {n: config[n] for n in servers if n not in selected or n == model}
    sci = exact_scicode_graph(sci_config)
    creators = exact_creator_graphs(creator_config)
    graphs = [sci, *creators]
    used, creator_rows, routes = set(), [], []
    for row in rows:
        ref = row.get("agent_ref")
        if not isinstance(ref, dict) or ref.get("type", "responses_api_agents") != "responses_api_agents":
            raise ConfigError("Mixed restart needs explicit materialized agent route")
        matches = [g for g in graphs if g["agent"]["name"] == ref.get("name")]
        if len(matches) != 1:
            raise ConfigError("Mixed restart row agent is unclaimed")
        graph = matches[0]
        expected_source = agent if graph is sci else graph["resource"]["name"]
        if row.get("task_source") != expected_source:
            raise ConfigError("Mixed restart row task source disagrees with route")
        relay = row.get("_ng_environment_server")
        if relay is not None and relay != graph["relay"]["name"]:
            raise ConfigError("Mixed restart row relay disagrees with route")
        used.add(graph["agent"]["name"])
        routes.append({"agent": ref["name"], "source": expected_source, "relay": graph["relay"]["name"]})
        if graph is not sci:
            creator_rows.append(row)
    if used != {g["agent"]["name"] for g in graphs}:
        raise ConfigError("Mixed restart contains unused configured chains")
    return sci_config, creator_config, creator_rows, routes


def qualify_mixed_scientific_dependencies(root: Path, config: dict, materialized_path: Path) -> dict:
    try:
        raw = materialized_path.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        sci_config, creator_config, creator_rows, routes = partition_routes(config, rows)
        scientific = qualify_scicode_dependencies(root, sci_config)
        # Existing creator admission reads a materialized file. Its receipt stores
        # only its content digest, never this ephemeral path.
        with TemporaryDirectory(prefix="gym-mixed-closure-") as temporary:
            subset = Path(temporary) / "creator-inputs.jsonl"
            subset.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in creator_rows))
            creators = qualify_creator_dependencies(root, creator_config, subset)
        return {
            "protocol": "mixed-scicode-inspected-creators-v1",
            "scicode": scientific,
            "creators": creators,
            "materialized_inputs_sha256": hashlib.sha256(raw).hexdigest(),
            "routes": routes,
            "tasks": len(rows),
        }
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ConfigError("Mixed restart input/dependency inspection failed") from error
