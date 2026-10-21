# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency closure for inspected stateless creator-scoring paths.

This is not arbitrary creator-code restart support. GenomeQA and MolJSON use
pinned, extracted Python functions with stdlib-only scoring; all references are
in the immutable materialized inputs. QCBench binds its complete inspected
stdlib-only scoring module by exact hash. Other branches need their own inspected
closure. This binding neither replays an unknown request nor restores a tool.
"""

import hashlib
import json
import subprocess
from pathlib import Path

from omegaconf import OmegaConf

from nemo_gym.config_types import ConfigError
from nemo_gym.scicode_restart import _hash_file, inspect_component_runtime


_SOURCE_FILES = {
    "qcbench": "eval.py",
    "genomeqa": "run.py",
    "moljson": "evaluation_scripts/utils/answer_checking.py",
}


def exact_creator_graphs(config: dict) -> list[dict]:
    roles = {
        ("responses_api_models", "openai_model"): "model",
        ("responses_api_agents", "simple_agent"): "agent",
        ("resources_servers", "campaign_creator_scoring"): "resource",
        ("environment_servers", "legacy_agent"): "relay",
    }
    found = {role: {} for role in roles.values()}
    for name, value in config.items():
        if not isinstance(value, dict):
            continue
        native_kinds = {pair[0] for pair in roles}
        if any(
            key.endswith(("_servers", "_agents", "_models", "_providers")) and key not in native_kinds for key in value
        ):
            raise ConfigError("Creator restart unknown execution component kind")
        for kind in native_kinds.intersection(value):
            if not isinstance(value[kind], dict):
                raise ConfigError("Creator restart malformed component mapping")
            for implementation, settings in value[kind].items():
                role = roles.get((kind, implementation))
                if role is None or name in found[role] or not isinstance(settings, dict):
                    raise ConfigError("Creator restart requires its exact inspected stateless graph")
                if settings.get("entrypoint", "app.py") != "app.py" or any(
                    settings.get(key) for key in ("skills", "mcp_servers", "tools")
                ):
                    raise ConfigError("Creator restart external execution overrides unsupported")
                found[role][name] = {"name": name, "component": f"{kind}/{implementation}", "settings": settings}
    if len(found["model"]) != 1 or not found["agent"] or not found["resource"] or not found["relay"]:
        raise ConfigError("Creator restart requires one shared model and complete inspected chains")
    all_names = [name for group in found.values() for name in group]
    if len(set(all_names)) != len(all_names):
        raise ConfigError("Creator restart requires distinct component names")

    def target(owner, field, role):
        reference = owner["settings"].get(field)
        if not isinstance(reference, dict) or reference.get("name") not in found[role]:
            raise ConfigError("Creator restart role points outside inspected graph")
        selected = found[role][reference["name"]]
        expected_type = selected["component"].split("/")[0]
        if reference.get("type", expected_type) != expected_type:
            raise ConfigError("Creator restart role reference has wrong type")
        return selected

    graphs = []
    for relay in found["relay"].values():
        agent = target(relay, "agent_server", "agent")
        resource = target(agent, "resources_server", "resource")
        model = target(agent, "model_server", "model")
        if agent["settings"].get("max_steps") != 1:
            raise ConfigError("Creator restart requires one tool-free generation step")
        graphs.append({"model": model, "agent": agent, "resource": resource, "relay": relay})
    for role in ("agent", "resource"):
        names = [graph[role]["name"] for graph in graphs]
        if len(names) != len(set(names)) or set(names) != set(found[role]):
            raise ConfigError("Creator restart requires unique fully used agent/scorer chains")
    return graphs


def exact_creator_graph(config: dict) -> dict:
    """Retain the single-route contract for callers that require exactly one chain."""
    graphs = exact_creator_graphs(config)
    if len(graphs) != 1:
        raise ConfigError("Creator restart requires exactly one inspected chain here")
    return graphs[0]


def creator_graph_for_row(graphs: list[dict], row: dict) -> dict:
    """Match the same materialized routing fields the native collector dispatches."""
    agent = row.get("agent_ref")
    relay = row.get("_ng_environment_server")
    candidates = graphs
    if agent is not None:
        if not isinstance(agent, dict):
            raise ConfigError("Creator checkpoint malformed row agent reference")
        candidates = [g for g in candidates if g["agent"]["name"] == agent.get("name")]
    if relay is not None:
        candidates = [g for g in candidates if g["relay"]["name"] == relay]
    if len(candidates) != 1:
        raise ConfigError("Creator checkpoint row has ambiguous or cross-wired route")
    graph = candidates[0]
    if len(graphs) > 1 and row.get("task_source") != graph["resource"]["name"]:
        raise ConfigError("Creator checkpoint task source differs from selected scorer")
    return graph


def creator_source_binding(root: Path, resource: dict) -> dict:
    from benchmarks.scientific_agent_campaign.deterministic import PINS

    kind = resource.get("benchmark")
    if kind not in _SOURCE_FILES:
        raise ConfigError("Creator restart scorer branch has no inspected dependency closure")
    location = resource.get("upstream_repo")
    if not isinstance(location, str) or not location:
        raise ConfigError("Creator restart requires explicit creator source")
    repo = Path(location)
    repo = repo if repo.is_absolute() else root / repo
    # Do not resolve away a final/ancestor symlink before inspecting source.
    if any(path.is_symlink() for path in [repo, *repo.parents]):
        raise ConfigError("Creator restart source symlinks unsupported")
    repo = repo.resolve()
    source = repo / _SOURCE_FILES[kind]
    if any(path.is_symlink() for path in [source, *source.parents]):
        raise ConfigError("Creator restart scorer symlinks unsupported")
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, timeout=10
        ).strip()
        if revision != PINS[kind]:
            raise ConfigError("Creator restart source revision differs from maintained pin")
        subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--"], check=True, timeout=10)
        original = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{revision}:{_SOURCE_FILES[kind]}"], timeout=10
        )
        actual = _hash_file(source)
        if actual != hashlib.sha256(original).hexdigest():
            raise ConfigError("Creator restart scorer bytes differ from pinned Git object")
        if kind == "qcbench":
            # This exact complete module is imported, not AST-extracted. Its only
            # import-time effect is Decimal precision=50; file I/O is main-guarded.
            if actual != "5d22c986ac0c08c0ba3f6ace7122d3d2e845eca6ce01e6b16a4a0d3cf1fe5b10":
                raise ConfigError("QCBench scorer differs from inspected stdlib-only module")
        maintained = {}
        # Worker imports deterministic -> prepare and recovered_releases. Binding
        # the whole small maintained package also catches future helper changes.
        for path in sorted((root / "benchmarks/scientific_agent_campaign").glob("*.py")):
            maintained[str(path.relative_to(root))] = _hash_file(path)
        required = {"deterministic.py", "prepare.py", "recovered_releases.py"}
        if not required.issubset({Path(path).name for path in maintained}):
            raise ConfigError("Creator restart maintained adapter source missing")
        return {
            "protocol": "stateless-stdlib-qcbench-module-v1"
            if kind == "qcbench"
            else "stateless-stdlib-ast-creator-v1",
            "benchmark": kind,
            "creator": {"root": str(repo), "revision": revision, "source": _SOURCE_FILES[kind], "sha256": actual},
            "maintained_source": maintained,
        }
    except (OSError, subprocess.SubprocessError) as error:
        raise ConfigError("Creator restart source unavailable, dirty, or unreadable") from error


def qualify_creator_dependencies(root: Path, config: dict, materialized_path: Path) -> dict:
    from nemo_gym.cli.env import _resolve_server_dir
    from nemo_gym.cli.setup_command import get_venv_path

    root = root.resolve()
    graphs = exact_creator_graphs(config)
    closures = {g["agent"]["name"]: creator_source_binding(root, g["resource"]["settings"]) for g in graphs}
    try:
        rows = [json.loads(line) for line in materialized_path.read_text().splitlines() if line.strip()]
        if not rows:
            raise ConfigError("Creator restart requires materialized tasks")
        used = set()
        for row in rows:
            graph = creator_graph_for_row(graphs, row)
            used.add(graph["agent"]["name"])
            closure = closures[graph["agent"]["name"]]
            # Legacy native rows carry the actual agent input at top level.
            request = row.get("responses_create_params")
            metadata = row.get("verifier_metadata")
            if not isinstance(request, dict) or request.get("tools") or not isinstance(metadata, dict):
                raise ConfigError("Creator restart requires tool-free materialized scorer inputs")
            if not isinstance(row.get("uuid"), str) or not row["uuid"]:
                raise ConfigError("Creator restart requires creator task identity")
            required = {"answer", "options_keys"} if closure["benchmark"] == "genomeqa" else {"answer"}
            if not required.issubset(metadata):
                raise ConfigError("Creator restart missing scorer reference inputs")
        if used != set(closures):
            raise ConfigError("Creator checkpoint contains an unused configured scientific chain")
        runtime = {}
        servers = {server["name"]: server for graph in graphs for server in graph.values()}
        names = ("SimpleAgentConfig", "SimpleModelServerConfig", "LegacyAgentEnvironmentServerConfig", "Config")
        for name, server in servers.items():
            component = root / server["component"]
            if _resolve_server_dir(Path(server["component"])).resolve() != component.resolve():
                raise ConfigError("Creator restart native component resolution is shadowed")
            python = get_venv_path(component, OmegaConf.create(config)) / "bin/python"
            receipt = inspect_component_runtime(root, component, python, config_class_names=names)
            allowed = set(receipt["config_fields"]) | {
                "datasets",
                "allowed_agents",
                "verified",
                "description",
                "value",
            }
            if not receipt["config_fields"] or set(server["settings"]) - allowed:
                raise ConfigError("Creator restart contains unrecognized component settings")
            runtime[name] = receipt
        if len(graphs) == 1:
            closure = closures[graphs[0]["agent"]["name"]]
            runtime = {role: runtime[server["name"]] for role, server in graphs[0].items()}
        else:
            closure = {"protocol": "mixed-inspected-stateless-creators-v1", "chains": closures}
        closure["component_runtimes"] = runtime
        closure["materialized_inputs_sha256"] = _hash_file(materialized_path)
        closure["tasks"] = len(rows)
        return closure
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ConfigError("Creator restart input/runtime dependency inspection failed") from error
