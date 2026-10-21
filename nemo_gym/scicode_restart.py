# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verified SciCode dependencies for existing batch execution binding, not a registry."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

from nemo_gym.config_types import ConfigError


def _hash_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ConfigError(f"SciCode restart requires a regular dependency file: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_image(image: str) -> dict:
    """Read local daemon metadata only; never pull or create a container."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image], capture_output=True, text=True, timeout=20, check=True
        )
        records = json.loads(result.stdout)
        if len(records) != 1:
            raise ValueError("Expected one image")
        return records[0]
    except (OSError, subprocess.SubprocessError, ValueError, TypeError) as error:
        raise ConfigError("SciCode restart image unavailable from the selected local Docker daemon") from error


def scicode_dependency_binding(root: Path, agent: dict, resource: dict, *, image_inspector=inspect_image) -> dict:
    """Bind existing prompt/H5/image configuration plus maintained executable source.

    Caller must separately enforce exactly the inspected server graph. This does not
    restore a sandbox, qualify scientific scores, or authorize a prior failed attempt.
    """
    root = root.resolve()
    image = resource.get("sandbox_image")
    if not isinstance(image, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ConfigError("SciCode restart requires its immutable sandbox image ID")
    metadata = image_inspector(image)
    if not isinstance(metadata, dict):
        raise ConfigError("SciCode restart image metadata must be an object")
    if metadata.get("Id") != image or metadata.get("Os") != "linux":
        raise ConfigError("SciCode restart image identity/platform mismatch")
    if metadata.get("Architecture") not in {"amd64", "arm64"}:
        raise ConfigError("SciCode restart image architecture unavailable or unsupported")
    rootfs = metadata.get("RootFS")
    layers = rootfs.get("Layers") if isinstance(rootfs, dict) else None
    if (
        not isinstance(layers, list)
        or not layers
        or any(not isinstance(layer, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", layer) for layer in layers)
    ):
        raise ConfigError("SciCode restart image layer identity missing")
    prompt = Path(agent.get("prompt_fpath") or "")
    prompt = prompt if prompt.is_absolute() else root / prompt
    if not prompt.resolve().is_relative_to(root / "benchmarks/scicode/prompts"):
        raise ConfigError("SciCode restart external prompt dependency is unsupported")
    data = Path(resource.get("test_data_fpath") or "")
    data = data if data.is_absolute() else root / data
    expected = resource.get("test_data_sha256")
    actual = _hash_file(data)
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected) or actual != expected:
        raise ConfigError("SciCode restart H5 checksum mismatch")
    files = {str(prompt.relative_to(root)): _hash_file(prompt)}
    for relative in (
        "responses_api_agents/scicode_agent",
        "resources_servers/scicode",
        "environment_servers/legacy_agent",
    ):
        directory = root / relative
        if not (directory / "app.py").is_file():
            raise ConfigError("SciCode restart maintained source missing")
        for path in sorted(directory.rglob("*")):
            parts = path.relative_to(directory).parts
            if any(part in {".venv", "__pycache__", "data", "tests", ".git"} for part in parts):
                continue
            if path.is_symlink():
                raise ConfigError("SciCode restart source symlinks are unsupported")
            if not path.is_file() and not path.is_dir():
                raise ConfigError("SciCode restart special source files are unsupported")
            if path.is_file():
                files[str(path.relative_to(root))] = _hash_file(path)
    return {
        "source_files": files,
        "test_data": {"path": str(data.resolve()), "sha256": actual},
        "image": {"id": image, "os": metadata["Os"], "architecture": metadata["Architecture"], "layers": layers},
    }


def inspect_component_runtime(root: Path, component: Path, python: Path) -> dict:
    """Import actual maintained component in its native interpreter; no server construction."""
    import os

    if not python.is_file():
        raise ConfigError("SciCode restart component interpreter is unavailable")
    script = """
import hashlib, importlib.metadata, json, pathlib, sys
from urllib.parse import urlparse, unquote
import app, nemo_gym
editable = []
for distribution in importlib.metadata.distributions():
    raw = distribution.read_text('direct_url.json')
    if raw:
        direct = json.loads(raw)
        if direct.get('dir_info', {}).get('editable'):
            url = urlparse(direct.get('url', ''))
            editable.append({'name': distribution.metadata['Name'], 'path': unquote(url.path) if url.scheme == 'file' else None})
modules = {}
for name, module in tuple(sys.modules.items()):
    filename = getattr(module, '__file__', None)
    if filename and (name == 'app' or name == 'step_utils' or name.startswith(('nemo_gym', 'scicode_integration', 'resources_servers.scicode'))):
        path = pathlib.Path(filename).resolve()
        if path.is_file():
            modules[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
print(json.dumps({'python': sys.version, 'packages': sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions() if d.metadata['Name']), 'modules': modules, 'editable': editable, 'config_fields': sorted({field for name, value in vars(app).items() if name in {'ScicodeAgentConfig', 'ScicodeResourcesServerConfig', 'LegacyAgentEnvironmentServerConfig', 'SimpleModelServerConfig'} for field in value.model_fields}), 'h5_default': getattr(app, 'ScicodeResourcesServerConfig', None).model_fields['test_data_sha256'].default if hasattr(app, 'ScicodeResourcesServerConfig') else None}))
"""
    env = dict(os.environ)
    # Match run_command: retain preexisting precedence instead of hiding shadows.
    paths = [str(component.resolve()), str(root.resolve())]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    roots = [entry for entry in env.get("NEMO_GYM_EXTRA_ROOTS", "").split(os.pathsep) if entry]
    if str(root.resolve()) not in roots:
        roots.append(str(root.resolve()))
    env["NEMO_GYM_EXTRA_ROOTS"] = os.pathsep.join(roots)
    try:
        result = subprocess.run(
            [str(python), "-c", script], cwd=component, env=env, capture_output=True, text=True, timeout=30, check=True
        )
        receipt = json.loads(result.stdout)
        for editable in receipt["editable"]:
            # Source precedence is explicit for Gym; other editable dependencies lack
            # a verified closure and must not be silently treated as wheel versions.
            if editable["name"].lower().replace("_", "-") != "nemo-gym":
                raise ValueError("Unsupported external editable dependency")
        modules = receipt["modules"]
        if Path(modules["app"]["path"]) != (component / "app.py").resolve():
            raise ValueError("Wrong component import")
        if Path(modules["nemo_gym"]["path"]) != (root / "nemo_gym/__init__.py").resolve():
            raise ValueError("Wrong Gym import")
        for value in modules.values():
            path = Path(value["path"])
            if not path.is_relative_to(root.resolve()) or _hash_file(path) != value["sha256"]:
                raise ValueError("Stale or external maintained import")
        return receipt
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as error:
        raise ConfigError("SciCode restart component runtime/source inspection failed") from error


def exact_scicode_graph(config: dict) -> dict:
    """Recognize only the inspected model→SciCode agent→resource/relay graph."""
    roles = {
        ("responses_api_models", "openai_model"): "model",
        ("responses_api_agents", "scicode_agent"): "agent",
        ("resources_servers", "scicode"): "resource",
        ("environment_servers", "legacy_agent"): "relay",
    }
    found = {}
    for name, value in config.items():
        if not isinstance(value, dict):
            continue
        for kind in {item[0] for item in roles}.intersection(value):
            if not isinstance(value[kind], dict):
                raise ConfigError("SciCode restart server section must be an object")
            for implementation, settings in value[kind].items():
                role = roles.get((kind, implementation))
                if role is None or role in found or not isinstance(settings, dict):
                    raise ConfigError("SciCode restart requires its exact inspected server graph")
                if (
                    settings.get("entrypoint", "app.py") != "app.py"
                    or settings.get("skills")
                    or settings.get("mcp_servers")
                ):
                    raise ConfigError("SciCode restart external execution overrides are unsupported")
                found[role] = {"name": name, "component": f"{kind}/{implementation}", "settings": settings}
    if len({item["name"] for item in found.values()}) != len(found):
        raise ConfigError("SciCode restart requires distinct server names")
    if set(found) != {"model", "agent", "resource", "relay"}:
        raise ConfigError("SciCode restart requires all inspected server roles")
    for owner, field, target in (
        ("agent", "model_server", "model"),
        ("agent", "resources_server", "resource"),
        ("relay", "agent_server", "agent"),
    ):
        reference = found[owner]["settings"].get(field)
        expected_type = found[target]["component"].split("/")[0]
        if (
            not isinstance(reference, dict)
            or reference.get("name") != found[target]["name"]
            or reference.get("type", expected_type) != expected_type
        ):
            raise ConfigError("SciCode restart role reference points outside inspected graph")
    return found


def qualify_scicode_dependencies(root: Path, config: dict) -> dict:
    """Inspect the native component interpreters and existing scientific assets.

    No setup/install happens here. Missing environment or daemon access is an
    unsupported restart disposition, never permission to skip verification.
    """
    from omegaconf import OmegaConf

    from nemo_gym import _resolve_under_cwd_or_install
    from nemo_gym.cli.env import _resolve_server_dir
    from nemo_gym.cli.setup_command import get_venv_path

    graph = exact_scicode_graph(config)
    runtime = {}
    for role, server in graph.items():
        component = root / server["component"]
        if _resolve_server_dir(Path(server["component"])).resolve() != component.resolve():
            raise ConfigError("SciCode restart native component resolution is shadowed")
        try:
            python = get_venv_path(component, OmegaConf.create(config)) / "bin/python"
        except (KeyError, TypeError, ValueError) as error:
            raise ConfigError("SciCode restart native component environment configuration is unavailable") from error
        receipt = inspect_component_runtime(root, component, python)
        # These native discovery/dataset metadata keys are not consumed by app;
        # dataset contents are separately fixed in the materialized batch hash.
        allowed = set(receipt["config_fields"]) | {"datasets", "allowed_agents", "verified", "description", "value"}
        if not allowed or set(server["settings"]) - allowed:
            raise ConfigError("SciCode restart contains unrecognized component configuration overrides")
        runtime[role] = receipt
    resource = dict(graph["resource"]["settings"])
    if "test_data_sha256" not in resource:
        resource["test_data_sha256"] = runtime["resource"]["h5_default"]
    prompt = Path(graph["agent"]["settings"].get("prompt_fpath", ""))
    expected_prompt = prompt if prompt.is_absolute() else root / prompt
    if _resolve_under_cwd_or_install(prompt).resolve() != expected_prompt.resolve():
        raise ConfigError("SciCode restart native prompt resolution is shadowed")
    closure = scicode_dependency_binding(root, graph["agent"]["settings"], resource)
    closure["component_runtimes"] = runtime
    return closure
