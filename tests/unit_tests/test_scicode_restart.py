# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
from copy import deepcopy

import pytest

from nemo_gym.config_types import ConfigError
from nemo_gym.scicode_restart import scicode_dependency_binding


@pytest.fixture
def closure(tmp_path):
    for directory in (
        "responses_api_agents/scicode_agent",
        "resources_servers/scicode",
        "environment_servers/legacy_agent",
    ):
        path = tmp_path / directory
        path.mkdir(parents=True)
        (path / "app.py").write_text("# maintained fixture\n")
    prompt = tmp_path / "benchmarks/scicode/prompts/background.yaml"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("user: '{problem}'\n")
    data = tmp_path / "test_data.h5"
    data.write_bytes(b"fixed data fixture")
    image = "sha256:" + "a" * 64
    agent = {"prompt_fpath": str(prompt)}
    resource = {
        "sandbox_image": image,
        "test_data_fpath": str(data),
        "test_data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
    }
    metadata = {"Id": image, "Os": "linux", "Architecture": "amd64", "RootFS": {"Layers": ["sha256:" + "b" * 64]}}
    return tmp_path, agent, resource, metadata


def test_exact_dependency_closure_and_mutation(closure):
    root, agent, resource, metadata = closure
    first = scicode_dependency_binding(root, agent, resource, image_inspector=lambda _: metadata)
    assert first["image"]["id"] == resource["sandbox_image"]
    assert len(first["source_files"]) == 4
    prompt = root / "benchmarks/scicode/prompts/background.yaml"
    prompt.write_text("user: changed\n")
    assert scicode_dependency_binding(root, agent, resource, image_inspector=lambda _: metadata) != first
    dependency = root / "resources_servers/scicode/requirements.txt"
    dependency.write_text("changed-dependency==1\n")
    assert (
        str(dependency.relative_to(root))
        in scicode_dependency_binding(root, agent, resource, image_inspector=lambda _: metadata)["source_files"]
    )


@pytest.mark.parametrize(
    "change",
    [
        "mutable",
        "wrong_image",
        "missing_layers",
        "bad_platform",
        "missing_h5",
        "wrong_h5",
        "external_prompt",
        "source_symlink",
        "missing_source",
    ],
)
def test_missing_or_changed_dependencies_refused(closure, change):
    root, agent, resource, metadata = deepcopy(closure)
    if change == "mutable":
        resource["sandbox_image"] = "latest"
    elif change == "wrong_image":
        metadata["Id"] = "sha256:" + "c" * 64
    elif change == "missing_layers":
        metadata.pop("RootFS")
    elif change == "bad_platform":
        metadata["Os"] = "windows"
    elif change == "missing_h5":
        (root / "test_data.h5").unlink()
    elif change == "wrong_h5":
        (root / "test_data.h5").write_bytes(b"changed")
    elif change == "external_prompt":
        agent["prompt_fpath"] = "/external/prompt.yaml"
    elif change == "source_symlink":
        (root / "resources_servers/scicode/extra.py").symlink_to(root / "test_data.h5")
    else:
        (root / "resources_servers/scicode/app.py").unlink()
    with pytest.raises(ConfigError):
        scicode_dependency_binding(root, agent, resource, image_inspector=lambda _: metadata)


def test_actual_component_interpreter_probe(tmp_path):
    import sys
    from pathlib import Path

    from nemo_gym.scicode_restart import inspect_component_runtime

    (tmp_path / "nemo_gym").mkdir()
    (tmp_path / "nemo_gym/__init__.py").write_text("# fixture package\n")
    component = tmp_path / "component"
    component.mkdir()
    (component / "app.py").write_text("# fixture app\n")
    receipt = inspect_component_runtime(tmp_path, component, Path(sys.executable))
    assert receipt["modules"]["app"]["path"] == str(component / "app.py")
    assert receipt["packages"]
    with pytest.raises(ConfigError, match="unavailable"):
        inspect_component_runtime(tmp_path, component, tmp_path / "missing-python")


@pytest.mark.parametrize(
    "metadata", [None, [], {"Id": "sha256:" + "a" * 64, "Os": "linux", "Architecture": "amd64", "RootFS": []}]
)
def test_malformed_image_receipt(closure, metadata):
    root, agent, resource, _ = closure
    with pytest.raises(ConfigError):
        scicode_dependency_binding(root, agent, resource, image_inspector=lambda _: metadata)


def test_exact_graph_rejects_extra_or_redirected_roles():
    from nemo_gym.scicode_restart import exact_scicode_graph

    graph = {
        "policy": {"responses_api_models": {"openai_model": {}}},
        "solver": {
            "responses_api_agents": {
                "scicode_agent": {"model_server": {"name": "policy"}, "resources_server": {"name": "science"}}
            }
        },
        "science": {"resources_servers": {"scicode": {}}},
        "relay": {"environment_servers": {"legacy_agent": {"agent_server": {"name": "solver"}}}},
    }
    assert set(exact_scicode_graph(graph)) == {"model", "agent", "resource", "relay"}
    for change in ("redirect", "extra", "entrypoint", "missing"):
        bad = deepcopy(graph)
        if change == "redirect":
            bad["solver"]["responses_api_agents"]["scicode_agent"]["model_server"]["name"] = "external"
        elif change == "extra":
            bad["other"] = {"resources_servers": {"mcqa": {}}}
        elif change == "entrypoint":
            bad["science"]["resources_servers"]["scicode"]["entrypoint"] = "/external.py"
        else:
            del bad["relay"]
        with pytest.raises(ConfigError):
            exact_scicode_graph(bad)


@pytest.mark.parametrize("corruption", ["external_editable", "stale_app"])
def test_runtime_probe_rejects_unbound_imports(tmp_path, monkeypatch, corruption):
    import json
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    import nemo_gym.scicode_restart as module

    component = tmp_path / "component"
    component.mkdir()
    (component / "app.py").write_text("# app\n")
    (tmp_path / "nemo_gym").mkdir()
    (tmp_path / "nemo_gym/__init__.py").write_text("# gym\n")
    paths = {"app": component / "app.py", "nemo_gym": tmp_path / "nemo_gym/__init__.py"}
    receipt = {
        "modules": {
            name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in paths.items()
        },
        "editable": [],
    }
    if corruption == "external_editable":
        receipt["editable"] = [{"name": "external-science", "path": "/other/source"}]
    else:
        receipt["modules"]["app"]["path"] = "/old/source/app.py"
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=json.dumps(receipt)))
    with pytest.raises(ConfigError, match="runtime/source inspection failed"):
        module.inspect_component_runtime(tmp_path, component, Path(sys.executable))


def test_malformed_graph_section_rejected():
    from nemo_gym.scicode_restart import exact_scicode_graph

    with pytest.raises(ConfigError, match="section must be an object"):
        exact_scicode_graph({"model": {"responses_api_models": []}})


def test_native_qualification_binds_each_interpreter_and_rejects_unknown_override(closure, monkeypatch):
    import nemo_gym.cli.setup_command as setup
    import nemo_gym.scicode_restart as module

    root, agent, resource, metadata = closure
    graph = {
        "uv_venv_dir": str(root),
        "policy": {"responses_api_models": {"openai_model": {}}},
        "solver": {
            "responses_api_agents": {
                "scicode_agent": {**agent, "model_server": {"name": "policy"}, "resources_server": {"name": "science"}}
            }
        },
        "science": {"resources_servers": {"scicode": resource}},
        "relay": {"environment_servers": {"legacy_agent": {"agent_server": {"name": "solver"}}}},
    }
    for component in (
        "responses_api_agents/scicode_agent",
        "resources_servers/scicode",
        "environment_servers/legacy_agent",
        "responses_api_models/openai_model",
    ):
        directory = root / component
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "requirements.txt").write_text("# fixture\n")
    monkeypatch.chdir(root)
    monkeypatch.delenv("NEMO_GYM_EXTRA_ROOTS", raising=False)
    called = []
    fields = set(agent) | set(resource) | {"model_server", "resources_server", "agent_server"}

    def runtime(source, component, python):
        called.append((component, python))
        return {
            "config_fields": sorted(fields),
            "h5_default": resource["test_data_sha256"],
            "packages": [[component.name, "1"]],
        }

    monkeypatch.setattr(setup, "get_venv_path", lambda component, config: component / ".venv")
    monkeypatch.setattr(module, "inspect_component_runtime", runtime)
    original = module.scicode_dependency_binding
    monkeypatch.setattr(
        module,
        "scicode_dependency_binding",
        lambda root, agent, resource: original(root, agent, resource, image_inspector=lambda _: metadata),
    )
    result = module.qualify_scicode_dependencies(root, graph)
    assert len(called) == 4
    assert all(python == component / ".venv/bin/python" for component, python in called)
    assert set(result["component_runtimes"]) == {"model", "agent", "resource", "relay"}
    shadow = root / "shadow"
    shadow_prompt = shadow / "benchmarks/scicode/prompts/background.yaml"
    shadow_prompt.parent.mkdir(parents=True)
    shadow_prompt.write_text("user: shadowed\n")
    graph["solver"]["responses_api_agents"]["scicode_agent"]["prompt_fpath"] = (
        "benchmarks/scicode/prompts/background.yaml"
    )
    monkeypatch.setenv("NEMO_GYM_EXTRA_ROOTS", str(shadow))
    with pytest.raises(ConfigError, match="prompt resolution is shadowed"):
        module.qualify_scicode_dependencies(root, graph)
    monkeypatch.delenv("NEMO_GYM_EXTRA_ROOTS")
    graph["science"]["resources_servers"]["scicode"]["external_runner"] = "/tmp/unbound.py"
    with pytest.raises(ConfigError, match="unrecognized"):
        module.qualify_scicode_dependencies(root, graph)


def test_native_external_component_shadow_is_rejected(closure, monkeypatch):
    import nemo_gym.scicode_restart as module

    root, agent, resource, _ = closure
    shadow = root / "external"
    directory = shadow / "responses_api_models/openai_model"
    directory.mkdir(parents=True)
    (directory / "requirements.txt").write_text("# external\n")
    graph = {
        "policy": {"responses_api_models": {"openai_model": {}}},
        "solver": {
            "responses_api_agents": {
                "scicode_agent": {**agent, "model_server": {"name": "policy"}, "resources_server": {"name": "science"}}
            }
        },
        "science": {"resources_servers": {"scicode": resource}},
        "relay": {"environment_servers": {"legacy_agent": {"agent_server": {"name": "solver"}}}},
    }
    monkeypatch.setenv("NEMO_GYM_EXTRA_ROOTS", str(shadow))
    with pytest.raises(ConfigError, match="shadowed"):
        module.qualify_scicode_dependencies(root, graph)
