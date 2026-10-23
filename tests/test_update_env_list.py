# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
import sys
from pathlib import Path

import pytest
import scripts.update_env_list as update_env_list


def test_training_server_info_includes_benchmark_configs(tmp_path: Path, monkeypatch) -> None:
    resources_servers = tmp_path / "resources_servers"
    responses_api_agents = tmp_path / "responses_api_agents"
    benchmarks = tmp_path / "benchmarks"
    resources_servers.mkdir()
    responses_api_agents.mkdir()
    benchmark_dir = benchmarks / "desktop_bench"
    benchmark_dir.mkdir(parents=True)
    agent_dir = responses_api_agents / "desktop_agent" / "configs"
    agent_dir.mkdir(parents=True)
    agent_config = agent_dir / "desktop_agent.yaml"
    agent_config.write_text(
        """
desktop_agent:
  responses_api_agents:
    desktop_agent:
      domain: agent
      description: Desktop benchmark
      value: GUI evaluation
""".lstrip(),
        encoding="utf-8",
    )
    (benchmark_dir / "config.yaml").write_text(
        f"""
config_paths:
  - {agent_config}

desktop_agent:
  responses_api_agents:
    desktop_agent:
      datasets:
        - name: validation
          type: validation
          license: Apache 2.0
          huggingface_identifier:
            repo_id: example/desktop-bench
""".lstrip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(update_env_list, "RESOURCES_SERVERS_FOLDER", resources_servers)
    monkeypatch.setattr(update_env_list, "RESPONSES_API_AGENTS_FOLDER", responses_api_agents)
    monkeypatch.setattr(update_env_list, "BENCHMARKS_FOLDER", benchmarks)

    servers = update_env_list.get_training_server_info()

    assert len(servers) == 1
    server = servers[0]
    assert server.name == "desktop_bench"
    assert server.display_name == "Desktop Agent"
    assert server.config_path == "benchmarks/desktop_bench/config.yaml"
    assert server.readme_path == "benchmarks/desktop_bench/README.md"
    assert server.config_metadata.domain == "agent"
    assert server.config_metadata.description == "Desktop benchmark"
    assert server.config_metadata.value == "GUI evaluation"
    assert server.config_metadata.types == ["validation"]


@pytest.mark.parametrize("skip", [False, True])
def test_cli_without_markers_is_optional_and_preserves_readme(tmp_path: Path, skip: bool) -> None:
    readme = tmp_path / "README.md"
    original = b"# Life sciences evaluation report\n\nResearch summary.\n"
    readme.write_bytes(original)
    command = [sys.executable, str(Path(update_env_list.__file__).resolve())]
    if skip:
        command.append("--skip-without-markers")
    command.append("README.md")  # Pre-commit supplies changed filenames.
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == (0 if skip else 1), result.stderr
    assert readme.read_bytes() == original
    assert ("skipping" in result.stdout) if skip else ("markers" in result.stderr)


@pytest.mark.parametrize(
    "marker",
    ["<!-- START_TRAINING_SERVERS_TABLE -->", "<!-- END_TRAINING_SERVERS_TABLE -->"],
)
def test_cli_partial_markers_still_fail(tmp_path: Path, marker: str) -> None:
    readme = tmp_path / "README.md"
    original = f"# Report\n{marker}\n".encode()
    readme.write_bytes(original)
    result = subprocess.run(
        [sys.executable, str(Path(update_env_list.__file__).resolve()), "--skip-without-markers"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "markers" in result.stderr
    assert readme.read_bytes() == original


@pytest.mark.parametrize("skip", [False, True])
def test_cli_updates_existing_marked_table(tmp_path: Path, skip: bool) -> None:
    readme = tmp_path / "README.md"
    start = "<!-- START_TRAINING_SERVERS_TABLE -->"
    end = "<!-- END_TRAINING_SERVERS_TABLE -->"
    readme.write_text(f"# Report\n{start}\nold table\n{end}\nAfter.\n")
    for folder in ("resources_servers", "responses_api_agents", "benchmarks"):
        (tmp_path / folder).mkdir()
    command = [sys.executable, str(Path(update_env_list.__file__).resolve())]
    if skip:
        command.append("--skip-without-markers")
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    expected_table = update_env_list.generate_training_table([])
    assert readme.read_text() == f"# Report\n{start}\n{expected_table}\n{end}\nAfter.\n"
