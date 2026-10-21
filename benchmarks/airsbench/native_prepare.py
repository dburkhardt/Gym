# SPDX-License-Identifier: Apache-2.0
"""Native Harbor overlay using qualified input-bearing images and separate scoring."""

import json
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.airsbench.prepare import prepare as prepare_preserved
from nemo_gym.dataset_staging import stage_file


def runtime_config(text: str, image: str) -> str:
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("Immutable runtime image required")
    if text.count("docker_image = ") != 1 or text.count("[verifier]") != 1:
        raise ValueError("Unexpected preserved task configuration")
    text = re.sub(r'docker_image = "[^"]+"', f'docker_image = "{image}"', text)
    text = text.replace("allow_internet = false", 'network_mode = "no-network"')
    text = text.replace("[environment]\n", '[environment]\nworkdir = "/workspace"\n')
    text = text.replace("[verifier]\n", '[verifier]\nenvironment_mode = "separate"\n')
    return 'artifacts = ["/workspace"]\n' + text


def prepare() -> Path:
    roots = json.loads(os.environ["AIRS_TASK_ROOTS_JSON"])
    destination = Path(__file__).parent / "data"
    manifest = json.loads(Path(__file__).with_name("frozen_sources.json").read_text())
    with TemporaryDirectory() as temporary:
        staged = Path(temporary) / "preserved"
        dataset = prepare_preserved(roots, staged)
        for task in manifest["tasks"]:
            source = staged / "tasks" / task["name"]
            target = destination / "tasks" / task["name"]
            # The qualified image already contains the exact public inputs.
            # No Dockerfile build, mutable tag, or metadata-only upload replaces it.
            if (target / "environment").exists():
                raise ValueError("Unexpected old runtime overlay; use a fresh prepared destination")
            for file in source.rglob("*"):
                relative = file.relative_to(source)
                if not file.is_file() or relative.parts[0] == "environment" or file.name.endswith(".stage.lock"):
                    continue
                if relative == Path("task.toml"):
                    file.write_text(runtime_config(file.read_text(), task["image"]))
                published = stage_file(file, target / relative)
                if file.suffix == ".sh":
                    published.chmod(0o755)
        return stage_file(dataset, destination / "benchmark.jsonl")


if __name__ == "__main__":
    print(prepare())
