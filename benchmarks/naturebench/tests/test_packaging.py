# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
import yaml

from benchmarks.naturebench.packaging import MOLECULAR_TASK, restore_public_input_mount


def test_only_public_evaluator_mount_changes() -> None:
    source = (Path(__file__).parent / "fixtures/molecular-creator-compose.yaml").read_text()
    repaired = restore_public_input_mount(source, task_id=MOLECULAR_TASK)
    original_config, revised = yaml.safe_load(source), yaml.safe_load(repaired)
    assert revised["services"]["naturebench-eval"]["volumes"].pop() == "./input/data:/problem/data:ro"
    assert revised == original_config
    with pytest.raises(ValueError, match="Unexpected creator"):
        restore_public_input_mount(repaired, task_id=MOLECULAR_TASK)
    with pytest.raises(ValueError, match="qualified only"):
        restore_public_input_mount(source, task_id="another-task")
    with pytest.raises(ValueError, match="Unexpected creator"):
        restore_public_input_mount(source + "\n", task_id=MOLECULAR_TASK)
