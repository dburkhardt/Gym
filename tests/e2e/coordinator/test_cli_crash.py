# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from control import main


@pytest.mark.skipif(
    os.environ.get("NEMO_GYM_COORDINATOR_E2E") != "1", reason="starts real native CLI processes on loopback"
)
@pytest.mark.parametrize("logged", [False, True])
async def test_top_level_cli_crash_refuses_ambiguous_resume(tmp_path, logged):
    await main(tmp_path / "control", logged=logged)


@pytest.mark.skipif(os.environ.get("NEMO_GYM_COORDINATOR_E2E") != "1", reason="local native CLI processes")
async def test_completed_run_resume_retains_calls(tmp_path):
    await main(tmp_path / "control", logged=True, completed=True)


@pytest.mark.skipif(os.environ.get("NEMO_GYM_COORDINATOR_E2E") != "1", reason="local native CLI processes")
async def test_dataset_repeat_distribution_and_completed_resume(tmp_path):
    await main(tmp_path / "control", completed=True, distribution_control=True)


@pytest.mark.skipif(os.environ.get("NEMO_GYM_COORDINATOR_E2E") != "1", reason="Opt-in native subprocess control")
async def test_explicit_rejected_provider_restart(tmp_path):
    await main(tmp_path / "control", restart_control=True)


@pytest.mark.skipif(os.environ.get("NEMO_GYM_COORDINATOR_E2E") != "1", reason="local native CLI processes")
@pytest.mark.parametrize("rejection_status", [429, 503])
async def test_creator_resource_restart(tmp_path, rejection_status):
    from pathlib import Path

    repo = Path(os.environ["NEMO_GYM_CREATOR_RESTART_SOURCE"])
    await main(tmp_path / "control", restart_control=True, creator_repo=repo, rejection_status=rejection_status)
