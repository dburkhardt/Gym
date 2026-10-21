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
