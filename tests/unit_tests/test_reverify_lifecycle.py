# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym import rollout_reverification
from nemo_gym.cli import eval as eval_cli


@pytest.mark.parametrize("failure", [None, RuntimeError("verify failed"), KeyboardInterrupt()])
def test_reverify_shuts_down_owned_servers_on_terminal_outcome(monkeypatch, failure):
    runner = MagicMock()
    helper = MagicMock()
    helper.run_from_config = AsyncMock(side_effect=failure)
    monkeypatch.setattr(eval_cli, "RunHelper", lambda: runner)
    monkeypatch.setattr(eval_cli, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(rollout_reverification.RolloutReverificationConfig, "model_validate", lambda _: object())
    monkeypatch.setattr(rollout_reverification, "RolloutReverificationHelper", lambda: helper)
    if failure:
        with pytest.raises(type(failure)):
            eval_cli.reverify_rollouts()
    else:
        eval_cli.reverify_rollouts()
    runner.start.assert_called_once_with(None)
    runner.shutdown.assert_called_once_with()


def test_reverify_config_failure_after_start_still_shuts_down(monkeypatch):
    runner = MagicMock()
    monkeypatch.setattr(eval_cli, "RunHelper", lambda: runner)
    monkeypatch.setattr(eval_cli, "get_global_config_dict", MagicMock(side_effect=RuntimeError("invalid config")))
    with pytest.raises(RuntimeError, match="invalid config"):
        eval_cli.reverify_rollouts()
    runner.shutdown.assert_called_once_with()
