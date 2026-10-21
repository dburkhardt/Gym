# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from nemo_gym import server_utils
from nemo_gym.global_config import NEMO_GYM_RESERVED_TOP_LEVEL_KEYS


def setup(monkeypatch, config):
    ray = MagicMock()
    ray.is_initialized.return_value = False
    ray.get_runtime_context.return_value.gcs_address = "local:1234"
    monkeypatch.setattr(server_utils, "_get_ray", lambda: ray)
    monkeypatch.setattr(server_utils, "get_global_config_dict", lambda: OmegaConf.create(config))
    monkeypatch.setattr(server_utils, "_NEMO_GYM_STARTED_RAY_CLUSTER", False)
    return ray


def test_local_setting_uses_public_ray_option_without_task_memory_override(monkeypatch):
    ray = setup(monkeypatch, {"ray_local_object_store_memory_bytes": 2 * 1024**3})
    server_utils.initialize_ray()
    ray.init.assert_called_once_with(ignore_reinit_error=True, object_store_memory=2 * 1024**3)
    assert server_utils._NEMO_GYM_STARTED_RAY_CLUSTER
    assert "ray_local_object_store_memory_bytes" in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "2147483648"])
def test_invalid_new_cluster_setting_refused_before_init(monkeypatch, value):
    ray = setup(monkeypatch, {"ray_local_object_store_memory_bytes": value})
    with pytest.raises(ValueError, match="positive integer"):
        server_utils.initialize_ray()
    ray.init.assert_not_called()
    assert not server_utils._NEMO_GYM_STARTED_RAY_CLUSTER


def test_connecting_child_does_not_resize_existing_cluster(monkeypatch):
    ray = setup(
        monkeypatch, {"ray_head_node_address": "local:1234", "ray_local_object_store_memory_bytes": 2 * 1024**3}
    )
    server_utils.initialize_ray()
    ray.init.assert_called_once_with(ignore_reinit_error=True, address="local:1234")
    assert not server_utils._NEMO_GYM_STARTED_RAY_CLUSTER


def test_default_leaves_upstream_memory_policy_unchanged(monkeypatch):
    ray = setup(monkeypatch, {})
    server_utils.initialize_ray()
    ray.init.assert_called_once_with(ignore_reinit_error=True)
