# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual ServerClient/request/retry chain at its physical HTTP boundary."""

from unittest.mock import AsyncMock

import pytest
from aiohttp import ServerDisconnectedError
from omegaconf import OmegaConf

from nemo_gym import server_utils as su


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [None, False, True])
@pytest.mark.parametrize("checkpoint_run", [False, True])
async def test_resend_policy_actual_request_chain(monkeypatch, explicit, checkpoint_run):
    client = su.ServerClient(
        head_server_config={"host": "localhost", "port": 0}, global_config_dict=OmegaConf.create({})
    )
    client._checkpoint_enabled = checkpoint_run
    client._server_base_urls["target"] = "http://localhost:12345"
    physical = AsyncMock()
    response = object()
    physical.request.side_effect = [ServerDisconnectedError(), response]
    monkeypatch.setattr(su, "get_global_aiohttp_client", lambda: physical)
    monkeypatch.setattr(su, "is_span_group_enabled", lambda *_: False)
    kwargs = {} if explicit is None else {"_resend_after_disconnect": explicit}
    if checkpoint_run or explicit is False:
        with pytest.raises(ServerDisconnectedError):
            await client.post("target", "/run" if checkpoint_run else "/v1/responses", _max_num_tries=2, **kwargs)
        assert physical.request.await_count == 1
    else:
        assert await client.post("target", "/v1/responses", _max_num_tries=2, **kwargs) is response
        assert physical.request.await_count == 2
    assert all("_resend_after_disconnect" not in call.kwargs for call in physical.request.await_args_list)


@pytest.mark.asyncio
async def test_explicit_single_attempt_never_retransmits(monkeypatch):
    client = su.ServerClient(
        head_server_config={"host": "localhost", "port": 0}, global_config_dict=OmegaConf.create({})
    )
    client._server_base_urls["target"] = "http://localhost:12345"
    physical = AsyncMock()
    physical.request.side_effect = ServerDisconnectedError()
    monkeypatch.setattr(su, "get_global_aiohttp_client", lambda: physical)
    monkeypatch.setattr(su, "is_span_group_enabled", lambda *_: False)
    with pytest.raises(ServerDisconnectedError):
        await client.post("target", "/v1/responses", _max_num_tries=1, _resend_after_disconnect=True)
    assert physical.request.await_count == 1
