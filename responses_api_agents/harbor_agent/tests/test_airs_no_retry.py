# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from unittest.mock import AsyncMock, patch

import pytest
from harbor.agents.terminus_2.terminus_2 import Terminus2

from responses_api_agents.harbor_agent.custom_agents.airs_no_retry import AIRSNoRetryAgent, NoRetryNemoGymLLM


@pytest.mark.asyncio
async def test_failed_http_call_is_attempted_once_and_preserved(tmp_path):
    llm = NoRetryNemoGymLLM(model_name="test", api_base="http://localhost:1")
    llm.failure_receipt_path = tmp_path / "failure.json"
    with patch.object(
        llm, "_post_chat_completions", new=AsyncMock(side_effect=ConnectionError("uncertain dispatch"))
    ) as post:
        with pytest.raises(ConnectionError, match="uncertain dispatch"):
            await llm.call("question")
    assert post.await_count == 1
    assert json.loads(llm.failure_receipt_path.read_text()) == {
        "exception_type": "ConnectionError",
        "message": "uncertain dispatch",
        "retry_performed": False,
    }


@pytest.mark.asyncio
async def test_outer_native_query_decorator_cannot_repeat_failure():
    agent = object.__new__(AIRSNoRetryAgent)
    with patch.object(
        Terminus2._query_llm, "__wrapped__", new=AsyncMock(side_effect=ConnectionError("failed"))
    ) as query:
        with pytest.raises(ConnectionError):
            await agent._query_llm(None, "question", (None, None, None))
    assert query.await_count == 1
