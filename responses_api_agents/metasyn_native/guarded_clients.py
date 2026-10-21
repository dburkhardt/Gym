# SPDX-License-Identifier: Apache-2.0
"""Transport-only binding for the creator's synchronous OpenAI clients."""
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import httpx
from openai import OpenAI

from nemo_gym.sandbox.guarded_chat_transport import GuardedSyncChatTransport

MODEL = 'nvidia/nvidia/nemotron-3.5-super-vl'


@contextmanager
def guarded_clients(*, rag, llm_client, socket_path: str, token: str, lineage: str,
                    digest: str, output: Path, transport=None):
    """Patch SDK construction only; creator prompts and JSON retries stay native.

    After an ambiguous physical dispatch the shared guard remains poisoned, so
    an upstream parsing/retry loop cannot forward another request.
    """
    guarded = GuardedSyncChatTransport(socket_path=socket_path, token=token,
        lineage=lineage, digest=digest, namespace='metasyn', model=MODEL,
        output=output, transport=transport)
    with httpx.Client(transport=guarded, timeout=600, follow_redirects=False) as client:
        def factory(*args, **kwargs):
            if args:
                raise ValueError('Unexpected positional SDK construction')
            if set(kwargs) - {'api_key', 'base_url'}:
                raise ValueError('Unreviewed creator SDK options')
            return OpenAI(api_key='task-scoped-placeholder', base_url='http://gym-internal/v1',
                          http_client=client, max_retries=0)
        with patch.object(rag, 'OpenAI', factory), patch.object(llm_client, 'OpenAI', factory):
            yield
