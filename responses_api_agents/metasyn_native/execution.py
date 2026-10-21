# SPDX-License-Identifier: Apache-2.0
"""Shared lifecycle for trusted creator execution; no scientific protocol changes."""
import shlex
from pathlib import Path

from nemo_gym.sandbox.lifecycle import _publish
from nemo_gym.sandbox.providers.local.provider import LocalProvider
from nemo_gym.sandbox.scoring_workspace import scoring_workspace


async def run_creator(argv: list[str], output: Path, *, timeout_s: float = 3600) -> None:
    provider = LocalProvider(concurrency=1)
    command = shlex.join(argv) + ' > ' + shlex.quote(str(output / 'native-runtime.log')) + ' 2>&1'
    async with scoring_workspace(provider) as handle:
        result = await provider.exec(handle, command, timeout_s=timeout_s)
    _publish(output / 'execution.json', {
        'return_code': result.return_code, 'error_type': result.error_type,
        'timeout_s': timeout_s, 'process_group_settled': True,
        'scientific_score_assigned': False,
    })
    if result.error_type == 'timeout':
        raise TimeoutError('MetaSyn creator process exceeded its execution budget; process group settled')
    if result.return_code != 0 or result.error_type:
        raise RuntimeError(f'MetaSyn creator execution failed: {result.error_type or result.return_code}')
