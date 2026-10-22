# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit hash-bound saved-solver judging; never invokes the solver or viewer."""
import hashlib
import json
import secrets
import shlex
import shutil
from pathlib import Path

from benchmarks.scivisagentbench.managed import validate_native_python
from benchmarks.scivisagentbench.transport import MODEL
from nemo_gym.sandbox.model_bridge import bridge
from nemo_gym.sandbox.providers.local.provider import LocalProvider
from nemo_gym.sandbox.scoring_workspace import scoring_workspace


def prepare_saved(manifest_path: Path, task: dict, output: Path) -> tuple[dict, dict]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('version') != 1 or manifest.get('purpose') != 'saved-solver-judge-only':
        raise ValueError('Explicit saved solver judging manifest required')
    root = Path(manifest['root'])
    if root.is_symlink() or not root.is_dir() or not manifest.get('files'):
        raise ValueError('Regular hash-bound saved attempt required')
    for rel, sha in manifest['files'].items():
        p = root / rel
        if Path(rel).is_absolute() or '..' in Path(rel).parts:
            raise ValueError('Invalid saved artifact path')
        if any(q.is_symlink() for q in (p, *p.parents) if q.is_relative_to(root)):
            raise ValueError('Saved artifact symlink refused')
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            raise ValueError('Saved artifact hash mismatch')
    for directory in ('workspace', 'judge', 'solver'):
        actual = {str(p.relative_to(root)) for p in (root / directory).rglob('*') if p.is_file()}
        expected = {p for p in manifest['files'] if p.startswith(directory + '/')}
        if actual != expected:
            raise ValueError('Saved artifact coverage mismatch')
    required = ['identity.json', 'worker-input.json', 'solver/native-solver-result.json']
    if any(p not in manifest['files'] for p in required):
        raise ValueError('Missing saved solver evidence')
    identity = json.loads((root / 'identity.json').read_text())
    solver = json.loads((root / 'solver/native-solver-result.json').read_text())
    if identity['task'] != task or identity['model'] != MODEL or not solver.get('success'):
        raise ValueError('Saved task/model/success mismatch')
    if not any(p.startswith('judge/') and '/test_results/' in p for p in manifest['files']):
        raise ValueError('Original evaluator test result required')
    output.mkdir(parents=True, exist_ok=False)
    for rel in manifest['files']:
        if rel.startswith('workspace/'):
            target = output / 'exported-artifacts' / Path(rel).relative_to('workspace')
        elif rel.startswith('judge/') or rel.startswith('solver/'):
            target = output / rel
        else:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / rel, target)
    (output / 'saved-input-manifest.json').write_bytes(manifest_path.read_bytes())
    return json.loads((root / 'worker-input.json').read_text()), solver


async def execute_saved(config, task: dict, output: Path, endpoint: str) -> tuple[dict, dict]:
    payload, solver = prepare_saved(Path(config.saved_solver_manifest), task, output)
    source = Path(__file__).resolve().parents[2]
    interpreter = validate_native_python(Path(config.native_python))
    paths = [source / 'benchmarks/scivisagentbench' / (name + '.py')
             for name in ('native', 'transport', 'judge', 'evaluate', 'worker', 'saved_judge')]
    paths += [source / 'nemo_gym/guarded_chat_transport.py']
    identity = {'source': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                'interpreter': {'path': str(interpreter), 'resolved_path': str(interpreter.resolve()),
                                'sha256': hashlib.sha256(interpreter.read_bytes()).hexdigest()},
                'task': task, 'model': MODEL, 'max_tokens': config.max_tokens,
                'timeout_s': config.timeout_s, 'request_timeout_s': config.request_timeout_s,
                'saved_manifest_sha256': hashlib.sha256(Path(config.saved_solver_manifest).read_bytes()).hexdigest()}
    payload['saved_solver'] = solver
    payload['source_root'] = config.source_root
    payload['task']['judge_case_dir'] = str(output / 'judge' / task['case_name'])
    payload['task']['judge_data_dir'] = str(output / 'judge-data')
    (output / 'identity.json').write_text(json.dumps(identity, indent=2))
    (output / 'worker-input.json').write_text(json.dumps(payload, indent=2))
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    provider = LocalProvider(concurrency=1, workspace_root='/tmp')
    async with scoring_workspace(provider) as handle:
        socket = handle.raw['workspace'] / 'model.sock'
        token, lineage = secrets.token_urlsafe(32), secrets.token_hex(16)
        async with bridge(socket, endpoint=endpoint, token=token, lineage=lineage, digest=digest,
                          namespace='scivis', expected_model=MODEL, request_timeout_s=config.request_timeout_s):
            result = await provider.exec(
                handle, shlex.join([str(interpreter), '-m', 'benchmarks.scivisagentbench.worker', str(output)]),
                timeout_s=config.timeout_s + config.request_timeout_s + 60,
                env={'PYTHONPATH': str(source), 'NEMO_GYM_EXTRA_ROOTS': str(source),
                     'SCIVIS_SOCKET': str(socket), 'SCIVIS_TOKEN': token,
                     'SCIVIS_LINEAGE': lineage, 'SCIVIS_DIGEST': digest},
            )
            (output / 'worker.stdout').write_text(result.stdout or '')
            (output / 'worker.stderr').write_text(result.stderr or '')
            if result.return_code or result.error_type:
                raise RuntimeError('Saved solver judging failed; original attempt preserved')
    record = json.loads((output / 'worker-result.json').read_text())
    return record['solver'], record['evaluation']
