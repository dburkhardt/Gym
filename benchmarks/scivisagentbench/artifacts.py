# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export only task-owned regular outputs without changing producer permissions."""
import asyncio
import hashlib
import json
import shlex
import stat
import sys
from pathlib import Path


def artifact_paths(workspace: Path) -> list[str]:
    if workspace.is_symlink() or not workspace.is_dir():
        raise ValueError('Regular owned workspace required')
    result = []
    for p in sorted(workspace.rglob('*')):
        mode = p.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError('Artifact symlink or special file refused')
        result.append(str(p.relative_to(workspace)))
    return result


async def export_artifacts(descriptor: dict, workspace: Path, destination: Path) -> dict:
    # Imported only by the complete Gym interpreter, not the creator environment.
    from nemo_gym.sandbox import AsyncSandbox
    from nemo_gym.sandbox.providers.docker.provider import DockerProvider

    if not descriptor.get('ownership_labels'):
        raise ValueError('Attributed owned sandbox descriptor required')
    paths = artifact_paths(workspace)
    destination.mkdir(parents=True, exist_ok=False)
    sandbox = await AsyncSandbox.connect(descriptor, provider=DockerProvider(), owns_provider=False)
    # Borrow the parent's still-owned sandbox; never stop it here.
    code = '''import pathlib,stat,hashlib,json,sys
root=pathlib.Path('/workspace'); result={}
for rel in json.loads(sys.argv[1]):
 p=root/rel
 assert not pathlib.Path(rel).is_absolute() and '..' not in pathlib.Path(rel).parts
 assert all(not q.is_symlink() for q in (p,*p.parents) if q.is_relative_to(root))
 assert stat.S_ISREG(p.lstat().st_mode)
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 result[rel]={'sha256':h.hexdigest(),'mode':oct(stat.S_IMODE(p.stat().st_mode)),'uid':p.stat().st_uid,'size':p.stat().st_size}
print(json.dumps(result))
'''
    r = await sandbox.exec(shlex.join(['python', '-c', code, json.dumps(paths)]), timeout_s=60)
    if r.return_code:
        raise RuntimeError('Owned artifact identity inspection failed')
    original = json.loads(r.stdout)
    if set(original) != set(paths):
        raise ValueError('Artifact manifest mismatch')
    for rel in paths:
        target = destination / rel
        await sandbox.download('/workspace/' + rel, target)
        if target.is_symlink() or not target.is_file():
            raise ValueError('Downloaded artifact is not a regular file')
        if hashlib.sha256(target.read_bytes()).hexdigest() != original[rel]['sha256']:
            raise ValueError('Downloaded artifact hash mismatch')
    if artifact_paths(workspace) != paths:
        raise ValueError('Artifact set changed during export')
    return {'files': original, 'source_permissions_unchanged': True,
            'method': 'owned AsyncSandbox.download of explicit regular files; no source/data/reference mounts'}


async def main(output: Path):
    payload = json.loads((output / 'worker-input.json').read_text())
    receipt = await export_artifacts(payload['viewer_descriptor'], output / 'workspace', output / 'exported-artifacts')
    (output / 'artifact-export.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    asyncio.run(main(Path(sys.argv[1])))
