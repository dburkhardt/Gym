# SPDX-License-Identifier: Apache-2.0
"""Check the exact preserved creator Python tree and child import identities."""
import hashlib
import json
import sys
from pathlib import Path


def creator_identity(root: Path, expected: dict | None = None) -> dict:
    expected = expected if expected is not None else json.loads(Path(__file__).with_name('creator-python-files.json').read_text())
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Creator source must be a real directory')
    paths = list(root.rglob('*.py'))
    if any(p.is_symlink() or any(parent.is_symlink() for parent in p.parents if parent.is_relative_to(root)) for p in paths):
        raise ValueError('Creator Python source contains symlink')
    actual = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if actual != expected:
        raise ValueError('Creator Python source differs from preserved snapshot')
    return {'root':str(root.resolve()), 'files':actual}


def verify_child(output: Path, modules: list) -> None:
    from nemo_gym.sandbox.lifecycle import _publish
    expected=json.loads((output/'identity.json').read_text())
    root=Path(expected['creator']['root'])
    creator_identity(root, expected['creator']['files'])
    imports={m.__name__:str(Path(m.__file__).resolve()) for m in modules}
    if any(not Path(path).is_relative_to(root) for path in imports.values()):
        raise ValueError('Creator import resolved outside pinned tree')
    executable=Path(sys.executable).resolve()
    actual={'path':str(executable),'sha256':hashlib.sha256(executable.read_bytes()).hexdigest()}
    if actual != expected['interpreter']:
        raise ValueError('Creator interpreter differs from recorded runtime')
    for name, sha in expected['sources'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:
            raise ValueError('Wrapper source changed before dispatch')
    _publish(output/'runtime-imports.json',{'imports':imports,'interpreter':actual})
