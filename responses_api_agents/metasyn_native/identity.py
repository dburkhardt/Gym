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
    assets=expected['scientific_assets']
    if scientific_assets(Path(assets['data_root']), assets['review_id']) != assets:
        raise ValueError('Scientific assets changed before dispatch')
    prompt=Path(assets['data_root'])/'rag'/str(assets['review_id'])/'prompt.txt'
    if _sha256(prompt)!=expected['prompt_sha256'] or _sha256(Path(expected['reference_path']))!=expected['reference_sha256']:
        raise ValueError('Prompt or reference changed before dispatch')
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


def _sha256(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def scientific_assets(data_root: Path, review_id: int) -> dict:
    """Bind candidate selection and every local encoder artifact by content."""
    config_path = data_root / 'similarity-model.json'
    config = json.loads(config_path.read_text())
    model_root = Path(config['local_path'])
    if not model_root.is_dir():
        raise ValueError('Similarity encoder snapshot is unavailable')
    model_files = {str(p.relative_to(model_root)):_sha256(p)
                   for p in sorted(model_root.rglob('*')) if p.is_file()}
    if not model_files:
        raise ValueError('Similarity encoder snapshot is empty')
    return {'data_root':str(data_root.resolve()), 'review_id':review_id,
            'candidates_sha256':_sha256(data_root/'rag'/str(review_id)/'candidates.json'),
            'similarity_config_sha256':_sha256(config_path),
            'similarity_root':str(model_root.resolve()), 'similarity_files':model_files}
