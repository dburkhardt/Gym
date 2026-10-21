"""Offline consistency checks for the qualified Linux artifact closure."""
import json
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1] / 'dependencies'


def entries(name):
    return [line for line in (ROOT / name).read_text().splitlines() if line and not line.startswith('#')]


def test_install_and_download_hashes_and_source_mapping():
    install = entries('linux-cp311.lock')
    download = entries('linux-cp311-download.lock')
    assert len(install) == 71 and len(download) == 70
    wheel_hashes = {line.split()[0] for line in entries('wheels.sha256')}
    assert len(wheel_hashes) == 71
    for line in install + download:
        requirement, digest = line.split(' --hash=sha256:')
        assert digest in wheel_hashes
        parsed = Requirement(requirement)
        assert parsed.name
    assert set(download) - set(install) == {
        next(line for line in download if line.startswith('torch @ https://download-r2.pytorch.org/whl/cpu/'))
    }
    assert len(set(install) - set(download)) == 2
    assert not any('antlr4' in line for line in download)


def test_encoder_and_build_provenance():
    encoder = json.loads((ROOT / 'encoder.json').read_text())
    assert encoder['revision'] == '4328cf26390c98c5e3c738b4460a05b95f4911f5'
    assert len(encoder['files']) == 13
    for item in encoder['files']:
        assert f"/resolve/{encoder['revision']}/" in item['url']
        assert f"{item['sha256']}  {item['path'].removeprefix('encoder/')}" in entries('encoder.sha256')
    build = json.loads((ROOT / 'antlr-build.json').read_text())
    assert build['python_files_equal_sdist'] == 56
    assert build['version'] == '4.9.3'
    assert f"{build['sha256']}  {build['wheel']}" in entries('wheels.sha256')
    assert {k: v['version'] for k, v in build['build_inputs'].items()} == {'setuptools': '69.5.1', 'wheel': '0.43.0'}
