import copy
import json
from pathlib import Path
import pytest
from benchmarks.corebench import image_controls as controls
from benchmarks.corebench.qualification_compatibility import RETAINED_CONTROL_TRANSITIONS, reviewed_compatibility

@pytest.fixture(params=sorted(RETAINED_CONTROL_TRANSITIONS))
def pair(request):
    row=next(r for r in json.loads((Path(controls.__file__).parent/'frozen20.json').read_text())['rows'] if r['id']==request.param)
    _,digest,variant=RETAINED_CONTROL_TRANSITIONS[request.param]
    if variant is not None:row['harness_install_variant']=variant
    return row,{'qualification_sha256':digest}


def test_exact_retained_control(pair):
    row,receipt=pair;old=copy.deepcopy(receipt)
    assert reviewed_compatibility(row,receipt)['review']=='core-five-original-content-preserving-transitions-v1'
    assert old==receipt

@pytest.mark.parametrize('field',['reference_sha256','scorer_sha256','instruction_sha256','published_image'])
def test_changed_identity(pair,field):
    row,receipt=pair;row[field]+='changed'
    assert reviewed_compatibility(row,receipt) is None


def test_changed_source(pair):
    row,receipt=pair;row['medium_file_manifest'][0]['sha256']='0'*64
    assert reviewed_compatibility(row,receipt) is None

@pytest.mark.parametrize('field',['SCIENTIFIC_PROBE','HARNESS_PROBE'])
def test_changed_probe(pair,field,monkeypatch):
    row,receipt=pair;monkeypatch.setattr(controls,field,getattr(controls,field)+'changed')
    assert reviewed_compatibility(row,receipt) is None


def test_changed_recipe(pair,monkeypatch):
    row,receipt=pair;monkeypatch.setattr(controls,'recipe_digest',lambda r:'0'*64)
    assert reviewed_compatibility(row,receipt) is None


def test_changed_variant(pair):
    row,receipt=pair;row['harness_install_variant']='changed'
    assert reviewed_compatibility(row,receipt) is None


def test_unreviewed_capsule(pair):
    row,receipt=pair;row['id']='capsule-6133093'
    assert reviewed_compatibility(row,receipt) is None


def test_changed_engine(pair,monkeypatch,tmp_path):
    row,receipt=pair;p=tmp_path/'engine.py';p.write_text('changed');monkeypatch.setattr(controls,'__file__',str(p))
    assert reviewed_compatibility(row,receipt) is None


def test_changed_harness_runtime(pair,monkeypatch):
    from benchmarks.corebench import prepare
    row,receipt=pair;monkeypatch.setattr(prepare,'RUNTIME_SHA256','0'*64)
    assert reviewed_compatibility(row,receipt) is None
