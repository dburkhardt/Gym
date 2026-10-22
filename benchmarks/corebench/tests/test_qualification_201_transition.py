import copy
import json
from pathlib import Path
import pytest
from benchmarks.corebench import image_controls as controls
from benchmarks.corebench.qualification_compatibility import CORE201_PRIOR_QUALIFICATION, reviewed_compatibility


def inputs():
    row=next(r for r in json.loads((Path(controls.__file__).parent/'frozen20.json').read_text())['rows'] if r['id']=='capsule-2011424')
    return row, {'qualification_sha256':CORE201_PRIOR_QUALIFICATION}


def test_exact_transition_preserves_receipt():
    row,receipt=inputs(); original=copy.deepcopy(receipt)
    assert reviewed_compatibility(row,receipt)['review']=='core201-unchanged-probes-after-opt-in-891-v1'
    assert receipt==original


@pytest.mark.parametrize('field',['instruction_sha256','reference_sha256','scorer_sha256','published_image'])
def test_bound_content_change_rejected(field):
    row,receipt=inputs();row[field]+='changed'
    assert reviewed_compatibility(row,receipt) is None


@pytest.mark.parametrize('field',['SCIENTIFIC_PROBE','HARNESS_PROBE'])
def test_changed_probe_rejected(monkeypatch,field):
    row,receipt=inputs();monkeypatch.setattr(controls,field,getattr(controls,field)+'changed')
    assert reviewed_compatibility(row,receipt) is None


def test_changed_source_rejected():
    row,receipt=inputs();row['medium_file_manifest'][0]['sha256']='0'*64
    assert reviewed_compatibility(row,receipt) is None


def test_changed_recipe_rejected(monkeypatch):
    row,receipt=inputs();monkeypatch.setattr(controls,'recipe_digest',lambda row:'0'*64)
    assert reviewed_compatibility(row,receipt) is None


@pytest.mark.parametrize('field',['filesystem_layout_variant','external_input_variant','harness_install_variant'])
def test_new_variant_rejected(field):
    row,receipt=inputs();row[field]='new'
    assert reviewed_compatibility(row,receipt) is None


def test_unrelated_capsule_rejected():
    row,receipt=inputs();row['id']='capsule-6133093'
    assert reviewed_compatibility(row,receipt) is None


def test_changed_engine_rejected(monkeypatch,tmp_path):
    row,receipt=inputs();p=tmp_path/'engine.py';p.write_text('changed');monkeypatch.setattr(controls,'__file__',str(p))
    assert reviewed_compatibility(row,receipt) is None


def test_changed_harness_archive_rejected(monkeypatch):
    from benchmarks.corebench import prepare
    row,receipt=inputs();monkeypatch.setattr(prepare,'RUNTIME_SHA256','0'*64)
    assert reviewed_compatibility(row,receipt) is None
