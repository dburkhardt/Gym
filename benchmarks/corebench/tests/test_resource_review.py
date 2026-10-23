import copy
import hashlib
import json

import pytest

from benchmarks.corebench.image_controls import source_digest
from benchmarks.corebench.prepare_frozen20 import validate_qualification
from benchmarks.corebench.resource_review import validate_resource_review
from benchmarks.corebench.tests.test_frozen20 import fixture


def evidence(tmp_path, row, resources):
    expected = {'capsule': row['id'], 'creator_image': row['published_image'],
                'source_sha256': source_digest(row), 'runtime_environment_variant': row['runtime_environment_variant']}
    values = {
        'binding': {**expected, 'resources': resources, 'attempt_id':'fixture-attempt'},
        'result': {'return_code': 0, 'error_type': None, 'model_calls': 0, 'scientific_score': None},
        'lifecycle': {'cleanup_confirmed': True, 'handle': {'sandbox_id':'fixture-container','ownership_labels':{'nemo-gym.run':'fixture-attempt'}}},
        'supervisor': {'return_code': 0, 'error': None, 'timed_out': False, 'harness_forced_kill': False, 'cleanup_confirmed': True},
        'container': {'Name':'/fixture-container', 'Image': row['published_image'].split('@')[-1], 'HostConfig': {'Memory': resources['memory_mb']*1024**2, 'MemorySwap': resources['memory_mb']*1024**2, 'NanoCpus': resources['cpus']*10**9}, 'Config': {'Labels':{'nemo-gym.run':'fixture-attempt'}, 'Env': [k+'='+v for k,v in row['runtime_environment_variant'].items()]}},
        'source_manifest': {x['path']: x['sha256'] for x in row['medium_file_manifest']},
        'memory': {'peak_bytes': 1000000, 'oom': 0, 'oom_kill': 0},
    }
    review = {'version': 'core-resource-review-v1', 'attempt_id':'fixture-attempt', **expected, 'artifacts': {}}
    for role, value in values.items():
        p=tmp_path/(role+'.json');p.write_text(json.dumps(value))
        review['artifacts'][role]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    mutate(review,'binding',lambda d:d.update(control_artifacts={k:v['sha256'] for k,v in review['artifacts'].items() if k!='binding'}))
    return review


def mutate(review, role, fn):
    from pathlib import Path
    p=Path(review['artifacts'][role]['path']);value=json.loads(p.read_text());fn(value);p.write_text(json.dumps(value))
    review['artifacts'][role]['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()


def test_review_is_not_image_qualification_and_truthy_bypass_rejected(tmp_path):
    _,_,row,q,_=fixture(tmp_path);row['status']='measured_oom_requires_resource_review'
    for bad in [None, True, {}, {'version':'core-resource-review-v1'}]:
        with pytest.raises(ValueError):validate_resource_review(row,q['resources'],bad)
    review=evidence(tmp_path,row,q['resources']);q['oom_resource_review']=review
    validate_qualification(row,q)
    q['harness_python_ok']=False
    with pytest.raises(ValueError,match='incomplete'):validate_qualification(row,q)


@pytest.mark.parametrize('role,change',[
    ('binding',lambda d:d.update(capsule='wrong')),
    ('binding',lambda d:d.update(creator_image='wrong')),
    ('binding',lambda d:d.update(source_sha256='wrong')),
    ('binding',lambda d:d.update(runtime_environment_variant={'PATH':'wrong'})),
    ('result',lambda d:d.update(return_code=1)),
    ('result',lambda d:d.update(return_code=False)),
    ('result',lambda d:d.update(model_calls=False)),
    ('result',lambda d:d.pop('scientific_score')),
    ('result',lambda d:d.pop('error_type')),
    ('supervisor',lambda d:d.update(return_code=False)),
    ('supervisor',lambda d:d.pop('error')),
    ('container',lambda d:d.update(Name='/unrelated-container')),
    ('container',lambda d:d['Config']['Labels'].update({'nemo-gym.run':'other-attempt'})),
    ('result',lambda d:d.update(error_type='timeout')),
    ('result',lambda d:d.update(model_calls=1)),
    ('result',lambda d:d.update(scientific_score=1)),
    ('lifecycle',lambda d:d.update(cleanup_confirmed=False)),
    ('supervisor',lambda d:d.update(timed_out=True)),
    ('supervisor',lambda d:d.update(harness_forced_kill=True)),
    ('container',lambda d:d.update(Image='wrong')),
    ('container',lambda d:d['HostConfig'].update(Memory=8192)),
    ('source_manifest',lambda d:d.update({'code/run.py':'wrong'})),
    ('memory',lambda d:d.update(oom_kill=1)),
    ('memory',lambda d:d.update(peak_bytes=2**60)),
])
def test_invalid_actual_controls_rejected(tmp_path,role,change):
    _,_,row,q,_=fixture(tmp_path);review=evidence(tmp_path,row,q['resources']);mutate(review,role,change)
    if role != 'binding':
        mutate(review,'binding',lambda d:d['control_artifacts'].update({role:review['artifacts'][role]['sha256']}))
    with pytest.raises(ValueError):validate_resource_review(row,q['resources'],review)


def test_insufficient_memory_and_tampered_or_missing_artifacts(tmp_path):
    from pathlib import Path
    _,_,row,q,_=fixture(tmp_path);review=evidence(tmp_path,row,q['resources'])
    with pytest.raises(ValueError,match='insufficient'):validate_resource_review(row,{**q['resources'],'memory_mb':1024},review)
    p=Path(review['artifacts']['result']['path']);p.write_text('{}')
    with pytest.raises(ValueError,match='hash'):validate_resource_review(row,q['resources'],review)
    p.unlink()
    with pytest.raises(ValueError,match='unavailable'):validate_resource_review(row,q['resources'],review)


@pytest.mark.asyncio
async def test_producer_carries_review_and_runs_all_image_probes(tmp_path,monkeypatch):
    from benchmarks.corebench import image_controls as controls,prepare
    medium,refs,row,q,_=fixture(tmp_path);row['status']='measured_oom_requires_resource_review'
    review=evidence(tmp_path,row,q['resources']);runtime=tmp_path/'runtime.tar';runtime.write_bytes(b'fixture')
    monkeypatch.setattr(prepare,'RUNTIME_SHA256',hashlib.sha256(runtime.read_bytes()).hexdigest())
    calls=[]
    async def probe(image,command,destination,**kwargs):
        calls.append(destination.name)
        return {'image':{'Id':'sha256:'+'b'*64},'stdout':'harness-python-ok reference-isolation-ok' if destination.name=='harness' else ('image-source-ok' if destination.name=='image-source' else 'same science')}
    monkeypatch.setattr(controls,'probe_image',probe)
    receipt=await controls.qualify_images(row,medium,refs,runtime,'sha256:'+'b'*64,tmp_path/'qualified',resources=q['resources'],oom_resource_review=review)
    assert calls==['creator','final','harness','image-source']
    assert receipt['oom_resource_review']==review
    validate_qualification(row,receipt)
