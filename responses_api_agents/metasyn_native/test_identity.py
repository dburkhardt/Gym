# SPDX-License-Identifier: Apache-2.0
import json
import pytest
from responses_api_agents.metasyn_native.identity import scientific_assets,creator_identity


def test_assets_capture_candidate_config_and_weights(tmp_path):
    model=tmp_path/'encoder';model.mkdir();(model/'weights.bin').write_bytes(b'weights')
    (tmp_path/'rag/203').mkdir(parents=True)
    candidates=tmp_path/'rag/203/candidates.json';candidates.write_text('[]')
    config=tmp_path/'similarity-model.json';config.write_text(json.dumps({'local_path':str(model)}))
    initial=scientific_assets(tmp_path,203)
    candidates.write_text('[{"id":1}]');assert scientific_assets(tmp_path,203)!=initial
    candidates.write_text('[]');(model/'weights.bin').write_bytes(b'other')
    assert scientific_assets(tmp_path,203)!=initial
    (model/'weights.bin').write_bytes(b'weights');(model/'new-config.json').write_text('{}')
    assert scientific_assets(tmp_path,203)!=initial


def test_missing_encoder_is_not_an_empty_identity(tmp_path):
    (tmp_path/'similarity-model.json').write_text(json.dumps({'local_path':str(tmp_path/'missing')}))
    with pytest.raises(ValueError,match='unavailable'):
        scientific_assets(tmp_path,203)


def test_child_rechecks_prompt_and_reference(tmp_path):
    import hashlib,sys
    from types import SimpleNamespace
    from pathlib import Path
    from responses_api_agents.metasyn_native.identity import verify_child
    def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    creator=tmp_path/'creator';creator.mkdir();module=creator/'rag.py';module.write_text('# pinned')
    encoder=tmp_path/'encoder';encoder.mkdir();(encoder/'weights').write_text('weights')
    (tmp_path/'similarity-model.json').write_text(json.dumps({'local_path':str(encoder)}))
    (tmp_path/'rag/203').mkdir(parents=True);prompt=tmp_path/'rag/203/prompt.txt';prompt.write_text('prompt')
    (tmp_path/'rag/203/candidates.json').write_text('[]');refs=tmp_path/'refs';refs.write_text('reference')
    executable=Path(sys.executable).resolve()
    record={'scientific_assets':scientific_assets(tmp_path,203),'creator':{'root':str(creator),'files':{'rag.py':sha(module)}},
            'prompt_sha256':sha(prompt),'reference_path':str(refs),'reference_sha256':sha(refs),
            'interpreter':{'path':str(executable),'sha256':sha(executable)},'sources':{}}
    (tmp_path/'identity.json').write_text(json.dumps(record))
    modules=[SimpleNamespace(__name__='metasyn.rag',__file__=str(module))]
    verify_child(tmp_path,modules)
    prompt.write_text('changed')
    with pytest.raises(ValueError,match='Prompt or reference'):verify_child(tmp_path,modules)
    prompt.write_text('prompt');refs.write_text('changed')
    with pytest.raises(ValueError,match='Prompt or reference'):verify_child(tmp_path,modules)
