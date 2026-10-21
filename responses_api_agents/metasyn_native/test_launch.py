# SPDX-License-Identifier: Apache-2.0
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from responses_api_agents.metasyn_native import app


@pytest.mark.asyncio
async def test_disabled_launch_does_not_touch_creator():
    with pytest.raises(ValueError, match='qualified-runtime'):
        await app.MetaSynAgent.run(SimpleNamespace(config=SimpleNamespace(inference_enabled=False)), None, None)


@pytest.mark.asyncio
async def test_parent_passes_scoped_bridge_and_preserves_native_result(tmp_path, monkeypatch):
    data=tmp_path/'data';(data/'rag/203').mkdir(parents=True)
    (data/'rag/203/prompt.txt').write_text('frozen prompt')
    refs=tmp_path/'references.jsonl';refs.write_text('reference')
    output=tmp_path/'output';output.mkdir()
    config=SimpleNamespace(output_root=output,materialized_root=data,reference_file=refs,
        runner_script=Path(app.__file__).with_name('native_runner.py'),native_python=sys.executable,creator_source_root=data)
    events=[]
    @asynccontextmanager
    async def bridge(socket, **kwargs):
        events.append(('open',kwargs))
        try: yield
        finally: events.append(('close',None))
    async def creator(argv, out, *, timeout_s, env):
        assert env['METASYN_BRIDGE_LINEAGE']=='rollout'
        assert env['METASYN_DATA']==str(data)
        assert env['OMP_NUM_THREADS']=='1'
        assert env['HF_HUB_OFFLINE']=='1'
        assert timeout_s==3600
        assert argv[0]==sys.executable
        identity=json.loads((out/'identity.json').read_text())
        assert identity['review_id']==203 and identity['model']==app.MODEL
        assert events[0][1]['expected_model']==app.MODEL
        (out/'evaluation.json').write_text(json.dumps({'status':'succeeded','metrics':{'inclusion_f1':0.8}}))
    monkeypatch.setattr(app,'scientific_assets',lambda root, review_id: {'review_id':review_id})
    monkeypatch.setattr(app,'creator_identity',lambda root: {'root':str(root),'files':{}})
    monkeypatch.setattr(app,'bridge',bridge)
    monkeypatch.setattr(app,'run_creator',creator)
    out,result=await app.MetaSynAgent._evaluate(SimpleNamespace(config=config),203,'http://scoped/v1','rollout')
    assert result['metrics']['inclusion_f1']==0.8
    assert [x[0] for x in events]==['open','close']
    with pytest.raises(FileExistsError):
        await app.MetaSynAgent._evaluate(SimpleNamespace(config=config),203,'http://scoped/v1','rollout')
