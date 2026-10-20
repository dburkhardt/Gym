import json
from types import SimpleNamespace
import pytest
from benchmarks.moosechem import native


def client(tmp_path, create):
    c = object.__new__(native.RecordedClient)
    c.stage = 'control'
    c.trace = tmp_path / 'model-traces.jsonl'
    c.replay = None
    c.output_token_budget = 16384
    c.inner = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return c


def test_intent_exists_before_dispatch_and_matches_terminal(tmp_path):
    def create(**kwargs):
        intent = json.loads((tmp_path/'dispatch-intents.jsonl').read_text())
        assert intent['request'] == kwargs
        assert not (tmp_path/'model-traces.jsonl').exists()
        return SimpleNamespace(model=native.MODEL, model_dump=lambda: {'model':native.MODEL})
    c = client(tmp_path, create)
    c.create(model=native.MODEL, max_tokens=8192, messages=[])
    intent = json.loads((tmp_path/'dispatch-intents.jsonl').read_text())
    terminal = json.loads(c.trace.read_text())
    assert intent['request_id'] == terminal['request_id']
    assert 'response' in terminal


@pytest.mark.parametrize('fail_at, expected_calls', [(1,0),(3,1)])
def test_receipt_io_failure_escapes_native_retry(tmp_path, monkeypatch, fail_at, expected_calls):
    calls = []
    syncs = []
    real = native.os.fsync
    def fsync(fd):
        syncs.append(fd)
        if len(syncs) == fail_at:
            raise OSError('injected receipt failure')
        real(fd)
    monkeypatch.setattr(native.os, 'fsync', fsync)
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(model=native.MODEL, model_dump=lambda: {'model':native.MODEL})
    c = client(tmp_path, create)
    with pytest.raises(native.ReceiptAbort):
        for _ in range(3):
            try:
                c.create(model=native.MODEL,max_tokens=8192,messages=[])
            except Exception:
                continue
    assert len(calls) == expected_calls


def test_run_claim_refuses_second_start(tmp_path):
    native.claim_run(tmp_path)
    with pytest.raises(FileExistsError):
        native.claim_run(tmp_path)


@pytest.mark.parametrize('interrupt',[KeyboardInterrupt,SystemExit])
def test_interrupt_retains_unknown_outcome(tmp_path,interrupt):
    def create(**kwargs):
        raise interrupt()
    c=client(tmp_path,create)
    with pytest.raises(interrupt):
        c.create(model=native.MODEL,max_tokens=8192,messages=[])
    r=json.loads(c.trace.read_text())
    assert r['outcome']=='unknown' and r['requires_reconciliation']
    assert 'response' not in r

def test_remaining_two_excludes_completed38(tmp_path):
    from benchmarks.moosechem.remaining_two import tasks_from_sample
    p=tmp_path/'sample.jsonl';p.write_text('\n'.join(json.dumps({'id':i}) for i in [18,26,38,0,33]))
    assert tasks_from_sample(p)==[0,33]
    p.write_text('\n'.join(json.dumps({'id':i}) for i in [18,26,0,38,33]))
    with pytest.raises(AssertionError):tasks_from_sample(p)

def test_admission_routing_retains_native_no_retry(tmp_path,monkeypatch):
    import openai
    captured={}
    def fake(**kwargs):captured.update(kwargs);return SimpleNamespace()
    monkeypatch.setattr(openai,'OpenAI',fake)
    monkeypatch.setenv('MOOSE_ADMISSION_URL','http://127.0.0.1:49394/v1')
    native.RecordedClient(stage='evaluate',trace=tmp_path/'trace',base_url='http://127.0.0.1:49180/v1',api_key='mock')
    assert captured['base_url']=='http://127.0.0.1:49394/v1'
    assert captured['default_headers']['X-Campaign-Benchmark']=='MOOSE-Chem'
    assert captured['max_retries']==0
    monkeypatch.setenv('MOOSE_ADMISSION_URL','https://unapproved.invalid')
    with pytest.raises(AssertionError):native.RecordedClient(stage='evaluate',trace=tmp_path/'trace',base_url='http://127.0.0.1:49180/v1',api_key='mock')
