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
