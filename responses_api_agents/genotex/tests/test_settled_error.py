import hashlib
import json

import pytest

from responses_api_agents.genotex.checkpoint import remaining_budget, restore_checkpoint


@pytest.fixture
def packet(tmp_path):
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return {"filename": str(path), "digest": hashlib.sha256(path.read_bytes()).hexdigest()}

    paths = {}
    for name in ("public", "prediction", "workspace", "reference", "grading"):
        p = tmp_path / name
        p.mkdir()
        paths[name] = str(p)
    state = {
        "task": {"id": "frozen"},
        "paths": paths,
        "history": [{"role": "user", "content": "original"}],
        "pending": {"kind": "model", "action_id": "original-action", "step": 74},
        "next_step": 74,
        "elapsed_seconds": 708.0,
        "pending_calls": [],
        "trajectory": [{"tool": "already completed"}],
        "runtime_identity": {"image": "original"},
        "guard_config_sha256": "original-config",
    }
    checkpoint = tmp_path / "checkpoint.json"
    save(checkpoint, state)
    request = {"input": state["history"], "model": "original-model"}
    evidence = {
        "version": 1,
        "wire_encoding": "python-json-dumps-default-utf8-v1",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "action_id": "original-action",
    }
    evidence["request"] = save(tmp_path / "capture/request.json", request)
    evidence["response"] = save(tmp_path / "capture/response.json", {"error": {"code": "503"}})
    evidence["started"] = save(tmp_path / "capture/started.json", {"utc": "1970-01-01T00:00:01+00:00"})
    evidence["receipt"] = save(
        tmp_path / "receipt.json",
        {
            "request_artifact_id": "capture",
            "status": 503,
            "requested_model": "original-model",
            "utc": "1970-01-01T00:00:04+00:00",
        },
    )
    evidence["gateway"] = save(
        tmp_path / "gateway.json",
        [
            {
                "event": "started",
                "request_id": "unique",
                "request_sha256": hashlib.sha256(json.dumps(request).encode()).hexdigest(),
                "time": 2,
                "model": "original-model",
            },
            {"event": "finished", "request_id": "unique", "http_status": 503, "time": 3},
        ],
    )
    ep = tmp_path / "evidence.json"
    resolution = {
        "outcome": "settled_error",
        "checkpoint": str(checkpoint),
        "action_id": "original-action",
        "evidence": str(ep),
        "evidence_sha256": save(ep, evidence)["digest"],
    }
    return tmp_path, checkpoint, state, evidence, resolution, save


def restore(p):
    root, checkpoint, state, _, resolution, _ = p
    return restore_checkpoint(checkpoint, work_root=root, task=state["task"], resolution=resolution)


def test_explicit_settlement_preserves_original_and_budget(packet):
    before = packet[1].read_bytes()
    restored = restore(packet)
    assert restored["pending"] is None
    assert restored["history"] == packet[2]["history"]
    assert restored["trajectory"] == packet[2]["trajectory"]
    assert remaining_budget(restored, timeout_s=7200, max_steps=80) == (6492, 6)
    assert restored["settled_model_errors"][0]["outcome"] == "settled_error"
    assert packet[1].read_bytes() == before


@pytest.mark.parametrize(
    "case",
    [
        "wire",
        "terminal",
        "duplicate",
        "receipt",
        "response",
        "runtime",
        "checkpoint",
        "tool",
        "config",
        "encoding",
        "malformed_error",
    ],
)
def test_mismatched_or_incomplete_evidence_refused(packet, case):
    root, checkpoint, state, evidence, resolution, save = packet
    if case in ("runtime", "tool", "config"):
        if case == "runtime":
            state.pop("runtime_identity")
        elif case == "config":
            state.pop("guard_config_sha256")
        else:
            state["pending_calls"] = [{"id": "unfinished-tool"}]
        save(checkpoint, state)
        evidence["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    elif case == "encoding":
        evidence["wire_encoding"] = "unknown"
    elif case == "checkpoint":
        evidence["checkpoint_sha256"] = "0" * 64
    else:
        key = (
            "gateway"
            if case in ("wire", "terminal", "duplicate")
            else "response"
            if case == "malformed_error"
            else case
        )
        path = root / (
            "gateway.json" if key == "gateway" else "receipt.json" if key == "receipt" else "capture/response.json"
        )
        value = json.loads(path.read_text())
        if case == "wire":
            value[0]["request_sha256"] = "wrong"
        elif case == "terminal":
            value[1]["event"] = "transport_error"
        elif case == "duplicate":
            value.append(value[0])
        elif case == "receipt":
            value["request_artifact_id"] = "another-call"
        elif case == "malformed_error":
            value["error"] = "bad"
        else:
            value["error"]["code"] = "429"
        evidence[key] = save(path, value)
    resolution["evidence_sha256"] = save(root / "evidence.json", evidence)["digest"]
    with pytest.raises(ValueError):
        restore(packet)


def test_no_automatic_resolution(packet):
    root, checkpoint, state, *_ = packet
    with pytest.raises(ValueError, match="receipt review"):
        restore_checkpoint(checkpoint, work_root=root, task=state["task"])
