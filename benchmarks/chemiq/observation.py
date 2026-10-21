"""Observe the pinned creator's swallowed dependency failures without retrying."""

import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class _Proxy:
    def __init__(self, target):
        self._target = target

    def __getattr__(self, name):
        return getattr(self._target, name)


class _Response(_Proxy):
    def __init__(self, target, events, phase):
        super().__init__(target)
        self.events = events
        self.phase = phase

    def raise_for_status(self):
        self.phase[0] = "http_response"
        try:
            return self._target.raise_for_status()
        except Exception:
            status = self._target.status_code
            self.events.append(
                {"kind": "http_error", "status": status, "dependency_failure": status == 429 or 500 <= status < 600}
            )
            raise
        finally:
            self.phase[0] = None

    def json(self, *args, **kwargs):
        self.phase[0] = "http_response"
        try:
            data = self._target.json(*args, **kwargs)
            if not isinstance(data, dict):
                self.events.append(
                    {"kind": "non_object_json", "dependency_failure": 200 <= self._target.status_code < 300}
                )
            return data
        except ValueError:
            self.events.append({"kind": "malformed_json", "dependency_failure": 200 <= self._target.status_code < 300})
            raise
        finally:
            self.phase[0] = None


class _Requests(_Proxy):
    def __init__(self, target, events, phase):
        super().__init__(target)
        self.events = events
        self.phase = phase

    def get(self, *args, **kwargs):
        self.events.append({"kind": "request", "dependency_failure": False})
        self.phase[0] = "http_request"
        try:
            return _Response(self._target.get(*args, **kwargs), self.events, self.phase)
        except (self._target.exceptions.ConnectionError, self._target.exceptions.Timeout) as exc:
            self.events.append({"kind": type(exc).__name__, "dependency_failure": True})
            raise
        finally:
            self.phase[0] = None


class _Signal(_Proxy):
    def __init__(self, target, events, phase):
        super().__init__(target)
        self.events = events
        self.phase = phase
        self.installing = True

    def signal(self, number, handler):
        if self.installing and number == self._target.SIGALRM:
            self.installing = False
            original = handler

            def observed(signum, frame):
                self.events.append(
                    {
                        "kind": "creator_alarm",
                        "phase": self.phase[0] or "local_verifier",
                        "dependency_failure": self.phase[0] is not None,
                        "verifier_failure": self.phase[0] is None,
                    }
                )
                return original(signum, frame)

            handler = observed
        return self._target.signal(number, handler)


def check(module, verifier, task, parsed, *, source_pin):
    actual_question = verifier.question_dict[task["uuid"]]
    if task["verifier_metadata"] != actual_question:
        raise ValueError("ChemIQ supplied reference differs from pinned creator question")
    events = []
    requests, signal = module.requests, module.signal
    cached = parsed in verifier._opsin_cache if isinstance(parsed, str) else False
    evidence = {
        "source_pin": source_pin,
        "creator_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        "observer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "task_sha256": digest(task),
        "actual_question_sha256": digest(actual_question),
        "question_file_sha256": hashlib.sha256(Path(verifier._question_file).read_bytes()).hexdigest(),
        "parsed_answer_sha256": digest(parsed),
        "cache_hit": cached,
        "cached_value_sha256": digest(verifier._opsin_cache[parsed]) if cached else None,
        "events": events,
    }
    phase = [None]
    module.requests, module.signal = _Requests(requests, events, phase), _Signal(signal, events, phase)
    try:
        result = verifier.check_answer(task["uuid"], parsed)
    finally:
        module.requests, module.signal = requests, signal
    return result, evidence
