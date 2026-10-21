# SPDX-License-Identifier: Apache-2.0
"""Observe one pinned creator evaluation; never compute replacement verdicts."""

import ast
import copy
import hashlib
import json
import marshal
import math
import numbers
import sys
import types
from pathlib import Path


SCORER_SHA256 = "88a132c1196bfd9f1f8cf32d23e64cb6e0b26d98437ab5ff72f5e77ef56b53bb"
COUNTERS = ("correct_written_answers", "correct_vision_answers")


def _layout(scorer):
    code = scorer.__code__
    data = Path(code.co_filename).read_bytes()
    if hashlib.sha256(data).hexdigest() != SCORER_SHA256:
        raise ValueError("unsupported_scorer_source")
    module = compile(data, code.co_filename, "exec")
    originals = [v for v in module.co_consts if isinstance(v, types.CodeType) and v.co_name == "eval_result_json"]
    if len(originals) != 1 or marshal.dumps(originals[0]) != marshal.dumps(code):
        raise ValueError("loaded_scorer_differs")
    function = next(n for n in ast.parse(data).body if isinstance(n, ast.FunctionDef) and n.name == code.co_name)
    loops = [
        n for n in ast.walk(function) if isinstance(n, ast.For) and ast.unparse(n.iter) == "reported_result.keys()"
    ]
    if len(loops) != 2:
        raise ValueError("unsupported_scorer_structure")
    loop = max(loops, key=lambda n: n.lineno)
    if len(loop.body) != 1 or not isinstance(loop.body[0], ast.If):
        raise ValueError("unsupported_scorer_structure")
    comparisons = {
        n.lineno
        for n in ast.walk(loop)
        if isinstance(n, ast.If)
        and any(isinstance(v, ast.AugAssign) for v in ast.walk(n))
        and isinstance(n.test, ast.Compare)
        and not any(isinstance(op, ast.In) for op in n.test.ops)
    }
    if len(comparisons) != 3:
        raise ValueError("unsupported_scorer_structure")
    return loop.lineno, loop.body[0].lineno, comparisons


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return value if math.isfinite(value) else {"nonfinite": str(value)}
    return value


class Observation:
    def __init__(self, scorer, references, reported):
        self.code = scorer.__code__
        self.loop, self.start, self.comparisons = _layout(scorer)
        self.references = copy.deepcopy(references)
        self.raw = copy.deepcopy(reported)
        self.events = {}
        self.active = None
        self.final = None
        self.error = None

    def finish(self, local):
        if self.active is not None:
            key, before = self.active
            self.events[key]["original_counter_increment"] = sum(local[k] for k in COUNTERS) - before
            self.active = None

    def observe(self, frame, event, arg):
        if frame.f_code is not self.code:
            return None
        try:
            local = frame.f_locals
            if event == "line":
                if frame.f_lineno == self.loop:
                    self.finish(local)
                if frame.f_lineno == self.start:
                    key = local["key"]
                    self.active = (key, sum(local[k] for k in COUNTERS))
                    self.events[key] = {"comparison_executed": False, "exception": None}
                if self.active is not None and frame.f_lineno in self.comparisons:
                    self.events[self.active[0]]["comparison_executed"] = True
            elif event == "exception" and self.active is not None:
                self.events[self.active[0]]["exception"] = arg[0].__name__
            elif event == "return":
                self.finish(local)
                self.final = copy.deepcopy(
                    {
                        k: local.get(k)
                        for k in (
                            "prediction_interval_bounds",
                            "reported_result",
                            "numeric_keys",
                            "string_keys",
                            "list_keys",
                        )
                    }
                )
        except Exception as error:
            self.error = type(error).__name__
        return self.observe

    def report(self, result):
        if self.error or self.final is None:
            raise ValueError("observer_state_unavailable")
        final = self.final
        items = []
        for key in self.references[0]:
            observed = self.events.get(key)
            kind = next((name for name in ("numeric", "string", "list") if key in (final[name + "_keys"] or [])), None)
            if key not in self.raw:
                status = "not_reported"
            elif observed is None:
                status = "not_reached"
            elif observed["exception"]:
                status = "comparison_exception"
            elif observed["comparison_executed"]:
                status = "evaluated"
            else:
                status = "unrecognized_reference_type"
            interval = (final["prediction_interval_bounds"] or {}).get(key)
            items.append(
                {
                    "question": key,
                    "category": "figure" if "fig" in key else "written",
                    "reference_type": kind,
                    "reported": key in self.raw,
                    "prediction_raw": self.raw.get(key),
                    "prediction_normalized": (final["reported_result"] or {}).get(key),
                    "reference_observations": [v.get(key) for v in self.references],
                    "numeric_interval": interval,
                    "interval_nonfinite": interval is not None and any(not math.isfinite(x) for x in interval),
                    "status": status,
                    "observation": observed,
                    "original_verdict": bool(observed["original_counter_increment"])
                    if status == "evaluated"
                    else None,
                }
            )
        return _json_value(
            {
                "status": "available",
                "original_components": result,
                "items": items,
                "unrecognized_report_keys": [k for k in self.raw if k not in self.references[0]],
            }
        )


def evaluate(scorer, references, reported):
    """Return the original result and best-effort diagnostics, calling scorer once."""
    observer = None
    diagnostic = {"status": "unavailable", "reason": "preexisting_trace"}
    previous = sys.gettrace()
    installed = False
    if previous is None:
        try:
            observer = Observation(scorer, references, reported)
            sys.settrace(observer.observe)
            installed = True
        except Exception as error:
            diagnostic = {"status": "unavailable", "reason": str(error)}
    try:
        result = scorer(references, reported)
    finally:
        if installed:
            sys.settrace(previous)
    if observer is not None and installed:
        try:
            diagnostic = observer.report(result)
        except Exception as error:
            diagnostic = {"status": "unavailable", "reason": str(error)}
    return result, diagnostic


def publish(diagnostic, destination, *, scorer_path, reference_path, report_path):
    """Only evaluator logs receive references; diagnostic I/O cannot change reward."""
    try:
        paths = {"scorer": scorer_path, "reference": reference_path, "report": report_path, "observer": __file__}
        diagnostic = {
            **diagnostic,
            "version": 1,
            "sha256": {
                name: hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).exists() else None
                for name, path in paths.items()
            },
        }
        Path(destination).write_text(json.dumps(diagnostic, allow_nan=False, indent=2))
    except Exception as error:
        try:
            print("CORE item diagnostics unavailable: " + type(error).__name__, file=sys.stderr)
        except Exception:
            pass
