# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

from benchmarks.corebench.rendered_kfold import CLASSIFIERS, FIGURES, METRICS, extract, png_metadata, validate


def stdout():
    lines = ["+ python -u NewData_ML_Kfold.py", "(11500, 180)"]
    for metric in METRICS:
        for classifier in CLASSIFIERS:
            lines.append(f"{metric}-{classifier}: " + str([0.9] * 10))
    for classifier in CLASSIFIERS:
        lines.append(f"{classifier}: " + " ".join([str(sum([0.9] * 10) / 10)] * 3))
    return "\n".join(lines + ["+ wait", "start: (11500, 178)"])


def png():
    def chunk(name, body):
        return struct.pack(">I", len(body)) + name + body + struct.pack(">I", zlib.crc32(name + body) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\0\xff\0\0"))
        + chunk(b"IEND", b"")
    )


def test_exact_creator_print_grammar_and_arithmetic():
    value = extract(stdout())
    assert len(value["folds"]) == 12
    assert len(value["means"]) == 4
    assert value["means"]["RF"]["Acc"] == pytest.approx(0.9)


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "nine", "eleven", "nan", "range", "mean", "order", "malformed"]
)
def test_invalid_measurements_refused(change):
    text = stdout()
    if change == "missing":
        text = "\n".join(l for l in text.splitlines() if not l.startswith("PP-RF:"))
    elif change == "duplicate":
        text += "\n" + next(l for l in text.splitlines() if l.startswith("Acc-RF:"))
    elif change == "nine":
        text = text.replace(str([0.9] * 10), str([0.9] * 9), 1)
    elif change == "eleven":
        text = text.replace(str([0.9] * 10), str([0.9] * 11), 1)
    elif change == "nan":
        text = text.replace("[0.9", "[nan", 1)
    elif change == "range":
        text = text.replace("[0.9", "[1.1", 1)
    elif change == "mean":
        text = "\n".join("XGboost: 0.8 0.9 0.9" if line.startswith("XGboost:") else line for line in text.splitlines())
    elif change == "order":
        text = (
            text.replace("Acc-XGboost:", "Acc-SWAP:")
            .replace("Acc-SVM:", "Acc-XGboost:")
            .replace("Acc-SWAP:", "Acc-SVM:")
        )
    else:
        text = text.replace("Acc-XGboost: [", "Acc-XGboost: nope [")
    with pytest.raises(ValueError):
        extract(text)


def test_original_six_figures_and_cli(tmp_path):
    for name in FIGURES:
        (tmp_path / name).write_bytes(png())
    value = validate(stdout(), tmp_path)
    assert len(value["figures"]) == 6
    assert value["figures"]["Box.png"]["sha256"] == hashlib.sha256(png()).hexdigest()
    log = tmp_path / "stdout.txt"
    log.write_text(stdout())
    output = tmp_path / "measurements.json"
    script = Path(__file__).parents[1] / "rendered_kfold.py"
    subprocess.run([sys.executable, str(script), str(log), str(tmp_path), str(output)], check=True)
    assert json.loads(output.read_text()) == value
    (tmp_path / "Box.png").unlink()
    with pytest.raises(ValueError, match="figure set"):
        validate(stdout(), tmp_path)


@pytest.mark.parametrize("data", [b"", b"PNG", png()[:-1], png() + b"junk", png().replace(b"IDAT", b"IDAX")])
def test_invalid_png_refused(tmp_path, data):
    path = tmp_path / "bad.png"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        png_metadata(path)


def test_original_workflow_and_isolated_parser_contract():
    script = (Path(__file__).parents[1] / "controls/6746514/solve.sh").read_text()
    assert "bash run >" in script
    assert "/opt/corebench-harness/bin/python3 /solution/rendered_kfold.py" in script
    assert "find -L /results" in script
    assert "rm " not in script
    assert "printf '{}\\n'" in script
