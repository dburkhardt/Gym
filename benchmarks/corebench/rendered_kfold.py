# SPDX-License-Identifier: Apache-2.0
"""Validate CORE674's emitted fold metrics and original figures, never visual answers."""

import hashlib
import json
import math
import re
import struct
import sys
import zlib
from pathlib import Path


CLASSIFIERS = ("XGboost", "SVM", "DT", "RF")
METRICS = ("Acc", "Se", "PP")
FIGURES = ("Amplitude.png", "Box.png", "Box_1.png", "number_class.png", "zhu_1.png", "zhu_2.png")
NUMBER = r"(?:0|1)(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?"


def extract(text):
    prefixes = tuple(f"{m}-{c}:" for m in METRICS for c in CLASSIFIERS) + tuple(c + ":" for c in CLASSIFIERS)
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith(prefixes)]
    expected = [f"{m}-{c}:" for m in METRICS for c in CLASSIFIERS] + [c + ":" for c in CLASSIFIERS]
    if len(lines) != 16 or [line.split(":", 1)[0] + ":" for line in lines] != expected:
        raise ValueError("Missing, duplicated or reordered creator metrics")
    folds = {}
    for prefix, line in zip(expected[:12], lines[:12]):
        match = re.fullmatch(re.escape(prefix) + r" \[(" + NUMBER + r"(?:, " + NUMBER + r"){9})\]", line)
        if not match:
            raise ValueError("Expected exactly ten finite creator fold values")
        values = [float(x) for x in match[1].split(", ")]
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in values):
            raise ValueError("Fold value outside metric range")
        folds[prefix[:-1]] = values
    means = {}
    for classifier, line in zip(CLASSIFIERS, lines[12:]):
        match = re.fullmatch(re.escape(classifier) + r": (" + NUMBER + r") (" + NUMBER + r") (" + NUMBER + r")", line)
        if not match:
            raise ValueError("Malformed creator mean record")
        means[classifier] = {}
        for metric, raw in zip(METRICS, match.groups()):
            value = float(raw)
            recomputed = sum(folds[f"{metric}-{classifier}"]) / 10
            if not math.isfinite(value) or not math.isclose(value, recomputed, rel_tol=0, abs_tol=1e-12):
                raise ValueError("Creator mean differs from its ten emitted folds")
            means[classifier][metric] = value
    return {
        "folds": folds,
        "means": means,
        "scope": "Emitted fold arithmetic only; not independent predictions or visual answers.",
    }


def png_metadata(path):
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Invalid PNG signature")
    offset, compressed, header, ended = 8, bytearray(), None, False
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError("Truncated PNG")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        body = data[offset + 8 : offset + 8 + length]
        if offset + 12 + length > len(data):
            raise ValueError("Truncated PNG chunk")
        crc = struct.unpack(">I", data[offset + 8 + length : offset + 12 + length])[0]
        if zlib.crc32(kind + body) & 0xFFFFFFFF != crc:
            raise ValueError("PNG CRC mismatch")
        if kind == b"IHDR":
            if header is not None or length != 13:
                raise ValueError("Invalid PNG header")
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            compressed.extend(body)
        elif kind == b"IEND":
            ended = True
            if length != 0 or offset + 12 != len(data):
                raise ValueError("Invalid PNG end")
        offset += 12 + length
    if not header or not ended:
        raise ValueError("Incomplete PNG")
    width, height, depth, color, compression, filtering, interlace = header
    if not (
        0 < width * height <= 100_000_000
        and depth == 8
        and color in (0, 2, 3, 4, 6)
        and compression == filtering == interlace == 0
    ):
        raise ValueError("Unsupported or oversized PNG")
    stride = width * {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color]
    limit = (stride + 1) * height
    decoder = zlib.decompressobj()
    raw = decoder.decompress(compressed, limit + 1)
    if (
        len(raw) != limit
        or not decoder.eof
        or decoder.unused_data
        or any(raw[i] > 4 for i in range(0, limit, stride + 1))
    ):
        raise ValueError("Invalid PNG scanlines")
    return {"width": width, "height": height, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def validate(text, results):
    report = extract(text)
    found = {p.name for p in results.glob("*.png")}
    if found != set(FIGURES):
        raise ValueError("Missing or unexpected original figure set")
    report["figures"] = {name: png_metadata(results / name) for name in FIGURES}
    report["figure_scope"] = (
        "Original scripts include fixed published arrays; figures are not fresh classifier predictions."
    )
    return report


if __name__ == "__main__":
    Path(sys.argv[3]).write_text(json.dumps(validate(Path(sys.argv[1]).read_text(), Path(sys.argv[2])), indent=2))
