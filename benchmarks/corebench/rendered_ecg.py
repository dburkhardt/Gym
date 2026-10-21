# SPDX-License-Identifier: Apache-2.0
"""Parse creator CORE020 output; these are emitted metrics, not recomputed predictions."""

import json
import re
import sys
from pathlib import Path


METRICS = {"sen": "sensitivity", "spe": "specificity", "acc": "accuracy", "auc": "AUC"}


def extract(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 11 or not re.fullmatch(r"running time: [0-9]+\.[0-9]{3}s", lines[0]):
        raise ValueError("Incomplete or ambiguous creator metric output")
    report = {}
    for offset, level in [(1, "sample"), (6, "patient")]:
        if lines[offset] != level + " level result":
            raise ValueError("Missing or reordered metric section")
        for line, (short, name) in zip(lines[offset + 1 : offset + 5], METRICS.items()):
            match = re.fullmatch(re.escape(short) + r": ([0-9]+\.[0-9]{3})", line)
            if not match or not 0 <= float(match[1]) <= 1:
                raise ValueError("Invalid or reordered creator metric")
            report[f"Report the {name} at the '{level}-level'."] = float(match[1])
    return report


if __name__ == "__main__":
    Path(sys.argv[2]).write_text(json.dumps(extract(Path(sys.argv[1]).read_text())))
