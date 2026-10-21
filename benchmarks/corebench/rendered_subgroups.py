# SPDX-License-Identifier: Apache-2.0
"""CORE190 control: read computed subgroup tables, never evaluator answers."""

import json
import math
import re
import sys
from html.parser import HTMLParser
from pathlib import Path


class Blocks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.blocks = []
        self.active = None

    def handle_starttag(self, tag, attrs):
        if tag == "pre":
            if self.active is not None:
                raise ValueError("Nested pre blocks")
            self.active = ""

    def handle_data(self, data):
        if self.active is not None:
            self.active += data

    def handle_endtag(self, tag):
        if tag == "pre" and self.active is not None:
            self.blocks.append(self.active)
            self.active = None


def extract(html):
    parser = Blocks()
    parser.feed(html)
    result = {}
    for category, group in [("pop_categories", "adults"), ("country_categories", "Italy")]:
        anchors = [
            i for i, text in enumerate(parser.blocks) if re.search(r"split\s*\(\s*~" + category + r"\s*\)", text)
        ]
        if len(anchors) != 1 or anchors[0] + 1 >= len(parser.blocks):
            raise ValueError("Missing or ambiguous computed subgroup block")
        text = parser.blocks[anchors[0] + 1]
        sections = re.split(r"(?m)^##\s+\$", text)
        matches = [s for s in sections[1:] if s.splitlines()[0].strip() == group]
        if len(matches) != 1:
            raise ValueError("Missing or ambiguous subgroup")
        lines = [line.removeprefix("##").strip() for line in matches[0].splitlines()[1:]]
        rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines if line.startswith("|")]
        if len(rows) != 3 or rows[0].count("Delta") != 1 or len(rows[0]) != len(rows[2]):
            raise ValueError("Malformed computed subgroup table")
        value = float(rows[2][rows[0].index("Delta")])
        if not math.isfinite(value):
            raise ValueError("Nonfinite subgroup estimate")
        result[f"Report the Delta value for {group}."] = value
    return result


if __name__ == "__main__":
    Path(sys.argv[2]).write_text(json.dumps(extract(Path(sys.argv[1]).read_text())))
