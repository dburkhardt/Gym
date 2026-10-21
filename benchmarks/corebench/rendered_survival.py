# SPDX-License-Identifier: Apache-2.0
"""Read the original CORE389 rendered PFS table; never evaluator answers."""

import json
import math
import sys
from html.parser import HTMLParser
from pathlib import Path


class RenderedTables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.events = []
        self.active = None
        self.text = ""

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        kind = "pre" if tag == "pre" else None
        if tag == "script" and "data-pagedtable-source" in attrs:
            if attrs.get("type") != "application/json":
                raise ValueError("Unsupported rendered table encoding")
            kind = "table"
        if kind:
            if self.active is not None:
                raise ValueError("Nested rendered blocks")
            self.active, self.text = kind, ""

    def handle_data(self, data):
        if self.active:
            self.text += data

    def handle_endtag(self, tag):
        if (tag == "pre" and self.active == "pre") or (tag == "script" and self.active == "table"):
            self.events.append((self.active, self.text))
            self.active = None


def extract(html):
    parser = RenderedTables()
    parser.feed(html)
    parser.close()
    if parser.active:
        raise ValueError("Truncated rendered block")
    anchors = [
        i
        for i, (kind, text) in enumerate(parser.events)
        if kind == "pre" and text.strip().splitlines()[-1:] == ["pfs_results"]
    ]
    if len(anchors) != 1 or anchors[0] + 1 >= len(parser.events):
        raise ValueError("Missing or ambiguous rendered pfs_results")
    kind, text = parser.events[anchors[0] + 1]
    if kind != "table":
        raise ValueError("Missing computed PFS table")
    table = json.loads(text)
    columns = table["columns"]
    keys = [str(c["name"][0]) for c in columns if c["label"] == ["p.value"] and len(c["name"]) == 1]
    if len(keys) != 1 or len({str(c["name"][0]) for c in columns}) != len(columns):
        raise ValueError("Missing or ambiguous unadjusted p.value column")
    rows = [r for r in table["data"] if r.get("_rn_") == "Lesions.Contoured"]
    if len(rows) != 1:
        raise ValueError("Missing or ambiguous Lesions.Contoured row")
    raw = rows[0][keys[0]]
    if isinstance(raw, bool):
        raise ValueError("Invalid p value")
    value = float(raw)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Invalid p value")
    return {"Report the p value for Lesions.Contoured.": value}


if __name__ == "__main__":
    Path(sys.argv[2]).write_text(json.dumps(extract(Path(sys.argv[1]).read_text())))
