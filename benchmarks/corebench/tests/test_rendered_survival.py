# SPDX-License-Identifier: Apache-2.0
import json
import pytest
from benchmarks.corebench.rendered_survival import extract


def rendered(*, value="0.12", duplicate=False, label="p.value", anchor="pfs_results"):
    row = {"_rn_": "Lesions.Contoured", "4": value, "6": "0.6825"}
    table = {
        "columns": [
            {"label": [""], "name": ["_rn_"]},
            {"label": [label], "name": [4]},
            {"label": ["p.adjust"], "name": [6]},
        ],
        "data": [row, row] if duplicate else [row],
    }
    return (
        "<pre><code>"
        + anchor
        + '</code></pre><script data-pagedtable-source type="application/json">'
        + json.dumps(table)
        + "</script>"
    )


def test_unadjusted_value_from_computed_table():
    assert extract(rendered()) == {"Report the p value for Lesions.Contoured.": 0.12}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"duplicate": True},
        {"label": "p.adjust"},
        {"anchor": "os_results"},
        {"value": "NaN"},
        {"value": "1.2"},
        {"value": True},
    ],
)
def test_missing_ambiguous_or_invalid_refused(kwargs):
    with pytest.raises((ValueError, KeyError)):
        extract(rendered(**kwargs))


def test_duplicate_pfs_block_refused():
    with pytest.raises(ValueError):
        extract(rendered() + rendered())


def test_source_only_is_not_computed_output():
    with pytest.raises(ValueError):
        extract('<pre>pfs_results</pre><pre>{"p.value":0.12}</pre>')
