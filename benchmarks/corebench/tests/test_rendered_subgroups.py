# SPDX-License-Identifier: Apache-2.0
import pytest
from benchmarks.corebench.rendered_subgroups import extract


def rendered(value="2.75"):
    return "".join(
        f"<pre>dat |&gt; split(~{category})</pre><pre>## ${group}\n## | N (studies)| Delta| se|\n## |---:|---:|---:|\n## |3|{value}|0.1|</pre>"
        for category, group in [("pop_categories", "adults"), ("country_categories", "Italy")]
    )


def test_computed_values_not_reference_constants():
    assert set(extract(rendered()).values()) == {2.75}


@pytest.mark.parametrize(
    "html",
    [
        "",
        rendered() + rendered(),
        rendered().replace("$Italy", "$Other"),
        rendered("nan"),
        rendered().replace("Delta", "wrong"),
    ],
)
def test_missing_ambiguous_invalid_refused(html):
    with pytest.raises(ValueError):
        extract(html)
