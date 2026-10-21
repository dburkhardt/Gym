# SPDX-License-Identifier: Apache-2.0
import pytest

from benchmarks.corebench.harness_install import select_sources, validate_packages
from benchmarks.corebench.image_controls import dockerfile, recipe_digest


def test_sources_exclude_third_party_and_untrusted_options():
    safe = "deb http://archive.ubuntu.com/ubuntu bionic main universe"
    text = "\n".join(
        [
            safe,
            "deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran40/",
            "deb http://deb.codeocean.com/rstudio-server-bionic/ ubuntu main",
            "deb [trusted=yes] http://archive.ubuntu.com/ubuntu bionic main",
            "deb http://archive.ubuntu.com.evil/ubuntu bionic main",
            "deb http://archive.ubuntu.com/ubuntu focal main",
        ]
    )
    assert select_sources(text) == safe + "\n"
    with pytest.raises(ValueError, match="No exact"):
        select_sources("deb [trusted=yes] http://archive.ubuntu.com/ubuntu bionic main")


@pytest.mark.parametrize("after", [{"R": "2", "tmux": "1"}, {"tmux": "1"}, {"R": "1", "tmux": "1", "unexpected": "1"}])
def test_package_drift_removal_or_new_dependency_refused(after):
    with pytest.raises(ValueError):
        validate_packages({"R": "1"}, after)


def test_only_new_harness_packages_allowed():
    validate_packages(
        {"r-base": "1", "libtinfo5:amd64": "1"},
        {
            "r-base": "1",
            "libtinfo5:amd64": "1",
            "tmux": "2.6",
            "libevent-2.1-6:amd64": "2.1.8",
            "libutempter0": "1.1.6",
        },
    )


def test_recipe_variant_is_explicit_and_bound():
    row = {"id": "capsule-6133093", "published_image": "pinned", "runtime_environment_variant": {}}
    historical = recipe_digest(row)
    assert "apt-get update &&" in dockerfile(row)
    variant = {**row, "harness_install_variant": "ubuntu-bionic-isolated-v1"}
    assert recipe_digest(variant) != historical
    assert "harness_install_sha256=" in dockerfile(variant)
    assert "COPY harness_install.py" in dockerfile(variant)
    with pytest.raises(ValueError, match="Unsupported"):
        dockerfile({**variant, "id": "capsule-other"})


def test_1906954_variant_requires_exact_published_identity():
    import json
    from pathlib import Path

    from benchmarks.corebench.image_controls import dockerfile, recipe_digest

    rows = json.loads((Path(__file__).parents[1] / "frozen20.json").read_text())["rows"]
    original = next(row for row in rows if row["id"] == "capsule-1906954")
    variant = {**original, "harness_install_variant": "ubuntu-bionic-isolated-v1"}
    assert recipe_digest(original) != recipe_digest(variant)
    assert "corebench-harness-install.py" in dockerfile(variant)
    assert "apt-get update &&" not in dockerfile(variant)
    for key, value in [("published_image", "unreviewed"), ("id", "capsule-other")]:
        with pytest.raises(ValueError, match="Unsupported capsule-specific"):
            dockerfile({**variant, key: value})
    assert "harness_install_variant" not in original


def test_1683542_exact_variant_and_observed_sources():
    import json
    from pathlib import Path

    rows = json.loads((Path(__file__).parents[1] / "frozen20.json").read_text())["rows"]
    original = next(row for row in rows if row["id"] == "capsule-1683542")
    variant = {**original, "harness_install_variant": "ubuntu-bionic-isolated-v1"}
    assert recipe_digest(variant) != recipe_digest(original)
    assert "COPY harness_install.py" in dockerfile(variant)
    assert "apt-get update &&" not in dockerfile(variant)
    for key, value in [
        ("id", "capsule-1683543"),
        ("published_image", original["published_image"][:-1] + "0"),
        ("harness_install_variant", "unreviewed"),
    ]:
        with pytest.raises(ValueError, match="Unsupported capsule-specific"):
            dockerfile({**variant, key: value})
    ubuntu = "deb http://archive.ubuntu.com/ubuntu/ bionic main restricted"
    observed = "\n".join(
        [
            ubuntu,
            "deb http://deb.codeocean.com/rstudio-server-bionic/ ubuntu main",
            "deb http://deb.codeocean.com/shiny-server-trusty/ ubuntu main",
            "deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/",
        ]
    )
    assert select_sources(observed) == ubuntu + "\n"
    assert "harness_install_variant" not in original


def test_3137115_exact_signed_variant():
    import json
    from pathlib import Path

    rows = json.loads((Path(__file__).parents[1] / "frozen20.json").read_text())["rows"]
    row = next(r for r in rows if r["id"] == "capsule-3137115")
    variant = dict(row, harness_install_variant="ubuntu-bionic-isolated-v1")
    assert "COPY harness_install.py" in dockerfile(variant)
    assert "harness_install_variant" not in row
    with pytest.raises(ValueError):
        dockerfile(dict(variant, published_image=row["published_image"] + "x"))
    with pytest.raises(ValueError):
        dockerfile(dict(variant, id="capsule-other"))
    ubuntu = "deb http://archive.ubuntu.com/ubuntu/ bionic main restricted"
    assert (
        select_sources(ubuntu + "\ndeb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran40/") == ubuntu + "\n"
    )
