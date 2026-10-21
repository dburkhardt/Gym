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
