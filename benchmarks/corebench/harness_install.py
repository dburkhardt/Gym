# SPDX-License-Identifier: Apache-2.0
"""Explicit bionic harness-only apt variant; never relax signature validation."""

import json
import re
import subprocess
import tempfile
from pathlib import Path


ALLOWED_NEW = {"tmux", "libevent-2.1-6", "libutempter0"}
SOURCE = re.compile(
    r"deb\s+https?://(?:archive|security)\.ubuntu\.com/ubuntu/?\s+(bionic(?:-updates|-security|-backports)?)\s+((?:main|restricted|universe|multiverse)(?:\s+(?:main|restricted|universe|multiverse))*)\s*"
)


def select_sources(text):
    lines = [line.strip() for line in text.splitlines() if SOURCE.fullmatch(line.strip())]
    if not lines:
        raise ValueError("No exact signed Ubuntu bionic source lines")
    return "\n".join(lines) + "\n"


def validate_packages(before, after):
    if any(after.get(name) != version for name, version in before.items()):
        raise ValueError("Existing package version/removal changed")
    unexpected = {name for name in after.keys() - before.keys() if name.split(":")[0] not in ALLOWED_NEW}
    if unexpected:
        raise ValueError("Unexpected new packages: " + ", ".join(sorted(unexpected)))
    if not any(name.split(":")[0] == "tmux" for name in after):
        raise ValueError("tmux is absent")


def inventory():
    result = subprocess.check_output(
        ["dpkg-query", "-W", "-f=${binary:Package}\t${Version}\t${db:Status-Status}\n"], text=True
    )
    return {
        name: version
        for name, version, status in (line.split("\t") for line in result.splitlines())
        if status == "installed"
    }


def install():
    selected = select_sources(Path("/etc/apt/sources.list").read_text())
    before = inventory()
    metadata = Path("/opt/corebench-build-metadata")
    metadata.mkdir(exist_ok=True)
    (metadata / "dpkg-before.json").write_text(json.dumps(before, sort_keys=True, indent=2))
    (metadata / "ubuntu-sources.list").write_text(selected)
    with tempfile.TemporaryDirectory(prefix="corebench-apt-") as temporary:
        root = Path(temporary)
        (root / "sources.list").write_text(selected)
        (root / "lists/partial").mkdir(parents=True)
        options = [
            "-o",
            "Dir::Etc::sourcelist=" + str(root / "sources.list"),
            "-o",
            "Dir::Etc::sourceparts=-",
            "-o",
            "Dir::State::lists=" + str(root / "lists"),
            "-o",
            "Acquire::AllowInsecureRepositories=false",
            "-o",
            "APT::Get::AllowUnauthenticated=false",
            "-o",
            "Acquire::AllowDowngradeToInsecureRepositories=false",
        ]
        subprocess.run(["apt-get", *options, "update"], check=True)
        subprocess.run(
            ["apt-get", *options, "install", "-y", "--no-upgrade", "--no-install-recommends", "tmux"], check=True
        )
    after = inventory()
    (metadata / "dpkg-after.json").write_text(json.dumps(after, sort_keys=True, indent=2))
    validate_packages(before, after)


if __name__ == "__main__":
    install()
