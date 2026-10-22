# SPDX-License-Identifier: Apache-2.0
"""Explicit current-upstream input variant; never historical-data equivalence."""

import hashlib
from pathlib import Path


VARIANT = "8912293-tse-2016-current-v1"
INPUT_SHA256 = "0e511c46a2e22c1766c21dd5d8d5d9f50596e1316cc8243a52af74b384784ddd"
CREATOR_IMAGE = (
    "registry.codeocean.com/published/2a0bef51-a61e-4c00-82c5-c79023c606d3"
    "@sha256:e72a3d557a31640b2340714199641b1674925aa5039ceb3e36ef717126654fca"
)
PROFILE = f'''local({{
  input <- "/opt/corebench-external/perfil_eleitorado_2016.zip"
  checksum <- system2("sha256sum", input, stdout=TRUE)
  stopifnot(length(checksum) == 1L,
            substr(checksum, 1L, 64L) == "{INPUT_SHA256}")
  stopifnot(file.copy(input, file.path(tempdir(), "perfil_eleitorado_2016.zip")))
}})
'''


def recipe(row: dict[str, object]) -> str:
    """Return an opt-in cache recipe bound to the exact capsule and input."""
    if row.get("external_input_variant") is None:
        return ""
    if (
        row["external_input_variant"] != VARIANT
        or row["id"] != "capsule-8912293"
        or row["published_image"] != CREATOR_IMAGE
        or row.get("filesystem_layout_variant") != "8912293-relative-data-v1"
    ):
        raise ValueError("Unsupported external input variant")
    profile_sha = hashlib.sha256(PROFILE.encode()).hexdigest()
    return (
        f"# external-input-sha256={INPUT_SHA256}; profile-sha256={profile_sha}\n"
        'RUN test "$HOME" = /root && test ! -e /root/.Rprofile && test ! -L /root/.Rprofile\n'
        "COPY external-input.zip /opt/corebench-external/perfil_eleitorado_2016.zip\n"
        f"RUN echo '{INPUT_SHA256}  /opt/corebench-external/perfil_eleitorado_2016.zip' | sha256sum -c -\n"
        "COPY external-input.Rprofile /root/.Rprofile\n"
    )


def stage_files(row: dict[str, object], source: Path | None, destination: Path) -> None:
    """Stage verified bytes only; no download and no edit to creator scripts.

    Caller supplies a private context under its existing atomic staging lock.
    A temporary cache keeps staging lock files outside the Docker COPY tree.
    """
    import os
    from tempfile import TemporaryDirectory

    from nemo_gym.dataset_staging import stage_file

    if not recipe(row):
        if source is not None:
            raise ValueError("External input supplied without explicit variant")
        return
    if source is None:
        raise ValueError("Explicit variant requires the pinned external input")
    with TemporaryDirectory(dir=destination.parent) as temporary:
        checked = Path(temporary) / "checked.zip"
        stage_file(source, checked, expected_sha256=INPUT_SHA256)
        os.replace(checked, destination / "external-input.zip")
    (destination / "external-input.Rprofile").write_text(PROFILE)
