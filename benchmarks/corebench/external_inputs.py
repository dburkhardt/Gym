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
    from benchmarks.corebench import model_cache

    if row.get("external_input_variant") == model_cache.VARIANT:
        return model_cache.recipe(row)
    if row.get("external_input_variant") is None:
        return ""
    if (
        row["external_input_variant"] not in (VARIANT, CLOSURE_VARIANT)
        or row["id"] != "capsule-8912293"
        or row["published_image"] != CREATOR_IMAGE
        or row.get("filesystem_layout_variant") != "8912293-relative-data-v1"
    ):
        raise ValueError("Unsupported external input variant")
    if row["external_input_variant"] == CLOSURE_VARIANT:
        return (
            "# explicit current-upstream external input closure; historical identity unverified\n"
            'RUN test "$HOME" = /root && test ! -e /root/.Rprofile && test ! -L /root/.Rprofile\n'
            + "".join(
                f"COPY {name} /opt/corebench-external/{name}\n"
                f"RUN echo '{sha}  /opt/corebench-external/{name}' | sha256sum -c -\n"
                for name, sha in CLOSURE_INPUTS.items()
            )
            + "COPY external-input.Rprofile /root/.Rprofile\n"
        )
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
    from benchmarks.corebench import model_cache

    if row["external_input_variant"] == model_cache.VARIANT:
        model_cache.stage(source, destination)
        return
    if row["external_input_variant"] == CLOSURE_VARIANT:
        if not source.is_dir():
            raise ValueError("Complete external input variant requires a directory")
        with TemporaryDirectory(dir=destination.parent) as temporary:
            verified = Path(temporary)
            for name, sha in CLOSURE_INPUTS.items():
                stage_file(source / name, verified / name, expected_sha256=sha)
            for name in CLOSURE_INPUTS:
                os.replace(verified / name, destination / name)
        (destination / "external-input.Rprofile").write_text(closure_profile())
        return
    with TemporaryDirectory(dir=destination.parent) as temporary:
        checked = Path(temporary) / "checked.zip"
        stage_file(source, checked, expected_sha256=INPUT_SHA256)
        os.replace(checked, destination / "external-input.zip")
    (destination / "external-input.Rprofile").write_text(PROFILE)


# Complete dynamic-input closure used by the original five-script workflow.
# These are current upstream bytes, not proof of the historical release inputs.
# Diagnostic only: 2016 column 27 is now SQ_COLIGACAO, but the original helper
# labels it as nominal votes. This closure is NOT scientifically qualified;
# do not admit a full workflow merely because all cache/parser probes succeed.
CLOSURE_VARIANT = "8912293-tse-all-current-v2"
CLOSURE_INPUTS = {
    "perfil_eleitorado_2016.zip": INPUT_SHA256,
    "votacao_partido_munzona_2004.zip": "dfc48a3f3a6098b4e0b365bfc71d0699f2d378682fc2b3150efe549b9ee48cc3",
    "votacao_partido_munzona_2008.zip": "37de8e24af94fb726002fbb6bc3e1bc27fc51d75116573a5be2a25da4c786bb2",
    "votacao_partido_munzona_2012.zip": "8ccb50040abf361aa451df4b584fa7b0e636cd114026d1f311be18c92d9b7ec1",
    "votacao_partido_munzona_2016.zip": "7cf2353594ee332b2cc8e2f2edb6c25e1da06f22d8125a2a9278967e264fa8d5",
}


def closure_profile() -> str:
    """Warm only the creator package's existing filename cache; no parser edits."""
    return (
        "local({\n"
        + "".join(
            f'  input <- "/opt/corebench-external/{name}"\n'
            '  checksum <- system2("sha256sum", input, stdout=TRUE)\n'
            f'  stopifnot(length(checksum) == 1L, substr(checksum, 1L, 64L) == "{sha}")\n'
            f'  stopifnot(file.copy(input, file.path(tempdir(), "{name}")))\n'
            for name, sha in CLOSURE_INPUTS.items()
        )
        + "})\n"
    )
