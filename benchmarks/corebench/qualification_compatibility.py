# SPDX-License-Identifier: Apache-2.0
"""One reviewed source transition; never rewrite or upgrade historical receipts.

188bc960c -> b395eb6de changes only dockerfile()'s explicit bionic allowlist
for capsule1906954. Capsule8197429's default recipe and all probes are identical.
The full current implementation hash must match this review before using it.
"""

import hashlib
import json
from pathlib import Path


PRIOR_IMPLEMENTATION = "c6e4ce4f173d80780107693cb272307b16952b2492720744b64a358ba57c016d"
REVIEWED_IMPLEMENTATION = "3dd5667fcd8de0d360df04d85d074ff6856dd92c8a41b2fb248432c77c87a403"
PRIOR_QUALIFICATION = "950b338f7d7210f3b40636524c2c0a57ce6059c890ccbe1831fc97a30fe3bb3b"


def reviewed_compatibility(row, receipt):
    from benchmarks.corebench import image_controls as controls
    from benchmarks.corebench.prepare import RUNTIME_SHA256

    transition = reviewed_313_addition(row, receipt)
    if transition is not None:
        return transition
    if (
        row["id"] != "capsule-8197429"
        or row.get("harness_install_variant") is not None
        or receipt.get("qualification_sha256") != PRIOR_QUALIFICATION
        or hashlib.sha256(Path(controls.__file__).read_bytes()).hexdigest() != REVIEWED_IMPLEMENTATION
    ):
        return None
    payload = {
        "implementation_sha256": PRIOR_IMPLEMENTATION,
        "schema": "corebench-image-qualification-v2",
        "recipe": controls.recipe_digest(row),
        "source": controls.source_digest(row),
        "instruction": row["instruction_sha256"],
        "reference": row["reference_sha256"],
        "scorer": row["scorer_sha256"],
        "harness_archive": RUNTIME_SHA256,
        "scientific_probe": controls.SCIENTIFIC_PROBE,
        "harness_probe": controls.HARNESS_PROBE,
    }
    if hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != PRIOR_QUALIFICATION:
        return None
    return {
        "review": "core819-default-bionic-allowlist-only-v1",
        "original_qualification_sha256": PRIOR_QUALIFICATION,
        "original_implementation_sha256": PRIOR_IMPLEMENTATION,
        "reviewed_implementation_sha256": REVIEWED_IMPLEMENTATION,
    }


# Exact eight-line dockerfile allowlist addition only; no probes or previous recipes changed.
PRE_313_ENGINE = "e6da1cd94c28988ee19adbf20b01a4da47e1cda3ae3d6fb032e865f909d9e8f7"
POST_313_ENGINE = "e756392591a45a0cf3afb685b40ee5f552dcd9908b59ed24a5e56352d16b8417"
PRE_313_CAPSULES = frozenset({"capsule-1683542", "capsule-6746514"})


def reviewed_313_addition(row, receipt):
    from benchmarks.corebench import image_controls as controls
    from benchmarks.corebench.prepare import RUNTIME_SHA256

    if (
        row["id"] not in PRE_313_CAPSULES
        or hashlib.sha256(Path(controls.__file__).read_bytes()).hexdigest() != POST_313_ENGINE
    ):
        return None
    try:
        recipe = controls.recipe_digest(row)
    except ValueError:
        return None
    payload = {
        "implementation_sha256": PRE_313_ENGINE,
        "schema": "corebench-image-qualification-v2",
        "recipe": recipe,
        "source": controls.source_digest(row),
        "instruction": row["instruction_sha256"],
        "reference": row["reference_sha256"],
        "scorer": row["scorer_sha256"],
        "harness_archive": RUNTIME_SHA256,
        "scientific_probe": controls.SCIENTIFIC_PROBE,
        "harness_probe": controls.HARNESS_PROBE,
    }
    prior = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if receipt.get("qualification_sha256") != prior:
        return None
    return {
        "review": "core313-exact-allowlist-addition-v1",
        "original_qualification_sha256": prior,
        "original_implementation_sha256": PRE_313_ENGINE,
        "reviewed_implementation_sha256": POST_313_ENGINE,
    }
