# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Narrow compatibility fix for the pinned Harbor single-step regrade path.

Harbor 96ff9c6 preserves links for multistep replay but dereferences them for
single-step replay. Image-internal absolute links are normally dangling on the
controller; dereferencing also risks importing unrelated controller files.
Remove this shim once the pinned upstream implements link-preserving replay.
"""

import hashlib
import inspect


_UPSTREAM_SEED_SHA256 = "149d426d2a23b72076eb64f595993a1f21a19c002f5edd0c01eb93ac0c333127"


def _seed_preserving_links(self):
    self._replay_recorded_outputs(
        step_name=None,
        declared_artifacts=[*self.task.config.artifacts, *self.config.artifacts],
        preserve_symlinks=True,
    )


def install_regrade_compat():
    """Install only on the reviewed upstream implementation; fail closed on drift."""
    from harbor.trial.regrade import RegradeTrial

    current = RegradeTrial._seed_from_source
    if current is _seed_preserving_links:
        return
    digest = hashlib.sha256(inspect.getsource(current).encode()).hexdigest()
    if digest != _UPSTREAM_SEED_SHA256:
        raise RuntimeError("Harbor regrade implementation changed; review symlink compatibility before replay")
    RegradeTrial._seed_from_source = _seed_preserving_links
