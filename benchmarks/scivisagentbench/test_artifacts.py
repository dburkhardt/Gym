# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os

import pytest

from benchmarks.scivisagentbench.artifacts import artifact_paths


def test_private_regular_artifact_is_enumerated_without_changing_permissions(tmp_path):
    p = tmp_path / 'screenshot.png'
    p.write_bytes(b'exact screenshot')
    p.chmod(0o600)
    assert artifact_paths(tmp_path) == ['screenshot.png']
    assert p.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize('kind', ['absolute', 'dangling', 'fifo'])
def test_artifact_export_refuses_non_regular_files(tmp_path, kind):
    p = tmp_path / 'unsafe'
    if kind == 'fifo':
        os.mkfifo(p)
    else:
        p.symlink_to('/etc/passwd' if kind == 'absolute' else tmp_path / 'missing')
    with pytest.raises(ValueError, match='symlink or special'):
        artifact_paths(tmp_path)
