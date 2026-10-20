# SPDX-License-Identifier: Apache-2.0
import pytest

from benchmarks.corral.harbor_native import copy_state, run_phase


def test_saved_state_rejects_links_before_copy(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "link").symlink_to(tmp_path / "private")
    with pytest.raises(ValueError, match="symlinks"):
        copy_state(source, tmp_path / "copy")
    assert not (tmp_path / "copy").exists()


@pytest.mark.asyncio
async def test_scorer_without_saved_state_fails_before_allocation(tmp_path):
    with pytest.raises(ValueError, match="saved state"):
        await run_phase(
            task_id="make_1_lvl3",
            phase="scorer",
            snapshot=tmp_path / "missing",
            endpoint="",
            output=tmp_path / "output",
        )
    assert not (tmp_path / "output").exists()
