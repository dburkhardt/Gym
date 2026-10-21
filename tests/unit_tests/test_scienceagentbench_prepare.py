import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.scienceagentbench.prepare import digest, enumerated_paths, format_task, load_frozen, prepare


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / "scienceagentbench-upstream"
PARQUET = ROOT.parents[1] / "outputs/broad-eval-campaign/scientific-agents/scienceagentbench-verified.parquet"


@pytest.fixture
def frozen():
    if not SOURCE.is_dir() or not PARQUET.is_file():
        pytest.skip("Retained pinned creator source and verified parquet required")
    return load_frozen(SOURCE, PARQUET)


def test_exact_frozen_fifty_and_creator_format_parity(frozen):
    manifest, rows = frozen
    assert len(rows) == len({row[1]["instance_id"] for row in rows}) == 50
    assert manifest["source_revision"] != manifest["dataset_revision"]
    tree = ast.parse((SOURCE / "run_infer.py").read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "format_task_dict"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "pinned-format-task", "exec"), namespace)
    for selected, row in rows:
        for knowledge in (False, True):
            assert format_task(row, knowledge) == namespace["format_task_dict"](
                row, SimpleNamespace(datasets_path="benchmark/datasets/", use_knowledge=knowledge)
            )
        assert selected["instance_id"] == row["instance_id"]
        assert not {"gold_program_name", "eval_script_name"} & format_task(row).keys()


def assets_fixture(root, rows):
    # Deliberately synthetic prerequisite files, not a scientific runtime.
    for _, row in rows:
        directory = root / format_task(row)["dataset_path"].removeprefix("benchmark/")
        directory.mkdir(parents=True, exist_ok=True)
        for relative, is_directory in enumerated_paths(row["dataset_folder_tree"]):
            path = root / relative
            if is_directory:
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture only")
        for relative in ("gold_programs/" + row["gold_program_name"], "eval_programs/" + row["eval_script_name"]):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# prerequisite fixture only\n")


def test_missing_assets_before_publication(tmp_path, frozen):
    with pytest.raises(ValueError, match="Missing required creator asset"):
        prepare(SOURCE, PARQUET, tmp_path / "missing", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_prerequisite_inventory_and_reference_separation(tmp_path, frozen):
    assets = tmp_path / "assets"
    assets_fixture(assets, frozen[1])
    output = prepare(SOURCE, PARQUET, assets, tmp_path / "out")
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 50
    receipt = json.loads((output.parent / "asset-preflight.json").read_text())
    assert receipt["status"] == "required_paths_present_not_runtime_or_archive_provenance_qualified"
    assert not any("gold_program" in json.dumps(row["creator_task"]) for row in rows)
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        prepare(SOURCE, PARQUET, assets, output.parent)
    assert output.read_bytes() == original


def test_source_tamper_and_asset_escape_refused(tmp_path, frozen):
    import shutil

    source = tmp_path / "source"
    for relative in frozen[0]["source_files"]:
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / relative, target)
    (source / "agent.py").write_text("# changed")
    with pytest.raises(ValueError, match="Creator source changed"):
        load_frozen(source, PARQUET)
    assets = tmp_path / "assets"
    assets_fixture(assets, frozen[1])
    path = assets / "gold_programs" / frozen[1][0][1]["gold_program_name"]
    path.unlink()
    path.symlink_to(SOURCE / "agent.py")
    with pytest.raises(ValueError, match="symlinks"):
        prepare(SOURCE, PARQUET, assets, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_enumerated_missing_file_and_expected_checksum_rejection(tmp_path, frozen):
    assets = tmp_path / "assets"
    assets_fixture(assets, frozen[1])
    path = next(
        assets / relative
        for relative, directory in enumerated_paths(frozen[1][0][1]["dataset_folder_tree"])
        if not directory
    )
    old = path.read_bytes()
    path.unlink()
    with pytest.raises(ValueError, match="Missing required creator asset"):
        prepare(SOURCE, PARQUET, assets, tmp_path / "out")
    path.write_bytes(old)
    expected = tmp_path / "expected.json"
    expected.write_text(
        json.dumps({p.relative_to(assets).as_posix(): digest(p) for p in assets.rglob("*") if p.is_file()})
    )
    expected_sha = digest(expected)
    prepare(SOURCE, PARQUET, assets, tmp_path / "valid", asset_manifest=expected, asset_manifest_sha256=expected_sha)
    path.write_text("tampered")
    with pytest.raises(ValueError, match="Asset checksum changed"):
        prepare(SOURCE, PARQUET, assets, tmp_path / "out", asset_manifest=expected, asset_manifest_sha256=expected_sha)
    assert not (tmp_path / "out").exists()
