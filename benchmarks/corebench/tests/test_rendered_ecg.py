# SPDX-License-Identifier: Apache-2.0
import pytest

from benchmarks.corebench.rendered_ecg import extract


OUTPUT = """running time: 263.501s

sample level result
sen: 0.966
spe: 0.994
acc: 0.992
auc: 0.998

patient level result
sen: 1.000
spe: 0.972
acc: 0.978
auc: 0.998
"""


def test_creator_output():
    report = extract(OUTPUT)
    assert len(report) == 8
    assert report["Report the sensitivity at the 'sample-level'."] == 0.966
    assert report["Report the sensitivity at the 'patient-level'."] == 1
    assert report["Report the AUC at the 'patient-level'."] == 0.998


@pytest.mark.parametrize(
    "text",
    [
        "",
        OUTPUT + OUTPUT,
        OUTPUT.replace("spe: 0.994", "sen: 0.994"),
        OUTPUT.replace("0.994", "nan"),
        OUTPUT.replace("0.994", "1.001"),
        OUTPUT.replace("patient level result", "sample level result"),
        OUTPUT.replace("auc: 0.998\n", "", 1),
        OUTPUT + "unexpected result\n",
    ],
)
def test_refuse_malformed_or_ambiguous(text):
    with pytest.raises(ValueError):
        extract(text)


def test_solution_refuses_stale_output_before_workflow(tmp_path):
    import subprocess
    from pathlib import Path

    solution = Path(__file__).parents[1] / "controls/0201225/solve.sh"
    # Execute the actual precondition, substituting only its absolute output path.
    text = solution.read_text()
    checks = "\n".join(
        line for line in text.splitlines() if line.startswith("test !") and "/results/output.txt" in line
    )
    assert checks and text.index(checks.splitlines()[0]) < text.index("bash run")
    output = tmp_path / "output.txt"
    command = checks.replace("/results/output.txt", str(output))
    assert subprocess.run(["bash", "-ec", command]).returncode == 0
    output.write_text("historical evidence")
    assert subprocess.run(["bash", "-ec", command]).returncode != 0
    assert output.read_text() == "historical evidence"
    output.unlink()
    output.symlink_to(tmp_path / "missing")
    assert subprocess.run(["bash", "-ec", command]).returncode != 0
