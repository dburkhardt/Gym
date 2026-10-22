"""Keep a successful GLX probe from becoming a pipefail/SIGPIPE failure."""
import os
import subprocess
import sys
from pathlib import Path


def test_glx_probe_consumes_large_output(tmp_path):
    commands = {
        "Xvfb": "#!/bin/sh\nexec /bin/sleep 60\n",
        "xdpyinfo": f"#!{sys.executable}\nimport sys\nsys.stdout.write('GLX\\n' + 'x' * 1000000)\n",
        "python": "#!/bin/sh\nprintf '%s' \"$1\" > \"$MARKER\"\n",
    }
    for name, text in commands.items():
        path = tmp_path / name
        path.write_text(text)
        path.chmod(0o755)
    marker = tmp_path / "viewer-invocation"
    result = subprocess.run(
        ["bash", str(Path(__file__).with_name("start_viewer.sh"))],
        env={**os.environ, "PATH": str(tmp_path) + ":" + os.environ["PATH"], "DISPLAY": ":99", "MARKER": str(marker)},
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert marker.read_text() == "/control/viewer.py"
