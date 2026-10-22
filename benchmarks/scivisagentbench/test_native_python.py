from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from benchmarks.scivisagentbench.managed import validate_native_python


def test_python310_rejected_before_runtime_can_use_asyncio_timeout():
    with patch("benchmarks.scivisagentbench.managed.subprocess.run", return_value=SimpleNamespace(stdout="[3, 10]")):
        with pytest.raises(ValueError, match="requires Python 3.11"):
            validate_native_python(Path("/qualified/python"))


def test_current_controller_interpreter_is_supported():
    import sys

    validate_native_python(Path(sys.executable))
