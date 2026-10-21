# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-local observed 429 deadlines, shared by explicitly named provider scopes.

No request/token budget is inferred. Locks protect only short state operations;
callers wait outside them. State is not suitable for cross-host/network filesystems.
"""

import fcntl
import hashlib
import json
import math
import os
import platform
import stat
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit


def boot_identity() -> str:
    """Identify the kernel clock epoch so a reboot cannot reuse monotonic deadlines."""
    if platform.system() == "Linux":
        raw = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    elif platform.system() == "Darwin":
        raw = subprocess.check_output(["/usr/sbin/sysctl", "-n", "kern.bootsessionuuid"], text=True, timeout=5).strip()
    else:
        raise ValueError("Shared cooldown supports host-local Linux/macOS clocks only")
    if not raw:
        raise ValueError("Kernel boot identity unavailable")
    return hashlib.sha256(raw.encode()).hexdigest()


class SharedCooldown:
    """Atomic maximum deadline, scoped without persisting credentials or raw URLs."""

    def __init__(self, directory: Path, namespace: str, endpoint: str, model: str) -> None:
        parsed = urlsplit(endpoint)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Shared cooldown endpoint must not contain userinfo, query or fragment")
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Shared cooldown requires an absolute HTTP(S) endpoint")
        if not namespace or not model:
            raise ValueError("Shared cooldown namespace and model must be nonempty")
        directory = directory.absolute()
        if any(p.is_symlink() for p in (directory, *directory.parents)):
            raise ValueError("Shared cooldown directory must not traverse symlinks")
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory = directory
        name = hashlib.sha256(namespace.encode()).hexdigest()
        self.path = directory / f"{name}.json"
        self.lock_path = directory / f"{name}.lock"
        self.scope = hashlib.sha256(
            json.dumps([endpoint.rstrip("/"), model], separators=(",", ":")).encode()
        ).hexdigest()
        self.boot = boot_identity()
        self.extend(0.0)  # Bind the scope before the first provider request, even without a429.

    @contextmanager
    def _lock(self):
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("Cooldown lock must be a regular file")
            limit = time.monotonic() + 1.0
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= limit:
                        raise TimeoutError("Shared provider cooldown state lock unavailable after one second")
                    time.sleep(0.01)
            yield
        finally:
            os.close(fd)

    def _read(self) -> dict | None:
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            return None
        with os.fdopen(fd) as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode) or os.fstat(stream.fileno()).st_size > 4096:
                raise ValueError("Invalid cooldown state file")
            record = json.load(stream)
        if not isinstance(record, dict):
            raise ValueError("Cooldown state must be a JSON object")
        if record.get("version") != 1 or record.get("scope_sha256") != self.scope:
            raise ValueError("Cooldown namespace is bound to a different endpoint/model or state version")
        deadline = record.get("until_monotonic")
        if type(deadline) not in (float, int) or not math.isfinite(deadline) or deadline < 0:
            raise ValueError("Invalid cooldown deadline")
        if not isinstance(record.get("boot_sha256"), str) or len(record["boot_sha256"]) != 64:
            raise ValueError("Invalid cooldown boot identity")
        return record

    def remaining(self) -> float:
        """Read the current host-epoch deadline; no lock survives this return."""
        with self._lock():
            record = self._read()
            if record is None or record["boot_sha256"] != self.boot:
                return 0.0
            return max(0.0, record["until_monotonic"] - time.monotonic())

    def extend(self, seconds: float) -> None:
        """Publish the maximum observed delay atomically, never shorten a peer's wait."""
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid provider cooldown")
        with self._lock():
            previous = self._read()
            deadline = time.monotonic() + seconds
            if not math.isfinite(deadline):
                raise ValueError("Provider cooldown cannot be represented")
            if previous and previous["boot_sha256"] == self.boot:
                deadline = max(deadline, previous["until_monotonic"])
            record = {
                "version": 1,
                "scope_sha256": self.scope,
                "boot_sha256": self.boot,
                "until_monotonic": deadline,
                "updated_wall_time": time.time(),
            }
            fd, temporary = tempfile.mkstemp(prefix=".cooldown-", dir=self.directory)
            try:
                with os.fdopen(fd, "w") as stream:
                    json.dump(record, stream, allow_nan=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
                directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                Path(temporary).unlink(missing_ok=True)
