# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cooperative, per-user local listener ownership; not an OS socket reservation."""

import fcntl
import os
import stat
from pathlib import Path


class PortOwnership:
    """Hold local port locks until the native supervisor has stopped its services.

    Native services are launched on this host. Deliberately key by port, not a
    hostname string: localhost, loopback addresses and wildcard binds can overlap.
    This conservatively also excludes disjoint local interfaces. Other users and
    non-Gym listeners do not participate; their OS bind races still fail at startup.
    Never unlink lock files: replacing their inode would split existing ownership.
    """

    def __init__(self, listeners, directory=None):
        self._files = []
        ports = []
        for label, host, port in listeners:
            if not isinstance(host, str) or not host or type(port) is not int or not 1 <= port <= 65535:
                raise ValueError(f"Invalid native listener for {label}: {host!r}:{port!r}")
            if port in ports:
                raise RuntimeError(f"Native listeners overlap on local port {port}; check head and component ports")
            ports.append(port)
        # Fixed POSIX host-local namespace: TMPDIR can differ between supervisors.
        # Resolve the platform /tmp alias only; never resolve supplied lock paths.
        root = Path(directory) if directory is not None else Path("/tmp").resolve() / f"nemo-gym-ports-{os.getuid()}"
        root = root.absolute()
        if any(p.is_symlink() for p in (root, *root.parents)):
            raise RuntimeError("Native port ownership directory must not traverse symlinks")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = root.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise RuntimeError("Native port ownership directory must be private and owned by the current user")
        try:
            for port in sorted(ports):
                path = root / f"{port}.lock"
                fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, 0o600)
                stream = os.fdopen(fd, "a+")
                self._files.append(stream)
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
                    raise RuntimeError("Native port lock must be a private regular file owned by the current user")
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError(
                        f"Another native Gym supervisor owns local port {port}; use a disjoint port configuration"
                    ) from exc
        except BaseException:
            self.close()
            raise

    def close(self):
        for stream in reversed(self._files):
            stream.close()
        self._files.clear()
