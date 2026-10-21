# SPDX-License-Identifier: Apache-2.0
"""Durable native-service ownership; never a model dispatch or session restorer."""

import copy
import fcntl
import hashlib
import json
import math
import os
import signal
import socket
import sys
import uuid
from pathlib import Path

import psutil

from nemo_gym.sandbox.lifecycle import _publish


TOKEN = "NEMO_GYM_SERVICE_OWNER_TOKEN"


def component_binding(components):
    from nemo_gym.provider_dispatch import configuration_digest

    result = copy.deepcopy(components)
    for component in result.values():
        for kind in ("responses_api_models", "responses_api_agents", "resources_servers", "environment_servers"):
            for server in component.get(kind, {}).values():
                # Exactly BaseServerConfig deployment fields, never nested task/tool settings.
                for field in ("host", "port", "num_workers"):
                    server.pop(field, None)
    return configuration_digest(result)


def host_identity():
    return {"host": socket.gethostname(), "boot_time": psutil.boot_time(), "uid": os.getuid()}


def identity(process):
    return {"pid": process.pid, "created": process.create_time()}


def matching(value):
    if not isinstance(value, dict) or set(value) != {"pid", "created"}:
        raise RuntimeError("Malformed service process identity")
    if (
        type(value["pid"]) is not int
        or value["pid"] <= 1
        or type(value["created"]) not in (int, float)
        or not math.isfinite(value["created"])
        or value["created"] < 0
    ):
        raise RuntimeError("Invalid service process identity")
    try:
        process = psutil.Process(value["pid"])
        return (
            process if process.create_time() == value["created"] and process.status() != psutil.STATUS_ZOMBIE else None
        )
    except psutil.NoSuchProcess:
        return None


def checked_directory(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise RuntimeError("Service ownership directory must not traverse symlinks")
    path.mkdir(parents=True, exist_ok=True)
    if path.stat().st_uid != os.getuid():
        raise RuntimeError("Service ownership directory owner mismatch")
    return path


def owned_members(token, created_after):
    members = []
    for process in psutil.process_iter():
        try:
            if (
                process.create_time() < created_after
                or process.uids().real != os.getuid()
                or process.status() == psutil.STATUS_ZOMBIE
            ):
                continue
            if process.environ().get(TOKEN) == token:
                members.append(process)
        except (psutil.NoSuchProcess, ProcessLookupError):
            continue
        except psutil.AccessDenied:
            # A process can exit between the status and environment reads.
            # Only a fresh observation of termination permits omission; a live
            # inaccessible process still prevents ownership confirmation.
            try:
                if process.status() == psutil.STATUS_ZOMBIE:
                    continue
            except (psutil.NoSuchProcess, ProcessLookupError):
                continue
            raise
    return members


class ServiceOwnership:
    """Hold the existing run output's ownership lease and reconcile old services."""

    def __init__(self, directory, binding):
        self.directory = checked_directory(directory)
        self.binding = binding
        self.published = []
        lock = self.directory / ".lock"
        if lock.is_symlink():
            raise RuntimeError("Service ownership lock is a symlink")
        self.lock = lock.open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.reconcile()
        except BaseException:
            self.lock.close()
            raise

    def reconcile(self):
        # Validate every record before signaling any process. Never infer ownership
        # from process names, ports, or a PID without its creation identity/token.
        records = []
        for path in sorted(self.directory.glob("*.json")):
            if path.is_symlink():
                raise RuntimeError("Service ownership receipt is a symlink")
            value = json.loads(path.read_text())
            if (
                value.get("version") != 1
                or value.get("binding") != self.binding
                or value.get("host") != host_identity()
            ):
                raise RuntimeError(
                    f"Service ownership source/config/host binding mismatch: recorded={value.get('binding')} expected={self.binding}"
                )
            token = value.get("token")
            if not isinstance(token, str) or uuid.UUID(token).hex != token:
                raise RuntimeError("Invalid service ownership token")
            coordinator = matching(value.get("coordinator"))
            if coordinator is not None and value.get("cleanup_confirmed") is not True:
                raise RuntimeError("Original service coordinator is still alive")
            root = matching(value["root"]) if value.get("root") else None
            members = owned_members(token, value["coordinator"]["created"])
            if root is not None:
                members = list({p.pid: p for p in [root, *root.children(recursive=True), *members]}.values())
            if coordinator is not None and members:
                raise RuntimeError("Completed ownership receipt still has live services")
            records.append((path, value, members))
        for path, value, members in records:
            for process in members:
                try:
                    process.send_signal(signal.SIGTERM)
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(members, timeout=5)
            for process in alive:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(alive, timeout=5)
            alive = [p for p in alive if p.is_running() and p.status() != psutil.STATUS_ZOMBIE]
            if alive or owned_members(value["token"], value["coordinator"]["created"]):
                raise RuntimeError("Owned native services did not settle")
            value["cleanup_confirmed"] = True
            _publish(path, value)

    def command(self, command, env):
        token = uuid.uuid4().hex
        path = self.directory / f"{token}.json"
        value = {
            "version": 1,
            "binding": self.binding,
            "host": host_identity(),
            "token": token,
            "coordinator": identity(psutil.Process()),
            "root": None,
            "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
            "cleanup_confirmed": False,
        }
        _publish(path, value)
        self.published.append(path)
        env[TOKEN] = token
        return [sys.executable, str(Path(__file__).absolute()), str(path), command]

    def close(self):
        try:
            for path in self.published:
                value = json.loads(path.read_text())
                if (value.get("root") and matching(value["root"]) is not None) or owned_members(
                    value["token"], value["coordinator"]["created"]
                ):
                    raise RuntimeError("Cannot confirm native services stopped")
                value["cleanup_confirmed"] = True
                _publish(path, value)
        finally:
            self.lock.close()


def launch(path, command):
    path = Path(path)
    value = json.loads(path.read_text())
    if value["host"] != host_identity() or value["token"] != os.environ.get(TOKEN):
        raise RuntimeError("Service launch ownership mismatch")
    if value["command_sha256"] != hashlib.sha256(command.encode()).hexdigest():
        raise RuntimeError("Service command identity mismatch")
    if matching(value["coordinator"]) is None:
        raise RuntimeError("Coordinator exited before service launch")
    if os.getpgrp() != os.getpid():
        raise RuntimeError("Service wrapper must own its POSIX process group")
    value["root"] = identity(psutil.Process())
    _publish(path, value)
    os.execve("/bin/bash", ["/bin/bash", "-c", command], os.environ)


if __name__ == "__main__":
    launch(*sys.argv[1:])
