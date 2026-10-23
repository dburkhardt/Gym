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
import subprocess
import sys
import time
import uuid
from pathlib import Path

import psutil

from nemo_gym.sandbox.lifecycle import _publish


TOKEN = "NEMO_GYM_SERVICE_OWNER_TOKEN"
MODE_ENV = "NEMO_GYM_SERVICE_OWNERSHIP_MODE"


def _preflight_systemd_user_manager() -> None:
    """Require an accessible user manager before any native services start."""
    message = (
        "Systemd service ownership requires a reachable user manager before native startup. "
        "Check `systemctl --user show --property=SystemState --value` in the same launch environment "
        "and explicitly configure the correct XDG_RUNTIME_DIR/DBUS_SESSION_BUS_ADDRESS for that user. "
        "Gym does not infer bus addresses, start a manager, or fall back to process ownership."
    )
    try:
        result = subprocess.run(
            ["systemctl", "--user", "show", "--property=SystemState", "--value"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(message) from error
    # `is-system-running` exits nonzero for degraded managers, even though they
    # can manage scopes. Query the manager property instead of system health.
    if result.returncode != 0 or result.stdout.strip() not in {"running", "degraded"}:
        raise RuntimeError(message)


def _unit(value):
    return f"nemogym-{value['token']}.scope"


def _description(value):
    payload = {k: value[k] for k in ("binding", "host", "token", "command_sha256")}
    return "NeMoGym owned " + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _unit_properties(unit):
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "--property=Id,LoadState,ActiveState,SubState,InvocationID,ControlGroup,Description",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    props = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if props.get("LoadState") == "not-found":
        # Missing-unit stdout alone is not cleanup evidence: failed or partial
        # manager responses must not turn an unresolved scope into an absence.
        if (
            result.returncode not in {0, 4}
            or props.get("Id") != unit
            or props.get("ActiveState") != "inactive"
            or props.get("SubState") != "dead"
            or props.get("ControlGroup") != ""
        ):
            raise RuntimeError("Cannot establish absence of dedicated systemd ownership unit")
        return None
    if result.returncode or not props:
        raise RuntimeError("Cannot inspect dedicated systemd ownership unit")
    return props


def _scope_identity(value):
    recorded = value.get("systemd")
    if recorded is not None and (
        not isinstance(recorded, dict)
        or set(recorded) != {"unit", "invocation_id", "control_group"}
        or recorded["unit"] != _unit(value)
        or not isinstance(recorded["invocation_id"], str)
        or len(recorded["invocation_id"]) != 32
        or any(c not in "0123456789abcdef" for c in recorded["invocation_id"])
        or not isinstance(recorded["control_group"], str)
        or not recorded["control_group"].startswith(
            f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/"
        )
        or Path(recorded["control_group"]).name != _unit(value)
        or ".." in Path(recorded["control_group"]).parts
    ):
        raise RuntimeError("Malformed dedicated systemd scope receipt")
    props = _unit_properties(_unit(value))
    if props is None:
        if value.get("root") and matching(value["root"]) is not None:
            raise RuntimeError("Owned root remains alive outside missing systemd unit")
        if recorded and _scope_populated(recorded["control_group"]):
            raise RuntimeError("Missing systemd unit retains populated cgroup")
        return None
    group = props.get("ControlGroup", "")
    invocation = props.get("InvocationID", "")
    if (
        props.get("Id") != _unit(value)
        or props.get("Description") != _description(value)
        or not group.startswith(f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/")
        or Path(group).name != _unit(value)
        or ".." in Path(group).parts
        or len(invocation) != 32
        or any(c not in "0123456789abcdef" for c in invocation)
    ):
        raise RuntimeError("Dedicated systemd scope identity mismatch")
    actual = {"unit": _unit(value), "invocation_id": invocation, "control_group": group}
    if recorded is not None and actual != recorded:
        raise RuntimeError("Dedicated systemd scope invocation changed")
    return actual


def _scope_populated(group):
    path = Path("/sys/fs/cgroup") / group.lstrip("/")
    if ".." in Path(group).parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise RuntimeError("Invalid owned cgroup path")
    try:
        fields = dict(line.split() for line in (path / "cgroup.events").read_text().splitlines())
    except FileNotFoundError:
        return False
    if fields.get("populated") not in {"0", "1"}:
        raise RuntimeError("Cannot establish owned cgroup population")
    return fields["populated"] == "1"


def _stop_scope(value):
    actual = _scope_identity(value)
    if actual is None:
        return
    # Revalidate the durable invocation immediately before stopping this unique
    # unit. Never stop the caller's SSH scope or enumerate other users' processes.
    bound = {**value, "systemd": actual}
    _scope_identity(bound)
    result = subprocess.run(["systemctl", "--user", "stop", actual["unit"]], capture_output=True, timeout=15)
    if result.returncode or _scope_populated(actual["control_group"]):
        raise RuntimeError("Dedicated systemd scope did not settle")
    if value.get("root") and matching(value["root"]) is not None:
        raise RuntimeError("Owned service root outlived its dedicated scope")
    props = _unit_properties(actual["unit"])
    if props is not None and props.get("ActiveState") not in {"inactive", "failed"}:
        raise RuntimeError("Dedicated systemd scope remains active")


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


def inspect_service_ownership(directory, binding):
    """Read recorded service identities without acquiring a lease or signaling.

    This is a point-in-time native-service check, not a port, Docker, or detached
    process census. Process mode cannot establish descendant absence without the
    global scan deliberately excluded here.
    """
    result = {
        "version": 1,
        "status": "unknown",
        "scope": "native_services_only",
        "external_containers_checked": False,
        "observed_at_unix_s": time.time(),
        "records": [],
    }
    try:
        directory = Path(directory).absolute()
        if any(p.is_symlink() for p in (directory, *directory.parents)):
            raise RuntimeError("Ownership directory traverses symlinks")
        if not directory.is_dir() or directory.stat().st_uid != os.getuid():
            raise RuntimeError("Ownership directory missing or owner mismatch")
        paths = sorted(directory.glob("*.json"))
        if not paths:
            raise RuntimeError("No service ownership records")
        for path in paths:
            entry = {"receipt": path.name, "status": "unknown"}
            result["records"].append(entry)
            try:
                if path.is_symlink() or path.stat().st_uid != os.getuid():
                    raise RuntimeError("Ownership receipt symlink or owner mismatch")
                value = json.loads(path.read_text())
                if (
                    type(value.get("version")) is not int
                    or value["version"] != 1
                    or value.get("binding") != binding
                    or value.get("host") != host_identity()
                ):
                    raise RuntimeError("Ownership source/config/host binding mismatch")
                token = value.get("token")
                if not isinstance(token, str) or uuid.UUID(token).hex != token:
                    raise RuntimeError("Invalid ownership token")
                coordinator = matching(value.get("coordinator"))
                root = matching(value["root"]) if value.get("root") else None
                entry.update(coordinator_alive=coordinator is not None, root_alive=root is not None)
                mode = value.get("mode", "process")
                entry["mode"] = mode
                if mode == "systemd":
                    actual = _scope_identity(value)
                    group = actual or value.get("systemd")
                    populated = bool(group and _scope_populated(group["control_group"]))
                    entry["cgroup_populated"] = populated
                    if root is not None or populated:
                        entry["status"] = "live"
                    elif value.get("root") is None or value.get("systemd") is None:
                        entry["reason"] = (
                            "Incomplete launch identity; never-started state not independently established"
                        )
                    elif actual is not None:
                        entry["reason"] = "Dedicated scope still exists"
                    elif value.get("cleanup_confirmed") is not True:
                        entry["reason"] = "Cleanup was not confirmed by the owner"
                    else:
                        entry["status"] = "clean"
                elif mode == "process":
                    entry["status"] = "live" if root is not None else "unknown"
                    entry["reason"] = "Detached descendant absence requires separate process-mode evidence"
                else:
                    raise RuntimeError("Unsupported ownership mode")
            except Exception as error:
                entry.update(status="unknown", error_type=type(error).__name__, reason=str(error))
        states = {entry["status"] for entry in result["records"]}
        result["status"] = "live" if "live" in states else "clean" if states == {"clean"} else "unknown"
    except Exception as error:
        result.update(error_type=type(error).__name__, reason=str(error))
    return result


class ServiceOwnership:
    """Hold the existing run output's ownership lease and reconcile old services."""

    def __init__(self, directory, binding, *, mode=None):
        self.mode = mode or os.environ.get(MODE_ENV, "process")
        if self.mode not in {"process", "systemd"}:
            raise RuntimeError("Unsupported native service ownership mode")
        if self.mode == "systemd" and sys.platform != "linux":
            raise RuntimeError("Systemd ownership requires Linux user scopes")
        if self.mode == "systemd":
            _preflight_systemd_user_manager()
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
                or value.get("mode", "process") != self.mode
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
            if self.mode == "systemd":
                actual = _scope_identity(value)
                if coordinator is not None and actual and _scope_populated(actual["control_group"]):
                    raise RuntimeError("Completed ownership receipt still has live services")
                records.append((path, value, None))
                continue
            root = matching(value["root"]) if value.get("root") else None
            members = owned_members(token, value["coordinator"]["created"])
            if root is not None:
                members = list({p.pid: p for p in [root, *root.children(recursive=True), *members]}.values())
            if coordinator is not None and members:
                raise RuntimeError("Completed ownership receipt still has live services")
            records.append((path, value, members))
        for path, value, members in records:
            if self.mode == "systemd":
                _stop_scope(value)
                value["cleanup_confirmed"] = True
                _publish(path, value)
                continue
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
        if self.mode == "systemd":
            value.update(mode="systemd", systemd=None)
        _publish(path, value)
        self.published.append(path)
        env[TOKEN] = token
        bootstrap = [sys.executable, str(Path(__file__).absolute()), str(path), command]
        if self.mode == "systemd":
            return [
                "systemd-run",
                "--user",
                "--scope",
                "--quiet",
                "--no-ask-password",
                f"--unit={_unit(value)}",
                f"--description={_description(value)}",
                "--property=KillMode=control-group",
                "--property=TimeoutStopSec=5s",
                "--",
                *bootstrap,
            ]
        return bootstrap

    def close(self):
        try:
            for path in self.published:
                value = json.loads(path.read_text())
                if self.mode == "systemd":
                    _stop_scope(value)
                    value["cleanup_confirmed"] = True
                    _publish(path, value)
                    continue
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
    if value.get("mode") == "systemd":
        actual = _scope_identity(value)
        if actual is None:
            raise RuntimeError("Dedicated systemd scope missing before service exec")
        if Path("/proc/self/cgroup").read_text().strip() != "0::" + actual["control_group"]:
            raise RuntimeError("Service bootstrap is outside its dedicated cgroup")
        value["systemd"] = actual
    value["root"] = identity(psutil.Process())
    _publish(path, value)
    os.execve("/bin/bash", ["/bin/bash", "-c", command], os.environ)


if __name__ == "__main__":
    launch(*sys.argv[1:])
