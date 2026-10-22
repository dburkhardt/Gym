# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit infrastructure restart admission using existing batch status identity."""

import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

from nemo_gym.config_types import ConfigError
from nemo_gym.episode_types import EpisodeId
from nemo_gym.provider_dispatch import configuration_digest


BINDING_KEY = "execution_binding"
ROW_BINDING_KEY = "_ng_batch_execution_sha256"
_INFRASTRUCTURE_KINDS = {
    "provider_unavailable",
    "provider_quota_exhausted",
    "provider_oom_killed",
    "session_lost",
    "session_expired",
}
_SERVER_TYPES = {"responses_api_models", "responses_api_agents", "resources_servers", "environment_servers"}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def bind_execution(
    tracker,
    global_config,
    materialized_path: Path,
    *,
    resume: bool,
    require_prior: bool,
    collector_config=None,
    capture_dirs=(),
):
    """Extend native batch status with resolved server protocol and exact input binding."""
    config = (
        OmegaConf.to_container(global_config, resolve=True) if OmegaConf.is_config(global_config) else global_config
    )
    from nemo_gym.base_responses_api_model import CaptureStore

    servers = {}
    source_roots = {Path(__file__).parent}
    supported = {
        ("responses_api_models", "openai_model"),
        ("responses_api_agents", "simple_agent"),
        ("environment_servers", "legacy_agent"),
        ("resources_servers", "mcqa"),
        ("resources_servers", "example_single_tool_call"),
    }
    qualified = True

    for name, value in config.items():
        if not isinstance(value, dict):
            continue
        for kind in _SERVER_TYPES & value.keys():
            if not isinstance(value[kind], dict):
                raise ConfigError("Malformed server implementation mapping in batch binding")
            for implementation, settings in value[kind].items():
                if not isinstance(settings, dict):
                    raise ConfigError("Malformed server settings in batch binding")
                root = Path(kind) / implementation
                entrypoint = root / settings.get("entrypoint", "app.py")
                if (kind, implementation) not in supported or settings.get("entrypoint", "app.py") != "app.py":
                    qualified = False
                if settings.get("skills") or settings.get("mcp_servers"):
                    qualified = False
                if not entrypoint.resolve().is_relative_to(root.resolve()) or not entrypoint.is_file():
                    raise ConfigError("Batch execution binding changed or implementation source unavailable")
                source_roots.add(root)
            servers[name] = {
                kind: {
                    implementation: configuration_digest(settings) for implementation, settings in value[kind].items()
                }
            }
    # The native CLI relays legacy agents through this built-in environment.
    # Qualify only the unchanged relay to an admitted simple agent, never an
    # arbitrary target selected through the otherwise generic AgentServerRef.
    for value in config.values():
        if not isinstance(value, dict):
            continue
        environments = value.get("environment_servers", {})
        if "legacy_agent" not in environments:
            continue
        relay = environments["legacy_agent"]
        reference = relay.get("agent_server")
        if not isinstance(reference, dict) or not isinstance(reference.get("name"), str):
            raise ConfigError("Malformed legacy relay agent reference in batch binding")
        target = config.get(reference["name"], {})
        if (
            reference.get("type") != "responses_api_agents"
            or not isinstance(target, dict)
            or set(target.get("responses_api_agents", {})) != {"simple_agent"}
        ):
            qualified = False
    if not servers:
        raise ConfigError("Guarded batch binding requires resolved server configurations")
    collector = dict(collector_config or {})
    # Execution placement/reporting changes are not scientific protocol changes.
    for key in (
        "resume_from_cache",
        "guarded_restart_receipt_fpath",
        "eval_checkpoint_dir",
        "eval_restore_checkpoint",
        "num_samples_in_parallel",
        "dispatch_longest_first",
        "dispatch_budget_s",
        "drain_margin_s",
        "worker_lease_deadline_unix_s",
        "worker_lease_cleanup_reserve_s",
        "worker_lease_startup_reserve_s",
        "retain_results_in_memory",
        "health_check_ignored_checks",
        "disable_health_check",
    ):
        collector.pop(key, None)
    source_hashes = {}
    for root in sorted(source_roots):
        for folder, directories, files in os.walk(root):
            directories[:] = sorted(
                d for d in directories if d not in {".venv", "__pycache__", "tests", "data", ".git"}
            )
            for filename in sorted(files):
                if filename.endswith(".py"):
                    path = Path(folder) / filename
                    source_hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    stores = {
        str(Path(directory).resolve()): CaptureStore(directory, provider_dispatch_guard=True).identity()
        for directory in capture_dirs
    }
    if require_prior and not stores:
        raise ConfigError("Explicit restart requires original durable capture stores")
    global_science = {
        key: value
        for key, value in config.items()
        if key not in servers and key not in {"rollout_collection", "e2e_rollout_collection", "collect_rollouts"}
    }
    # These global scalar values only describe local serving/observability placement.
    for key in (
        "host",
        "port",
        "num_workers",
        "observability_enabled",
        "model_call_capture_dir",
        "provider_dispatch_guard",
        "guarded_restart_receipt_fpath",
        "eval_checkpoint_dir",
        "eval_restore_checkpoint",
        "resume_from_cache",
        "reuse_existing_data_preparation",
        "worker_lease_deadline_unix_s",
        "worker_lease_cleanup_reserve_s",
        "worker_lease_startup_reserve_s",
        "disallowed_ports",
    ):
        global_science.pop(key, None)
    scientific_closure = None
    resource_implementations = {
        implementation
        for value in config.values()
        if isinstance(value, dict) and isinstance(value.get("resources_servers"), dict)
        for implementation in value["resources_servers"]
    }
    if {"scicode", "campaign_creator_scoring"} <= resource_implementations:
        from nemo_gym.mixed_scientific_restart import qualify_mixed_scientific_dependencies

        try:
            scientific_closure = qualify_mixed_scientific_dependencies(Path.cwd(), config, materialized_path)
            qualified = True
        except ConfigError as error:
            if require_prior:
                raise
            scientific_closure = {"unsupported": str(error)}
            qualified = False
    elif any(
        "scicode" in value.get("resources_servers", {})
        for value in config.values()
        if isinstance(value, dict) and isinstance(value.get("resources_servers", {}), dict)
    ):
        from nemo_gym.scicode_restart import qualify_scicode_dependencies

        try:
            scientific_closure = qualify_scicode_dependencies(Path.cwd(), config)
            qualified = True
        except ConfigError as error:
            if require_prior:
                raise
            # Ordinary first runs remain possible, but an unavailable dependency
            # closure can never authorize a later explicit restart.
            scientific_closure = {"unsupported": str(error)}
            qualified = False
    elif any(
        "campaign_creator_scoring" in value.get("resources_servers", {})
        for value in config.values()
        if isinstance(value, dict) and isinstance(value.get("resources_servers", {}), dict)
    ):
        from nemo_gym.creator_restart import qualify_creator_dependencies

        try:
            scientific_closure = qualify_creator_dependencies(Path.cwd(), config, materialized_path)
            qualified = True
        except ConfigError as error:
            if require_prior:
                raise
            scientific_closure = {"unsupported": str(error)}
            qualified = False
    if collector.get("skills") or collector.get("prompt_config"):
        qualified = False
    if require_prior and not qualified:
        raise ConfigError(
            "Explicit restart unsupported for this external dependency/harness closure; preserve failure and reconcile separately"
        )
    binding = {
        "manifest_sha256": tracker.manifest_sha256,
        "materialized_sha256": hashlib.sha256(materialized_path.read_bytes()).hexdigest(),
        "server_protocol_sha256": digest(servers),
        "collector_protocol_sha256": configuration_digest(collector),
        "global_options_sha256": configuration_digest(global_science),
        "global_option_digests": {key: configuration_digest({key: value}) for key, value in global_science.items()},
        "implementation_sha256": digest(source_hashes),
        "capture_stores": stores,
        "restart_scope_qualified": qualified,
        "runtime_sha256": digest(
            {
                "python": sys.version,
                "packages": sorted(
                    (d.metadata["Name"], d.version) for d in importlib.metadata.distributions() if d.metadata["Name"]
                ),
            }
        ),
    }
    if scientific_closure is not None:
        binding["scientific_dependencies"] = scientific_closure
    prior = getattr(tracker, "execution_binding", None)
    if require_prior and (not resume or prior is None):
        raise ConfigError(
            "Explicit restart requires a previously bound native batch; old unbound runs cannot be relabeled"
        )
    if prior is not None and prior != binding:
        changed = sorted(key for key in set(prior) | set(binding) if prior.get(key) != binding.get(key))
        details = ""
        if "global_option_digests" in changed:
            old = prior.get("global_option_digests", {})
            new = binding["global_option_digests"]
            details = (
                " (global keys: "
                + ", ".join(sorted(key for key in set(old) | set(new) if old.get(key) != new.get(key)))
                + ")"
            )
        raise ConfigError("Batch execution binding changed: " + ", ".join(changed) + details)
    tracker.execution_binding = binding
    return digest(binding)


def _captured_provider_quota_rejection(key, capture_dirs):
    """Qualify only a fully rejected 429 attempt, never an outer relay error."""
    from nemo_gym.base_responses_api_model import CaptureStore

    attempts = []
    for directory in capture_dirs:
        store = CaptureStore(directory, provider_dispatch_guard=True)
        store.assert_dispatch_reconciled([key])
        if store.is_incomplete(key) or store.has_pending(key):
            return False
        calls = store.read(key)
        records = store.dispatch_records(key)
        if len(calls) != len(records):
            return False
        for record in records:
            if not record["provider_attempts"]:
                return False
            attempts.extend(record["provider_attempts"])
    return bool(attempts) and all(
        attempt.get("outcome") == "rejected" and attempt.get("status_code") == 429 for attempt in attempts
    )


def admitted_prior_attempts(
    receipt_path, *, binding_sha256, failures_path, pending_rows, completed_keys, capture_dirs=()
):
    """Return exact prior capture keys; receipt is an explicit operator choice, not a retry policy."""
    from nemo_gym.base_responses_api_model import maybe_rollout_id_from_run_body

    receipt = json.loads(Path(receipt_path).read_text())
    if (
        set(receipt) != {"execution_binding_sha256", "failures"}
        or receipt["execution_binding_sha256"] != binding_sha256
    ):
        raise ConfigError("Restart receipt execution binding mismatch")
    failures = [json.loads(line) for line in Path(failures_path).read_text().splitlines() if line.strip()]
    by_key = {}
    for row in failures:
        key = maybe_rollout_id_from_run_body(row)
        if key in by_key:
            raise ConfigError("Duplicate failed-attempt identity; explicit reconciliation required")
        by_key[key] = row
    pending = {
        EpisodeId.from_capture_key(maybe_rollout_id_from_run_body(row)).rollout_id: EpisodeId.from_capture_key(
            maybe_rollout_id_from_run_body(row)
        ).attempt
        for row in pending_rows
    }
    allowed = set()
    for entry in receipt["failures"]:
        if set(entry) != {"capture_key", "failure_sha256"} or entry["capture_key"] in allowed:
            raise ConfigError("Malformed or duplicate restart admission")
        key = entry["capture_key"]
        row = by_key.get(key)
        if row is None or digest(row) != entry["failure_sha256"] or row.get(ROW_BINDING_KEY) != binding_sha256:
            raise ConfigError("Restart failure evidence changed or is unbound")
        logical = EpisodeId.from_capture_key(key)
        if (row.get("_ng_task_index"), row.get("_ng_rollout_index")) in completed_keys:
            raise ConfigError("Completed scientific output cannot be selected for restart")
        if pending.get(logical.rollout_id, -1) <= logical.attempt:
            raise ConfigError("Restart must be a monotonically new native attempt")
        failure_kind = row.get("failure_kind")
        if failure_kind is None and _captured_provider_quota_rejection(key, capture_dirs):
            failure_kind = "provider_quota_exhausted"
        if (
            failure_kind not in _INFRASTRUCTURE_KINDS
            or not row.get("_ng_failure_class")
            or row.get("reward") is not None
        ):
            raise ConfigError(
                "Restart needs a positively qualified infrastructure failure, not a scientific score or generic timeout"
            )
        allowed.add(key)
    if not allowed:
        raise ConfigError("Restart receipt contains no failed attempts")
    return allowed


def reject_bound_overwrite(manifest_path, *, resume):
    """Run before materialization or output cleanup; preserve every existing artifact."""
    if manifest_path and not resume:
        status = Path(manifest_path).with_name("batch_status.json")
        if status.exists() and json.loads(status.read_text()).get("execution_binding") is not None:
            raise ConfigError("Existing bound batch cannot be overwritten; resume or choose a distinct run directory")
