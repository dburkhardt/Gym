# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Allowlisted resolved policy evidence; no admission, cooldown I/O, or capacity claim.

Role means the configured model-server instance name, not an inferred scientific
role. Unsupported model implementations remain explicit. None retains its native
meaning (including unbounded admission); this report imposes no campaign policy.
"""

import hashlib
import importlib
import inspect
import json
import math
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf


def _fingerprint(value):
    return hashlib.sha256(value.encode()).hexdigest() if value is not None else None


_SUPPORTED_HARBOR_AGENTS = {
    "responses_api_agents.harbor_agent.custom_agents.terminus_2_nemo_gym:Terminus2NemoGym",
    "responses_api_agents.harbor_agent.custom_agents.terminus_2_images_nemo_gym:Terminus2ImagesNemoGym",
}


def _harbor_timeout_contract(path):
    """Inspect only known managed adapters; optional Harbor dependencies may be absent."""
    if "litellm" not in sys.modules and os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() != "true":
        raise RuntimeError("Offline Harbor contract import requires local cost map")
    from responses_api_agents.harbor_agent.custom_agents.llms.nemo_gym_llm import NemoGymLLM
    from responses_api_agents.harbor_agent.custom_agents.terminus_2_nemo_gym import Terminus2NemoGym

    module, name = path.split(":")
    cls = getattr(importlib.import_module(module), name)
    if not issubclass(cls, Terminus2NemoGym):
        raise ValueError("Unknown managed adapter")
    default = inspect.signature(Terminus2NemoGym.__init__).parameters["nemo_model_server_timeout_sec"].default
    if cls.options_model.model_fields["nemo_model_server_timeout_sec"].default != default:
        raise ValueError("Constructor and options defaults disagree")
    sources = {}
    for implementation in (Terminus2NemoGym, cls, NemoGymLLM):
        key = implementation.__module__ + ":" + implementation.__name__
        sources[key] = hashlib.sha256(Path(inspect.getfile(implementation)).read_bytes()).hexdigest()
    return default, sources


def _positive_seconds(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _harbor_transport_links(config, roles):
    links = []
    for agent, instance in config.items():
        if not OmegaConf.is_dict(instance) or "responses_api_agents" not in instance:
            continue
        settings = instance.responses_api_agents.get("harbor_agent")
        if not OmegaConf.is_dict(settings):
            continue
        row = {"agent": agent, "status": "unknown"}
        links.append(row)
        try:
            values = OmegaConf.to_container(settings, resolve=True)
            path = values.get("harbor_agent_import_path")
            if path not in _SUPPORTED_HARBOR_AGENTS:
                row.update(status="unsupported", reason="unsupported_harness")
                continue
            row["harness"] = path
            binding = values.get("model_server") or {}
            if binding.get("type") != "responses_api_models" or not isinstance(binding.get("name"), str):
                row["reason"] = "missing_exact_model_binding"
                continue
            row["model_server"] = binding["name"]
            matches = [r for r in roles if r["role"] == binding["name"]]
            if len(matches) != 1 or matches[0]["status"] != "validated":
                row["reason"] = "unsupported_or_missing_model_policy"
                continue
            model = matches[0]
            try:
                default, sources = _harbor_timeout_contract(path)
            except Exception:
                row["reason"] = "harbor_runtime_contract_unavailable_or_offline_import_not_enabled"
                continue
            kwargs = values.get("harbor_agent_kwargs") or {}
            timeout = kwargs.get("nemo_model_server_timeout_sec", default)
            row.update(
                timeout_origin="configured" if "nemo_model_server_timeout_sec" in kwargs else "constructor_default",
                runtime_source_sha256=sources,
            )
            if not _positive_seconds(timeout):
                row["reason"] = "invalid_client_timeout"
                continue
            row["client_timeout_seconds"] = timeout
            row["client_timeout_semantics"] = "httpx_phase_or_inactivity_timeout_not_absolute_wall_time"
            if kwargs.get("llm") is not None:
                row["reason"] = "custom_llm_override"
                continue
            if config.get("provider_dispatch_guard") is not True:
                row["reason"] = "unguarded_agent_retry_schedule"
                continue
            if model["max_attempts"] != 1 or model["upstream_max_num_tries"] != 1:
                row["reason"] = "multiple_or_unknown_attempt_schedule"
                continue
            # NeMoGymAsyncOpenAI.max_num_tries=1 overrides its max_http_attempts fallback
            # and disables transport retries (openai_utils._request_with_retries).
            row["effective_outbound_attempts"] = 1
            row["configured_max_http_attempts_fallback"] = model["max_http_attempts"]
            queue, provider = model["admission_timeout_seconds"], model["provider_timeout_seconds"]
            if not _positive_seconds(queue) or not _positive_seconds(provider):
                row["reason"] = "unbounded_or_unknown_server_window"
                continue
            window = queue + provider
            row.update(
                status="client_shorter_than_server_window"
                if timeout < window
                else "covers_declared_queue_plus_provider_only",
                server_window_seconds=window,
                margin_seconds=timeout - window,
                scope="single_attempt_queue_plus_provider_only_not_end_to_end_deadline",
                excluded_overhead=["pre_request_jitter", "serialization", "network_and_response_delivery"],
                pre_request_jitter_seconds=model["pre_request_jitter_seconds"],
            )
        except Exception:
            # Interpolation or malformed options can contain secrets. Never echo them.
            row["reason"] = "unresolved_transport_configuration"
    return sorted(links, key=lambda row: row["agent"])


def resolved_admission_evidence(config):
    roles = []
    for role, instance in config.items():
        if not OmegaConf.is_dict(instance) or "responses_api_models" not in instance:
            continue
        for implementation, settings in instance.responses_api_models.items():
            record = {"role": role, "implementation": implementation}
            if implementation != "openai_model":
                roles.append(record | {"status": "unsupported"})
                continue
            from responses_api_models.openai_model.app import SimpleModelServerConfig

            try:
                values = OmegaConf.to_container(settings, resolve=True)
                model = SimpleModelServerConfig.model_validate({**values, "name": role})
            except Exception:
                # Pydantic/interpolation errors may contain credentials or entire input mappings.
                raise ValueError(f"Cannot validate admission policy for model role {role!r}") from None
            record.update(
                status="validated",
                model=model.openai_model,
                endpoint_sha256=_fingerprint(model.openai_base_url.rstrip("/")),
                max_attempts=model.upstream_retry_policy.max_attempts,
                # No typed retryable-status field exists: cooldown mode retries
                # only 429; legacy mode catches general exceptions except its
                # terminal controls. Null is intentionally not an empty set.
                retry_mode="observed_429" if model.upstream_rate_limit_cooldown_seconds is not None else "legacy",
                retryable_http_status_codes=[429] if model.upstream_rate_limit_cooldown_seconds is not None else None,
                terminal_http_status_codes=sorted(model.upstream_retry_policy.terminal_http_status_codes),
                pre_request_jitter_seconds=model.upstream_retry_policy.pre_request_jitter_seconds,
                backoff_initial_seconds=model.upstream_retry_policy.backoff_initial_seconds,
                backoff_multiplier=model.upstream_retry_policy.backoff_multiplier,
                backoff_jitter_fraction=model.upstream_retry_policy.backoff_jitter_fraction,
                upstream_max_num_tries=model.upstream_max_num_tries,
                max_http_attempts=model.max_http_attempts,
                provider_timeout_seconds=model.upstream_request_timeout_seconds,
                admission_timeout_seconds=model.upstream_admission_timeout_seconds,
                cooldown_seconds=model.upstream_rate_limit_cooldown_seconds,
                shared_cooldown=model.upstream_rate_limit_shared_directory is not None,
                namespace_sha256=_fingerprint(model.upstream_rate_limit_shared_namespace),
                max_concurrent_requests=model.max_concurrent_requests,
            )
            if model.upstream_exhausted_token_cooldown_seconds is not None:
                record["exhausted_token_cooldown_seconds"] = model.upstream_exhausted_token_cooldown_seconds
                record["trusted_exhaustion_header"] = "x-ratelimit-remaining-tokens"
            if model.propagate_upstream_http_status_codes:
                record["propagate_upstream_http_status_codes"] = sorted(model.propagate_upstream_http_status_codes)
            roles.append(record)
    roles.sort(key=lambda row: (row["role"], row["implementation"]))
    evidence = {"version": 2, "roles": roles, "harbor_transport_links": _harbor_transport_links(config, roles)}
    return evidence | {
        "sha256": hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    }
