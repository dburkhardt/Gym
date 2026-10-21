# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Allowlisted resolved policy evidence; no admission, cooldown I/O, or capacity claim.

Role means the configured model-server instance name, not an inferred scientific
role. Unsupported model implementations remain explicit. None retains its native
meaning (including unbounded admission); this report imposes no campaign policy.
"""

import hashlib
import json

from omegaconf import OmegaConf


def _fingerprint(value):
    return hashlib.sha256(value.encode()).hexdigest() if value is not None else None


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
            roles.append(record)
    roles.sort(key=lambda row: (row["role"], row["implementation"]))
    return {
        "version": 1,
        "roles": roles,
        "sha256": hashlib.sha256(json.dumps(roles, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }
