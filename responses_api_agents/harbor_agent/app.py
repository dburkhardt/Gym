# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import logging
import math
import re
import sys
from asyncio import Semaphore
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

import ray
from fastapi import Body, FastAPI
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import HarborShutdownConfig, ModelServerRef
from nemo_gym.failure_kinds import AGENT_RUN_ERROR
from nemo_gym.global_config import (
    get_first_server_config_dict,
    get_global_config_dict,
)
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.sandbox.lifecycle import _publish
from nemo_gym.server_utils import (
    GlobalAIOHTTPAsyncClientConfig,
    global_aiohttp_client_exit_async,
    set_global_aiohttp_client,
)
from responses_api_agents.harbor_agent.failure_diagnostics import (
    intent_bound_provider_rejection,
    measurement_failure,
    required_artifact_failure,
    terminus_bound_provider_rejection,
)
from responses_api_agents.harbor_agent.measurement_review import apply_measurement_review
from responses_api_agents.harbor_agent.observability import read_terminus_observations
from responses_api_agents.harbor_agent.resource_admission import StaticRayResourceError, static_resource_feasibility
from responses_api_agents.harbor_agent.utils import HarborAgentUtils, HarborResourcePolicy


logger = logging.getLogger(__name__)


class HarborDatasetSourceConfig(BaseModel):
    local_dataset_path: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_version: Optional[str] = None
    workdir: Optional[str] = None


class HarborAgentConfig(HarborShutdownConfig, BaseResponsesAPIAgentConfig):
    concurrency: int
    # Opt-in phase-budget enforcement; the actual collector drain is bound separately.
    harbor_admission_finalization_allowance_s: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)

    # CPUs Ray reserves per Harbor job. Jobs are I/O-bound, so a fractional
    # value runs far more of them than Ray's default of 1 CPU/task, which caps
    # concurrency at the driver's CPU count.
    harbor_ray_task_num_cpus: Optional[float] = None
    # Logical reservation for the whole trial on its Ray worker, including local
    # Docker children and sidecars. This is admission, not a container memory limit.
    harbor_ray_task_memory_bytes: Optional[int] = Field(default=None, gt=0, strict=True)

    # Explicit static placement only; not suitable for unconstrained autoscaling pools.
    harbor_ray_static_resource_preflight: bool = False

    # --- Harbor agent settings ---
    # Name of a built-in Harbor agent (e.g. "terminus-2", "claude-code", "aider").
    harbor_agent_name: Optional[str] = "terminus-2"
    # Python import path for a custom agent class (e.g. "my_pkg.my_mod:MyAgent").
    # Overrides harbor_agent_name when set.
    harbor_agent_import_path: Optional[str] = None
    # Extra kwargs forwarded to the Harbor AgentConfig (e.g. collect_rollout_details,
    # model_info). See harbor_agent.yaml for examples.
    harbor_agent_kwargs: Optional[dict[str, Any]] = None

    # --- Dataset routing ---
    # Map of dataset aliases to source definitions. Each alias must define exactly
    # one source:
    # 1) local: {"local_dataset_path": "..."}
    # 2) registry: {"dataset_name": "...", "dataset_version": "..."} (version optional)
    # Requests must provide instance_id in the form "<dataset_alias>::<task_name>".
    harbor_datasets: dict[str, HarborDatasetSourceConfig]

    # --- Environment ---
    # Harbor environment type: "singularity", "docker", "daytona", "modal", etc.
    harbor_environment_type: Optional[str] = "singularity"
    # Python import path for a custom environment class (e.g. "my_pkg.my_mod:MyEnv").
    # Overrides harbor_environment_type when set.
    harbor_environment_import_path: Optional[str] = None
    # Extra kwargs forwarded to the Harbor EnvironmentConfig (e.g.
    # singularity_image_cache_dir, singularity_force_pull).
    harbor_environment_kwargs: Optional[dict[str, Any]] = None
    # Explicit acknowledgement of creator omissions; no resource overrides are added.
    harbor_resource_policy: HarborResourcePolicy = "require_explicit"

    # Native Harbor custom verifier hook. Model route is injected for guarded judges.
    harbor_verifier_import_path: Optional[str] = None
    # Optional independent judge transport; absent preserves the solver model route.
    judge_model_server: ModelServerRef | None = None
    harbor_verifier_kwargs: Optional[dict[str, Any]] = None
    # Required single-step captures; optional diagnostic artifacts remain best-effort.
    harbor_required_artifact_sources: list[str] = Field(default_factory=list)
    harbor_measurement_review_import_path: str | None = None

    # --- Timeouts ---
    # Override agent timeout (seconds). Replaces the task's own timeout entirely.
    # Use this to set a fixed timeout for all tasks regardless of task.toml.
    harbor_agent_override_timeout: Optional[int] = None
    # Cap agent timeout (seconds). Uses the task's own timeout but clamps it
    # to this maximum. Respects shorter per-task timeouts unlike harbor_agent_override_timeout.
    harbor_agent_max_timeout: Optional[int] = None
    # Override verifier timeout (seconds). Replaces the task's own verifier timeout.
    harbor_verifier_override_timeout: Optional[int] = None
    # Cap verifier timeout (seconds). Uses the task's own verifier timeout but
    # clamps it to this maximum.
    harbor_verifier_max_timeout: Optional[int] = None
    # Multiplier applied to all Harbor timeouts after override/cap. None = 1.0.
    harbor_timeout_multiplier: Optional[float] = None

    # --- Job output ---
    # Directory where Harbor writes job results and trial artifacts.
    harbor_jobs_dir: str = "jobs"

    # Keep Docker runtime images between trials (maps to EnvironmentConfig.delete=False).
    harbor_no_delete: bool = True

    # --- Model routing ---
    # NeMo Gym model server reference used to resolve Harbor model base URL.
    model_server: ModelServerRef


class HarborRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    instance_id: str


class HarborVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


def _find_trial_dir_with_result(job_dir: Path) -> Optional[Path]:
    if not job_dir.exists():
        return None
    for trial_dir in job_dir.iterdir():
        if trial_dir.is_dir() and (trial_dir / "result.json").exists():
            return trial_dir
    return None


async def run_harbor_job(
    job_config_dict: dict,
    admission_budget: Optional[dict] = None,
    *,
    resource_policy: HarborResourcePolicy = "require_explicit",
) -> str:
    """Runs a single Harbor Job and returns the *absolute* trial directory path.

    The trial directory contains:
    - result.json: Summary result with reward, agent_result, verifier_result, etc.
    - agent/trajectory.json: Full ATIF trajectory with per-step messages, tool
      calls, observations, and per-token logprobs.

    Harbor writes result.json and trajectory.json to disk even when the trial
    fails (e.g. verifier timeout, reward file not found, OOM).  We recover the
    trial directory after an exception so the caller can still use the partial
    trajectory for training.

    This runs inside a Ray remote worker (see ``runner_ray_remote``), whose cwd
    is inherited from wherever the Ray cluster was started (the Gym repo root,
    via ``ng_run``) -- NOT the `harbor_agent` server process's own cwd (which
    `ng_run` `cd`s into `responses_api_agents/harbor_agent/` before launching).
    `config.jobs_dir` is a relative path (e.g. `responses_api_agents/harbor_agent/
    jobs/...`), so it only resolves correctly from the *former*. We therefore
    MUST resolve to an absolute path before returning, since the caller
    (`HarborAgent.run()`) re-opens `result.json` from the server process's own
    cwd -- a relative path here silently resolves to the wrong (nonexistent,
    doubled) location there, and every trial looks like it failed even though
    it ran and wrote everything correctly to disk.
    """
    from harbor.job import Job
    from harbor.models.job.config import JobConfig

    if job_config_dict.get("source_jobs"):
        from responses_api_agents.harbor_agent.regrade_compat import install_regrade_compat

        install_regrade_compat()
    config = JobConfig(**job_config_dict)
    job = await Job.create(config)
    # The reviewed Harbor pin resolves/cache-downloads tasks in create(). Inspect
    # those exact paths before starting containers; newer Harbor no longer gives
    # omitted resources the old 1 CPU / 2048 MiB defaults.
    for download in job._task_download_results.values():
        HarborAgentUtils.validate_task_resources(download.path, config.environment, resource_policy=resource_policy)

    if admission_budget is not None:
        from responses_api_agents.harbor_agent.phase_budget import validate_job_admission

        validate_job_admission(job, **admission_budget)

    job_error = None
    try:
        await job.run()
    except Exception as e:
        job_error = e

    # Find the trial directory from the job output directory. Harbor writes
    # result.json before propagating most exceptions, so we can usually
    # recover the trial even when job.run() raised.
    job_dir = config.jobs_dir / config.job_name
    trial_dir = _find_trial_dir_with_result(job_dir)
    if trial_dir is not None:
        return str(trial_dir.resolve())

    # No trial directory found — re-raise the original error if there was one,
    # otherwise raise FileNotFoundError.
    if job_error is not None:
        raise job_error
    raise FileNotFoundError(f"No trial result found in {job_dir.resolve()}")


_RAY_WORKER_EVENT_LOOP: Optional[asyncio.AbstractEventLoop] = None


async def _run_harbor_job_with_transport(
    job_config_dict: dict,
    transport_config: dict,
    admission_budget: Optional[dict] = None,
    *,
    resource_policy: HarborResourcePolicy = "require_explicit",
) -> str:
    # Only the typed parent-selected pool settings cross the Ray boundary. Never
    # parse worker argv or forward the credential-bearing global configuration.
    cfg = GlobalAIOHTTPAsyncClientConfig.model_validate(transport_config)
    set_global_aiohttp_client(cfg)
    primary_error = None
    try:
        kwargs = {}
        if admission_budget is not None:
            kwargs["admission_budget"] = admission_budget
        if resource_policy != "require_explicit":
            kwargs["resource_policy"] = resource_policy
        return await run_harbor_job(job_config_dict, **kwargs)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            await global_aiohttp_client_exit_async()
        except BaseException as cleanup_error:
            if primary_error is None:
                raise
            primary_error.add_note(f"Worker HTTP cleanup failed: {type(cleanup_error).__name__}")
            logger.exception("Worker HTTP cleanup failed while preserving the original job error")


def _run_harbor_job_sync(
    job_config_dict: dict,
    transport_config: Optional[dict] = None,
    admission_budget: Optional[dict] = None,
    *,
    resource_policy: HarborResourcePolicy = "require_explicit",
) -> str:
    """Synchronous wrapper for run_harbor_job for use in Ray remote.

    Ray workers are long-lived processes. Reusing a single event loop per worker
    avoids cross-loop issues with global async state (e.g., LiteLLM logging worker
    queues) when multiple jobs execute sequentially in the same process.
    """
    global _RAY_WORKER_EVENT_LOOP
    if _RAY_WORKER_EVENT_LOOP is None or _RAY_WORKER_EVENT_LOOP.is_closed():
        _RAY_WORKER_EVENT_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_RAY_WORKER_EVENT_LOOP)
    task = _RAY_WORKER_EVENT_LOOP.create_task(
        _run_harbor_job_with_transport(
            job_config_dict, transport_config or {}, admission_budget, resource_policy=resource_policy
        )
    )
    try:
        return _RAY_WORKER_EVENT_LOOP.run_until_complete(task)
    except KeyboardInterrupt:
        # Ray cooperative cancellation interrupts the synchronous entrypoint.
        # Translate it once into native asyncio cancellation so Harbor can recover
        # artifacts and finalize its owned environment before this worker exits.
        task.cancel()
        cleanup_error = None
        while not task.done():
            try:
                _RAY_WORKER_EVENT_LOOP.run_until_complete(task)
            except KeyboardInterrupt:
                # A repeated cooperative signal must not inject a second asyncio
                # cancellation into a finalizer already recovering artifacts.
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                cleanup_error = type(exc).__name__
                logger.exception("Harbor cleanup failed during Ray cancellation")
                break
        if task.done() and not task.cancelled():
            error = task.exception()
            if error is not None:
                cleanup_error = type(error).__name__
        _write_cancellation_receipt(
            _cancellation_path(job_config_dict, "worker"),
            {"phase": "native_coroutine_settled", "cleanup_error": cleanup_error, "docker_cleanup_confirmed": False},
        )
        raise


@ray.remote(
    scheduling_strategy="SPREAD",
    runtime_env={
        "py_executable": sys.executable,
    },
)
def runner_ray_remote(runner: Callable, params: dict[str, Any]) -> Any:
    return runner(**params)


def _cancellation_path(config: dict, role: str) -> Path | None:
    if not config.get("jobs_dir") or not config.get("job_name"):
        return None
    return Path(config["jobs_dir"]) / config["job_name"] / f"gym-cancellation-{role}.json"


def _write_cancellation_receipt(path: Path | None, value: dict) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _publish(
            path, {**value, "job_directory": str(path.parent), "timestamp": datetime.now(timezone.utc).isoformat()}
        )
    except Exception:
        logger.exception("Could not persist owned Harbor cancellation receipt at %s", path)


async def _await_owned_ray_job(future, *, settle_timeout_s: float, receipt_path: Path | None = None) -> str:
    async def wait_for_result():
        return await future

    waiter = asyncio.create_task(wait_for_result())
    try:
        return await asyncio.shield(waiter)
    except asyncio.CancelledError:
        requested_at = datetime.now(timezone.utc)
        _write_cancellation_receipt(receipt_path, {"phase": "cancellation_requested", "cleanup_confirmed": False})
        # Never force-kill a worker: native finalizers need time to preserve
        # artifacts and stop only the environment belonging to this job.
        try:
            ray.cancel(future, force=False, recursive=True)
        except Exception:
            logger.exception("Could not request owned Harbor Ray job cancellation; cleanup remains unconfirmed")
        deadline = asyncio.get_running_loop().time() + settle_timeout_s
        worker_receipt = None
        worker_path = receipt_path.parent / "gym-cancellation-worker.json" if receipt_path else None
        while True:
            if worker_path is not None and worker_path.is_file():
                try:
                    candidate = json.loads(worker_path.read_text())
                    if (
                        isinstance(candidate, dict)
                        and "cleanup_error" in candidate
                        and (candidate["cleanup_error"] is None or isinstance(candidate["cleanup_error"], str))
                        and candidate.get("job_directory") == str(worker_path.parent)
                        and candidate.get("phase") == "native_coroutine_settled"
                        and datetime.fromisoformat(candidate["timestamp"]) >= requested_at
                    ):
                        worker_receipt = candidate
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            if waiter.done() and (worker_path is None or worker_receipt is not None):
                break
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                logger.error("Owned Harbor Ray job cancellation did not settle; cleanup remains unconfirmed")
                break
            try:
                if waiter.done():
                    await asyncio.sleep(min(0.05, remaining))
                else:
                    await asyncio.wait_for(asyncio.shield(waiter), timeout=remaining)
            except asyncio.CancelledError:
                continue
            except TimeoutError:
                logger.error("Owned Harbor Ray job cancellation did not settle; cleanup remains unconfirmed")
                break
            except Exception:
                # Ray can mark the ObjectRef cancelled before native artifact
                # collection and environment teardown finish in the worker.
                continue
        _write_cancellation_receipt(
            receipt_path,
            {
                "phase": (
                    "native_finalizer_error"
                    if worker_receipt and worker_receipt.get("cleanup_error")
                    else "native_finalizer_settled"
                    if worker_receipt
                    else "ray_wait_settled"
                    if waiter.done() and worker_path is None
                    else "settlement_unknown"
                ),
                "worker_receipt": worker_receipt,
                "settlement_timed_out": worker_path is not None and worker_receipt is None,
                "cleanup_confirmed": False,
                "note": "Ray wait completion alone does not prove Docker or native finalizer cleanup",
            },
        )
        if waiter.done():
            if not waiter.cancelled():
                waiter.exception()  # Retrieve the expected remote cancellation/failure.
        else:
            # Observe a late failure without cancelling the underlying ObjectRef.
            waiter.add_done_callback(lambda task: None if task.cancelled() else task.exception())
        raise


class HarborAgent(SimpleResponsesAPIAgent):
    ray_enabled = True
    config: HarborAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        app.post("/v1/responses")(self.responses)
        app.post("/run")(self.run)
        # Registered explicitly because this override replaces (rather than extends)
        # SimpleResponsesAPIAgent.setup_webserver(), which would otherwise add this route
        # by default. HarborAgent doesn't override compute_metrics/get_key_metrics, so this
        # uses AggregateMetricsMixin's generic reward-based aggregation (mean/max/min/median/std
        # via RewardProfiler) over the `reward` field already present on every /run response.
        app.post("/aggregate_metrics")(self.aggregate_metrics)
        return app

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        raise NotImplementedError

    async def run(self, body: HarborRunRequest) -> HarborVerifyResponse:
        async with self.sem:
            global_config_dict = get_global_config_dict()

            policy_model_name = self._resolve_model_name(global_config_dict)
            base_url = self._resolve_model_base_url(global_config_dict)
            from nemo_gym.rollout_correlation import current_rollout_id, maybe_rollout_id_from_run_body
            from nemo_gym.server_utils import rollout_path_prefix

            capture_id = current_rollout_id() or maybe_rollout_id_from_run_body(body)
            if global_config_dict.get("provider_dispatch_guard") and capture_id is None:
                raise ValueError("Guarded Harbor runs require a rollout capture identity")
            if capture_id is not None:
                base_url = base_url.removesuffix("/v1") + rollout_path_prefix(capture_id) + "/v1"
            judge_route = self._resolve_judge_route(global_config_dict, capture_id)
            run_timestamp = datetime.now(timezone.utc)
            run_id = self._build_run_id(run_timestamp)

            instance_id = body.instance_id
            dataset_alias, task_name = self._parse_instance_id(instance_id)

            output_file_dir = self._get_results_output_dir(policy_model_name, dataset_alias, run_timestamp)
            jobs_dir = self._get_jobs_output_dir(policy_model_name, dataset_alias, run_timestamp)
            job_name = self._build_job_name(run_id)

            responses_create_params = body.responses_create_params.model_dump(
                exclude_unset=True,
                exclude_none=True,
            )

            job_config_dict = self._build_job_config(
                dataset_alias,
                task_name,
                policy_model_name,
                base_url,
                job_name=job_name,
                jobs_dir=jobs_dir,
                responses_create_params=responses_create_params,
                gym_provider_dispatch_guard=global_config_dict.get("provider_dispatch_guard") is True,
                judge_route=judge_route,
            )

            failure_reason = None
            failure_kind = None
            failure_metadata = {}
            resource_admission = None
            observations = None
            try:
                params = dict(
                    job_config_dict=job_config_dict,
                    transport_config=GlobalAIOHTTPAsyncClientConfig.model_validate(global_config_dict).model_dump(
                        mode="json"
                    ),
                )
                if self.config.harbor_resource_policy != "require_explicit":
                    params["resource_policy"] = self.config.harbor_resource_policy
                if self.config.harbor_admission_finalization_allowance_s is not None:
                    drain = global_config_dict.get("drain_margin_s")
                    if (
                        isinstance(drain, bool)
                        or not isinstance(drain, (int, float))
                        or not math.isfinite(drain)
                        or drain <= 0
                    ):
                        raise ValueError("Harbor phase admission requires explicit positive collector drain_margin_s")
                    params["admission_budget"] = {
                        "drain_seconds": drain,
                        "operational_allowance_seconds": self.config.harbor_admission_finalization_allowance_s,
                    }
                runner = runner_ray_remote
                resource_options = {}
                if self.config.harbor_ray_task_num_cpus is not None:
                    resource_options["num_cpus"] = self.config.harbor_ray_task_num_cpus
                if self.config.harbor_ray_task_memory_bytes is not None:
                    resource_options["memory"] = self.config.harbor_ray_task_memory_bytes
                if self.config.harbor_ray_static_resource_preflight:
                    resource_admission = static_resource_feasibility(
                        await asyncio.to_thread(ray.nodes), resource_options
                    )
                    admission_path = jobs_dir / f"{job_name}.ray-admission.json"
                    admission_path.parent.mkdir(parents=True, exist_ok=True)
                    _publish(admission_path, resource_admission)
                    if not resource_admission["feasible"]:
                        raise StaticRayResourceError(
                            f"No ALIVE node in declared static Ray pool can satisfy {resource_admission['requested']}; "
                            f"see {admission_path}"
                        )
                if resource_options:
                    runner = runner_ray_remote.options(**resource_options)
                future = runner.remote(_run_harbor_job_sync, params)
                # Await the ObjectRef directly; to_thread(ray.get, ...) would pin
                # one executor thread per in-flight trial, so completed trials
                # would queue behind long-running ones.
                trial_dir_path = await _await_owned_ray_job(
                    future,
                    settle_timeout_s=self.config.harbor_ray_cancel_settle_timeout_s,
                    receipt_path=jobs_dir / job_name / "gym-cancellation-request.json",
                )
                trial_dir = Path(trial_dir_path)

                def _read_trial_files():
                    traj_path = trial_dir / "agent" / "trajectory.json"
                    flags_path = trial_dir / "agent" / "agent_error_flags.json"
                    return (
                        json.loads((trial_dir / "result.json").read_text()),
                        json.loads(traj_path.read_text()) if traj_path.exists() else None,
                        json.loads(flags_path.read_text()) if flags_path.exists() else {},
                    )

                # Trajectories can be many MB; parse off the event loop.
                trial_result, trajectory, agent_error_flags = await asyncio.to_thread(_read_trial_files)
                try:
                    observed = await asyncio.to_thread(
                        read_terminus_observations,
                        trial_dir=trial_dir,
                        capture_dir=global_config_dict.get("model_call_capture_dir"),
                        rollout_id=capture_id,
                    )
                    observations = observed.model_dump(mode="json")
                except Exception as exc:
                    # Optional reporting must not mask a native scientific result.
                    # Cancellation is a BaseException and still propagates.
                    observations = {
                        "source": "harbor:terminus_dispatch_cursor",
                        "records": [],
                        "gaps": [{"code": "harbor_solver_call_bindings_unavailable", "detail": type(exc).__name__}],
                    }

                # Extract reward from verifier result
                verifier_result = trial_result.get("verifier_result")
                failure_kind, failure_reason = measurement_failure(
                    HarborAgentUtils.with_native_validity(trial_result),
                    llm_failure=agent_error_flags.get("llm_call_failure"),
                )
                if failure_kind == "harbor:unclassified" and global_config_dict.get("provider_dispatch_guard") is True:
                    rejection = await asyncio.to_thread(
                        intent_bound_provider_rejection,
                        trial_result,
                        trial_dir=trial_dir,
                        capture_dir=global_config_dict.get("model_call_capture_dir"),
                        rollout_id=capture_id,
                        endpoint=base_url.rstrip("/") + "/chat/completions",
                    )
                    if rejection is None:
                        rejection = await asyncio.to_thread(
                            terminus_bound_provider_rejection,
                            trial_result,
                            trial_dir=trial_dir,
                            capture_dir=global_config_dict.get("model_call_capture_dir"),
                            rollout_id=capture_id,
                            endpoint=base_url.rstrip("/") + "/chat/completions",
                        )
                    if rejection is not None:
                        failure_kind, failure_reason = rejection
                if self.config.harbor_required_artifact_sources:
                    manifest_path = trial_dir / "artifacts" / "manifest.json"
                    try:
                        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
                    except (OSError, ValueError):
                        manifest = None
                    capture_kind, capture_reason = required_artifact_failure(
                        manifest, self.config.harbor_required_artifact_sources
                    )
                    trial_result = {**trial_result, "artifact_capture_manifest": manifest}
                    if capture_reason:
                        trial_result["artifact_capture_failure"] = capture_reason
                        if failure_reason is None:
                            failure_kind, failure_reason = capture_kind, capture_reason
                trial_result, review_kind, review_reason = apply_measurement_review(
                    trial_result,
                    trial_dir=trial_dir,
                    instance_id=instance_id,
                    import_path=self.config.harbor_measurement_review_import_path,
                )
                if review_reason and failure_reason is None:
                    failure_kind, failure_reason = review_kind, review_reason
                # Keep native rewards in metadata even when the Gym numeric field
                # needs a finite placeholder for an invalid measurement.
                try:
                    reward = HarborAgentUtils.extract_reward(verifier_result)
                except (TypeError, ValueError, OverflowError):
                    reward = 0.0
                if not math.isfinite(reward):
                    reward = 0.0

                # Convert Harbor outputs to NeMo Gym response items:
                # keep rich trajectory details, then overlay rollout token details when present.
                output_items = HarborAgentUtils.trial_result_to_responses(trial_result, trajectory)

                # Extract the initial instruction from the trajectory as input messages
                input_messages = HarborAgentUtils.extract_input_from_trajectory(trajectory)

                # Populate usage from trajectory final_metrics or agent_result
                usage, usage_provenance = HarborAgentUtils.extract_usage_with_provenance(trial_result, trajectory)
                trial_result = {**trial_result, "usage_provenance": usage_provenance}
                if agent_error_flags.get("llm_call_failure"):
                    # Preserve inference diagnostics alongside native scores.
                    # measurement_failure distinguishes provider failures from
                    # scored benchmark budget exhaustion.
                    trial_result = {**trial_result, "agent_error_flags": agent_error_flags}

            except Exception as e:
                logger.exception("Error running Harbor job for %s", instance_id)
                failure_reason = f"{type(e).__name__}: {e}"
                failure_kind = AGENT_RUN_ERROR
                # A wrapper error can happen after model execution. Preserve the
                # artifact location without claiming that no inference occurred.
                failure_metadata = {
                    "wrapper_error": {
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                        "job_dir": str(jobs_dir / job_name),
                        "model_execution_status": "not_started"
                        if isinstance(e, StaticRayResourceError)
                        else "unknown",
                    }
                }
                if resource_admission is not None:
                    failure_metadata["ray_resource_admission"] = resource_admission
                trial_result = None
                trajectory = None
                agent_error_flags = {}
                output_items = []
                input_messages = []
                usage = None
                reward = 0.0

            response = HarborAgentUtils.get_default_response_object()
            response["model"] = policy_model_name
            response["temperature"] = responses_create_params.get("temperature")
            response["top_p"] = responses_create_params.get("top_p")
            response["output"] = output_items
            if usage:
                response["usage"] = usage

            # Update responses_create_params with the actual input sent to the agent
            updated_params = body.responses_create_params
            if input_messages:
                updated_params = body.responses_create_params.model_copy(update={"input": input_messages})

            verify_response = HarborVerifyResponse(
                responses_create_params=updated_params,
                reward=reward,
                response=response,
                instance_id=instance_id,
                metadata=trial_result if trial_result else failure_metadata,
                mask_sample=failure_reason is not None,
                failure_kind=failure_kind,
                failure_reason=failure_reason,
                context_length_exceeded_error=int(agent_error_flags.get("context_length_exceeded", False)),
                memory_limit_exceeded_error=int(agent_error_flags.get("memory_limit_exceeded", False)),
                agent_timeout_error=int(
                    ((trial_result or {}).get("exception_info") or {}).get("exception_type") == "AgentTimeoutError"
                ),
            )

            if observations is not None:
                verify_response = verify_response.model_copy(update={"ng_agent_observations": observations})

            # Save result to disk (folder = run_id, file = task name)
            output_path = output_file_dir / run_id
            output_path.mkdir(parents=True, exist_ok=True)

            safe_instance_id = self._sanitize_path_component(instance_id)
            with open(output_path / f"{safe_instance_id}.json", "w") as f:
                json.dump(verify_response.model_dump(), f, indent=2)

            return verify_response

    def _get_results_output_dir(self, policy_model_name: str, dataset_alias: str, run_timestamp: datetime) -> Path:
        """Build immutable run output directory grouped by date/dataset/model."""
        date_key = run_timestamp.strftime("%Y%m%d")
        dataset_key = self._sanitize_path_component(dataset_alias)
        model_key = self._sanitize_path_component(self._extract_model_name(policy_model_name))
        return Path.cwd() / "results" / "runs" / date_key / dataset_key / model_key

    def _get_jobs_output_dir(self, policy_model_name: str, dataset_alias: str, run_timestamp: datetime) -> Path:
        """Build Harbor jobs directory grouped by date/dataset/model."""
        date_key = run_timestamp.strftime("%Y%m%d")
        dataset_key = self._sanitize_path_component(dataset_alias)
        model_key = self._sanitize_path_component(self._extract_model_name(policy_model_name))
        return Path(self.config.harbor_jobs_dir) / date_key / dataset_key / model_key

    @staticmethod
    def _parse_instance_id(instance_id: str) -> tuple[str, str]:
        """Parse instance id in the required form: <dataset_alias>::<task_name>."""
        dataset_alias, sep, task_name = instance_id.partition("::")
        dataset_alias = dataset_alias.strip()
        task_name = task_name.strip()
        if not sep or not dataset_alias or not task_name:
            raise ValueError(f"instance_id must be in the form '<dataset_alias>::<task_name>' (got: {instance_id!r})")
        return dataset_alias, task_name

    def _build_run_id(self, run_timestamp: datetime) -> str:
        """Build a compact run id (time + short hash) for immutable file naming."""
        time_key = run_timestamp.strftime("%H%M%S")
        return f"{time_key}_{uuid4().hex[:8]}"

    def _build_job_name(self, run_id: str) -> str:
        """Build a Harbor job name from run id only."""
        return run_id

    @staticmethod
    def _extract_model_name(policy_model_name: str) -> str:
        """Extract the final model name from a full path or HF-style identifier.

        '/lustre/.../nano-v3-sft-...-hf'  -> 'nano-v3-sft-...-hf'
        'Qwen/Qwen3-8B'                   -> 'Qwen3-8B'
        'my-model'                         -> 'my-model'
        """
        return Path(policy_model_name).name or policy_model_name

    def _sanitize_path_component(self, value: str) -> str:
        """Sanitize path components to avoid accidental nested directories."""
        sanitized = value.replace("/", "__").replace("\\", "__").replace(":", "__")
        sanitized = re.sub(r"\s+", "_", sanitized)
        sanitized = sanitized.strip("._")
        return sanitized or "unknown"

    def _resolve_model_name(self, global_config_dict: Any) -> str:
        """Bind output and Harbor identity to the selected model service."""
        service = get_first_server_config_dict(global_config_dict, self.config.model_server.name)
        declared = [service.get(key) for key in ("openai_model", "model") if service.get(key) is not None]
        alias = global_config_dict.get("policy_model_name")
        if any(not isinstance(value, str) or not value.strip() for value in declared):
            raise ValueError("Selected model service has an invalid model identity")
        if len(set(declared)) > 1:
            raise ValueError("Selected model service declares conflicting model identities")
        selected = declared[0] if declared else alias
        if not isinstance(selected, str) or not selected.strip():
            raise ValueError("Selected model service needs a model identity or legacy policy_model_name")
        if alias is not None and alias != selected:
            raise ValueError("policy_model_name disagrees with the selected model service")
        return selected

    def _resolve_model_base_url(self, global_config_dict: Any) -> str:
        """Resolve model base URL from required model_server reference."""
        server_name = self.config.model_server.name
        model_server_config = get_first_server_config_dict(
            global_config_dict,
            server_name,
        )
        return f"http://{model_server_config['host']}:{model_server_config['port']}/v1"

    def _resolve_judge_route(self, global_config_dict: dict, capture_id: str | None) -> tuple[str, str] | None:
        """Resolve an explicit judge service with the same rollout capture attribution."""
        ref = self.config.judge_model_server
        if ref is None:
            return None
        if not self.config.harbor_verifier_import_path:
            raise ValueError("judge_model_server requires a custom Harbor verifier")
        target = global_config_dict.get(ref.name)
        if not isinstance(target, Mapping) or set(target) != {"responses_api_models"}:
            raise ValueError("judge_model_server must reference an existing model service")
        try:
            service = get_first_server_config_dict(global_config_dict, ref.name)
        except (KeyError, IndexError, AttributeError, TypeError) as exc:
            raise ValueError("judge_model_server must reference an existing model service") from exc
        declared = [service.get(key) for key in ("openai_model", "model") if service.get(key) is not None]
        if not declared or any(not isinstance(value, str) or not value.strip() for value in declared):
            raise ValueError("Judge model service needs an explicit model identity")
        if len(set(declared)) != 1:
            raise ValueError("Judge model service declares conflicting model identities")
        host, port = service.get("host"), service.get("port")
        if (
            not isinstance(host, str)
            or not host
            or not isinstance(port, int)
            or isinstance(port, bool)
            or not 0 < port < 65536
        ):
            raise ValueError("Judge model service needs a valid host and port")
        if global_config_dict.get("provider_dispatch_guard") and capture_id is None:
            raise ValueError("Guarded Harbor judges require a rollout capture identity")
        from nemo_gym.server_utils import rollout_path_prefix

        prefix = rollout_path_prefix(capture_id) if capture_id is not None else ""
        return f"http://{host}:{port}{prefix}/v1", declared[0]

    def _build_job_config(
        self,
        dataset_alias: str,
        task_name: str,
        model_name: str,
        api_base: str,
        job_name: str,
        jobs_dir: Path,
        responses_create_params: Optional[dict[str, Any]] = None,
        gym_provider_dispatch_guard: bool = False,
        *,
        judge_route: tuple[str, str] | None = None,
    ) -> dict:
        """Build a Harbor JobConfig dict for a single task."""
        from harbor.models.job.config import DatasetConfig, JobConfig, RetryConfig
        from harbor.models.trial.config import (
            AgentConfig,
            EnvironmentConfig,
            VerifierConfig,
        )

        agent_kwargs: dict[str, Any] = {"api_base": api_base}
        if responses_create_params:
            agent_kwargs["responses_create_params"] = responses_create_params
            # Terminus-2 accepts temperature as a top-level kwarg for trajectory metadata.
            if "temperature" in responses_create_params:
                agent_kwargs["temperature"] = responses_create_params["temperature"]
        if self.config.harbor_agent_kwargs:
            agent_kwargs.update(self.config.harbor_agent_kwargs)

        if self.config.harbor_agent_import_path:
            # Bind the server's guard decision; never parse Hydra's CLI in a Ray worker.
            agent_kwargs["gym_provider_dispatch_guard"] = gym_provider_dispatch_guard is True

        agent_config = AgentConfig(
            name=self.config.harbor_agent_name if not self.config.harbor_agent_import_path else None,
            import_path=self.config.harbor_agent_import_path,
            model_name=model_name,
            override_timeout_sec=(
                float(self.config.harbor_agent_override_timeout)
                if self.config.harbor_agent_override_timeout is not None
                else None
            ),
            max_timeout_sec=(
                float(self.config.harbor_agent_max_timeout)
                if self.config.harbor_agent_max_timeout is not None
                else None
            ),
            kwargs=agent_kwargs,
        )

        dataset_source = self.config.harbor_datasets.get(dataset_alias)
        if dataset_source is None:
            available = ", ".join(sorted(self.config.harbor_datasets.keys()))
            raise ValueError(
                f"Unknown dataset alias in instance_id: {dataset_alias!r}. Available aliases: [{available}]"
            )

        has_local = bool(dataset_source.local_dataset_path)
        has_registry = bool(dataset_source.dataset_name)
        if has_local == has_registry:
            raise ValueError(
                f"Dataset alias {dataset_alias!r} must define exactly one source: "
                "local_dataset_path OR dataset_name[/dataset_version]."
            )

        environment_kwargs = {}
        if self.config.harbor_environment_kwargs:
            environment_kwargs.update(self.config.harbor_environment_kwargs)
        # Dataset alias-level workdir overrides global harbor_environment_kwargs.workdir.
        if dataset_source.workdir is not None:
            environment_kwargs["workdir"] = dataset_source.workdir

        environment_config = EnvironmentConfig(
            type=self.config.harbor_environment_type if not self.config.harbor_environment_import_path else None,
            import_path=self.config.harbor_environment_import_path,
            delete=not self.config.harbor_no_delete,
            kwargs=environment_kwargs,
        )

        verifier_kwargs = dict(self.config.harbor_verifier_kwargs or {})
        if self.config.harbor_verifier_import_path:
            verifier_kwargs.update(
                gym_model_api_base=judge_route[0] if judge_route else api_base,
                gym_model_name=judge_route[1] if judge_route else model_name,
                gym_provider_dispatch_guard=gym_provider_dispatch_guard is True,
            )
        verifier_config = VerifierConfig(
            import_path=self.config.harbor_verifier_import_path,
            kwargs=verifier_kwargs,
            override_timeout_sec=(
                float(self.config.harbor_verifier_override_timeout)
                if self.config.harbor_verifier_override_timeout is not None
                else None
            ),
            max_timeout_sec=(
                float(self.config.harbor_verifier_max_timeout)
                if self.config.harbor_verifier_max_timeout is not None
                else None
            ),
        )

        if has_registry:
            dataset_config = DatasetConfig(
                name=dataset_source.dataset_name,
                version=dataset_source.dataset_version,
                task_names=[task_name],
            )
        else:
            dataset_config = DatasetConfig(
                path=Path(dataset_source.local_dataset_path),
                task_names=[task_name],
            )

        job_config = JobConfig(
            job_name=job_name,
            jobs_dir=jobs_dir,
            timeout_multiplier=(
                self.config.harbor_timeout_multiplier if self.config.harbor_timeout_multiplier is not None else 1.0
            ),
            n_concurrent_trials=1,
            quiet=True,
            retry=RetryConfig(max_retries=0),
            environment=environment_config,
            verifier=verifier_config,
            agents=[agent_config],
            datasets=[dataset_config],
        )

        return job_config.model_dump(mode="json")


if __name__ == "__main__":
    HarborAgent.run_webserver()
