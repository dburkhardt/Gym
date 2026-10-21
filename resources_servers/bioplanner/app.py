import json
import os
import shlex
from pathlib import Path
from typing import ClassVar

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.sandbox.providers.local.provider import LocalProvider
from nemo_gym.sandbox.scoring_workspace import scoring_workspace


class Config(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class Request(BaseVerifyRequest):
    id: str
    verifier_metadata: dict


class Response(BaseVerifyResponse, Request):
    metrics: dict | None
    scoring_note: str
    scoring_diagnostics: dict


async def run_score(
    payload: dict, *, timeout_s: float = 300, python: str | None = None, worker: Path | None = None
) -> dict:
    """Run trusted creator metrics with shared owned-process cleanup, never generated code."""
    provider = LocalProvider(concurrency=2)
    async with scoring_workspace(provider) as handle:
        request = handle.raw["workspace"] / "request.json"
        request.write_text(json.dumps(payload))
        command = shlex.join(
            [
                python or os.environ["BIOPLANNER_SCORER_PYTHON"],
                str((worker or Path(__file__).with_name("score_worker.py")).resolve()),
            ]
        )
        result = await provider.exec(handle, command + " < " + shlex.quote(str(request)), timeout_s=timeout_s)
        if result.error_type or result.return_code != 0:
            return {
                "metrics": None,
                "scorer_error": result.stderr or result.error_type or "Native scorer failed",
                "scorer_return_code": result.return_code,
                "scorer_error_type": result.error_type,
                "scorer_stdout": result.stdout,
                "scorer_stderr": result.stderr,
            }
        try:
            parsed = json.loads(result.stdout)
            if not isinstance(parsed, dict):
                raise ValueError("Scorer output must be an object")
            return parsed
        except (ValueError, TypeError) as error:
            return {
                "metrics": None,
                "scorer_error": "Malformed scorer output: " + str(error),
                "scorer_stdout": result.stdout,
                "scorer_stderr": result.stderr,
            }


class Server(SimpleResourcesServer):
    config: Config
    ray_enabled = False

    async def verify(self, body: Request) -> Response:
        answer = body.response.output_text or ""
        extraction = "raw"
        if os.environ.get("BIOPLANNER_EXTRACT_CODE_FENCE") == "1":
            from resources_servers.bioplanner.normalization import extract_single_code_block

            answer, extraction = extract_single_code_block(answer)
        payload = {**body.verifier_metadata, "answer": answer}
        result = await run_score(payload)
        result["extraction"] = extraction
        result["scored_code"] = answer
        metrics = result.get("metrics")
        value = metrics.get("argument_recall") if metrics else None
        extra = (
            {}
            if value is not None
            else {
                "mask_sample": True,
                "failure_kind": "bioplanner:scorer_or_reference_invalid",
                "failure_reason": str(result),
            }
        )
        return Response(
            **body.model_dump(by_alias=True),
            reward=value if value is not None else 0.0,
            metrics=metrics,
            scoring_diagnostics={k: v for k, v in result.items() if k != "metrics"},
            scoring_note="Reward is argument recall bookkeeping only; report all native metrics. Nulls are not zero. Provided definitions inflate function metrics.",
            **extra,
        )


if __name__ == "__main__":
    Server.run_webserver()
