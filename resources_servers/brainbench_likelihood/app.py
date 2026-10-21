import json
from typing import ClassVar

from pydantic import ConfigDict

from benchmarks.brainbench.protocol import mean_nll, score_pair
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)


class Config(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class VerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    uuid: str
    verifier_metadata: dict


class VerifyResponse(BaseVerifyResponse, VerifyRequest):
    scoring_details: dict


class Server(SimpleResourcesServer):
    config: Config
    ray_enabled = False

    async def verify(self, body: VerifyRequest) -> VerifyResponse:
        data = json.loads(body.response.output_text)
        values = [mean_nll(r["response"]) for r in data["receipts"]]
        assert values == data["mean_nll"]
        details = score_pair(values, body.verifier_metadata["target"])
        payload = body.model_dump()
        if payload.get("failure_kind") == "verifier_error":
            payload.update(mask_sample=False, failure_kind=None, failure_reason=None)
        payload.update(reward=details["reward"], scoring_details=details)
        return VerifyResponse.model_validate(payload)


if __name__ == "__main__":
    Server.run_webserver()
