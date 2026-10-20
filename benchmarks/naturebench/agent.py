"""Harbor 0.24 Terminus adapter using the existing Gym LLM transport."""
import os

from harbor.agents.terminus_2.terminus_2 import Terminus2
from responses_api_agents.harbor_agent.custom_agents.llms.nemo_gym_llm import NemoGymLLM

MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"


class NatureBenchNemotronAgent(Terminus2):
    def __init__(self, *args, **kwargs):
        if kwargs.get("enable_summarize", False):
            raise ValueError("Campaign summarization is disabled")
        kwargs["enable_summarize"] = False
        super().__init__(*args, **kwargs)

    def _init_llm(self, *, model_name, api_base, collect_rollout_details,
                  model_info, llm_kwargs, **kwargs):
        expected = os.environ.get("CAMPAIGN_SOLVER_BASE_URL")
        if model_name != MODEL or not expected or api_base != expected:
            raise ValueError("Explicit campaign model and solver endpoint required")
        options = dict(llm_kwargs or {})
        allowed = {"responses_create_params", "timeout_sec"}
        if set(options) - allowed:
            raise ValueError("Unrecognized Gym transport options")
        return NemoGymLLM(model_name=MODEL, api_base=api_base,
                          collect_rollout_details=collect_rollout_details,
                          model_info=model_info, **options)
