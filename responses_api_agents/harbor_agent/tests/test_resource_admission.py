# SPDX-License-Identifier: Apache-2.0
import pytest

from responses_api_agents.harbor_agent.resource_admission import static_resource_feasibility


def node(cpu=1, memory=22, alive=True):
    return {"Alive": alive, "NodeID": "node", "Resources": {"CPU": cpu, "memory": memory}}


def test_requires_one_alive_node_not_aggregate_capacity():
    assert not static_resource_feasibility([node(memory=11), node(memory=11)], {"memory": 22})["feasible"]
    assert not static_resource_feasibility([node(alive=False)], {"memory": 22})["feasible"]
    assert not static_resource_feasibility([], {"memory": 22})["feasible"]


def test_exact_boundary_and_busy_total_remain_feasible():
    busy = node()
    busy["AvailableResources"] = {"CPU": 0, "memory": 0}
    receipt = static_resource_feasibility([busy], {"memory": 22})
    assert receipt["feasible"] and receipt["requested"] == {"CPU": 1, "memory": 22}
    assert not static_resource_feasibility([node(cpu=0.5)], {"memory": 22})["feasible"]
    assert static_resource_feasibility([node(cpu=0.5)], {"num_cpus": 0.5, "memory": 22})["feasible"]


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf"), "22"])
def test_invalid_metadata_fails_closed(value):
    with pytest.raises(ValueError):
        static_resource_feasibility([node(memory=value)], {"memory": 22})
    with pytest.raises(ValueError):
        static_resource_feasibility([node()], {"memory": value})
