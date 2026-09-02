from pathlib import Path

import numpy as np

from spacepdhcg.orbitweaver import load_paper2_instance_contract


def _contract():
    return load_paper2_instance_contract(Path(__file__).resolve().parents[1])


def test_paper2_physical_contract_is_schema_valid_and_deterministic() -> None:
    first = _contract()
    second = _contract()
    assert first.contract_id == "orbitweaver-earth-leo-v1"
    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64

    state = first.target_state(17, 23)
    replay = second.target_state(17, 23)
    np.testing.assert_array_equal(state.position, replay.position)
    np.testing.assert_array_equal(state.velocity, replay.velocity)
    assert state.epoch == replay.epoch


def test_generated_targets_are_physical_circular_earth_orbits() -> None:
    contract = _contract()
    state = contract.target_state(31, 7)
    radius = float(np.linalg.norm(state.position))
    speed = float(np.linalg.norm(state.velocity))
    body = contract.payload["central_body"]
    generator = contract.payload["target_generator"]

    assert generator["minimum_radius"] <= radius <= generator["maximum_radius"]
    assert abs(float(state.position @ state.velocity)) <= 1e-5
    assert abs(speed - np.sqrt(body["gravitational_parameter"] / radius)) <= 1e-10
    assert not np.array_equal(state.position, contract.target_state(31, 8).position)


def test_robust_scenarios_are_symmetric_and_probability_normalised() -> None:
    contract = _contract()
    scenarios = [contract.scenario_factors(index, 5) for index in range(5)]
    assert sum(item.probability for item in scenarios) == 1.0
    assert scenarios[0].gravity_scale < 1.0 < scenarios[-1].gravity_scale
    assert scenarios[0].thrust_scale > 1.0 > scenarios[-1].thrust_scale
    assert scenarios[0].service_time_scale < 1.0 < scenarios[-1].service_time_scale
