"""Planner problem schema: positive examples, negative cases, unit normalisation, defaults."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from spacepdhcg.planner import (
    FAMILIES,
    FAMILY_INFO,
    ProblemValidationError,
    load_problem,
    normalise_problem,
    schema_errors,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "planner"
EXAMPLE_FILES = {
    "hcw": "hcw_rendezvous.json",
    "powered_descent_3dof": "powered_descent_3dof.json",
    "powered_descent_6dof": "powered_descent_6dof.json",
    "low_thrust": "low_thrust.json",
}


def example(family: str) -> dict:
    return load_problem(EXAMPLES / EXAMPLE_FILES[family])


@pytest.mark.parametrize("family", FAMILIES)
def test_examples_validate_and_normalise(family: str) -> None:
    document = example(family)
    assert schema_errors(document) == []
    canonical = normalise_problem(document)
    info = FAMILY_INFO[family]
    assert "units" not in canonical
    assert len(canonical["initial_state"]) == info.state_dimension
    assert canonical["terminal"]["fixed"] == list(info.terminal_pattern)
    assert canonical["horizon"]["intervals"] >= 2
    # Canonical documents re-validate and normalise idempotently.
    assert normalise_problem(canonical) == canonical


def test_degrees_and_kilonewtons_are_converted_to_canonical_units() -> None:
    document = example("powered_descent_3dof")
    document["units"] = {
        "angle": "deg",
        "thrust": "kN",
        "position": "km",
        "velocity": "km/s",
        "time": "min",
    }
    document["vehicle"]["thrust"] = {"minimum": 0.0, "maximum": 15.0}
    document["initial_state"] = [0.03, -0.02, 0.15, 0.0, 0.0, -0.01, 2000.0]
    document["horizon"]["final_time"] = 20.0 / 60.0
    canonical = normalise_problem(document)
    assert canonical["vehicle"]["thrust"]["maximum"] == pytest.approx(15000.0)
    assert canonical["constraints"]["maximum_tilt"] == pytest.approx(math.radians(30.0))
    assert canonical["constraints"]["glide_slope"] == pytest.approx(math.radians(60.0))
    assert canonical["initial_state"][:3] == pytest.approx([30.0, -20.0, 150.0])
    assert canonical["initial_state"][5] == pytest.approx(-10.0)
    assert canonical["horizon"]["final_time"] == pytest.approx(20.0)


def test_specific_impulse_alternatives_are_exclusive() -> None:
    document = example("powered_descent_3dof")
    document["vehicle"]["exhaust_velocity"] = 2000.0
    with pytest.raises(ProblemValidationError, match="at most one"):
        normalise_problem(document)


def test_terminal_fixed_defaults_to_the_frozen_pattern() -> None:
    document = example("low_thrust")
    del document["terminal"]["fixed"]
    canonical = normalise_problem(document)
    assert canonical["terminal"]["fixed"] == [True] * 6 + [False]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda d: d.pop("family"), "family"),
        (lambda d: d.__setitem__("family", "orbit_raise"), "family"),
        (lambda d: d.__setitem__("schema_version", "0.9.0"), "schema_version"),
        (lambda d: d["horizon"].__setitem__("intervals", 1), "intervals"),
        (lambda d: d["horizon"].__setitem__("final_time", -5.0), "final_time"),
        (lambda d: d["horizon"].__setitem__("free_final_time", True), "free_final_time"),
        (lambda d: d.__setitem__("initial_state", [0.0] * 6), "initial_state"),
        (lambda d: d["terminal"].__setitem__("fixed", [True] * 7), "must be free"),
        (
            lambda d: d["terminal"].__setitem__(
                "fixed", [False, True, True, True, True, True, False]
            ),
            "must be fixed",
        ),
        (lambda d: d["constraints"].__setitem__("minimum_altitude", 5.0), "minimum_altitude"),
        (lambda d: d["solver"].__setitem__("backend", "cusolver"), "backend"),
        (lambda d: d["solver"].__setitem__("preset", "fixed_tight_pdhcg"), "pure_qoco requires"),
        (lambda d: d["units"].__setitem__("position", "furlong"), "position"),
        (
            lambda d: d.__setitem__("initial_state", [30.0, -20.0, 150.0, 0.0, 0.0, -10.0, 900.0]),
            "dry mass",
        ),
        (
            lambda d: d.__setitem__("warm_start", {"states": [[0.0] * 7], "controls": []}),
            "warm_start",
        ),
        (lambda d: d.__setitem__("vehicle", {"principal_inertia": [1.0, 1.0, 1.0]}), "6dof"),
        (lambda d: d.__setitem__("unknown_block", {}), "unknown_block"),
    ],
)
def test_negative_documents_fail_closed(mutate, match: str) -> None:
    document = example("powered_descent_3dof")
    mutate(document)
    with pytest.raises(ProblemValidationError, match=match):
        normalise_problem(document)


def test_six_dof_quaternion_must_be_unit() -> None:
    document = example("powered_descent_6dof")
    document["initial_state"][6] = 0.5
    with pytest.raises(ProblemValidationError, match="unit norm"):
        normalise_problem(document)


def test_low_thrust_target_inside_minimum_radius_is_rejected() -> None:
    document = example("low_thrust")
    document["terminal"]["state"][:3] = [6000.0, 0.0, 0.0]
    with pytest.raises(ProblemValidationError, match="minimum radius"):
        normalise_problem(document)


def test_hcw_box_bound_and_units_normalise() -> None:
    document = example("hcw")
    document["units"] = {"acceleration": "mm/s^2", "angular_rate": "deg/s"}
    document["constraints"] = {"maximum_acceleration": 50.0, "acceleration_bound": "box"}
    document["environment"] = {"mean_motion": math.degrees(1.13e-3)}
    canonical = normalise_problem(document)
    assert canonical["constraints"]["maximum_acceleration"] == pytest.approx(0.05)
    assert canonical["environment"]["mean_motion"] == pytest.approx(1.13e-3)
    assert canonical["constraints"]["acceleration_bound"] == "box"


def test_schema_errors_are_reported_together() -> None:
    document = example("hcw")
    document["horizon"]["intervals"] = 0
    document["solver"]["tolerance"] = -1.0
    errors = schema_errors(document)
    assert len(errors) >= 2
    assert any("intervals" in error for error in errors)
    assert any("tolerance" in error for error in errors)


def test_load_problem_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ProblemValidationError, match="JSON object"):
        load_problem(path)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ProblemValidationError, match="not valid JSON"):
        load_problem(path)


@pytest.mark.parametrize("family", FAMILIES)
def test_python_family_metadata_matches_native_defaults(
    planner_native_library: Path, family: str
) -> None:
    from spacepdhcg.planner.native_library import native_default_document

    native = native_default_document(family)
    info = FAMILY_INFO[family]
    assert native["state_order"] == list(info.state_names)
    assert native["control_order"] == list(info.control_names)
    assert native["terminal_fixed"] == list(info.terminal_pattern)
    for key, unit in info.units.items():
        if key in native["units"]:
            assert native["units"][key] == unit
    # The example documents only use the native defaults (or override them explicitly).
    canonical = normalise_problem(example(family))
    for block in ("vehicle", "environment", "constraints"):
        for key, value in canonical.get(block, {}).items():
            if (
                key in native[block]
                and not isinstance(value, (dict, list))
                and key != "acceleration_bound"
            ):
                if key in {"mass_flow_coefficient"} and "specific_impulse" in canonical.get(
                    "vehicle", {}
                ):
                    continue
                assert value == pytest.approx(native[block][key], rel=1e-9), (block, key)


def test_normalise_does_not_mutate_the_input() -> None:
    document = example("powered_descent_6dof")
    frozen = copy.deepcopy(document)
    normalise_problem(document)
    assert json.dumps(document, sort_keys=True) == json.dumps(frozen, sort_keys=True)
