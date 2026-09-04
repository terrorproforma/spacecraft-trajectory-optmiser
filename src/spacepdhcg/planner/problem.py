"""Planner problem documents: schema validation, unit normalisation, family metadata.

A *user* document may use the optional ``units`` block. :func:`normalise_problem`
converts it into the *canonical* document consumed by the native code (``spacepdhcg_plan``
and the C ABI transcription); canonical documents never carry a ``units`` block.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

SCHEMA_VERSION = "1.0.0"
FAMILIES = ("hcw", "powered_descent_3dof", "powered_descent_6dof", "low_thrust")
BACKENDS = ("pure_qoco", "pdhcg", "pdhcg_recovery", "cpu_reference")
PRESETS = ("frozen_adaptive_pure_qoco", "frozen_adaptive_pdhcg", "fixed_tight_pdhcg")
DEFAULT_TIME_LIMIT_SECONDS = 600.0


class ProblemValidationError(ValueError):
    """Raised for schema or semantic problems in a planner document."""


@dataclass(frozen=True, slots=True)
class FamilyInfo:
    """Static metadata for one planning family (canonical units)."""

    name: str
    state_dimension: int
    control_dimension: int
    terminal_dimension: int
    state_names: tuple[str, ...]
    control_names: tuple[str, ...]
    state_kinds: tuple[str, ...]
    control_kinds: tuple[str, ...]
    units: Mapping[str, str]
    physical_family: str
    frame: str

    @property
    def terminal_pattern(self) -> tuple[bool, ...]:
        return tuple(index < self.terminal_dimension for index in range(self.state_dimension))


FAMILY_INFO: dict[str, FamilyInfo] = {
    "hcw": FamilyInfo(
        name="hcw",
        state_dimension=6,
        control_dimension=3,
        terminal_dimension=6,
        state_names=("x", "y", "z", "vx", "vy", "vz"),
        control_names=("ax", "ay", "az"),
        state_kinds=("position",) * 3 + ("velocity",) * 3,
        control_kinds=("acceleration",) * 3,
        units={
            "position": "m",
            "velocity": "m/s",
            "acceleration": "m/s^2",
            "angular_rate": "rad/s",
            "time": "s",
        },
        physical_family="Hill-Clohessy-Wiltshire rendezvous",
        frame="Hill/LVLH relative Cartesian (x radial, y along-track, z cross-track)",
    ),
    "powered_descent_3dof": FamilyInfo(
        name="powered_descent_3dof",
        state_dimension=7,
        control_dimension=4,
        terminal_dimension=6,
        state_names=("x", "y", "z", "vx", "vy", "vz", "mass"),
        control_names=("thrust_x", "thrust_y", "thrust_z", "sigma"),
        state_kinds=("position",) * 3 + ("velocity",) * 3 + ("mass",),
        control_kinds=("thrust",) * 4,
        units={
            "position": "m",
            "velocity": "m/s",
            "mass": "kg",
            "thrust": "N",
            "angle": "rad",
            "time": "s",
        },
        physical_family="3-DoF powered descent",
        frame="local-level inertial Cartesian (z up)",
    ),
    "powered_descent_6dof": FamilyInfo(
        name="powered_descent_6dof",
        state_dimension=14,
        control_dimension=7,
        terminal_dimension=13,
        state_names=(
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "q0",
            "q1",
            "q2",
            "q3",
            "wx",
            "wy",
            "wz",
            "mass",
        ),
        control_names=(
            "thrust_x",
            "thrust_y",
            "thrust_z",
            "torque_x",
            "torque_y",
            "torque_z",
            "sigma",
        ),
        state_kinds=("position",) * 3
        + ("velocity",) * 3
        + ("dimensionless",) * 4
        + ("angular_rate",) * 3
        + ("mass",),
        control_kinds=("thrust",) * 3 + ("torque",) * 3 + ("thrust",),
        units={
            "position": "m",
            "velocity": "m/s",
            "mass": "kg",
            "thrust": "N",
            "torque": "N*m",
            "angle": "rad",
            "angular_rate": "rad/s",
            "inertia": "kg*m^2",
            "time": "s",
        },
        physical_family="6-DoF powered descent",
        frame="local-level inertial Cartesian (z up); body thrust/torque; scalar-first quaternion",
    ),
    "low_thrust": FamilyInfo(
        name="low_thrust",
        state_dimension=7,
        control_dimension=4,
        terminal_dimension=6,
        state_names=("x", "y", "z", "vx", "vy", "vz", "mass"),
        control_names=("thrust_x", "thrust_y", "thrust_z", "sigma"),
        state_kinds=("position",) * 3 + ("velocity",) * 3 + ("mass",),
        control_kinds=("thrust",) * 4,
        units={
            "position": "km",
            "velocity": "km/s",
            "mass": "kg",
            "thrust": "N",
            "gravitational_parameter": "km^3/s^2",
            "time": "s",
        },
        physical_family="low-thrust two-body transfer",
        frame="central-body inertial Cartesian",
    ),
}

# Multiplicative factors that convert a *user* unit into the SI-based reference unit
# for its quantity kind (metres, seconds, kilograms, newtons, radians).
_TO_SI: dict[str, dict[str, float]] = {
    "position": {"m": 1.0, "km": 1.0e3},
    "velocity": {"m/s": 1.0, "km/s": 1.0e3},
    "acceleration": {"m/s^2": 1.0, "mm/s^2": 1.0e-3, "km/s^2": 1.0e3},
    "mass": {"kg": 1.0, "t": 1.0e3},
    "thrust": {"N": 1.0, "kN": 1.0e3},
    "torque": {"N*m": 1.0, "kN*m": 1.0e3},
    "angle": {"rad": 1.0, "deg": math.pi / 180.0},
    "angular_rate": {"rad/s": 1.0, "deg/s": math.pi / 180.0},
    "time": {"s": 1.0, "min": 60.0, "h": 3600.0},
}


def _schema() -> dict[str, Any]:
    text = (
        resources.files("spacepdhcg.planner.schema")
        .joinpath("problem.schema.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(text)


_VALIDATOR: Draft202012Validator | None = None


def schema_validator() -> Draft202012Validator:
    global _VALIDATOR
    if _VALIDATOR is None:
        _VALIDATOR = Draft202012Validator(_schema())
    return _VALIDATOR


def load_problem(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Read a problem document from a path or mapping without validating it."""

    if isinstance(source, Mapping):
        return copy.deepcopy(dict(source))
    path = Path(source)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ProblemValidationError(f"cannot read problem file {path}: {error}") from error
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise ProblemValidationError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(document, dict):
        raise ProblemValidationError(f"{path} must contain a JSON object")
    return document


def schema_errors(document: Mapping[str, Any]) -> list[str]:
    """Return human-readable JSON-Schema violations (empty when valid)."""

    errors = []
    for error in sorted(schema_validator().iter_errors(document), key=lambda item: list(item.path)):
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        errors.append(f"{location}: {error.message}")
    return errors


def family_info(family: str) -> FamilyInfo:
    try:
        return FAMILY_INFO[family]
    except KeyError as error:
        raise ProblemValidationError(
            f"unknown family {family!r}; expected one of {FAMILIES}"
        ) from error


def _factor(units: Mapping[str, str], kind: str, canonical: Mapping[str, str]) -> float:
    """Factor converting a user value of ``kind`` into the family canonical unit."""

    if kind == "dimensionless":
        return 1.0
    canonical_unit = canonical.get(kind)
    if canonical_unit is None:
        return 1.0
    user_unit = units.get(kind, canonical_unit)
    table = _TO_SI.get(kind)
    if table is None:
        return 1.0
    if user_unit not in table:
        raise ProblemValidationError(f"unsupported unit {user_unit!r} for {kind}")
    if canonical_unit not in table:
        return 1.0
    return table[user_unit] / table[canonical_unit]


def _scale_list(values: list[float], factor: float) -> list[float]:
    return [float(value) * factor for value in values]


def normalise_problem(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a user document and return the canonical-unit document.

    Raises :class:`ProblemValidationError` with every schema violation joined, or with the
    first semantic problem (dimension mismatch, unsupported free/fixed pattern, ...).
    """

    errors = schema_errors(document)
    if errors:
        raise ProblemValidationError(
            "problem document failed schema validation:\n  " + "\n  ".join(errors)
        )
    result: dict[str, Any] = copy.deepcopy(dict(document))
    family = family_info(result["family"])
    units: Mapping[str, str] = result.pop("units", {}) or {}
    canonical = family.units

    def factor(kind: str) -> float:
        return _factor(units, kind, canonical)

    # Dimensions and terminal pattern (fail closed before any unit conversion).
    initial = result["initial_state"]
    if len(initial) != family.state_dimension:
        raise ProblemValidationError(
            f"initial_state must have {family.state_dimension} components for family "
            f"{family.name} ({', '.join(family.state_names)}); received {len(initial)}"
        )
    terminal = result["terminal"]
    if len(terminal["state"]) != family.state_dimension:
        raise ProblemValidationError(
            f"terminal.state must have {family.state_dimension} components for family {family.name}"
        )
    fixed = terminal.get("fixed")
    if fixed is None:
        terminal["fixed"] = list(family.terminal_pattern)
    else:
        if len(fixed) != family.state_dimension:
            raise ProblemValidationError(
                f"terminal.fixed must have {family.state_dimension} booleans for family "
                f"{family.name}"
            )
        for index, (flag, supported) in enumerate(zip(fixed, family.terminal_pattern, strict=True)):
            if bool(flag) != supported:
                raise ProblemValidationError(
                    f"terminal component {family.state_names[index]!r} must be "
                    f"{'fixed' if supported else 'free'} for family {family.name}: the frozen "
                    f"transcription pins exactly the first {family.terminal_dimension} components "
                    "and leaves the remainder (terminal mass) free"
                )
    for value in list(initial) + list(terminal["state"]):
        if not math.isfinite(float(value)):
            raise ProblemValidationError("initial_state and terminal.state must be finite")
    if result["horizon"].get("free_final_time", False):
        raise ProblemValidationError("free final time is not supported by schema 1.0.0")

    # Unit conversion of state-like arrays.
    result["initial_state"] = [
        float(value) * factor(kind) for value, kind in zip(initial, family.state_kinds, strict=True)
    ]
    terminal["state"] = [
        float(value) * factor(kind)
        for value, kind in zip(terminal["state"], family.state_kinds, strict=True)
    ]
    result["horizon"]["final_time"] = float(result["horizon"]["final_time"]) * factor("time")

    vehicle = result.get("vehicle", {}) or {}
    if "dry_mass" in vehicle:
        vehicle["dry_mass"] = float(vehicle["dry_mass"]) * factor("mass")
    if "exhaust_velocity" in vehicle:
        vehicle["exhaust_velocity"] = float(vehicle["exhaust_velocity"]) * factor("velocity")
    if "mass_flow_coefficient" in vehicle:
        # kg/s per thrust unit -> canonical kg/s per N (or per canonical thrust unit)
        vehicle["mass_flow_coefficient"] = (
            float(vehicle["mass_flow_coefficient"])
            * factor("mass")
            / factor("time")
            / factor("thrust")
        )
    if "thrust" in vehicle:
        thrust = vehicle["thrust"]
        for key in ("minimum", "maximum"):
            if key in thrust:
                thrust[key] = float(thrust[key]) * factor("thrust")
    if "maximum_torque" in vehicle:
        vehicle["maximum_torque"] = float(vehicle["maximum_torque"]) * factor("torque")
    if vehicle:
        result["vehicle"] = vehicle

    environment = result.get("environment", {}) or {}
    if "gravity" in environment:
        environment["gravity"] = _scale_list(environment["gravity"], factor("acceleration"))
    if "mean_motion" in environment:
        environment["mean_motion"] = float(environment["mean_motion"]) * factor("angular_rate")
    if "gravitational_parameter" in environment:
        environment["gravitational_parameter"] = (
            float(environment["gravitational_parameter"])
            * factor("position") ** 3
            / factor("time") ** 2
        )
    if environment:
        result["environment"] = environment

    constraints = result.get("constraints", {}) or {}
    for key, kind in (
        ("maximum_tilt", "angle"),
        ("glide_slope", "angle"),
        ("minimum_altitude", "position"),
        ("minimum_radius", "position"),
        ("maximum_angular_rate", "angular_rate"),
        ("maximum_acceleration", "acceleration"),
    ):
        if key in constraints:
            constraints[key] = float(constraints[key]) * factor(kind)
    if constraints:
        result["constraints"] = constraints

    solver = result.get("solver", {}) or {}
    if "time_limit_seconds" in solver:
        solver["time_limit_seconds"] = float(solver["time_limit_seconds"]) * factor("time")
    result["solver"] = solver

    warm = result.get("warm_start")
    if warm is not None:
        intervals = int(result["horizon"]["intervals"])
        states = warm.get("states", [])
        controls = warm.get("controls", [])
        if len(states) != intervals + 1 or any(
            len(row) != family.state_dimension for row in states
        ):
            raise ProblemValidationError(
                f"warm_start.states must be {intervals + 1} rows of {family.state_dimension} values"
            )
        if len(controls) != intervals or any(
            len(row) != family.control_dimension for row in controls
        ):
            raise ProblemValidationError(
                f"warm_start.controls must be {intervals} rows of {family.control_dimension} values"
            )
        for row in list(states) + list(controls):
            for value in row:
                if not math.isfinite(float(value)):
                    raise ProblemValidationError("warm_start arrays must be finite")
        warm["states"] = [[float(value) for value in row] for row in states]
        warm["controls"] = [[float(value) for value in row] for row in controls]

    # Family-specific semantic checks mirrored from the native parser so users get
    # the same message before any native code runs.
    _semantic_checks(result, family)
    return result


def _semantic_checks(document: dict[str, Any], family: FamilyInfo) -> None:
    vehicle = document.get("vehicle", {}) or {}
    constraints = document.get("constraints", {}) or {}
    initial = document["initial_state"]
    target = document["terminal"]["state"]
    if family.name in {"powered_descent_3dof", "powered_descent_6dof", "low_thrust"}:
        dry_mass = float(vehicle.get("dry_mass", 1000.0 if family.name != "low_thrust" else 200.0))
        if initial[-1] <= dry_mass:
            raise ProblemValidationError(
                f"initial mass {initial[-1]} must exceed the vehicle dry mass {dry_mass}"
            )
        thrust = vehicle.get("thrust", {}) or {}
        if "minimum" in thrust and "maximum" in thrust and thrust["minimum"] > thrust["maximum"]:
            raise ProblemValidationError(
                "vehicle.thrust.minimum may not exceed vehicle.thrust.maximum"
            )
        supplied = [
            key
            for key in ("mass_flow_coefficient", "exhaust_velocity", "specific_impulse")
            if key in vehicle
        ]
        if len(supplied) > 1:
            raise ProblemValidationError(
                "specify at most one of mass_flow_coefficient, exhaust_velocity, specific_impulse"
            )
    if family.name in {"powered_descent_3dof", "powered_descent_6dof"}:
        if initial[2] < 0.0 or target[2] < 0.0:
            raise ProblemValidationError("powered-descent altitude (z) must be non-negative")
        if constraints.get("minimum_altitude", 0.0) != 0.0:
            raise ProblemValidationError(
                "minimum_altitude other than 0 is not supported by the frozen powered-descent "
                "transcription (altitude bound is z >= 0)"
            )
        half_pi = math.pi / 2.0
        for key in ("maximum_tilt", "glide_slope"):
            if key in constraints and not 0.0 < constraints[key] < half_pi:
                raise ProblemValidationError(f"{key} must lie strictly inside (0, pi/2) radians")
        if family.name == "powered_descent_3dof" and (
            "principal_inertia" in vehicle
            or "maximum_torque" in vehicle
            or "maximum_angular_rate" in constraints
        ):
            raise ProblemValidationError(
                "inertia, torque, and angular-rate parameters apply only to powered_descent_6dof"
            )
        if family.name == "powered_descent_6dof":
            for label, state in (("initial", initial), ("target", target)):
                norm = math.sqrt(sum(float(state[index]) ** 2 for index in range(6, 10)))
                if abs(norm - 1.0) > 1.0e-9:
                    raise ProblemValidationError(
                        f"{label} quaternion [q0, q1, q2, q3] must have unit norm "
                        f"(found {norm:.12g})"
                    )
    if family.name == "low_thrust":
        minimum_radius = float(constraints.get("minimum_radius", 6500.0))
        for label, state in (("initial", initial), ("target", target)):
            radius = math.sqrt(sum(float(state[index]) ** 2 for index in range(3)))
            if radius < minimum_radius:
                raise ProblemValidationError(
                    f"{label} radius {radius:.6g} km lies inside the minimum radius "
                    f"{minimum_radius} km"
                )
        thrust = vehicle.get("thrust", {}) or {}
        if thrust.get("minimum", 0.0) != 0.0:
            raise ProblemValidationError(
                "a positive thrust.minimum is not supported by the frozen low-thrust transcription"
            )
    if family.name == "hcw" and constraints.get("acceleration_bound", "norm") not in {
        "norm",
        "box",
    }:
        raise ProblemValidationError("acceleration_bound must be 'norm' or 'box'")
    solver = document.get("solver", {}) or {}
    backend = solver.get("backend", "pure_qoco")
    preset = solver.get("preset")
    if preset is not None:
        if backend == "pure_qoco" and preset != "frozen_adaptive_pure_qoco":
            raise ProblemValidationError(
                "backend pure_qoco requires preset frozen_adaptive_pure_qoco"
            )
        if backend == "pdhcg_recovery" and preset != "fixed_tight_pdhcg":
            raise ProblemValidationError("backend pdhcg_recovery requires preset fixed_tight_pdhcg")
        if backend == "pdhcg" and preset == "frozen_adaptive_pure_qoco":
            raise ProblemValidationError(
                "backend pdhcg cannot use the pure-QOCO preset; choose frozen_adaptive_pdhcg or "
                "fixed_tight_pdhcg"
            )
    minimum = solver.get("minimum_outer_iterations", 1)
    maximum = solver.get("maximum_outer_iterations", 30)
    if minimum > maximum:
        raise ProblemValidationError(
            "minimum_outer_iterations may not exceed maximum_outer_iterations"
        )


def apply_defaults(
    document: dict[str, Any], *, time_limit_seconds: float | None = None
) -> dict[str, Any]:
    """Fill CLI-level defaults (time limit) into a canonical document in place."""

    solver = document.setdefault("solver", {})
    if "time_limit_seconds" not in solver:
        solver["time_limit_seconds"] = (
            DEFAULT_TIME_LIMIT_SECONDS if time_limit_seconds is None else float(time_limit_seconds)
        )
    elif time_limit_seconds is not None:
        solver["time_limit_seconds"] = float(time_limit_seconds)
    return document


def dump_canonical(document: Mapping[str, Any]) -> str:
    """Deterministic JSON text for the native executable / C ABI."""

    return json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
