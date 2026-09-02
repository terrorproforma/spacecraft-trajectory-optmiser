"""Versioned deterministic physical instances for the frozen Paper 2 matrix."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
_MASK64 = (1 << 64) - 1


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31)


def _unit_interval(seed: int, target: int, component: int) -> float:
    mixed = _splitmix64(seed ^ (target * 0xD1342543DE82EF95) ^ component)
    return (mixed >> 11) * (1.0 / (1 << 53))


@dataclass(frozen=True, slots=True)
class CartesianState:
    position: FloatArray
    velocity: FloatArray
    epoch: float


@dataclass(frozen=True, slots=True)
class ScenarioFactors:
    probability: float
    gravity_scale: float
    thrust_scale: float
    service_time_scale: float


@dataclass(frozen=True, slots=True)
class Paper2InstanceContract:
    payload: dict[str, Any]
    sha256: str

    @property
    def contract_id(self) -> str:
        return str(self.payload["contract_id"])

    def target_state(self, target: int, epoch_index: int) -> CartesianState:
        if target < 0 or epoch_index < 0:
            raise ValueError("target and epoch indices must be non-negative")
        generator = self.payload["target_generator"]
        body = self.payload["central_body"]
        epoch_grid = self.payload["epoch_grid"]
        seed = int(generator["seed"])
        radius = float(generator["minimum_radius"]) + _unit_interval(seed, target, 0) * (
            float(generator["maximum_radius"]) - float(generator["minimum_radius"])
        )
        inclination = float(generator["maximum_inclination"]) * _unit_interval(
            seed, target, 1
        )
        ascending_node = 2.0 * np.pi * _unit_interval(seed, target, 2)
        initial_phase = 2.0 * np.pi * _unit_interval(seed, target, 3)
        epoch = float(epoch_grid["initial_epoch"]) + epoch_index * float(epoch_grid["spacing"])
        mean_motion = np.sqrt(float(body["gravitational_parameter"]) / radius**3)
        phase = initial_phase + mean_motion * epoch

        in_plane_position = np.asarray(
            [radius * np.cos(phase), radius * np.sin(phase), 0.0],
            dtype=np.float64,
        )
        speed = np.sqrt(float(body["gravitational_parameter"]) / radius)
        in_plane_velocity = np.asarray(
            [-speed * np.sin(phase), speed * np.cos(phase), 0.0],
            dtype=np.float64,
        )
        cos_node, sin_node = np.cos(ascending_node), np.sin(ascending_node)
        cos_inc, sin_inc = np.cos(inclination), np.sin(inclination)
        rotation = np.asarray(
            [
                [cos_node, -sin_node * cos_inc, sin_node * sin_inc],
                [sin_node, cos_node * cos_inc, -cos_node * sin_inc],
                [0.0, sin_inc, cos_inc],
            ],
            dtype=np.float64,
        )
        return CartesianState(
            position=rotation @ in_plane_position,
            velocity=rotation @ in_plane_velocity,
            epoch=epoch,
        )

    def scenario_factors(self, scenario: int, count: int) -> ScenarioFactors:
        if count <= 0 or not 0 <= scenario < count:
            raise ValueError("scenario index must lie within a positive scenario count")
        generator = self.payload["scenario_generator"]
        quantile = 0.0 if count == 1 else 2.0 * scenario / (count - 1) - 1.0
        return ScenarioFactors(
            probability=1.0 / count,
            gravity_scale=1.0 + quantile * float(generator["gravity_relative_spread"]),
            thrust_scale=1.0 - quantile * float(generator["thrust_relative_spread"]),
            service_time_scale=1.0
            + quantile * float(generator["service_time_relative_spread"]),
        )


def load_paper2_instance_contract(repository: Path) -> Paper2InstanceContract:
    contract_path = repository / "benchmarks" / "paper2_instances.json"
    schema_path = (
        repository / "experiments" / "schema" / "paper2_instance_contract.schema.json"
    )
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(payload)
    if payload["target_generator"]["minimum_radius"] <= payload["central_body"][
        "equatorial_radius"
    ]:
        raise ValueError("target radius must remain above the central body")
    if payload["target_generator"]["maximum_radius"] <= payload["target_generator"][
        "minimum_radius"
    ]:
        raise ValueError("target radius range must be increasing")
    return Paper2InstanceContract(
        payload=payload,
        sha256=hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
    )
