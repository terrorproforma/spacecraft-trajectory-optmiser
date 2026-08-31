"""Strict standard-library validation for compact Paper 1 result records."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

PAPER1_SCHEMA_VERSION: Final = "1.0.0"
PAPER1_FAMILIES: Final = frozenset(
    {
        "P1-A-banded",
        "P1-B-hcw",
        "P1-C-pd3",
        "P1-D-pd6",
        "P1-E-low-thrust",
        "P1-F-robust-pd",
    }
)
PAPER1_SOLVERS: Final = frozenset(
    {
        "clarabel-cpu",
        "osqp-cpu",
        "pdhcg-upstream-one-shot",
        "spacepdhcg-persistent",
        "qoco-gpu",
        "cuclarabel",
        "structured-pipg",
        "hybrid-pdhcg-ipm",
    }
)
PAPER1_STATUSES: Final = frozenset(
    {"qualified", "unqualified", "unsupported", "oom", "timeout", "failed"}
)
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_REQUIRED_TOP_LEVEL: Final = frozenset(
    {
        "schema_version",
        "identity",
        "dimensions",
        "quality",
        "timing",
        "work",
        "resources",
        "aggregation",
        "artifacts",
    }
)
_ALLOWED_TOP_LEVEL: Final = _REQUIRED_TOP_LEVEL | {"notes"}
_REQUIRED_TIMING: Final = (
    "topology_seconds",
    "coefficient_seconds",
    "workspace_create_seconds",
    "update_seconds",
    "scaling_seconds",
    "h2d_seconds",
    "solve_seconds",
    "residual_seconds",
    "replay_seconds",
    "acceptance_seconds",
    "d2h_seconds",
    "collective_seconds",
    "cqp_total_seconds",
    "scvx_total_seconds",
)
_REQUIRED_QUALITY: Final = (
    "objective",
    "canonical_primal_residual",
    "canonical_dual_residual",
    "canonical_cone_residual",
    "canonical_gap",
    "dynamics_residual",
    "path_residual",
    "terminal_residual",
    "virtual_control_residual",
    "nonanticipativity_residual",
    "risk_epigraph_residual",
)
_REQUIRED_RESOURCE: Final = (
    "peak_device_bytes",
    "reserved_device_bytes",
    "h2d_bytes",
    "d2h_bytes",
    "collective_bytes",
    "collective_count",
    "energy_joules",
    "topology_allocation_count_after_create",
)
_REQUIRED_ARTIFACTS: Final = ("manifest", "raw", "stdout", "stderr")


class Paper1ResultError(ValueError):
    """Raised when a compact result violates the frozen Paper 1 contract."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Paper1ResultError(f"{name} must be an object")
    return value


def _require_keys(value: Mapping[str, Any], required: Sequence[str], name: str) -> None:
    missing = sorted(set(required) - set(value))
    if missing:
        raise Paper1ResultError(f"{name} is missing required keys: {', '.join(missing)}")


def _reject_unknown_keys(
    value: Mapping[str, Any],
    allowed: Sequence[str] | set[str] | frozenset[str],
    name: str,
) -> None:
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise Paper1ResultError(f"{name} contains unknown keys: {', '.join(unknown)}")


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise Paper1ResultError(f"{name} must be a non-empty string")
    return value


def _nonnegative_int_or_none(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Paper1ResultError(f"{name} must be a non-negative integer or null")
    return value


def _finite_or_none(value: Any, name: str, *, nonnegative: bool = False) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Paper1ResultError(f"{name} must be a finite number or null")
    result = float(value)
    if not math.isfinite(result):
        raise Paper1ResultError(f"{name} must be finite when supplied")
    if nonnegative and result < 0.0:
        raise Paper1ResultError(f"{name} must be non-negative when supplied")
    return result


def _validate_identity(identity: Mapping[str, Any]) -> None:
    required = (
        "run_id",
        "repository_commit",
        "family",
        "instance_id",
        "solver",
        "policy",
        "status",
    )
    allowed = (*required, "hardware_id", "precision", "warm_start", "cold_start")
    _require_keys(identity, required, "identity")
    _reject_unknown_keys(identity, allowed, "identity")
    _nonempty_string(identity["run_id"], "identity.run_id")
    commit = _nonempty_string(identity["repository_commit"], "identity.repository_commit")
    if _SHA40.fullmatch(commit) is None:
        raise Paper1ResultError("identity.repository_commit must be a 40-character lowercase SHA")
    if identity["family"] not in PAPER1_FAMILIES:
        raise Paper1ResultError(f"unknown Paper 1 family: {identity['family']!r}")
    if identity["solver"] not in PAPER1_SOLVERS:
        raise Paper1ResultError(f"unknown Paper 1 solver: {identity['solver']!r}")
    if identity["status"] not in PAPER1_STATUSES:
        raise Paper1ResultError(f"unknown Paper 1 status: {identity['status']!r}")
    _nonempty_string(identity["instance_id"], "identity.instance_id")
    _nonempty_string(identity["policy"], "identity.policy")
    if identity.get("hardware_id") is not None:
        _nonempty_string(identity["hardware_id"], "identity.hardware_id")
    if identity.get("precision") not in {None, "float32", "float64", "mixed"}:
        raise Paper1ResultError("identity.precision is invalid")
    for key in ("warm_start", "cold_start"):
        if identity.get(key) is not None and not isinstance(identity[key], bool):
            raise Paper1ResultError(f"identity.{key} must be boolean or null")


def _validate_dimensions(dimensions: Mapping[str, Any]) -> None:
    required = (
        "intervals",
        "scenarios",
        "gpus",
        "state_dimension",
        "control_dimension",
        "variables",
        "scalar_rows",
        "affine_rows",
        "q_nonzeros",
        "a_nonzeros",
        "f_nonzeros",
        "cone_inventory",
    )
    allowed = (*required, "topology_bytes", "numeric_bytes")
    _require_keys(dimensions, required, "dimensions")
    _reject_unknown_keys(dimensions, allowed, "dimensions")
    positive = ("intervals", "scenarios", "state_dimension", "control_dimension", "variables")
    for key in positive:
        value = _nonnegative_int_or_none(dimensions[key], f"dimensions.{key}")
        if value is None or value < 1:
            raise Paper1ResultError(f"dimensions.{key} must be a positive integer")
    for key in ("gpus", "scalar_rows", "affine_rows", "q_nonzeros", "a_nonzeros", "f_nonzeros"):
        if _nonnegative_int_or_none(dimensions[key], f"dimensions.{key}") is None:
            raise Paper1ResultError(f"dimensions.{key} may not be null")
    for key in ("topology_bytes", "numeric_bytes"):
        if key in dimensions:
            _nonnegative_int_or_none(dimensions[key], f"dimensions.{key}")
    inventory = _mapping(dimensions["cone_inventory"], "dimensions.cone_inventory")
    for key, value in inventory.items():
        _nonempty_string(key, "cone inventory key")
        if _nonnegative_int_or_none(value, f"cone_inventory.{key}") is None:
            raise Paper1ResultError(f"cone_inventory.{key} may not be null")


def _validate_quality(quality: Mapping[str, Any], status: str) -> None:
    allowed = {
        "qualified",
        *_REQUIRED_QUALITY,
        "reference_objective",
        "objective_gap",
        "native_primal_residual",
        "native_dual_residual",
        "ct_error_estimate",
        "requested_tolerance",
        "achieved_residual",
    }
    _require_keys(quality, ("qualified", *_REQUIRED_QUALITY), "quality")
    _reject_unknown_keys(quality, allowed, "quality")
    if not isinstance(quality["qualified"], bool):
        raise Paper1ResultError("quality.qualified must be boolean")
    for key, value in quality.items():
        if key == "qualified":
            continue
        _finite_or_none(value, f"quality.{key}", nonnegative=key != "objective")
    if status == "qualified" and not quality["qualified"]:
        raise Paper1ResultError("qualified identity status requires quality.qualified=true")
    if status != "qualified" and quality["qualified"]:
        raise Paper1ResultError("only identity.status=qualified may set quality.qualified=true")
    if quality["qualified"]:
        for key in _REQUIRED_QUALITY:
            if quality[key] is None:
                raise Paper1ResultError(f"qualified result may not omit quality.{key}")


def _validate_timing(timing: Mapping[str, Any]) -> None:
    allowed = (*_REQUIRED_TIMING, "accepted_trajectory_seconds")
    _require_keys(timing, _REQUIRED_TIMING, "timing")
    _reject_unknown_keys(timing, allowed, "timing")
    values = {
        key: _finite_or_none(value, f"timing.{key}", nonnegative=True)
        for key, value in timing.items()
    }
    cqp_total = values["cqp_total_seconds"]
    scvx_total = values["scvx_total_seconds"]
    if cqp_total is not None and scvx_total is not None and scvx_total + 1.0e-15 < cqp_total:
        raise Paper1ResultError("timing.scvx_total_seconds may not be below cqp_total_seconds")


def _validate_work(work: Mapping[str, Any]) -> None:
    integer_fields = (
        "outer_iterations",
        "inner_iterations",
        "matvecs",
        "cone_projections",
        "factorisations",
        "accepted_steps",
        "rejected_steps",
        "resolved_steps",
    )
    allowed = (*integer_fields, "polish_used", "scaling_refreshes")
    _require_keys(work, (*integer_fields, "polish_used"), "work")
    _reject_unknown_keys(work, allowed, "work")
    for key in integer_fields:
        _nonnegative_int_or_none(work[key], f"work.{key}")
    if "scaling_refreshes" in work:
        _nonnegative_int_or_none(work["scaling_refreshes"], "work.scaling_refreshes")
    polish = work["polish_used"]
    if polish is not None and not isinstance(polish, bool):
        raise Paper1ResultError("work.polish_used must be boolean or null")


def _validate_resources(resources: Mapping[str, Any]) -> None:
    allowed = {
        *_REQUIRED_RESOURCE,
        "load_imbalance",
        "throughput_per_second",
    }
    _require_keys(resources, _REQUIRED_RESOURCE, "resources")
    _reject_unknown_keys(resources, allowed, "resources")
    integer_fields = {
        "peak_device_bytes",
        "reserved_device_bytes",
        "h2d_bytes",
        "d2h_bytes",
        "collective_bytes",
        "collective_count",
        "topology_allocation_count_after_create",
    }
    for key in integer_fields:
        _nonnegative_int_or_none(resources[key], f"resources.{key}")
    for key in ("energy_joules", "load_imbalance", "throughput_per_second"):
        if key in resources:
            _finite_or_none(resources[key], f"resources.{key}", nonnegative=True)


def _validate_aggregation(aggregation: Mapping[str, Any], status: str) -> None:
    required = (
        "warmup_repeats",
        "measured_repeats",
        "statistic",
        "median",
        "q1",
        "q3",
        "minimum",
        "maximum",
        "coefficient_of_variation",
        "censored_count",
    )
    allowed = (*required, "bootstrap_low", "bootstrap_high")
    _require_keys(aggregation, required, "aggregation")
    _reject_unknown_keys(aggregation, allowed, "aggregation")
    warmup = _nonnegative_int_or_none(aggregation["warmup_repeats"], "aggregation.warmup_repeats")
    measured = _nonnegative_int_or_none(
        aggregation["measured_repeats"], "aggregation.measured_repeats"
    )
    censored = _nonnegative_int_or_none(
        aggregation["censored_count"], "aggregation.censored_count"
    )
    assert warmup is not None and measured is not None and censored is not None
    if aggregation["statistic"] != "median_iqr":
        raise Paper1ResultError("aggregation.statistic must be 'median_iqr'")
    ordered = [
        _finite_or_none(aggregation[name], f"aggregation.{name}", nonnegative=True)
        for name in ("minimum", "q1", "median", "q3", "maximum")
    ]
    supplied = [value for value in ordered if value is not None]
    if supplied and len(supplied) != len(ordered):
        raise Paper1ResultError("aggregation quantiles must be all supplied or all null")
    if supplied and any(right < left for left, right in pairwise(supplied)):
        raise Paper1ResultError("aggregation quantiles are not monotonically ordered")
    _finite_or_none(
        aggregation["coefficient_of_variation"],
        "aggregation.coefficient_of_variation",
        nonnegative=True,
    )
    for key in ("bootstrap_low", "bootstrap_high"):
        if key in aggregation:
            _finite_or_none(aggregation[key], f"aggregation.{key}")
    if status == "qualified" and measured < 5:
        raise Paper1ResultError("qualified deterministic summaries require at least five repeats")


def _validate_artifact(value: Any, name: str) -> None:
    artifact = _mapping(value, name)
    _require_keys(artifact, ("location", "sha256"), name)
    _reject_unknown_keys(artifact, ("location", "sha256"), name)
    _nonempty_string(artifact["location"], f"{name}.location")
    digest = _nonempty_string(artifact["sha256"], f"{name}.sha256")
    if _SHA256.fullmatch(digest) is None:
        raise Paper1ResultError(f"{name}.sha256 must be a lowercase SHA-256 digest")


def _validate_artifacts(artifacts: Mapping[str, Any]) -> None:
    allowed = {
        *_REQUIRED_ARTIFACTS,
        "nsys",
        "ncu",
        "compute_sanitizer",
        "energy_trace",
    }
    _require_keys(artifacts, _REQUIRED_ARTIFACTS, "artifacts")
    _reject_unknown_keys(artifacts, allowed, "artifacts")
    for key in _REQUIRED_ARTIFACTS:
        _validate_artifact(artifacts[key], f"artifacts.{key}")
    for key, value in artifacts.items():
        if key not in _REQUIRED_ARTIFACTS and value is not None:
            _validate_artifact(value, f"artifacts.{key}")


def validate_paper1_result(payload: Mapping[str, Any]) -> None:
    """Validate semantic invariants beyond the declarative JSON Schema."""

    if not isinstance(payload, Mapping):
        raise Paper1ResultError("Paper 1 result must be an object")
    missing = sorted(_REQUIRED_TOP_LEVEL - set(payload))
    if missing:
        raise Paper1ResultError(f"result is missing top-level keys: {', '.join(missing)}")
    _reject_unknown_keys(payload, _ALLOWED_TOP_LEVEL, "result")
    if payload["schema_version"] != PAPER1_SCHEMA_VERSION:
        raise Paper1ResultError(
            f"unsupported Paper 1 schema version {payload['schema_version']!r}"
        )
    identity = _mapping(payload["identity"], "identity")
    dimensions = _mapping(payload["dimensions"], "dimensions")
    quality = _mapping(payload["quality"], "quality")
    timing = _mapping(payload["timing"], "timing")
    work = _mapping(payload["work"], "work")
    resources = _mapping(payload["resources"], "resources")
    aggregation = _mapping(payload["aggregation"], "aggregation")
    artifacts = _mapping(payload["artifacts"], "artifacts")

    _validate_identity(identity)
    _validate_dimensions(dimensions)
    _validate_quality(quality, str(identity["status"]))
    _validate_timing(timing)
    _validate_work(work)
    _validate_resources(resources)
    _validate_aggregation(aggregation, str(identity["status"]))
    _validate_artifacts(artifacts)

    notes = payload.get("notes", [])
    if not isinstance(notes, list) or not all(isinstance(note, str) for note in notes):
        raise Paper1ResultError("notes must be an array of strings")

    solver = str(identity["solver"])
    gpus = int(dimensions["gpus"])
    if solver in {
        "pdhcg-upstream-one-shot",
        "spacepdhcg-persistent",
        "qoco-gpu",
        "cuclarabel",
        "hybrid-pdhcg-ipm",
    } and gpus < 1:
        raise Paper1ResultError(f"GPU solver {solver!r} requires dimensions.gpus >= 1")
    if solver == "spacepdhcg-persistent" and str(identity["status"]) == "qualified":
        allocations = resources["topology_allocation_count_after_create"]
        if allocations is None:
            raise Paper1ResultError(
                "qualified persistent result must record post-create topology allocations"
            )


def read_paper1_result(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise Paper1ResultError("Paper 1 result root must be an object")
    result = dict(payload)
    validate_paper1_result(result)
    return result


def write_paper1_result(payload: Mapping[str, Any], path: str | Path) -> Path:
    validate_paper1_result(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination
