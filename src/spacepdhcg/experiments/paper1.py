"""Strict standard-library validation for compact Paper 1 result records."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

from .g4 import (
    FAILURE_CLASSES,
    PATH_INVENTORY,
    QUALITY_TIERS,
    SCALING_MODES,
    WARM_MODES,
    validate_portability,
    validate_timing_identity,
)

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
    {
        "qualified",
        "unqualified",
        "hybrid_handoff_ineligible",
        "not_applicable",
        "unsupported",
        "oom",
        "timeout",
        "timeout_deterministic_replay",
        "numerical",
        "infeasible",
        "executor_defect",
        "unrun",
        "failed",
    }
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
_ALLOWED_TOP_LEVEL: Final = _REQUIRED_TOP_LEVEL | {"g4", "notes"}
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
_G4_IDENTITY: Final = (
    "gate",
    "campaign",
    "quality_tier",
    "conditioning",
    "scaling_mode",
    "warm_mode",
    "seed",
    "repeat_kind",
    "repeat",
    "solver_order",
    "failure_class",
    "failure_reason",
)


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


def _is_primary_g4(identity: Mapping[str, Any]) -> bool:
    return identity.get("gate") == "G4" and identity.get("campaign") == "g4-primary"


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
    allowed = (
        *required,
        "hardware_id",
        "precision",
        "warm_start",
        "cold_start",
        "record_scope",
        *_G4_IDENTITY,
    )
    _require_keys(identity, required, "identity")
    if _is_primary_g4(identity):
        _require_keys(identity, _G4_IDENTITY, "identity")
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
    if identity.get("record_scope") not in {
        None,
        "measured_attempt",
        "publication_aggregate",
    }:
        raise Paper1ResultError("identity.record_scope is invalid")
    if _is_primary_g4(identity):
        if identity["quality_tier"] not in QUALITY_TIERS:
            raise Paper1ResultError("identity.quality_tier is invalid")
        if (
            _finite_or_none(
                identity["conditioning"],
                "identity.conditioning",
                nonnegative=True,
            )
            is None
        ):
            raise Paper1ResultError("identity.conditioning may not be null")
        # ``not_applicable_ipm_native``: amendment single-gpu-v1.2 rule A, the pure IPM
        # baseline records that the PDHCG scaling axis never reaches QOCO.
        if identity["scaling_mode"] not in (*SCALING_MODES, "not_applicable_ipm_native"):
            raise Paper1ResultError("identity.scaling_mode is invalid")
        if identity["warm_mode"] not in WARM_MODES:
            raise Paper1ResultError("identity.warm_mode is invalid")
        for key in ("seed", "repeat", "solver_order"):
            if _nonnegative_int_or_none(identity[key], f"identity.{key}") is None:
                raise Paper1ResultError(f"identity.{key} may not be null")
        if identity["repeat_kind"] not in {"warmup", "measured"}:
            raise Paper1ResultError("identity.repeat_kind is invalid")
        if identity["failure_class"] not in FAILURE_CLASSES:
            raise Paper1ResultError("identity.failure_class is invalid")
        reason = identity["failure_reason"]
        if identity["status"] == "qualified":
            if identity["failure_class"] != "none" or reason is not None:
                raise Paper1ResultError("qualified G4 result may not report a failure")
        elif not isinstance(reason, str) or not reason:
            raise Paper1ResultError("non-qualified G4 result requires a failure reason")
        if (
            identity["status"]
            in {
                "hybrid_handoff_ineligible",
                "not_applicable",
                "unsupported",
                "executor_defect",
            }
            and identity["failure_class"] != identity["status"]
        ):
            raise Paper1ResultError("exact G4 terminal status and failure class must match")


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
    allowed = (
        *required,
        "topology_bytes",
        "numeric_bytes",
        "dispersion_class",
        "attitude_class",
        "rate_class",
        "transfer_class",
        "trust_class",
    )
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
    for key in ("dispersion_class", "attitude_class", "rate_class", "trust_class"):
        if key in dimensions:
            _finite_or_none(dimensions[key], f"dimensions.{key}", nonnegative=True)
    if "transfer_class" in dimensions and dimensions["transfer_class"] is not None:
        _nonempty_string(dimensions["transfer_class"], "dimensions.transfer_class")
    inventory = _mapping(dimensions["cone_inventory"], "dimensions.cone_inventory")
    for key, value in inventory.items():
        _nonempty_string(key, "cone inventory key")
        if _nonnegative_int_or_none(value, f"cone_inventory.{key}") is None:
            raise Paper1ResultError(f"cone_inventory.{key} may not be null")


def _validate_quality(
    quality: Mapping[str, Any],
    status: str,
    *,
    primary_g4: bool,
    family: str,
    quality_tier: str | None,
) -> None:
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
        "continuous_time_violation",
        "solver_status",
        "convergence_criteria_met",
        "objective_equivalent",
        "matched_quality_state",
        "independent_replay",
        "path_inventory_complete",
        "uses_solver_cached_residuals",
        "path_inventory",
    }
    _require_keys(quality, ("qualified", *_REQUIRED_QUALITY), "quality")
    if primary_g4:
        _require_keys(
            quality,
            (
                "continuous_time_violation",
                "solver_status",
                "convergence_criteria_met",
                "objective_equivalent",
                "matched_quality_state",
                "independent_replay",
                "path_inventory_complete",
                "uses_solver_cached_residuals",
                "path_inventory",
            ),
            "quality",
        )
    _reject_unknown_keys(quality, allowed, "quality")
    if not isinstance(quality["qualified"], bool):
        raise Paper1ResultError("quality.qualified must be boolean")
    nonnumeric = {
        "qualified",
        "solver_status",
        "convergence_criteria_met",
        "objective_equivalent",
        "matched_quality_state",
        "independent_replay",
        "path_inventory_complete",
        "uses_solver_cached_residuals",
        "path_inventory",
    }
    for key, value in quality.items():
        if key in nonnumeric:
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
    if primary_g4:
        for key in (
            "convergence_criteria_met",
            "objective_equivalent",
            "independent_replay",
            "path_inventory_complete",
            "uses_solver_cached_residuals",
        ):
            if not isinstance(quality[key], bool):
                raise Paper1ResultError(f"quality.{key} must be boolean")
        if quality["solver_status"] not in {
            "converged",
            "max_iterations",
            "numerical_failure",
            "infeasible",
            "executor_defect",
            "hybrid_handoff_ineligible",
            "not_applicable",
            "unsupported",
        }:
            raise Paper1ResultError("quality.solver_status is invalid")
        if quality["matched_quality_state"] not in {"matched", "unqualified"}:
            raise Paper1ResultError("quality.matched_quality_state is invalid")
        inventory = _mapping(quality["path_inventory"], "quality.path_inventory")
        _require_keys(inventory, PATH_INVENTORY[family], "quality.path_inventory")
        for name in PATH_INVENTORY[family]:
            check = _mapping(inventory[name], f"quality.path_inventory.{name}")
            _require_keys(
                check,
                ("violation", "independent"),
                f"quality.path_inventory.{name}",
            )
            _reject_unknown_keys(
                check,
                ("violation", "independent"),
                f"quality.path_inventory.{name}",
            )
            _finite_or_none(
                check["violation"],
                f"quality.path_inventory.{name}.violation",
                nonnegative=True,
            )
            if not isinstance(check["independent"], bool):
                raise Paper1ResultError(
                    f"quality.path_inventory.{name}.independent must be boolean"
                )
        if quality["qualified"] and (
            quality["solver_status"] != "converged"
            or not quality["convergence_criteria_met"]
            or not quality["objective_equivalent"]
            or not quality["independent_replay"]
            or not quality["path_inventory_complete"]
            or quality["uses_solver_cached_residuals"]
            or quality["matched_quality_state"] != "matched"
        ):
            raise Paper1ResultError("qualified G4 result fails matched-quality state")
        if quality["qualified"]:
            assert quality_tier is not None
            if quality.get("reference_objective") is None:
                raise Paper1ResultError("qualified G4 result requires a reference objective")
            objective_difference = abs(
                float(quality["objective"]) - float(quality["reference_objective"])
            )
            objective_limit = 1e-6 + 1e-4 * abs(float(quality["reference_objective"]))
            if objective_difference > objective_limit:
                raise Paper1ResultError("qualified G4 objective is not practically equivalent")
            tolerance = {
                "coarse": 1e-3,
                "medium": 1e-4,
                "tight": 1e-6,
                "ipm": 1e-8,
            }[quality_tier]
            for key in (
                "canonical_primal_residual",
                "canonical_dual_residual",
                "canonical_cone_residual",
                "canonical_gap",
                "dynamics_residual",
                "path_residual",
                "terminal_residual",
            ):
                if float(quality[key]) > tolerance:
                    raise Paper1ResultError(f"qualified G4 result exceeds quality tier at {key}")
            for name in PATH_INVENTORY[family]:
                check = inventory[name]
                if not check["independent"] or float(check["violation"]) > tolerance:
                    raise Paper1ResultError(
                        f"qualified G4 result fails path inventory check {name}"
                    )
            if float(quality["virtual_control_residual"]) > 1e-6:
                raise Paper1ResultError("qualified G4 result exceeds virtual-control gate")
            if float(quality["continuous_time_violation"]) > 1e-6:
                raise Paper1ResultError("qualified G4 result exceeds continuous-time gate")


def _validate_timing(timing: Mapping[str, Any], *, primary_g4: bool) -> None:
    g4_fields = (
        "recovery_seconds",
        "hybrid_conversion_seconds",
        "hybrid_setup_seconds",
        "polish_seconds",
        "cuda_startup_seconds",
        "cuda_startup_included",
        "accepted_trajectory_count",
        "accepted_timing_boundary",
        "cqp_total_identity",
        "scvx_total_identity",
    )
    allowed = (*_REQUIRED_TIMING, "accepted_trajectory_seconds", *g4_fields)
    _require_keys(timing, _REQUIRED_TIMING, "timing")
    if primary_g4:
        _require_keys(
            timing,
            (
                "accepted_trajectory_seconds",
                "recovery_seconds",
                "hybrid_conversion_seconds",
                "hybrid_setup_seconds",
                "polish_seconds",
                "cuda_startup_seconds",
                "cuda_startup_included",
                "accepted_trajectory_count",
                "accepted_timing_boundary",
                "cqp_total_identity",
                "scvx_total_identity",
            ),
            "timing",
        )
    _reject_unknown_keys(timing, allowed, "timing")
    values = {
        key: _finite_or_none(value, f"timing.{key}", nonnegative=True)
        for key, value in timing.items()
        if key
        not in {
            "cuda_startup_included",
            "accepted_trajectory_count",
            "accepted_timing_boundary",
            "cqp_total_identity",
            "scvx_total_identity",
        }
    }
    cqp_total = values["cqp_total_seconds"]
    scvx_total = values["scvx_total_seconds"]
    if cqp_total is not None and scvx_total is not None and scvx_total + 1.0e-15 < cqp_total:
        raise Paper1ResultError("timing.scvx_total_seconds may not be below cqp_total_seconds")
    if primary_g4:
        try:
            validate_timing_identity(timing)
        except ValueError as error:
            raise Paper1ResultError(str(error)) from error


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


def _validate_resources(resources: Mapping[str, Any], *, primary_g4: bool) -> None:
    allowed = {
        *_REQUIRED_RESOURCE,
        "load_imbalance",
        "throughput_per_second",
        "energy_scope",
        "energy_sampling_interval_milliseconds",
        "energy_maximum_gap_seconds",
        "energy_sampling_gaps",
        "energy_valid",
        "shared_or_display_gpu",
    }
    _require_keys(resources, _REQUIRED_RESOURCE, "resources")
    if primary_g4:
        _require_keys(
            resources,
            (
                "energy_scope",
                "energy_sampling_interval_milliseconds",
                "energy_maximum_gap_seconds",
                "energy_sampling_gaps",
                "energy_valid",
                "shared_or_display_gpu",
            ),
            "resources",
        )
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
    if primary_g4:
        if resources["energy_scope"] != "GPU-only":
            raise Paper1ResultError("resources.energy_scope must be GPU-only for G4")
        interval = _nonnegative_int_or_none(
            resources["energy_sampling_interval_milliseconds"],
            "resources.energy_sampling_interval_milliseconds",
        )
        if interval is None or interval < 1:
            raise Paper1ResultError("energy sampling interval must be positive")
        _finite_or_none(
            resources["energy_maximum_gap_seconds"],
            "resources.energy_maximum_gap_seconds",
            nonnegative=True,
        )
        for key in ("energy_sampling_gaps", "energy_valid", "shared_or_display_gpu"):
            if not isinstance(resources[key], bool):
                raise Paper1ResultError(f"resources.{key} must be boolean")
        if resources["energy_sampling_gaps"] and resources["energy_valid"]:
            raise Paper1ResultError("energy with reported sampling gaps may not be marked valid")


def _validate_aggregation(
    aggregation: Mapping[str, Any],
    status: str,
    *,
    primary_g4: bool,
    record_scope: str | None,
) -> None:
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
    g4_fields = ("instance_count", "evaluation_seed_count", "paired_bootstrap_samples")
    allowed = (*required, "bootstrap_low", "bootstrap_high", *g4_fields)
    _require_keys(aggregation, required, "aggregation")
    if primary_g4:
        _require_keys(aggregation, g4_fields, "aggregation")
    _reject_unknown_keys(aggregation, allowed, "aggregation")
    warmup = _nonnegative_int_or_none(aggregation["warmup_repeats"], "aggregation.warmup_repeats")
    measured = _nonnegative_int_or_none(
        aggregation["measured_repeats"], "aggregation.measured_repeats"
    )
    censored = _nonnegative_int_or_none(aggregation["censored_count"], "aggregation.censored_count")
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
    if status == "qualified" and measured < 5 and record_scope != "measured_attempt":
        raise Paper1ResultError("qualified deterministic summaries require at least five repeats")
    if primary_g4:
        if record_scope == "measured_attempt":
            if warmup != 0 or measured != 1:
                raise Paper1ResultError(
                    "G4 measured attempts require 0 warmups and 1 measured observation"
                )
        elif warmup != 2 or measured != 7:
            raise Paper1ResultError(
                "primary G4 publication summaries require 2 warmups and 7 measured repeats"
            )
        for key in g4_fields:
            value = _nonnegative_int_or_none(aggregation[key], f"aggregation.{key}")
            if value is None:
                raise Paper1ResultError(f"aggregation.{key} may not be null")
        if record_scope == "measured_attempt":
            if aggregation["instance_count"] != 1 or aggregation["evaluation_seed_count"] != 1:
                raise Paper1ResultError(
                    "G4 measured attempts require exactly one evaluation instance/seed"
                )
            if aggregation["paired_bootstrap_samples"] != 0:
                raise Paper1ResultError("G4 measured attempts may not bootstrap")
        else:
            if aggregation["instance_count"] < 20 or aggregation["evaluation_seed_count"] < 20:
                raise Paper1ResultError(
                    "primary G4 publication summaries require 20 evaluation instances/seeds"
                )
            if aggregation["paired_bootstrap_samples"] != 10_000:
                raise Paper1ResultError(
                    "primary G4 publication summaries require 10000 bootstrap samples"
                )


def _validate_artifact(value: Any, name: str, *, primary_g4: bool) -> None:
    artifact = _mapping(value, name)
    _require_keys(artifact, ("location", "sha256"), name)
    if primary_g4:
        _require_keys(
            artifact,
            ("immutable_uri", "internal_index_sha256", "portable"),
            name,
        )
    _reject_unknown_keys(
        artifact,
        ("location", "sha256", "immutable_uri", "internal_index_sha256", "portable"),
        name,
    )
    _nonempty_string(artifact["location"], f"{name}.location")
    digest = _nonempty_string(artifact["sha256"], f"{name}.sha256")
    if _SHA256.fullmatch(digest) is None:
        raise Paper1ResultError(f"{name}.sha256 must be a lowercase SHA-256 digest")
    if primary_g4:
        _nonempty_string(artifact["immutable_uri"], f"{name}.immutable_uri")
        index_digest = _nonempty_string(
            artifact["internal_index_sha256"],
            f"{name}.internal_index_sha256",
        )
        if _SHA256.fullmatch(index_digest) is None:
            raise Paper1ResultError(
                f"{name}.internal_index_sha256 must be a lowercase SHA-256 digest"
            )
        if not isinstance(artifact["portable"], bool):
            raise Paper1ResultError(f"{name}.portable must be boolean")


def _validate_artifacts(artifacts: Mapping[str, Any], *, primary_g4: bool, qualified: bool) -> None:
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
        _validate_artifact(artifacts[key], f"artifacts.{key}", primary_g4=primary_g4)
    for key, value in artifacts.items():
        if key not in _REQUIRED_ARTIFACTS and value is not None:
            _validate_artifact(value, f"artifacts.{key}", primary_g4=primary_g4)
    if primary_g4 and qualified:
        contracts = {key: value for key, value in artifacts.items() if isinstance(value, Mapping)}
        portability = validate_portability(contracts, raise_on_error=False)
        if not portability["portable"]:
            raise Paper1ResultError(
                "qualified G4 evidence is not portable: " + "; ".join(portability["reasons"])
            )


def _validate_g4(g4: Mapping[str, Any], *, hybrid: bool, status: str) -> None:
    required = (
        "policy_sha256",
        "runtime_requested",
        "runtime_actual",
        "outer_iterations",
        "hybrid_permutation",
        "hybrid_dual_disposition",
    )
    _require_keys(g4, required, "g4")
    _reject_unknown_keys(g4, required, "g4")
    digest = _nonempty_string(g4["policy_sha256"], "g4.policy_sha256")
    if _SHA256.fullmatch(digest) is None:
        raise Paper1ResultError("g4.policy_sha256 must be a lowercase SHA-256 digest")
    for key in ("runtime_requested", "runtime_actual"):
        runtime = _mapping(g4[key], f"g4.{key}")
        _require_keys(
            runtime,
            ("policy", "quality_tier", "scaling_mode", "warm_mode"),
            f"g4.{key}",
        )
    iterations = g4["outer_iterations"]
    if not isinstance(iterations, list) or (
        not iterations and status not in {"not_applicable", "unsupported"}
    ):
        raise Paper1ResultError("g4.outer_iterations must be non-empty for launched dispositions")
    iteration_required = (
        "phase",
        "requested_tolerance",
        "achieved_residual",
        "forcing_satisfied",
        "trust_before",
        "trust_after",
        "trust_action",
        "re_solved",
        "cqp_fingerprint",
        "resolve_fingerprint",
        "fingerprint_match",
        "scaling_refreshed",
        "predicted_reduction",
        "actual_reduction",
        "reduction_ratio",
        "scaling_min",
        "scaling_max",
        "warm_mode_actual",
        "recovery_reason",
        "polish_handoff",
        "accepted",
    )
    for index, raw in enumerate(iterations):
        item = _mapping(raw, f"g4.outer_iterations[{index}]")
        _require_keys(item, iteration_required, f"g4.outer_iterations[{index}]")
        for key in (
            "requested_tolerance",
            "achieved_residual",
            "trust_before",
            "trust_after",
            "predicted_reduction",
            "scaling_min",
            "scaling_max",
        ):
            _finite_or_none(
                item[key],
                f"g4.outer_iterations[{index}].{key}",
                nonnegative=True,
            )
        for key in ("actual_reduction", "reduction_ratio"):
            _finite_or_none(
                item[key],
                f"g4.outer_iterations[{index}].{key}",
            )
        for key in (
            "forcing_satisfied",
            "re_solved",
            "fingerprint_match",
            "scaling_refreshed",
            "polish_handoff",
            "accepted",
        ):
            if not isinstance(item[key], bool):
                raise Paper1ResultError(f"g4.outer_iterations[{index}].{key} must be boolean")
        for key in ("cqp_fingerprint", "resolve_fingerprint"):
            _nonempty_string(item[key], f"g4.outer_iterations[{index}].{key}")
        if item["warm_mode_actual"] not in WARM_MODES:
            raise Paper1ResultError(f"g4.outer_iterations[{index}].warm_mode_actual is invalid")
        _nonempty_string(
            item["recovery_reason"],
            f"g4.outer_iterations[{index}].recovery_reason",
        )
    if hybrid and status not in {"not_applicable", "unsupported"}:
        if not isinstance(g4["hybrid_permutation"], list) or not g4["hybrid_permutation"]:
            raise Paper1ResultError("hybrid G4 result requires the applied permutation")
        if g4["hybrid_dual_disposition"] not in {
            "transferred",
            "discarded_unsupported",
            "discarded-unsupported",
            "not-produced",
        }:
            raise Paper1ResultError("hybrid G4 result has invalid dual disposition")
    elif not hybrid and (
        g4["hybrid_permutation"] is not None or g4["hybrid_dual_disposition"] is not None
    ):
        raise Paper1ResultError("non-hybrid G4 result must use null hybrid disposition")


def validate_paper1_result(payload: Mapping[str, Any]) -> None:
    """Validate semantic invariants beyond the declarative JSON Schema."""

    if not isinstance(payload, Mapping):
        raise Paper1ResultError("Paper 1 result must be an object")
    missing = sorted(_REQUIRED_TOP_LEVEL - set(payload))
    if missing:
        raise Paper1ResultError(f"result is missing top-level keys: {', '.join(missing)}")
    _reject_unknown_keys(payload, _ALLOWED_TOP_LEVEL, "result")
    if payload["schema_version"] != PAPER1_SCHEMA_VERSION:
        raise Paper1ResultError(f"unsupported Paper 1 schema version {payload['schema_version']!r}")
    identity = _mapping(payload["identity"], "identity")
    dimensions = _mapping(payload["dimensions"], "dimensions")
    quality = _mapping(payload["quality"], "quality")
    timing = _mapping(payload["timing"], "timing")
    work = _mapping(payload["work"], "work")
    resources = _mapping(payload["resources"], "resources")
    aggregation = _mapping(payload["aggregation"], "aggregation")
    artifacts = _mapping(payload["artifacts"], "artifacts")
    primary_g4 = _is_primary_g4(identity)

    _validate_identity(identity)
    if primary_g4 and identity["family"] not in PATH_INVENTORY:
        raise Paper1ResultError("primary G4 result uses a non-G4 family")
    _validate_dimensions(dimensions)
    _validate_quality(
        quality,
        str(identity["status"]),
        primary_g4=primary_g4,
        family=str(identity["family"]),
        quality_tier=str(identity["quality_tier"]) if primary_g4 else None,
    )
    _validate_timing(timing, primary_g4=primary_g4)
    if primary_g4 and identity["status"] in {
        "hybrid_handoff_ineligible",
        "not_applicable",
        "unsupported",
    }:
        for key in ("cqp_total_seconds", "scvx_total_seconds"):
            if timing[key] is None:
                raise Paper1ResultError(
                    f"identity.status={identity['status']} requires explicit timing.{key}"
                )
    _validate_work(work)
    _validate_resources(resources, primary_g4=primary_g4)
    _validate_aggregation(
        aggregation,
        str(identity["status"]),
        primary_g4=primary_g4,
        record_scope=(
            str(identity["record_scope"]) if identity.get("record_scope") is not None else None
        ),
    )
    _validate_artifacts(
        artifacts,
        primary_g4=primary_g4,
        qualified=bool(quality["qualified"]),
    )
    if primary_g4:
        if "g4" not in payload:
            raise Paper1ResultError("primary G4 result requires g4 telemetry")
        _validate_g4(
            _mapping(payload["g4"], "g4"),
            hybrid=identity["policy"] == "hybrid-pdhcg-ipm",
            status=str(identity["status"]),
        )
    elif "g4" in payload:
        raise Paper1ResultError("g4 telemetry is reserved for primary G4 results")

    notes = payload.get("notes", [])
    if not isinstance(notes, list) or not all(isinstance(note, str) for note in notes):
        raise Paper1ResultError("notes must be an array of strings")

    solver = str(identity["solver"])
    gpus = int(dimensions["gpus"])
    if (
        solver
        in {
            "pdhcg-upstream-one-shot",
            "spacepdhcg-persistent",
            "qoco-gpu",
            "cuclarabel",
            "hybrid-pdhcg-ipm",
        }
        and gpus < 1
    ):
        raise Paper1ResultError(f"GPU solver {solver!r} requires dimensions.gpus >= 1")
    if solver == "spacepdhcg-persistent" and str(identity["status"]) == "qualified":
        allocations = resources["topology_allocation_count_after_create"]
        if allocations is None:
            raise Paper1ResultError(
                "qualified persistent result must record post-create topology allocations"
            )


def validate_paper1_result_schema(
    payload: Mapping[str, Any],
    schema_path: str | Path,
) -> None:
    """Apply Draft 2020-12 structure and the strict semantic contract."""

    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import ValidationError
    except ImportError as error:
        raise Paper1ResultError(
            "strict Paper 1 schema validation requires the project dev dependencies"
        ) from error
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(payload)
    except ValidationError as error:
        raise Paper1ResultError(
            f"Paper 1 JSON schema validation failed: {error.message}"
        ) from error
    validate_paper1_result(payload)


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
