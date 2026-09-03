"""Audited Gate G4 policy, qualification, decision, and evidence contracts.

This module is deliberately CPU-only.  It prepares runtime configuration for an
external solver executable, validates the executable's reported behaviour, and
refuses to turn incomplete or non-portable evidence into a performance claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise, product
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlparse

POLICY_NAMES: Final = (
    "fixed-tight",
    "fixed-loose",
    "adaptive",
    "adaptive+polish",
    "pure-gpu-ipm",
    "hybrid-pdhcg-ipm",
)
QUALITY_TIERS: Final = ("coarse", "medium", "tight", "ipm")
SCALING_MODES: Final = ("always_refresh", "reuse", "refresh_if_needed")
WARM_MODES: Final = ("cold", "primal", "primal_dual")
FAILURE_CLASSES: Final = (
    "none",
    "hybrid_handoff_ineligible",
    "not_applicable",
    "unsupported",
    "oom",
    "timeout",
    "numerical",
    "infeasible",
    "max_iterations",
    "unrun",
    "evidence",
)
DISPOSITIONS: Final = (
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
    "unrun",
)

PATH_INVENTORY: Final = {
    "P1-C-pd3": ("thrust", "mass", "altitude", "glide_slope"),
    "P1-D-pd6": (
        "thrust",
        "torque",
        "pointing",
        "mass",
        "altitude",
        "glide_slope",
        "angular_rate",
        "quaternion",
    ),
    "P1-E-low-thrust": ("thrust", "mass", "altitude"),
}

CQP_TIMING_COMPONENTS: Final = (
    "coefficient_seconds",
    "workspace_create_seconds",
    "update_seconds",
    "scaling_seconds",
    "h2d_seconds",
    "solve_seconds",
    "recovery_seconds",
    "residual_seconds",
    "d2h_seconds",
    "collective_seconds",
    "hybrid_conversion_seconds",
    "hybrid_setup_seconds",
    "polish_seconds",
)
SCVX_EXTRA_TIMING_COMPONENTS: Final = ("replay_seconds", "acceptance_seconds")
ACCEPTED_TIMING_BOUNDARY: Final = (
    "coefficient-generation-through-independent-replay-and-acceptance;cuda-startup-excluded"
)


class G4ContractError(ValueError):
    """Raised when frozen G4 evidence violates an audited contract."""


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G4ContractError(message)


def _finite(value: Any, name: str, *, nonnegative: bool = False) -> float:
    _require(
        not isinstance(value, bool) and isinstance(value, (int, float)),
        f"{name} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{name} must be finite")
    _require(not nonnegative or result >= 0.0, f"{name} must be non-negative")
    return result


def validate_policy(policy: Mapping[str, Any]) -> None:
    """Validate the frozen policy's cross-file and internal invariants."""

    _require(policy.get("gate") == "G4", "policy gate must be G4")
    _require(tuple(policy.get("policies", {})) == POLICY_NAMES, "policy order/value drift")
    tiers = policy.get("quality_tiers", {})
    _require(tuple(tiers) == QUALITY_TIERS, "quality-tier order/value drift")
    _require(
        tuple(policy["matrix"]["quality_tiers"]) == QUALITY_TIERS,
        "matrix quality tiers drift from runtime tiers",
    )
    _require(
        tuple(policy["matrix"]["scaling_modes"]) == SCALING_MODES,
        "matrix scaling modes drift from runtime modes",
    )
    _require(
        tuple(policy["matrix"]["warm_start_modes"]) == WARM_MODES,
        "matrix warm modes drift from runtime modes",
    )
    _require(tuple(policy["warm_start_modes"]) == WARM_MODES, "warm mode drift")
    _require(
        tuple(policy["scaling_policy"]["modes"]) == SCALING_MODES,
        "scaling mode drift",
    )
    _require(set(tiers.values()) == {1e-3, 1e-4, 1e-6, 1e-8}, "quality values drift")
    _require(
        policy["policies"]["fixed-tight"]["inner_tolerance_rule"] == "selected-quality-tier",
        "fixed-tight quality-tier rule drift",
    )

    adaptive = policy["policies"]["adaptive"]
    _require(adaptive["epsilon_max"] == 1e-3, "adaptive epsilon_max drift")
    _require(adaptive["epsilon_floor"] == 1e-8, "adaptive epsilon_floor drift")
    _require(adaptive["coefficient"] == 0.2, "adaptive coefficient drift")
    _require(adaptive["alpha"] == 0.5, "adaptive alpha drift")
    _require(adaptive["gamma"] == 0.6, "adaptive gamma drift")
    _require(
        adaptive["phase_ceilings"]["polish"] == 1e-8,
        "adaptive polish ceiling drift",
    )
    resolve = policy["resolve_policy"]
    _require(resolve["trigger_multiple_of_requested_tolerance"] == 5.0, "resolve trigger drift")
    _require(resolve["refinement_factor"] == 0.1, "resolve factor drift")
    _require(resolve["maximum_resolves_per_outer_iteration"] == 1, "resolve count drift")

    split = policy["tuning_evaluation_split"]
    tuning = split["tuning_seeds"]
    evaluation = split["evaluation_seeds"]
    _require(len(tuning) == len(set(tuning)), "duplicate tuning seeds")
    _require(len(evaluation) == len(set(evaluation)), "duplicate evaluation seeds")
    _require(set(tuning).isdisjoint(evaluation), "tuning/evaluation seeds overlap")
    _require(
        len(evaluation) >= policy["matrix"]["randomised_instances_per_coordinate"],
        "evaluation seeds do not cover committed randomised instance count",
    )
    _require(policy["matrix"]["warmup_repeats"] == 2, "warmup repeat count drift")
    _require(policy["matrix"]["measured_repeats"] == 7, "measured repeat count drift")
    _require(
        policy["matrix"]["randomised_instances_per_coordinate"] == 20,
        "randomised instance count drift",
    )
    required_inventory = policy["matched_quality"]["required_path_inventory"]
    for family, expected in PATH_INVENTORY.items():
        _require(family in policy["matrix"]["families"], f"missing matrix family {family}")
        _require(
            tuple(required_inventory.get(family, ())) == expected,
            f"path inventory drift for {family}",
        )


@dataclass(frozen=True)
class LoadedPolicy:
    values: dict[str, Any]
    sha256: str
    path: Path


def load_policy(path: str | Path, *, expected_sha256: str | None = None) -> LoadedPolicy:
    source = Path(path)
    digest = sha256_path(source)
    if expected_sha256 is not None and digest != expected_sha256:
        raise G4ContractError(
            f"frozen G4 policy hash drift: expected {expected_sha256}, received {digest}"
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "G4 policy root must be an object")
    validate_policy(payload)
    return LoadedPolicy(payload, digest, source)


def _resolve_policy(values: Mapping[str, Any], policy_name: str) -> dict[str, Any]:
    _require(policy_name in POLICY_NAMES, f"unknown G4 policy {policy_name!r}")
    result = dict(values["policies"][policy_name])
    parent = result.pop("inherits", None)
    if parent is not None:
        inherited = dict(values["policies"][parent])
        inherited.update(result)
        result = inherited
    return result


def runtime_configuration(
    loaded: LoadedPolicy,
    *,
    family: str,
    policy_name: str,
    quality_tier: str,
    scaling_mode: str,
    warm_mode: str,
) -> dict[str, Any]:
    """Generate one immutable runtime request directly from the frozen policy."""

    values = loaded.values
    _require(family in values["matrix"]["families"], f"unknown G4 family {family!r}")
    _require(quality_tier in QUALITY_TIERS, f"unknown quality tier {quality_tier!r}")
    _require(scaling_mode in SCALING_MODES, f"unknown scaling mode {scaling_mode!r}")
    _require(warm_mode in WARM_MODES, f"unknown warm mode {warm_mode!r}")
    selected = _resolve_policy(values, policy_name)
    quality_tolerance = float(values["quality_tiers"][quality_tier])
    if policy_name == "fixed-tight":
        selected["inner_tolerance"] = quality_tolerance
    selected["final_quality_tolerance"] = quality_tolerance
    return {
        "schema_version": "1.0.0",
        "gate": "G4",
        "policy_sha256": loaded.sha256,
        "family": family,
        "policy": policy_name,
        "quality_tier": quality_tier,
        "quality_tolerance": quality_tolerance,
        "scaling_mode": scaling_mode,
        "warm_start_mode": warm_mode,
        "solver": selected,
        "resolve": dict(values["resolve_policy"]),
        "trust": dict(values["trust_policy"]),
        "scaling": {
            "mode": scaling_mode,
            "refresh_condition_ratio": values["scaling_policy"]["refresh_condition_ratio"],
            "record_extrema": values["scaling_policy"]["record_extrema"],
        },
    }


def verify_runtime_behavior(requested: Mapping[str, Any], reported: Mapping[str, Any]) -> list[str]:
    """Return drift reasons between requested and actual executable behaviour."""

    reasons: list[str] = []
    if reported.get("policy_sha256") != requested.get("policy_sha256"):
        reasons.append("executable policy hash differs from frozen policy")
    requested_report = reported.get("requested")
    actual_report = reported.get("actual")
    if not isinstance(requested_report, Mapping):
        reasons.append("executable omitted requested runtime behavior")
    if not isinstance(actual_report, Mapping):
        reasons.append("executable omitted actual runtime behavior")
        return reasons
    keys = (
        "policy",
        "quality_tier",
        "quality_tolerance",
        "scaling_mode",
        "warm_start_mode",
    )
    for key in keys:
        expected = requested.get(key)
        if isinstance(requested_report, Mapping) and requested_report.get(key) != expected:
            reasons.append(f"reported requested {key} drift")
        if actual_report.get(key) != expected:
            reasons.append(f"actual {key} differs from request")
    expected_solver = requested.get("solver")
    if actual_report.get("solver") != expected_solver:
        reasons.append("actual solver policy values differ from frozen request")
    if actual_report.get("resolve") != requested.get("resolve"):
        reasons.append("actual re-solve policy values differ from frozen request")
    return reasons


def objective_equivalent(
    objective: float,
    reference: float,
    *,
    absolute: float,
    relative: float,
) -> bool:
    difference = abs(_finite(objective, "objective") - _finite(reference, "reference objective"))
    return difference <= absolute + relative * abs(reference)


def _quality_tolerance(policy: Mapping[str, Any], tier: str) -> float:
    _require(tier in QUALITY_TIERS, f"unknown quality tier {tier!r}")
    return float(policy["quality_tiers"][tier])


def qualify_matched_quality(record: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    """Apply every matched-quality gate and retain all failure reasons."""

    reasons: list[str] = []
    family = str(record.get("family", ""))
    tier = str(record.get("quality_tier", ""))
    tolerance = _quality_tolerance(policy, tier)
    matched = policy["matched_quality"]

    if record.get("solver_status") != "converged":
        reasons.append("solver did not report converged status")
    if record.get("convergence_criteria_met") is not True:
        reasons.append("solver convergence criteria were not met")
    if record.get("failure_class") == "max_iterations":
        reasons.append("maximum-iteration termination is unqualified")

    quality = record.get("quality")
    if not isinstance(quality, Mapping):
        reasons.append("quality evidence is missing")
        quality = {}
    canonical_limit = tolerance * float(matched["canonical_multiplier"])
    nonlinear_limit = tolerance * float(matched["nonlinear_multiplier"])
    for key in (
        "canonical_primal_residual",
        "canonical_dual_residual",
        "canonical_cone_residual",
        "canonical_gap",
    ):
        value = quality.get(key)
        if value is None or _finite(value, f"quality.{key}", nonnegative=True) > canonical_limit:
            reasons.append(f"{key} exceeds the matched canonical gate")
    for key in ("dynamics_residual", "terminal_residual"):
        value = quality.get(key)
        if value is None or _finite(value, f"quality.{key}", nonnegative=True) > nonlinear_limit:
            reasons.append(f"{key} exceeds the matched nonlinear gate")
    virtual = quality.get("virtual_control_residual")
    if virtual is None or _finite(
        virtual, "quality.virtual_control_residual", nonnegative=True
    ) > float(matched["virtual_control_tolerance"]):
        reasons.append("virtual control exceeds its frozen gate")
    ct_violation = quality.get("continuous_time_violation")
    if ct_violation is None or _finite(
        ct_violation, "quality.continuous_time_violation", nonnegative=True
    ) > float(matched["continuous_time_tolerance"]):
        reasons.append("continuous-time violation exceeds its frozen gate")

    objective = quality.get("objective")
    reference = quality.get("reference_objective")
    if objective is None or reference is None:
        reasons.append("objective equivalence evidence is missing")
    elif not objective_equivalent(
        objective,
        reference,
        absolute=float(matched["objective_practical_equivalence_absolute"]),
        relative=float(matched["objective_practical_equivalence_relative"]),
    ):
        reasons.append("objective is not practically equivalent to the reference")

    checks = record.get("independent_checks")
    if not isinstance(checks, Mapping):
        reasons.append("independent replay evidence is missing")
        checks = {}
    if checks.get("replay_performed") is not True:
        reasons.append("independent replay was not performed")
    if checks.get("uses_solver_cached_residuals") is not False:
        reasons.append("independent checker isolation was not demonstrated")
    path = checks.get("path_inventory")
    required_path = tuple(matched["required_path_inventory"].get(family, ()))
    if not isinstance(path, Mapping):
        reasons.append("full path inventory is missing")
        path = {}
    for name in required_path:
        item = path.get(name)
        if not isinstance(item, Mapping):
            reasons.append(f"path check {name} is missing")
            continue
        if item.get("independent") is not True:
            reasons.append(f"path check {name} is not independent")
        violation = item.get("violation")
        if (
            violation is None
            or _finite(violation, f"path_inventory.{name}.violation", nonnegative=True)
            > nonlinear_limit
        ):
            reasons.append(f"path check {name} exceeds the matched nonlinear gate")
    if checks.get("path_inventory_complete") is not True:
        reasons.append("path inventory was not declared complete")

    timing = record.get("timing")
    if not isinstance(timing, Mapping):
        reasons.append("accepted-trajectory timing evidence is missing")
    else:
        try:
            validate_timing_identity(timing)
        except G4ContractError as error:
            reasons.append(str(error))
    runtime = record.get("runtime")
    requested = record.get("runtime_configuration")
    if not isinstance(runtime, Mapping) or not isinstance(requested, Mapping):
        reasons.append("runtime requested/actual telemetry is missing")
    else:
        reasons.extend(verify_runtime_behavior(requested, runtime))

    artifacts = record.get("artifacts", {})
    portable = validate_portability(artifacts, raise_on_error=False)
    if not portable["portable"]:
        reasons.extend(portable["reasons"])

    return {
        "qualified": not reasons,
        "state": "matched" if not reasons else "unqualified",
        "quality_tier": tier,
        "canonical_limit": canonical_limit,
        "nonlinear_limit": nonlinear_limit,
        "reasons": reasons,
    }


def timing_from_components(
    components: Mapping[str, Any],
    *,
    accepted_trajectories: int = 1,
) -> dict[str, Any]:
    _require(accepted_trajectories > 0, "accepted trajectory count must be positive")
    values = {
        key: _finite(components.get(key, 0.0), key, nonnegative=True)
        for key in (*CQP_TIMING_COMPONENTS, *SCVX_EXTRA_TIMING_COMPONENTS)
    }
    cqp_total = math.fsum(values[key] for key in CQP_TIMING_COMPONENTS)
    scvx_total = cqp_total + math.fsum(values[key] for key in SCVX_EXTRA_TIMING_COMPONENTS)
    return {
        **values,
        "cuda_startup_seconds": _finite(
            components.get("cuda_startup_seconds", 0.0),
            "cuda_startup_seconds",
            nonnegative=True,
        ),
        "cuda_startup_included": False,
        "cqp_total_seconds": cqp_total,
        "scvx_total_seconds": scvx_total,
        "accepted_trajectory_seconds": scvx_total / accepted_trajectories,
        "accepted_trajectory_count": accepted_trajectories,
        "accepted_timing_boundary": ACCEPTED_TIMING_BOUNDARY,
        "cqp_total_identity": list(CQP_TIMING_COMPONENTS),
        "scvx_total_identity": [*CQP_TIMING_COMPONENTS, *SCVX_EXTRA_TIMING_COMPONENTS],
    }


def validate_timing_identity(timing: Mapping[str, Any], *, tolerance: float = 1e-12) -> None:
    for key in (*CQP_TIMING_COMPONENTS, *SCVX_EXTRA_TIMING_COMPONENTS):
        _finite(timing.get(key), f"timing.{key}", nonnegative=True)
    _require(timing.get("cuda_startup_included") is False, "CUDA startup must be excluded")
    _require(
        timing.get("accepted_timing_boundary") == ACCEPTED_TIMING_BOUNDARY,
        "accepted-trajectory timing boundary is missing or inconsistent",
    )
    _require(
        tuple(timing.get("cqp_total_identity", ())) == CQP_TIMING_COMPONENTS,
        "CQP timing identity fields drift",
    )
    _require(
        tuple(timing.get("scvx_total_identity", ()))
        == (*CQP_TIMING_COMPONENTS, *SCVX_EXTRA_TIMING_COMPONENTS),
        "SCvx timing identity fields drift",
    )
    expected_cqp = math.fsum(float(timing[key]) for key in CQP_TIMING_COMPONENTS)
    expected_scvx = expected_cqp + math.fsum(
        float(timing[key]) for key in SCVX_EXTRA_TIMING_COMPONENTS
    )
    actual_cqp = _finite(timing.get("cqp_total_seconds"), "timing.cqp_total_seconds")
    actual_scvx = _finite(timing.get("scvx_total_seconds"), "timing.scvx_total_seconds")
    _require(
        math.isclose(actual_cqp, expected_cqp, rel_tol=tolerance, abs_tol=tolerance),
        "CQP timing sum identity failed",
    )
    _require(
        math.isclose(actual_scvx, expected_scvx, rel_tol=tolerance, abs_tol=tolerance),
        "SCvx timing sum identity failed",
    )
    count = timing.get("accepted_trajectory_count")
    _require(isinstance(count, int) and count > 0, "accepted trajectory count is invalid")
    accepted = _finite(
        timing.get("accepted_trajectory_seconds"),
        "timing.accepted_trajectory_seconds",
        nonnegative=True,
    )
    _require(
        math.isclose(accepted, actual_scvx / count, rel_tol=tolerance, abs_tol=tolerance),
        "accepted-trajectory timing identity failed",
    )


def _quantile(sorted_values: Sequence[float], fraction: float) -> float:
    index = (len(sorted_values) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def paired_bootstrap_interval(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    samples: int,
    seed: int,
    confidence: float,
) -> dict[str, float]:
    _require(len(baseline) == len(candidate) and len(baseline) >= 2, "paired samples required")
    paired_reductions: list[float] = []
    for left, right in zip(baseline, candidate, strict=True):
        left_value = _finite(left, "baseline time", nonnegative=True)
        right_value = _finite(right, "candidate time", nonnegative=True)
        _require(left_value > 0.0, "baseline time must be positive")
        paired_reductions.append((left_value - right_value) / left_value)
    generator = random.Random(seed)
    bootstrapped = sorted(
        statistics.median(
            paired_reductions[generator.randrange(len(paired_reductions))]
            for _ in paired_reductions
        )
        for _ in range(samples)
    )
    alpha = 0.5 * (1.0 - confidence)
    return {
        "median": statistics.median(paired_reductions),
        "low": _quantile(bootstrapped, alpha),
        "high": _quantile(bootstrapped, 1.0 - alpha),
    }


def _coordinate_evidence(
    row: Mapping[str, Any], policy: Mapping[str, Any], *, baseline_key: str, candidate_key: str
) -> dict[str, Any]:
    disposition = row.get("disposition")
    if disposition is not None and disposition != "qualified":
        return {
            "eligible": False,
            "reason": f"terminal disposition {disposition} is not winner-eligible",
            "family": row.get("family"),
            "scale": row.get("scale"),
            "censored": disposition in {"timeout", "oom", "timeout_deterministic_replay"},
            "terminal_disposition": disposition,
        }
    baseline = row.get(baseline_key, ())
    candidate = row.get(candidate_key, ())
    if not isinstance(baseline, Sequence) or isinstance(baseline, (str, bytes)):
        baseline = ()
    if not isinstance(candidate, Sequence) or isinstance(candidate, (str, bytes)):
        candidate = ()
    minimum = policy["matrix"]["measured_repeats"]
    if len(baseline) < minimum or len(candidate) != len(baseline):
        return {"eligible": False, "reason": "missing paired measured repeats"}
    interval = paired_bootstrap_interval(
        baseline,
        candidate,
        samples=int(policy["statistics"]["bootstrap_samples"]),
        seed=int(policy["statistics"]["bootstrap_seed"]) + int(row.get("scale", 0)),
        confidence=float(policy["statistics"]["confidence"]),
    )
    baseline_failures = int(row.get("baseline_failures", 0))
    candidate_failures = int(row.get("candidate_failures", 0))
    attempts = int(row.get("attempts", len(baseline)))
    failure_delta = (candidate_failures - baseline_failures) / max(attempts, 1)
    return {
        "eligible": True,
        **interval,
        "failure_rate_delta": failure_delta,
        "family": row.get("family"),
        "scale": row.get("scale"),
        "conditioning": row.get("conditioning"),
        "warm_mode": row.get("warm_mode"),
        "scaling_mode": row.get("scaling_mode"),
        "quality_tier": row.get("quality_tier"),
        "dispersion_class": row.get("dispersion_class"),
        "attitude_class": row.get("attitude_class"),
        "rate_class": row.get("rate_class"),
        "transfer_class": row.get("transfer_class"),
        "trust_class": row.get("trust_class"),
        "censored": bool(row.get("censored", False)),
    }


def _sustained(
    evidence: Sequence[Mapping[str, Any]],
    predicate: Any,
    required: int,
) -> dict[str, list[list[Any]]]:
    strata = (
        "family",
        "conditioning",
        "warm_mode",
        "scaling_mode",
        "quality_tier",
        "dispersion_class",
        "attitude_class",
        "rate_class",
        "transfer_class",
        "trust_class",
    )
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in evidence:
        grouped[tuple(row.get(field) for field in strata)].append(row)
    result: dict[str, list[list[Any]]] = {}
    for key, rows in grouped.items():
        family = str(key[0])
        ordered = sorted(rows, key=lambda item: float(item.get("scale", 0)))
        regions: list[list[Any]] = []
        for start in range(len(ordered)):
            window = ordered[start : start + required]
            scales = [item.get("scale") for item in window]
            if (
                len(window) == required
                and all(float(right) > float(left) for left, right in pairwise(scales))
                and all(predicate(item) for item in window)
            ):
                regions.append([item.get("scale") for item in window])
        if regions:
            result.setdefault(family, []).extend(regions)
    return result


def decide_h5(rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate H5 exactly from paired, matched-quality coordinate evidence."""

    threshold = policy["decision_thresholds"]["H5"]
    evidence: list[dict[str, Any]] = []
    censored = 0
    for row in rows:
        item = _coordinate_evidence(
            row, policy, baseline_key="fixed_tight_seconds", candidate_key="adaptive_seconds"
        )
        item["matched_quality"] = bool(row.get("matched_quality", False))
        item["objective_equivalent"] = bool(row.get("objective_equivalent", False))
        item["forcing_satisfied"] = bool(row.get("forcing_satisfied", False))
        if item.get("censored"):
            censored += 1
        evidence.append(item)
    eligible = [item for item in evidence if item["eligible"] and not item["censored"]]

    def supported(item: Mapping[str, Any]) -> bool:
        return bool(
            item["matched_quality"]
            and item["objective_equivalent"]
            and item["forcing_satisfied"]
            and item["failure_rate_delta"] <= threshold["maximum_failure_rate_increase"]
            and item["median"] >= threshold["supported_minimum_time_reduction"]
            and item["low"] > 0.0
        )

    def rejected(item: Mapping[str, Any]) -> bool:
        return bool(
            (item["median"] <= -threshold["rejected_minimum_slowdown"] and item["high"] < 0.0)
            or item["failure_rate_delta"] > threshold["maximum_failure_rate_increase"]
        )

    required = int(policy["statistics"]["sustained_coordinates"])
    support_regions = _sustained(eligible, supported, required)
    reject_regions = _sustained(eligible, rejected, required)
    if len(support_regions) >= threshold["minimum_supported_families"]:
        decision = "supported"
        reason = "two-family sustained adaptive advantage clears all preregistered gates"
    else:
        tested_families = set(policy["matrix"]["families"])
        if tested_families and tested_families <= set(reject_regions):
            decision = "rejected"
            reason = "adaptive forcing is sustained-adverse across every frozen nonlinear family"
        elif eligible:
            decision = "mixed"
            reason = "eligible coordinates do not establish uniform support or rejection"
        else:
            decision = "unresolved"
            reason = "no uncensored matched-quality paired coordinates are decision-eligible"
    return {
        "hypothesis": "H5",
        "decision": decision,
        "reason": reason,
        "supported_regions": support_regions,
        "rejected_regions": reject_regions,
        "eligible_coordinates": len(eligible),
        "censored_coordinates": censored,
        "coordinates": evidence,
    }


def decide_h6(rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate H6 from hybrid/IPM timing and hybrid/unpolished residual evidence."""

    threshold = policy["decision_thresholds"]["H6"]
    evidence: list[dict[str, Any]] = []
    censored = 0
    for row in rows:
        item = _coordinate_evidence(
            row, policy, baseline_key="ipm_seconds", candidate_key="hybrid_seconds"
        )
        hybrid = row.get("hybrid_residual")
        ipm = row.get("ipm_residual")
        unpolished = row.get("unpolished_residual")
        if hybrid is not None and ipm is not None and float(ipm) > 0.0:
            item["ipm_quality_factor"] = float(hybrid) / float(ipm)
        else:
            item["ipm_quality_factor"] = math.inf
        if hybrid is not None and unpolished is not None and float(hybrid) > 0.0:
            item["residual_decades"] = math.log10(float(unpolished) / float(hybrid))
        else:
            item["residual_decades"] = -math.inf
        item["matched_quality"] = bool(row.get("matched_quality", False))
        item["conversion_and_polish_included"] = bool(
            row.get("conversion_and_polish_included", False)
        )
        item["unpolished_failed_tier"] = bool(row.get("unpolished_failed_tier", False))
        item["transfer_reliable"] = bool(row.get("transfer_reliable", False))
        if item.get("censored"):
            censored += 1
        evidence.append(item)
    eligible = [item for item in evidence if item["eligible"] and not item["censored"]]

    def supported(item: Mapping[str, Any]) -> bool:
        return bool(
            item["matched_quality"]
            and item["conversion_and_polish_included"]
            and item["transfer_reliable"]
            and item["ipm_quality_factor"] <= threshold["maximum_hybrid_to_ipm_residual_factor"]
            and item["median"] >= threshold["supported_minimum_time_reduction"]
            and item["low"] > 0.0
            and (
                item["residual_decades"] >= threshold["minimum_unpolished_residual_decades"]
                or item["unpolished_failed_tier"]
            )
        )

    def rejected(item: Mapping[str, Any]) -> bool:
        return bool(
            (item["median"] <= 0.0 and item["high"] <= 0.0) or not item["transfer_reliable"]
        )

    required = int(policy["statistics"]["sustained_coordinates"])
    support_regions = _sustained(eligible, supported, required)
    reject_regions = _sustained(eligible, rejected, required)
    if support_regions:
        decision = "supported"
        reason = "sustained hybrid region clears quality, timing, and residual-decade gates"
    elif eligible and set(policy["matrix"]["families"]) <= set(reject_regions):
        decision = "rejected"
        reason = "hybrid polish has no reliable timing advantage at common feasible scales"
    elif eligible:
        decision = "mixed"
        reason = "hybrid evidence is eligible but not sustained across the tested regime"
    else:
        decision = "unresolved"
        reason = "no uncensored matched-quality hybrid/IPM coordinates are decision-eligible"
    return {
        "hypothesis": "H6",
        "decision": decision,
        "reason": reason,
        "supported_regions": support_regions,
        "rejected_regions": reject_regions,
        "eligible_coordinates": len(eligible),
        "censored_coordinates": censored,
        "coordinates": evidence,
    }


def g4_decision(
    h5_rows: Sequence[Mapping[str, Any]],
    h6_rows: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    h5 = decide_h5(h5_rows, policy)
    h6 = decide_h6(h6_rows, policy)
    passed = h5["decision"] == "supported" and h6["decision"] == "supported"
    return {
        "gate": "G4",
        "decision": "PASS" if passed else "FAIL",
        "g5_authorized": passed,
        "H5": h5,
        "H6": h6,
    }


def _family_classes(family: str, values: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    if family == "P1-C-pd3":
        for dispersion in values["dispersion_classes"]:
            yield {"dispersion_class": dispersion}
    elif family == "P1-D-pd6":
        for attitude, rate in product(
            values["attitude_dispersion_radians"], values["angular_rate_dispersion"]
        ):
            yield {"attitude_class": attitude, "rate_class": rate}
    elif family == "P1-E-low-thrust":
        for trust, transfer in product(values["trust_radii"], values["transfer_classes"]):
            yield {"trust_class": trust, "transfer_class": transfer}
    else:
        raise G4ContractError(f"unsupported G4 family {family!r}")


def _coverage_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["family"],
        row["intervals"],
        row["policy"],
        row["quality_tier"],
        row["conditioning"],
        row["scaling_mode"],
        row["warm_mode"],
        row.get("dispersion_class"),
        row.get("attitude_class"),
        row.get("rate_class"),
        row.get("transfer_class"),
        row.get("trust_class"),
        row["seed"],
        row["instance"],
        row["repeat_kind"],
        row["repeat"],
    )


def iter_coverage_ledger(
    policy: Mapping[str, Any],
    executed: Iterable[Mapping[str, Any]],
    *,
    supported_policies: Iterable[str] = POLICY_NAMES,
) -> Iterator[dict[str, Any]]:
    """Yield the complete frozen Cartesian ledger with explicit dispositions."""

    from .g4_execution_contract import physical_instance_id, solver_rotation

    supported = set(supported_policies)
    executed_by_key: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in executed:
        key = _coverage_key(row)
        _require(key not in executed_by_key, f"duplicate coverage coordinate {key!r}")
        disposition = row.get("disposition")
        _require(disposition in DISPOSITIONS, f"invalid disposition {disposition!r}")
        _require(
            row.get("failure_class") in FAILURE_CLASSES,
            "every coverage disposition requires an explicit failure class",
        )
        _require(bool(row.get("reason")), "every coverage disposition requires a reason")
        if disposition in {"hybrid_handoff_ineligible", "not_applicable", "unsupported"}:
            timing = row.get("timing")
            _require(
                isinstance(timing, Mapping)
                and isinstance(timing.get("elapsed_seconds"), (int, float))
                and timing["elapsed_seconds"] >= 0,
                f"{disposition} requires explicit terminal timing",
            )
        executed_by_key[key] = row

    matrix = policy["matrix"]
    seeds = policy["tuning_evaluation_split"]["evaluation_seeds"][
        : matrix["randomised_instances_per_coordinate"]
    ]
    repeats = [
        *(("warmup", index) for index in range(matrix["warmup_repeats"])),
        *(("measured", index) for index in range(matrix["measured_repeats"])),
    ]
    for family, family_values in matrix["families"].items():
        for classes in _family_classes(family, family_values):
            for (
                intervals,
                policy_name,
                quality_tier,
                conditioning,
                scaling_mode,
                warm_mode,
                seed,
                repeat,
            ) in product(
                family_values["intervals"],
                POLICY_NAMES,
                matrix["quality_tiers"],
                matrix["conditioning_log10_spans"],
                matrix["scaling_modes"],
                matrix["warm_start_modes"],
                seeds,
                repeats,
            ):
                repeat_kind, repeat_index = repeat
                base = {
                    "family": family,
                    "intervals": intervals,
                    "policy": policy_name,
                    "quality_tier": quality_tier,
                    "conditioning": conditioning,
                    "scaling_mode": scaling_mode,
                    "warm_mode": warm_mode,
                    **classes,
                    "seed": seed,
                    "repeat_kind": repeat_kind,
                    "repeat": repeat_index,
                }
                base["instance"] = physical_instance_id(base)
                rotation = solver_rotation(
                    int(policy["randomisation"]["solver_order_seed"]),
                    base,
                )
                base["solver_order"] = (POLICY_NAMES.index(policy_name) + rotation) % len(
                    POLICY_NAMES
                )
                prior = executed_by_key.pop(_coverage_key(base), None)
                if prior is not None:
                    yield {**base, **prior}
                elif policy_name not in supported:
                    yield {
                        **base,
                        "disposition": "unsupported",
                        "failure_class": "unsupported",
                        "reason": "solver policy is unavailable in this campaign",
                    }
                else:
                    yield {
                        **base,
                        "disposition": "unrun",
                        "failure_class": "unrun",
                        "reason": "frozen coordinate has not been executed",
                    }
    _require(not executed_by_key, "executed ledger contains out-of-matrix coordinates")


def coverage_count(policy: Mapping[str, Any]) -> int:
    matrix = policy["matrix"]
    family_coordinates = 0
    for family, values in matrix["families"].items():
        family_coordinates += len(values["intervals"]) * sum(
            1 for _ in _family_classes(family, values)
        )
    repeats = matrix["warmup_repeats"] + matrix["measured_repeats"]
    return (
        family_coordinates
        * len(POLICY_NAMES)
        * len(matrix["quality_tiers"])
        * len(matrix["conditioning_log10_spans"])
        * len(matrix["scaling_modes"])
        * len(matrix["warm_start_modes"])
        * matrix["randomised_instances_per_coordinate"]
        * repeats
    )


def validate_artifact_contract(artifact: Mapping[str, Any], name: str) -> list[str]:
    reasons: list[str] = []
    uri = artifact.get("immutable_uri")
    digest = artifact.get("sha256")
    index_digest = artifact.get("internal_index_sha256")
    if not isinstance(uri, str) or not uri:
        reasons.append(f"{name} immutable artifact URI is missing")
    else:
        parsed = urlparse(uri)
        if parsed.scheme in {"", "file"}:
            reasons.append(f"{name} URI is local-only and not clean-clone portable")
    for value, label in ((digest, "SHA-256"), (index_digest, "internal index SHA-256")):
        if not isinstance(value, str) or len(value) != 64:
            reasons.append(f"{name} {label} is missing or invalid")
        elif any(character not in "0123456789abcdef" for character in value):
            reasons.append(f"{name} {label} must be lowercase hexadecimal")
    if artifact.get("portable") is not True:
        reasons.append(f"{name} portability is absent")
    return reasons


def validate_portability(artifacts: Any, *, raise_on_error: bool = True) -> dict[str, Any]:
    reasons: list[str] = []
    if not isinstance(artifacts, Mapping) or not artifacts:
        reasons.append("artifact evidence is missing")
    else:
        for name, artifact in artifacts.items():
            if not isinstance(artifact, Mapping):
                reasons.append(f"{name} artifact contract is missing")
            else:
                reasons.extend(validate_artifact_contract(artifact, str(name)))
    result = {"portable": not reasons, "reasons": reasons}
    if reasons and raise_on_error:
        raise G4ContractError("; ".join(reasons))
    return result


def verify_artifact_payload(
    artifact: Mapping[str, Any],
    payload: bytes,
    internal_index: bytes,
) -> None:
    """Verify downloaded evidence bytes without relying on the producing checkout."""

    reasons = validate_artifact_contract(artifact, "artifact")
    if reasons:
        raise G4ContractError("; ".join(reasons))
    content_digest = hashlib.sha256(payload).hexdigest()
    index_digest = hashlib.sha256(internal_index).hexdigest()
    _require(content_digest == artifact["sha256"], "artifact content SHA-256 mismatch")
    _require(
        index_digest == artifact["internal_index_sha256"],
        "artifact internal index SHA-256 mismatch",
    )
