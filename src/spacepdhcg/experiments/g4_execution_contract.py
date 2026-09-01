"""Versioned G4 applicability, identity, grouping, and claim-core contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Final

from .g4 import (
    POLICY_NAMES,
    QUALITY_TIERS,
    SCALING_MODES,
    WARM_MODES,
    G4ContractError,
    sha256_path,
)

APPLICABILITY_SCHEMA_VERSION: Final = "1.0.0"
CLAIM_CORE_SCHEMA_VERSION: Final = "1.0.0"
APPLICABILITY_STATES: Final = ("executable", "not_applicable", "unsupported")
TERMINAL_DISPOSITIONS: Final = (
    "qualified",
    "unqualified",
    "hybrid_handoff_ineligible",
    "not_applicable",
    "unsupported",
    "oom",
    "timeout",
    "numerical",
    "infeasible",
    "unrun",
)
FAMILY_CLASS_KEYS: Final = (
    "dispersion_class",
    "attitude_class",
    "rate_class",
    "trust_class",
    "transfer_class",
)
FAMILY_REQUIRED_CLASSES: Final = {
    "P1-C-pd3": ("dispersion_class",),
    "P1-D-pd6": ("attitude_class", "rate_class"),
    "P1-E-low-thrust": ("trust_class", "transfer_class"),
}
ATTEMPT_KINDS: Final = (("warmup", 0), ("warmup", 1), *(("measured", index) for index in range(7)))
PUBLICATION_PRODUCT_IDS: Final = frozenset(
    {*(f"F{index:02d}" for index in range(1, 13)), *(f"T{index:02d}" for index in range(1, 9))}
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G4ContractError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def content_id(prefix: str, value: Any) -> str:
    return f"{prefix}-{hashlib.sha256(canonical_bytes(value)).hexdigest()}"


@dataclass(frozen=True, slots=True)
class LoadedApplicability:
    values: dict[str, Any]
    sha256: str
    path: Path


@dataclass(frozen=True, slots=True)
class Applicability:
    state: str
    reason: str
    axis_decisions: dict[str, dict[str, str]]
    effective_warm_mode: str
    dual_disposition: str | None


@dataclass(frozen=True, slots=True)
class ExecutionGroup:
    group_id: str
    physical_instance_id: str
    coordinate: dict[str, Any]
    attempts: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class LoadedClaimCore:
    values: dict[str, Any]
    sha256: str
    path: Path


def validate_applicability_contract(contract: Mapping[str, Any]) -> None:
    _require(
        contract.get("schema_version") == APPLICABILITY_SCHEMA_VERSION,
        "unsupported G4 applicability schema",
    )
    _require(contract.get("gate") == "G4", "applicability contract gate must be G4")
    _require(
        tuple(contract.get("coordinate_states", ())) == APPLICABILITY_STATES,
        "applicability state order/value drift",
    )
    semantics = contract.get("matrix_semantics", {})
    _require(semantics.get("logical_rows") == 24_883_200, "logical row count drift")
    _require(semantics.get("measured_rows") == 19_353_600, "measured row count drift")
    _require(semantics.get("warmup_rows") == 5_529_600, "warmup row count drift")
    _require(semantics.get("execution_groups") == 2_764_800, "execution group count drift")
    _require(semantics.get("attempts_per_group") == 9, "attempts-per-group drift")
    _require(
        semantics.get("full_matrix_reduction_permitted") is False,
        "applicability may not reduce the frozen full matrix",
    )
    family_policy = contract.get("family_policy", {})
    _require(set(family_policy) == set(FAMILY_REQUIRED_CLASSES), "family applicability drift")
    for family, policies in family_policy.items():
        _require(tuple(policies) == POLICY_NAMES, f"policy applicability drift for {family}")
        for policy_name, state in policies.items():
            _require(
                state in APPLICABILITY_STATES,
                f"invalid applicability state for {family}/{policy_name}",
            )
    family_axes = contract.get("family_axes", {})
    for family in FAMILY_REQUIRED_CLASSES:
        _require(
            set(family_axes.get(family, {})) == set(FAMILY_CLASS_KEYS), f"axis drift for {family}"
        )
        for axis, decision in family_axes[family].items():
            _require(
                decision.get("state") in APPLICABILITY_STATES,
                f"invalid axis state for {family}/{axis}",
            )
            _require(bool(decision.get("reason")), f"missing axis reason for {family}/{axis}")
    policy_axes = contract.get("policy_axes", {})
    _require(tuple(policy_axes) == POLICY_NAMES, "policy-axis applicability drift")
    for policy_name, axes in policy_axes.items():
        _require(
            set(axes) == {"quality_tier", "scaling_mode", "warm_mode"},
            f"policy-axis coverage drift for {policy_name}",
        )
        for axis, decision in axes.items():
            _require(
                decision.get("state") in APPLICABILITY_STATES,
                f"invalid policy-axis state for {policy_name}/{axis}",
            )
            _require(
                bool(decision.get("reason")), f"missing policy-axis reason for {policy_name}/{axis}"
            )
    terminal = contract.get("terminal_dispositions", {})
    for disposition in ("hybrid_handoff_ineligible", "not_applicable", "unsupported"):
        rules = terminal.get(disposition, {})
        _require(rules.get("reason_required") is True, f"{disposition} must require a reason")
        _require(rules.get("timing_required") is True, f"{disposition} must require timing")
        _require(rules.get("winner_eligible") is False, f"{disposition} may not win")
    censoring = contract.get("censoring", {})
    _require(
        censoring.get("predictive_timeout_or_oom") == "forbidden", "predictive censoring drift"
    )
    _require(
        censoring.get("terminal_timeout_or_oom_requires_launch") is True,
        "timeouts/OOM must require launch",
    )


def load_applicability(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> LoadedApplicability:
    source = Path(path)
    digest = sha256_path(source)
    if expected_sha256 is not None:
        _require(digest == expected_sha256, "G4 applicability hash drift")
    payload = json.loads(source.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "G4 applicability root must be an object")
    validate_applicability_contract(payload)
    return LoadedApplicability(payload, digest, source)


def evaluate_applicability(
    contract: Mapping[str, Any],
    coordinate: Mapping[str, Any],
) -> Applicability:
    family = str(coordinate.get("family", ""))
    policy_name = str(coordinate.get("policy", ""))
    _require(family in FAMILY_REQUIRED_CLASSES, f"unknown G4 family {family!r}")
    _require(policy_name in POLICY_NAMES, f"unknown G4 policy {policy_name!r}")
    _require(coordinate.get("quality_tier") in QUALITY_TIERS, "unknown quality tier")
    _require(coordinate.get("scaling_mode") in SCALING_MODES, "unknown scaling mode")
    _require(coordinate.get("warm_mode") in WARM_MODES, "unknown warm mode")
    family_policy_state = contract["family_policy"][family][policy_name]
    decisions = {axis: dict(contract["family_axes"][family][axis]) for axis in FAMILY_CLASS_KEYS}
    decisions.update(
        {
            axis: dict(contract["policy_axes"][policy_name][axis])
            for axis in ("quality_tier", "scaling_mode", "warm_mode")
        }
    )
    for axis in FAMILY_REQUIRED_CLASSES[family]:
        _require(axis in coordinate, f"physical coordinate is missing {axis}")
    forbidden = set(FAMILY_CLASS_KEYS) - set(FAMILY_REQUIRED_CLASSES[family])
    _require(
        not any(coordinate.get(axis) is not None for axis in forbidden),
        f"coordinate supplies not-applicable family axes for {family}",
    )
    axis_states = [decision["state"] for decision in decisions.values()]
    state = (
        "unsupported"
        if family_policy_state == "unsupported" or "unsupported" in axis_states
        else "not_applicable"
        if family_policy_state == "not_applicable"
        else "executable"
    )
    reason = (
        f"{family}/{policy_name} is executable under applicability contract "
        f"{contract['contract_id']}"
        if state == "executable"
        else (
            f"{family}/{policy_name} is {state} under applicability contract "
            f"{contract['contract_id']}"
        )
    )
    dual_disposition = None
    effective_warm = str(coordinate["warm_mode"])
    if coordinate["warm_mode"] == "primal_dual" and policy_name in {
        "pure-gpu-ipm",
        "hybrid-pdhcg-ipm",
    }:
        dual_disposition = "discarded_unsupported"
        if policy_name == "pure-gpu-ipm":
            effective_warm = "primal"
    return Applicability(state, reason, decisions, effective_warm, dual_disposition)


def physical_coordinate(coordinate: Mapping[str, Any]) -> dict[str, Any]:
    family = str(coordinate.get("family", ""))
    _require(family in FAMILY_REQUIRED_CLASSES, f"unknown G4 family {family!r}")
    intervals = coordinate.get("intervals")
    seed = coordinate.get("seed")
    _require(isinstance(intervals, int) and intervals > 0, "physical intervals must be positive")
    _require(isinstance(seed, int) and seed >= 0, "physical seed must be non-negative")
    result: dict[str, Any] = {"family": family, "intervals": intervals}
    for axis in FAMILY_REQUIRED_CLASSES[family]:
        _require(
            axis in coordinate and coordinate[axis] is not None,
            f"physical coordinate missing {axis}",
        )
        result[axis] = coordinate[axis]
    result["seed"] = seed
    return result


def physical_instance_id(coordinate: Mapping[str, Any]) -> str:
    """Hash every physical axis; repeat and solver policy intentionally stay separate."""

    return content_id("g4-instance-v2", physical_coordinate(coordinate))


def solver_rotation_digest(
    solver_order_seed: int,
    coordinate: Mapping[str, Any],
) -> str:
    """Hash the full physical and experimental coordinate, excluding policy/repeat."""

    material = {
        "solver_order_seed": solver_order_seed,
        "physical": physical_coordinate(coordinate),
        "quality_tier": coordinate["quality_tier"],
        "conditioning": coordinate["conditioning"],
        "scaling_mode": coordinate["scaling_mode"],
        "warm_mode": coordinate["warm_mode"],
    }
    return hashlib.sha256(canonical_bytes(material)).hexdigest()


def solver_rotation(
    solver_order_seed: int,
    coordinate: Mapping[str, Any],
) -> int:
    """Rotate policy order from the full coordinate digest."""

    return int(solver_rotation_digest(solver_order_seed, coordinate)[:8], 16) % len(POLICY_NAMES)


def execution_group_coordinate(coordinate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **physical_coordinate(coordinate),
        "policy": coordinate["policy"],
        "quality_tier": coordinate["quality_tier"],
        "quality_tolerance": coordinate["quality_tolerance"],
        "conditioning": coordinate["conditioning"],
        "scaling_mode": coordinate["scaling_mode"],
        "warm_mode": coordinate["warm_mode"],
        "solver_order": coordinate["solver_order"],
    }


def make_execution_group(coordinate: Mapping[str, Any]) -> ExecutionGroup:
    group_coordinate = execution_group_coordinate(coordinate)
    instance_id = physical_instance_id(group_coordinate)
    group_id = content_id("g4-group-v1", group_coordinate)
    attempts = tuple(
        {
            **group_coordinate,
            "group_id": group_id,
            "instance": instance_id,
            "repeat_kind": repeat_kind,
            "repeat": repeat,
            "statistics_eligible": repeat_kind == "measured",
        }
        for repeat_kind, repeat in ATTEMPT_KINDS
    )
    return ExecutionGroup(group_id, instance_id, group_coordinate, attempts)


def validate_attempt_record(record: Mapping[str, Any]) -> None:
    _require(record.get("record_kind") == "raw_attempt", "record kind must be raw_attempt")
    _require(bool(record.get("group_id")), "raw attempt requires a group ID")
    _require(bool(record.get("attempt_id")), "raw attempt requires an attempt ID")
    _require(record.get("repeat_kind") in {"warmup", "measured"}, "invalid attempt kind")
    repeat = record.get("repeat")
    _require(isinstance(repeat, int) and repeat >= 0, "invalid attempt repeat")
    _require(isinstance(record.get("launched"), bool), "raw attempt requires launched boolean")
    disposition = record.get("disposition")
    _require(disposition in TERMINAL_DISPOSITIONS, f"invalid terminal disposition {disposition!r}")
    _require(bool(record.get("reason")), "every terminal disposition requires a reason")
    timing = record.get("timing")
    _require(isinstance(timing, Mapping), "every terminal disposition requires timing")
    elapsed = timing.get("elapsed_seconds")
    _require(
        isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and elapsed >= 0,
        "terminal timing requires non-negative elapsed_seconds",
    )
    if disposition in {"timeout", "oom"}:
        _require(record["launched"] is True, "timeout/OOM disposition requires an actual launch")
    if disposition == "not_applicable":
        _require(record["launched"] is False, "not_applicable attempts may not be launched")
    if disposition == "hybrid_handoff_ineligible":
        _require(record["launched"] is True, "hybrid handoff eligibility requires an actual launch")
        _require(
            record.get("policy") == "hybrid-pdhcg-ipm",
            "hybrid_handoff_ineligible is reserved for the hybrid policy",
        )
    expected_statistics = record["repeat_kind"] == "measured"
    _require(
        record.get("statistics_eligible") is expected_statistics,
        "warmups must be excluded and measured attempts retained for statistics",
    )


def winner_eligible(record: Mapping[str, Any]) -> bool:
    return bool(
        record.get("repeat_kind") == "measured"
        and record.get("statistics_eligible") is True
        and record.get("disposition") == "qualified"
    )


def validate_claim_core(definition: Mapping[str, Any]) -> None:
    _require(
        definition.get("schema_version") == CLAIM_CORE_SCHEMA_VERSION, "claim-core schema drift"
    )
    _require(
        definition.get("campaign_kind") == "h5_h6_claim_resolution_core", "claim-core kind drift"
    )
    _require(definition.get("claims_resolved") == ["H5", "H6"], "claim-core claims drift")
    _require(
        definition.get("full_regime_matrix_substitute") is False, "claim core cannot replace G4"
    )
    _require(
        definition.get("paper1_regime_products_permitted") == [], "claim core cannot feed products"
    )
    repetitions = definition.get("repetitions", {})
    _require(
        repetitions == {"instances": 20, "warmups": 2, "measurements": 7}, "core repeats drift"
    )
    families = definition.get("families", [])
    _require(isinstance(families, list) and len(families) == 2, "claim core needs two families")
    policy_scale_pairs = 0
    seen: set[str] = set()
    for family in families:
        family_name = family.get("family")
        _require(family_name in FAMILY_REQUIRED_CLASSES, "claim core has unknown family")
        _require(family_name not in seen, "claim core families must be distinct")
        seen.add(family_name)
        scales = family.get("scales")
        policies = family.get("policies")
        _require(
            isinstance(scales, list) and len(scales) == 3, "each core family needs three scales"
        )
        _require(len(set(scales)) == 3 and scales == sorted(scales), "core scales must increase")
        _require(isinstance(policies, list) and policies, "core family needs policies")
        _require(set(policies) <= set(POLICY_NAMES), "core has unknown policy")
        policy_scale_pairs += len(scales) * len(policies)
    _require(policy_scale_pairs == 18, "claim core must contain exactly 18 policy-scale pairs")
    _require(claim_core_invocation_count(definition) == 3_240, "claim core count must be 3,240")


def load_claim_core(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> LoadedClaimCore:
    source = Path(path)
    digest = sha256_path(source)
    if expected_sha256 is not None:
        _require(digest == expected_sha256, "H5/H6 claim-core hash drift")
    payload = json.loads(source.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "claim-core root must be an object")
    validate_claim_core(payload)
    return LoadedClaimCore(payload, digest, source)


def claim_core_invocation_count(definition: Mapping[str, Any]) -> int:
    repetitions = definition["repetitions"]
    attempts = repetitions["warmups"] + repetitions["measurements"]
    policy_scale_pairs = sum(
        len(family["scales"]) * len(family["policies"]) for family in definition["families"]
    )
    return policy_scale_pairs * repetitions["instances"] * attempts


def iter_claim_core_groups(definition: Mapping[str, Any]) -> Iterator[ExecutionGroup]:
    seeds: Sequence[int] = definition["evaluation_seeds"]
    _require(len(seeds) == definition["repetitions"]["instances"], "claim-core seed count drift")
    defaults = definition["fixed_axes"]
    for family_definition in definition["families"]:
        family = family_definition["family"]
        classes = family_definition["classes"]
        for intervals, seed, policy_name in product(
            family_definition["scales"],
            seeds,
            family_definition["policies"],
        ):
            coordinate = {
                "family": family,
                "intervals": intervals,
                **classes,
                "seed": seed,
                "policy": policy_name,
                "quality_tier": defaults["quality_tier"],
                "quality_tolerance": defaults["quality_tolerance"],
                "conditioning": defaults["conditioning"],
                "scaling_mode": defaults["scaling_mode"],
                "warm_mode": defaults["warm_mode"],
            }
            coordinate["solver_order"] = solver_rotation(
                int(definition["solver_order_seed"]),
                coordinate,
            )
            yield make_execution_group(coordinate)


def claim_core_group_at(
    definition: Mapping[str, Any],
    index: int,
) -> ExecutionGroup:
    total = claim_core_invocation_count(definition) // len(ATTEMPT_KINDS)
    if index < 0 or index >= total:
        raise IndexError(f"claim-core group index {index} outside [0, {total})")
    for current, group in enumerate(iter_claim_core_groups(definition)):
        if current == index:
            return group
    raise AssertionError("validated claim-core group count drift")


def claim_core_may_populate_product(definition: Mapping[str, Any], product_id: str) -> bool:
    _require(product_id in PUBLICATION_PRODUCT_IDS, f"unknown Paper 1 product {product_id!r}")
    return product_id in definition["paper1_regime_products_permitted"]
