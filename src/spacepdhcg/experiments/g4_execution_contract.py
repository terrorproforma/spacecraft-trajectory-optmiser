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
AMENDMENT_SCHEMA_VERSION: Final = "1.0.0"
APPLICABILITY_STATES: Final = ("executable", "not_applicable", "unsupported")
TERMINAL_DISPOSITIONS: Final = (
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
)
EXECUTOR_DEFECT_DISPOSITION: Final = "executor_defect"
# Amendment single-gpu-v1.1 (claim core only). The original single-gpu-v1 rules stay readable
# through ``ORIGINAL_CENSORING``; the amendment never rewrites the claim-core definition.
AMENDMENT_ID: Final = "single-gpu-v1.1"
AMENDMENT_RECORD_FIELD: Final = "policy_amendment"
ORIGINAL_CENSORING: Final = {"attempt_deadline_seconds": 600, "inner_iteration_cap": 1_000_000}
CLAIM_CORE_STRATUM: Final = "claim_core"
SENSITIVITY_STRATUM: Final = "censoring_sensitivity"
CENSORING_STRATA: Final = (CLAIM_CORE_STRATUM, SENSITIVITY_STRATUM)
REPLAY_DISPOSITION: Final = "timeout_deterministic_replay"
TRACE_HASH_FIELDS: Final = (
    "disposition",
    "inner_iterations",
    "outer_iterations",
    "canonical_residual",
    "dynamics_residual",
    "path_residual",
    "terminal_residual",
    "virtual_control_residual",
    "checkpoints[phase,requested_tolerance,achieved_residual,accepted,re_solved]",
)
_FNV_OFFSET: Final = 14_695_981_039_346_656_037
_FNV_PRIME: Final = 1_099_511_628_211
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
    result = {
        **physical_coordinate(coordinate),
        "policy": coordinate["policy"],
        "quality_tier": coordinate["quality_tier"],
        "quality_tolerance": coordinate["quality_tolerance"],
        "conditioning": coordinate["conditioning"],
        "scaling_mode": coordinate["scaling_mode"],
        "warm_mode": coordinate["warm_mode"],
        "solver_order": coordinate["solver_order"],
    }
    # Amendment single-gpu-v1.1: a censoring-sensitivity twin shares every physical and
    # experimental axis of its claim-core group (same physical instance, same solver order) and
    # differs only in the censoring stratum, which therefore enters the group identity.
    stratum = coordinate.get("censoring_stratum")
    if stratum is not None:
        _require(stratum in CENSORING_STRATA, f"unknown censoring stratum {stratum!r}")
        if stratum != CLAIM_CORE_STRATUM:
            result["censoring_stratum"] = stratum
    return result


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
    if disposition == REPLAY_DISPOSITION:
        # Amendment single-gpu-v1.1: a replayed timeout is never launched; it references the
        # executed measured/0 attempt whose deterministic trace it repeats.
        _require(record["launched"] is False, "replayed timeouts are recorded without a launch")
        _require(record["repeat_kind"] == "measured", "only measured attempts may be replayed")
        _require(repeat >= 1, "measured/0 must be executed before any replay")
        source = record.get("replay_source_attempt_id")
        _require(
            isinstance(source, str) and source.endswith("/measured-0"),
            "replayed timeouts must reference the executed measured/0 attempt",
        )
        _require(
            record.get(AMENDMENT_RECORD_FIELD) == AMENDMENT_ID,
            "replayed timeouts exist only under amendment single-gpu-v1.1",
        )
    if disposition == "not_applicable":
        _require(record["launched"] is False, "not_applicable attempts may not be launched")
    if disposition == EXECUTOR_DEFECT_DISPOSITION:
        # The record itself is well formed (so the evidence is retained and auditable) but it
        # is invalid as an observation: the executor, not the solver, failed.
        _require(record["launched"] is True, "an executor defect is recorded on a launched attempt")
        _require(
            record.get("failure_class") == EXECUTOR_DEFECT_DISPOSITION,
            "executor_defect disposition and failure class must match",
        )
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


# ------------------------------------------------------------------------------------------------
# Amendment single-gpu-v1.1 (preregistered claim-core amendment)
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoadedAmendment:
    values: dict[str, Any]
    sha256: str
    path: Path


def fnv1a64(payload: bytes) -> int:
    """64-bit FNV-1a, bit-identical to ``hash_bytes`` in the native executor."""

    value = _FNV_OFFSET
    for byte in payload:
        value ^= byte
        value = (value * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return value


def _g17(value: Any) -> str:
    """``%.17g`` exactly as the executor prints doubles into the trace string."""

    number = float(value)
    _require(number == number and abs(number) != float("inf"), "trace values must be finite")
    return f"{number:.17g}"


def deterministic_trace_string(disposition: str, trace: Mapping[str, Any]) -> str:
    """Canonical trace text hashed by both the executor and this reference implementation.

    ``trace`` is the ``trace`` object of a raw attempt record: iteration counts, the final
    residual set, and one ``[phase, requested_tolerance, achieved_residual, accepted,
    re_solved]`` checkpoint per outer iteration.
    """

    parts = [
        str(disposition),
        str(int(trace["inner_iterations"])),
        str(int(trace["outer_iterations"])),
        _g17(trace["canonical_residual"]),
        _g17(trace["dynamics_residual"]),
        _g17(trace["path_residual"]),
        _g17(trace["terminal_residual"]),
        _g17(trace["virtual_control_residual"]),
    ]
    checkpoints = []
    for checkpoint in trace["checkpoints"]:
        phase, requested, achieved, accepted, re_solved = checkpoint
        checkpoints.append(
            f"{phase}:{_g17(requested)}:{_g17(achieved)}:{int(bool(accepted))}:"
            f"{int(bool(re_solved))}"
        )
    return "|".join(parts) + "|" + ";".join(checkpoints)


def deterministic_trace_hash(disposition: str, trace: Mapping[str, Any]) -> str:
    return f"{fnv1a64(deterministic_trace_string(disposition, trace).encode('utf-8')):016x}"


def deterministic_replay_eligible(records: Sequence[Mapping[str, Any]]) -> bool:
    """Amendment rule 1: warm-up/0, warm-up/1 and measured/0 all timed out with equal traces."""

    if len(records) != 3:
        return False
    expected = (("warmup", 0), ("warmup", 1), ("measured", 0))
    for record, (kind, repeat) in zip(records, expected, strict=True):
        if (record.get("repeat_kind"), record.get("repeat")) != (kind, repeat):
            return False
        if record.get("launched") is not True or record.get("disposition") != "timeout":
            return False
        if not isinstance(record.get("trace_hash"), str) or not isinstance(
            record.get("trace"), Mapping
        ):
            return False
        if deterministic_trace_hash("timeout", record["trace"]) != record["trace_hash"]:
            raise G4ContractError("raw attempt trace_hash does not match its trace")
    return len({record["trace_hash"] for record in records}) == 1


def censoring_sensitivity_group_ids(
    definition: Mapping[str, Any],
    amendment: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Deterministic hash-selected subset per family x scale x policy stratum.

    Within each stratum the twenty claim-core groups are ranked by
    ``sha256(selection_seed + "|" + group_id)`` and the lowest ``groups_per_stratum`` are
    selected; the choice depends only on the frozen group identities and the amendment seed.
    """

    stratum = amendment["censoring"]["sensitivity_stratum"]
    seed = str(stratum["selection_seed"])
    per_stratum = int(stratum["groups_per_stratum"])
    ranked: dict[tuple[str, int, str], list[tuple[str, str]]] = {}
    for group in iter_claim_core_groups(definition):
        coordinate = group.coordinate
        key = (coordinate["family"], coordinate["intervals"], coordinate["policy"])
        rank = hashlib.sha256(f"{seed}|{group.group_id}".encode()).hexdigest()
        ranked.setdefault(key, []).append((rank, group.group_id))
    selected: dict[str, list[str]] = {}
    for key in sorted(ranked, key=lambda item: (item[0], item[1], item[2])):
        chosen = sorted(ranked[key])[:per_stratum]
        selected["/".join(str(part) for part in key)] = [group_id for _, group_id in chosen]
    return selected


def sensitivity_group_for(group: ExecutionGroup) -> ExecutionGroup:
    """The 600 s / 1M twin of one claim-core group (distinct identity, same instance)."""

    _require(
        group.coordinate.get("censoring_stratum") is None,
        "sensitivity twins derive from claim-core groups only",
    )
    return make_execution_group({**group.coordinate, "censoring_stratum": SENSITIVITY_STRATUM})


def group_censoring_stratum(group: ExecutionGroup) -> str:
    return str(group.coordinate.get("censoring_stratum") or CLAIM_CORE_STRATUM)


def group_censoring(
    group: ExecutionGroup,
    amendment: Mapping[str, Any],
) -> dict[str, int]:
    """Attempt deadline and inner iteration cap in force for one scheduled group."""

    censoring = amendment["censoring"]
    if group_censoring_stratum(group) == SENSITIVITY_STRATUM:
        source = censoring["original"]
    else:
        source = censoring["claim_core"]
    deadline = int(source["attempt_deadline_seconds"])
    return {
        "attempt_deadline_seconds": deadline,
        "inner_iteration_cap": int(source["inner_iteration_cap"]),
        "group_deadline_seconds": 9 * deadline + 60,
    }


def amended_claim_core_groups(
    definition: Mapping[str, Any],
    amendment: Mapping[str, Any],
) -> tuple[ExecutionGroup, ...]:
    """Frozen execution schedule of the amended claim core.

    Group identities, solver-order values and the seven-plus-two repeat structure are those of
    the claim core. Only the execution order changes: policies run in the amendment's
    ``policy_priority`` (converging policies before fixed-tight) and each hash-selected
    censoring-sensitivity twin immediately follows its claim-core group so the preregistered
    acceptance rule can be checked as evidence accrues.
    """

    priority = list(amendment["schedule"]["policy_priority"])
    selected = {
        group_id
        for group_ids in amendment["censoring"]["sensitivity_stratum"]["group_ids"].values()
        for group_id in group_ids
    }
    core_groups = tuple(iter_claim_core_groups(definition))
    _require(
        selected <= {group.group_id for group in core_groups},
        "sensitivity stratum references groups outside the claim core",
    )
    ordered: list[ExecutionGroup] = []
    for policy_name in priority:
        for group in core_groups:
            if group.coordinate["policy"] != policy_name:
                continue
            ordered.append(group)
            if group.group_id in selected:
                ordered.append(sensitivity_group_for(group))
    _require(len(ordered) == len(core_groups) + len(selected), "amended schedule count drift")
    return tuple(ordered)


def amended_schedule_sha256(groups: Sequence[ExecutionGroup]) -> str:
    return hashlib.sha256(canonical_bytes([group.group_id for group in groups])).hexdigest()


def validate_claim_core_amendment(
    amendment: Mapping[str, Any],
    definition: Mapping[str, Any],
    *,
    claim_core_sha256: str,
    policy_sha256: str,
) -> None:
    _require(amendment.get("schema_version") == AMENDMENT_SCHEMA_VERSION, "amendment schema drift")
    _require(amendment.get("amendment_id") == AMENDMENT_ID, "amendment identity drift")
    _require(
        amendment.get("preregistered_before_results") is True,
        "amendment must be frozen before any group result is inspected",
    )
    amends = amendment.get("amends", {})
    _require(amends.get("campaign_scope_id") == "single-gpu-v1", "amendment scope drift")
    _require(
        amends.get("claim_core_campaign_id") == definition.get("campaign_id"),
        "amendment references a different claim core",
    )
    _require(amends.get("claim_core_sha256") == claim_core_sha256, "amendment core hash drift")
    _require(amends.get("policy_sha256") == policy_sha256, "amendment policy hash drift")
    record_field = amendment.get("record_field", {})
    _require(
        record_field == {"name": AMENDMENT_RECORD_FIELD, "value": AMENDMENT_ID},
        "amendment record field drift",
    )
    contamination = amendment.get("contamination", {})
    _require(contamination.get("policy") == "run_and_flag", "contamination policy drift")
    _require(
        contamination.get("attempt_flag") == "contaminated"
        and contamination.get("rerun_required") is False,
        "contaminated attempts must be flagged in place without a re-run",
    )
    replay = amendment.get("deterministic_replay", {})
    _require(replay.get("disposition") == REPLAY_DISPOSITION, "replay disposition drift")
    _require(
        tuple(replay.get("trace_hash_fields", ())) == TRACE_HASH_FIELDS,
        "replay trace fields drift",
    )
    _require(replay.get("trace_hash_algorithm") == "fnv1a64", "replay hash algorithm drift")
    censoring = amendment.get("censoring", {})
    _require(
        censoring.get("original") == ORIGINAL_CENSORING,
        "original single-gpu-v1 censoring values must remain readable and unchanged",
    )
    core = censoring.get("claim_core", {})
    _require(
        core == {"attempt_deadline_seconds": 120, "inner_iteration_cap": 200_000},
        "claim-core censoring values drift",
    )
    stratum = censoring.get("sensitivity_stratum", {})
    _require(stratum.get("name") == SENSITIVITY_STRATUM, "sensitivity stratum name drift")
    _require(
        stratum.get("stratification") == ["family", "intervals", "policy"],
        "sensitivity stratification drift",
    )
    _require(stratum.get("groups_per_stratum") == 2, "sensitivity stratum must hold 10%")
    expected_ids = censoring_sensitivity_group_ids(definition, amendment)
    _require(
        stratum.get("group_ids") == expected_ids,
        "committed sensitivity group IDs differ from the deterministic selection",
    )
    total = sum(len(ids) for ids in expected_ids.values())
    _require(stratum.get("group_count") == total == 36, "sensitivity group count drift")
    schedule = amendment.get("schedule", {})
    _require(
        schedule.get("policy_priority")
        == ["pure-gpu-ipm", "adaptive", "hybrid-pdhcg-ipm", "fixed-tight"],
        "schedule policy priority drift",
    )
    _require(
        schedule.get("solver_order_identity_unchanged") is True,
        "the frozen solver-order rotation must remain recorded unchanged",
    )
    groups = amended_claim_core_groups(definition, amendment)
    _require(
        schedule.get("group_count") == len(groups) == 396, "amended schedule cardinality drift"
    )
    _require(
        schedule.get("schedule_sha256") == amended_schedule_sha256(groups),
        "amended schedule hash drift",
    )
    unchanged = amendment.get("unchanged", [])
    _require(isinstance(unchanged, list) and len(unchanged) >= 6, "amendment must list invariants")


def load_claim_core_amendment(
    path: str | Path,
    definition: Mapping[str, Any],
    *,
    claim_core_sha256: str,
    policy_sha256: str,
    expected_sha256: str | None = None,
) -> LoadedAmendment:
    source = Path(path)
    digest = sha256_path(source)
    if expected_sha256 is not None:
        _require(digest == expected_sha256, "claim-core amendment hash drift")
    payload = json.loads(source.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "amendment root must be an object")
    validate_claim_core_amendment(
        payload,
        definition,
        claim_core_sha256=claim_core_sha256,
        policy_sha256=policy_sha256,
    )
    return LoadedAmendment(payload, digest, source)
