#!/usr/bin/env python3
"""Validate, aggregate, and decide the preregistered G4 H5/H6 claim core.

The claim core (``benchmarks/g4_h5_h6_claim_core.json``) resolves H5 and H6 only. This script

1. re-validates every completed execution group against the raw-attempt and Paper 1 schemas and
   against the frozen group definition (identity, order, repeat set);
2. builds one strict ``record_scope=publication_aggregate`` Paper 1 record per policy/scale pair
   from the schema-valid measured attempts (warm-ups never enter a statistic);
3. pairs fixed-tight/adaptive (H5) and pure-GPU-IPM/hybrid (H6) measured attempts by physical
   instance and repeat index and applies the frozen decision functions in
   ``spacepdhcg.experiments.g4`` with the policy's bootstrap seed, sample count and thresholds.

No timing enters a decision unless both sides of a pair are ``qualified``; nothing is imputed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from spacepdhcg.experiments.g4 import (  # noqa: E402
    G4ContractError,
    g4_decision,
    load_policy,
    sha256_path,
)
from spacepdhcg.experiments.g4_execution_contract import (  # noqa: E402
    ATTEMPT_KINDS,
    ExecutionGroup,
    iter_claim_core_groups,
    load_claim_core,
    validate_attempt_record,
)
from spacepdhcg.experiments.paper1 import (  # noqa: E402
    Paper1ResultError,
    validate_paper1_result_schema,
)

RESIDUAL_FIELDS = (
    "canonical_primal_residual",
    "canonical_dual_residual",
    "canonical_cone_residual",
    "canonical_gap",
    "dynamics_residual",
    "path_residual",
    "terminal_residual",
)
CQP_COMPONENTS = (
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
EXTRA_COMPONENTS = ("replay_seconds", "acceptance_seconds")
PAIRING_RULE = (
    "measured attempts are paired by identical physical instance (evaluation seed) and repeat "
    "index; a pair enters the paired statistic only when both attempts are qualified; warm-ups, "
    "unqualified, censored and contaminated attempts never enter a statistic and are counted "
    "as failures or censoring instead"
)


class ClaimCoreDecisionError(ValueError):
    """Raised when claim-core evidence is incomplete or violates a frozen contract."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()


def content_id(prefix: str, value: Any) -> str:
    return f"{prefix}-{hashlib.sha256(canonical_bytes(value)).hexdigest()}"


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _median_int(values: Sequence[int]) -> int | None:
    return int(statistics.median_low(values)) if values else None


def _quartiles(values: Sequence[float]) -> tuple[float, float]:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0], ordered[0]
    quartiles = statistics.quantiles(ordered, n=4, method="inclusive")
    return quartiles[0], quartiles[2]


def final_residual(quality: Mapping[str, Any]) -> float:
    return max(float(quality[name]) for name in RESIDUAL_FIELDS if quality.get(name) is not None)


# --------------------------------------------------------------------------------------------
# Campaign loading and re-validation
# --------------------------------------------------------------------------------------------


def load_campaign(campaign: Path) -> tuple[dict[str, str], list[sqlite3.Row]]:
    database = sqlite3.connect(f"file:{campaign / 'checkpoint.sqlite3'}?mode=ro", uri=True)
    database.row_factory = sqlite3.Row
    metadata = {
        row["key"]: row["value"] for row in database.execute("SELECT key, value FROM metadata")
    }
    rows = list(
        database.execute(
            """
            SELECT c.ordinal, c.coordinate_id, c.state, a.attempt_id, a.disposition, a.reason,
                   a.run_directory
            FROM coordinates AS c
            JOIN attempts AS a ON a.attempt_id = c.latest_attempt_id
            ORDER BY c.ordinal
            """
        )
    )
    database.close()
    return metadata, rows


def validate_group_evidence(
    group: ExecutionGroup,
    result: Mapping[str, Any],
    raw_validator: Draft202012Validator,
    paper1_schema: Path,
) -> list[dict[str, Any]]:
    """Return the seven validated measured attempts of one completed group."""

    if result.get("record_kind") != "execution_group_result":
        raise ClaimCoreDecisionError("group result record kind drift")
    if result.get("group_id") != group.group_id:
        raise ClaimCoreDecisionError("group result identity differs from the claim core")
    attempts = result.get("raw_attempts", [])
    if len(attempts) != len(ATTEMPT_KINDS):
        raise ClaimCoreDecisionError(f"group {group.group_id} retained {len(attempts)} attempts")
    by_repeat = {(item["repeat_kind"], item["repeat"]): item for item in attempts}
    if set(by_repeat) != set(ATTEMPT_KINDS):
        raise ClaimCoreDecisionError(f"group {group.group_id} repeat set drift")
    measured: list[dict[str, Any]] = []
    for planned in group.attempts:
        record = by_repeat[(planned["repeat_kind"], planned["repeat"])]
        for field in (
            "group_id",
            "family",
            "intervals",
            "policy",
            "instance",
            "seed",
            "repeat_kind",
            "repeat",
            "statistics_eligible",
        ):
            if record.get(field) != planned[field]:
                raise ClaimCoreDecisionError(
                    f"group {group.group_id} attempt {planned['repeat_kind']}/{planned['repeat']} "
                    f"field {field} differs from the frozen definition"
                )
        try:
            raw_validator.validate(record)
            validate_attempt_record(record)
        except (ValidationError, G4ContractError) as error:
            raise ClaimCoreDecisionError(f"raw attempt invalid: {error}") from error
        if planned["repeat_kind"] != "measured":
            continue
        paper1 = record.get("paper1_result")
        if not isinstance(paper1, Mapping):
            raise ClaimCoreDecisionError("measured attempt lacks paper1_result")
        try:
            validate_paper1_result_schema(paper1, paper1_schema)
        except Paper1ResultError as error:
            raise ClaimCoreDecisionError(f"measured Paper 1 record invalid: {error}") from error
        identity = paper1["identity"]
        expected_identity = {
            "record_scope": "measured_attempt",
            "family": planned["family"],
            "policy": planned["policy"],
            "instance_id": planned["instance"],
            "seed": planned["seed"],
            "repeat": planned["repeat"],
            "repeat_kind": "measured",
            "status": record["disposition"],
            "quality_tier": planned["quality_tier"],
            "scaling_mode": planned["scaling_mode"],
            "warm_mode": planned["warm_mode"],
            "solver_order": planned["solver_order"],
        }
        observed = {key: identity.get(key) for key in expected_identity}
        if observed != expected_identity:
            raise ClaimCoreDecisionError(
                f"measured identity drift for {group.group_id}: {observed!r}"
            )
        if paper1["dimensions"]["intervals"] != planned["intervals"]:
            raise ClaimCoreDecisionError("measured dimensions differ from the frozen definition")
        measured.append(record)
    return measured


# --------------------------------------------------------------------------------------------
# Publication aggregates
# --------------------------------------------------------------------------------------------


def build_publication_aggregate(
    key: tuple[str, int, str],
    measured: Sequence[Mapping[str, Any]],
    *,
    source_commit: str,
    group_coordinate: Mapping[str, Any],
    archive: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate 20 instances x 7 measured attempts into one strict publication record."""

    family, intervals, policy = key
    results = [dict(item["paper1_result"]) for item in measured]
    if not results:
        raise ClaimCoreDecisionError(f"no measured attempts for {key}")
    seeds = sorted({int(item["seed"]) for item in measured})
    instances = sorted({str(item["instance"]) for item in measured})
    qualified = [item for item in results if item["identity"]["status"] == "qualified"]
    statuses = Counter(item["identity"]["status"] for item in results)
    all_qualified = len(qualified) == len(results)
    if all_qualified:
        status, failure_class, solver_status = "qualified", "none", "converged"
    else:
        failing = [item for item in results if item["identity"]["status"] != "qualified"]
        status = Counter(item["identity"]["status"] for item in failing).most_common(1)[0][0]
        same_status = [item for item in failing if item["identity"]["status"] == status]
        failure_class = Counter(
            item["identity"]["failure_class"] for item in same_status
        ).most_common(1)[0][0]
        solver_status = Counter(
            item["quality"]["solver_status"] for item in same_status
        ).most_common(1)[0][0]
    template = results[0]
    representative_pool = qualified or results
    ordered = sorted(
        representative_pool, key=lambda item: float(item["timing"]["scvx_total_seconds"])
    )
    representative = ordered[len(ordered) // 2]

    def median_of(section: str, name: str, pool: Sequence[Mapping[str, Any]]) -> float | None:
        values = [
            float(item[section][name]) for item in pool if item[section].get(name) is not None
        ]
        return _median(values)

    timing_pool = qualified or results
    timing: dict[str, Any] = {
        "topology_seconds": median_of("timing", "topology_seconds", timing_pool) or 0.0
    }
    for name in (*CQP_COMPONENTS, *EXTRA_COMPONENTS):
        timing[name] = median_of("timing", name, timing_pool) or 0.0
    cqp_total = math.fsum(timing[name] for name in CQP_COMPONENTS)
    scvx_total = cqp_total + math.fsum(timing[name] for name in EXTRA_COMPONENTS)
    timing.update(
        {
            "cqp_total_seconds": cqp_total,
            "scvx_total_seconds": scvx_total,
            "accepted_trajectory_seconds": scvx_total,
            "cuda_startup_seconds": median_of("timing", "cuda_startup_seconds", timing_pool) or 0.0,
            "cuda_startup_included": False,
            "accepted_trajectory_count": 1,
            "accepted_timing_boundary": template["timing"]["accepted_timing_boundary"],
            "cqp_total_identity": list(template["timing"]["cqp_total_identity"]),
            "scvx_total_identity": list(template["timing"]["scvx_total_identity"]),
        }
    )
    quality_pool = qualified or results
    quality: dict[str, Any] = {"qualified": all_qualified}
    for name in (
        "objective",
        "reference_objective",
        "objective_gap",
        *RESIDUAL_FIELDS,
        "native_primal_residual",
        "native_dual_residual",
        "virtual_control_residual",
        "nonanticipativity_residual",
        "risk_epigraph_residual",
        "ct_error_estimate",
        "requested_tolerance",
        "achieved_residual",
        "continuous_time_violation",
    ):
        quality[name] = median_of("quality", name, quality_pool)
    quality.update(
        {
            "solver_status": solver_status,
            "convergence_criteria_met": all_qualified,
            "objective_equivalent": all(
                bool(item["quality"]["objective_equivalent"]) for item in results
            )
            and all_qualified,
            "matched_quality_state": "matched" if all_qualified else "unqualified",
            "independent_replay": all(
                bool(item["quality"]["independent_replay"]) for item in results
            ),
            "path_inventory_complete": all(
                bool(item["quality"]["path_inventory_complete"]) for item in results
            ),
            "uses_solver_cached_residuals": any(
                bool(item["quality"]["uses_solver_cached_residuals"]) for item in results
            ),
            "path_inventory": {
                name: {
                    "violation": _median(
                        [
                            float(item["quality"]["path_inventory"][name]["violation"])
                            for item in quality_pool
                            if item["quality"]["path_inventory"][name]["violation"] is not None
                        ]
                    ),
                    "independent": all(
                        bool(item["quality"]["path_inventory"][name]["independent"])
                        for item in results
                    ),
                }
                for name in template["quality"]["path_inventory"]
            },
        }
    )
    work = {
        name: _median_int(
            [int(item["work"][name]) for item in results if item["work"][name] is not None]
        )
        for name in (
            "outer_iterations",
            "inner_iterations",
            "matvecs",
            "cone_projections",
            "factorisations",
            "accepted_steps",
            "rejected_steps",
            "resolved_steps",
        )
    }
    work["polish_used"] = bool(template["work"]["polish_used"])
    energy_values = [
        float(item["resources"]["energy_joules"])
        for item in results
        if item["resources"].get("energy_joules") is not None
    ]
    gaps = [
        float(item["resources"]["energy_maximum_gap_seconds"])
        for item in results
        if item["resources"].get("energy_maximum_gap_seconds") is not None
    ]
    energy_valid = all(bool(item["resources"]["energy_valid"]) for item in results)
    resources = {
        name: _median_int(
            [
                int(item["resources"][name])
                for item in results
                if item["resources"][name] is not None
            ]
        )
        for name in (
            "peak_device_bytes",
            "reserved_device_bytes",
            "h2d_bytes",
            "d2h_bytes",
            "collective_bytes",
            "collective_count",
            "topology_allocation_count_after_create",
        )
    }
    resources.update(
        {
            "energy_joules": _median(energy_values),
            "energy_scope": "GPU-only",
            "energy_sampling_interval_milliseconds": 50,
            "energy_maximum_gap_seconds": max(gaps) if gaps else None,
            "energy_sampling_gaps": not energy_valid,
            "energy_valid": energy_valid,
            "shared_or_display_gpu": True,
        }
    )
    totals = [float(item["timing"]["scvx_total_seconds"]) for item in qualified]
    if totals:
        q1, q3 = _quartiles(totals)
        median = statistics.median(totals)
        coefficient = (
            statistics.pstdev(totals) / statistics.fmean(totals) if len(totals) > 1 else 0.0
        )
        aggregation: dict[str, Any] = {
            "median": median,
            "q1": q1,
            "q3": q3,
            "minimum": min(totals),
            "maximum": max(totals),
            "coefficient_of_variation": coefficient,
        }
    else:
        aggregation = {
            "median": None,
            "q1": None,
            "q3": None,
            "minimum": None,
            "maximum": None,
            "coefficient_of_variation": None,
        }
    aggregation.update(
        {
            "warmup_repeats": 2,
            "measured_repeats": 7,
            "statistic": "median_iqr",
            "censored_count": len(results) - len(qualified),
            "instance_count": len(instances),
            "evaluation_seed_count": len(seeds),
            "paired_bootstrap_samples": 10_000,
        }
    )
    instance_set = content_id("g4-instance-set-v1", instances)
    run_id = content_id(
        "g4-claim-core-aggregate-v1",
        {"family": family, "intervals": intervals, "policy": policy, "instances": instances},
    )
    identity = {
        **{
            key_: template["identity"][key_]
            for key_ in (
                "repository_commit",
                "family",
                "solver",
                "policy",
                "hardware_id",
                "precision",
                "warm_start",
                "cold_start",
                "gate",
                "campaign",
                "quality_tier",
                "conditioning",
                "scaling_mode",
                "warm_mode",
                "solver_order",
            )
        },
        "run_id": run_id,
        "instance_id": instance_set,
        "status": status,
        "record_scope": "publication_aggregate",
        "seed": seeds[0],
        "repeat_kind": "measured",
        "repeat": 0,
        "failure_class": failure_class,
        "failure_reason": None
        if all_qualified
        else "; ".join(
            f"{name}={count}" for name, count in sorted(statuses.items()) if name != "qualified"
        ),
    }
    if identity["repository_commit"] != source_commit:
        raise ClaimCoreDecisionError("measured records were produced by a different source commit")
    # Mirrors the executor's convention for measured attempts: content-addressed identifiers
    # that the strict contract accepts. The archive itself remains local-only; see notes.
    artifacts = {
        name: {
            "location": f"{archive['location']}#{name}",
            "sha256": archive["sha256"],
            "immutable_uri": f"{archive['uri']}#{name}",
            "internal_index_sha256": archive["index_sha256"],
            "portable": True,
        }
        for name in ("manifest", "raw", "stdout", "stderr")
    }
    record = {
        "schema_version": "1.0.0",
        "identity": identity,
        "dimensions": {
            **{
                key_: value
                for key_, value in template["dimensions"].items()
                if key_ not in {"topology_bytes", "numeric_bytes"}
            },
            **{
                key_: group_coordinate[key_]
                for key_ in (
                    "dispersion_class",
                    "attitude_class",
                    "rate_class",
                    "trust_class",
                    "transfer_class",
                )
                if key_ in group_coordinate
            },
        },
        "quality": quality,
        "timing": timing,
        "work": work,
        "resources": resources,
        "aggregation": aggregation,
        "artifacts": artifacts,
        "g4": {
            **representative["g4"],
        },
        "notes": [
            "H5/H6 claim-core publication aggregate: 20 physical instances x 7 measured attempts "
            "after 2 same-session warm-ups; warm-ups never enter the statistic.",
            f"evaluation seeds aggregated: {seeds}; identity.seed/repeat carry the first seed and "
            "repeat 0 only because the schema requires scalar values.",
            "timing components are medians over qualified attempts and the totals are their "
            "identity sums; aggregation quantiles are over per-attempt scvx_total_seconds.",
            "g4.outer_iterations is the trace of the median-time qualified attempt "
            f"({representative['identity']['run_id']}).",
            f"measured attempt statuses: {dict(sorted(statuses.items()))}.",
            "artifacts are content-addressed identifiers of the local-only sealed claim-core "
            f"checkpoint ({archive['note']}); no network-resolvable immutable URI exists.",
            "This aggregate resolves H5/H6 only and may not populate any F01-F12/T01-T08 product.",
        ],
    }
    validate_paper1_result_schema(
        record, REPOSITORY / "experiments/schema/paper1_result.schema.json"
    )
    return record


# --------------------------------------------------------------------------------------------
# Decision rows
# --------------------------------------------------------------------------------------------


def _by_pair(measured: Iterable[Mapping[str, Any]]) -> dict[tuple[int, int], Mapping[str, Any]]:
    table: dict[tuple[int, int], Mapping[str, Any]] = {}
    for item in measured:
        key = (int(item["seed"]), int(item["repeat"]))
        if key in table:
            raise ClaimCoreDecisionError(f"duplicate measured attempt for seed/repeat {key}")
        table[key] = item
    return table


def _dominant_failure(records: Iterable[Mapping[str, Any]]) -> str | None:
    counts = Counter(item["disposition"] for item in records if item["disposition"] != "qualified")
    if not counts:
        return None
    for preferred in ("timeout", "oom"):
        if counts[preferred] and counts[preferred] == sum(counts.values()):
            return preferred
    return counts.most_common(1)[0][0]


def _final_forcing(paper1: Mapping[str, Any]) -> bool:
    iterations = paper1.get("g4", {}).get("outer_iterations", [])
    return bool(iterations) and bool(iterations[-1].get("forcing_satisfied"))


def _row_base(coordinate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "family": coordinate["family"],
        "scale": coordinate["intervals"],
        "conditioning": coordinate["conditioning"],
        "warm_mode": coordinate["warm_mode"],
        "scaling_mode": coordinate["scaling_mode"],
        "quality_tier": coordinate["quality_tier"],
        "dispersion_class": coordinate.get("dispersion_class"),
        "attitude_class": coordinate.get("attitude_class"),
        "rate_class": coordinate.get("rate_class"),
        "transfer_class": coordinate.get("transfer_class"),
        "trust_class": coordinate.get("trust_class"),
    }


def build_h5_rows(
    measured_by_key: Mapping[tuple[str, int, str], Sequence[Mapping[str, Any]]],
    coordinates_by_key: Mapping[tuple[str, int, str], Mapping[str, Any]],
    core: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family_definition in core["families"]:
        family = family_definition["family"]
        if set(family_definition["claim_roles"]["H5"]) != {"fixed-tight", "adaptive"}:
            continue
        for intervals in family_definition["scales"]:
            baseline = _by_pair(measured_by_key.get((family, intervals, "fixed-tight"), ()))
            candidate = _by_pair(measured_by_key.get((family, intervals, "adaptive"), ()))
            coordinate = coordinates_by_key[(family, intervals, "fixed-tight")]
            fixed_tight: list[float] = []
            adaptive: list[float] = []
            objective_equivalent = True
            forcing_satisfied = True
            for pair_key in sorted(set(baseline) & set(candidate)):
                left, right = baseline[pair_key], candidate[pair_key]
                if left["disposition"] != "qualified" or right["disposition"] != "qualified":
                    continue
                fixed_tight.append(float(left["paper1_result"]["timing"]["scvx_total_seconds"]))
                adaptive.append(float(right["paper1_result"]["timing"]["scvx_total_seconds"]))
                objective_equivalent &= bool(
                    left["paper1_result"]["quality"]["objective_equivalent"]
                ) and bool(right["paper1_result"]["quality"]["objective_equivalent"])
                forcing_satisfied &= _final_forcing(right["paper1_result"])
            failures = [
                item
                for item in (*baseline.values(), *candidate.values())
                if item["disposition"] != "qualified"
            ]
            row = {
                **_row_base(coordinate),
                "fixed_tight_seconds": fixed_tight,
                "adaptive_seconds": adaptive,
                "baseline_failures": sum(
                    item["disposition"] != "qualified" for item in baseline.values()
                ),
                "candidate_failures": sum(
                    item["disposition"] != "qualified" for item in candidate.values()
                ),
                "attempts": max(len(baseline), len(candidate), 1),
                "pair_count": len(fixed_tight),
                "matched_quality": bool(fixed_tight),
                "objective_equivalent": bool(fixed_tight) and objective_equivalent,
                "forcing_satisfied": bool(fixed_tight) and forcing_satisfied,
                "fixed_tight_dispositions": dict(
                    Counter(item["disposition"] for item in baseline.values())
                ),
                "adaptive_dispositions": dict(
                    Counter(item["disposition"] for item in candidate.values())
                ),
                "pairing_rule": PAIRING_RULE,
            }
            if not fixed_tight:
                row["disposition"] = _dominant_failure(failures) or "unrun"
            rows.append(row)
    return rows


def build_h6_rows(
    measured_by_key: Mapping[tuple[str, int, str], Sequence[Mapping[str, Any]]],
    coordinates_by_key: Mapping[tuple[str, int, str], Mapping[str, Any]],
    core: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family_definition in core["families"]:
        family = family_definition["family"]
        if set(family_definition["claim_roles"]["H6"]) != {"pure-gpu-ipm", "hybrid-pdhcg-ipm"}:
            continue
        for intervals in family_definition["scales"]:
            ipm = _by_pair(measured_by_key.get((family, intervals, "pure-gpu-ipm"), ()))
            hybrid = _by_pair(measured_by_key.get((family, intervals, "hybrid-pdhcg-ipm"), ()))
            unpolished = _by_pair(measured_by_key.get((family, intervals, "fixed-tight"), ()))
            coordinate = coordinates_by_key[(family, intervals, "pure-gpu-ipm")]
            ipm_seconds: list[float] = []
            hybrid_seconds: list[float] = []
            hybrid_residuals: list[float] = []
            ipm_residuals: list[float] = []
            for pair_key in sorted(set(ipm) & set(hybrid)):
                left, right = ipm[pair_key], hybrid[pair_key]
                if left["disposition"] != "qualified" or right["disposition"] != "qualified":
                    continue
                ipm_seconds.append(float(left["paper1_result"]["timing"]["scvx_total_seconds"]))
                hybrid_seconds.append(float(right["paper1_result"]["timing"]["scvx_total_seconds"]))
                ipm_residuals.append(final_residual(left["paper1_result"]["quality"]))
                hybrid_residuals.append(final_residual(right["paper1_result"]["quality"]))
            unpolished_qualified = [
                final_residual(item["paper1_result"]["quality"])
                for item in unpolished.values()
                if item["disposition"] == "qualified"
            ]
            hybrid_records = list(hybrid.values())
            transfer_reliable = (
                bool(hybrid_records)
                and not any(
                    item["disposition"] == "hybrid_handoff_ineligible" for item in hybrid_records
                )
                and all(
                    item.get("session", {}).get("dual_disposition")
                    in {"transferred", "discarded_unsupported", "not-produced"}
                    for item in hybrid_records
                )
            )
            failures = [
                item
                for item in (*ipm.values(), *hybrid.values())
                if item["disposition"] != "qualified"
            ]
            row = {
                **_row_base(coordinate),
                "ipm_seconds": ipm_seconds,
                "hybrid_seconds": hybrid_seconds,
                "baseline_failures": sum(
                    item["disposition"] != "qualified" for item in ipm.values()
                ),
                "candidate_failures": sum(
                    item["disposition"] != "qualified" for item in hybrid.values()
                ),
                "attempts": max(len(ipm), len(hybrid), 1),
                "pair_count": len(ipm_seconds),
                "hybrid_residual": _median(hybrid_residuals),
                "ipm_residual": _median(ipm_residuals),
                "unpolished_residual": _median(unpolished_qualified),
                "unpolished_failed_tier": bool(unpolished) and not unpolished_qualified,
                "matched_quality": bool(ipm_seconds),
                "conversion_and_polish_included": bool(ipm_seconds)
                and all(
                    {"hybrid_conversion_seconds", "hybrid_setup_seconds", "polish_seconds"}
                    <= set(item["paper1_result"]["timing"]["scvx_total_identity"])
                    for item in hybrid_records
                    if item["disposition"] == "qualified"
                ),
                "transfer_reliable": transfer_reliable,
                "ipm_dispositions": dict(Counter(item["disposition"] for item in ipm.values())),
                "hybrid_dispositions": dict(
                    Counter(item["disposition"] for item in hybrid.values())
                ),
                "pairing_rule": PAIRING_RULE,
            }
            if not ipm_seconds:
                row["disposition"] = _dominant_failure(failures) or "unrun"
            rows.append(row)
    return rows


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------


def write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capabilities", type=Path, required=True)
    parser.add_argument(
        "--archive-uri",
        default=None,
        help="identifier recorded for the evidence archive; defaults to a content address",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write a clearly labelled preview from a partial ledger; never a decision record",
    )
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    campaign = arguments.campaign.resolve()
    output = arguments.output.resolve()

    policy_lock = (repository / "benchmarks/g4_policy.sha256").read_text().split()[0]
    policy = load_policy(repository / "benchmarks/g4_policy.json", expected_sha256=policy_lock)
    core_lock = (repository / "benchmarks/g4_h5_h6_claim_core.sha256").read_text().split()[0]
    core = load_claim_core(
        repository / "benchmarks/g4_h5_h6_claim_core.json", expected_sha256=core_lock
    )
    groups = tuple(iter_claim_core_groups(core.values))
    capability = json.loads(arguments.capabilities.read_text(encoding="utf-8"))

    metadata, rows = load_campaign(campaign)
    if metadata.get("schedule_kind") != "claim_core_execution_groups":
        raise ClaimCoreDecisionError("campaign is not a claim-core grouped checkpoint")
    if metadata.get("schedule_sha256") != core.sha256:
        raise ClaimCoreDecisionError("campaign schedule hash differs from the locked claim core")
    if metadata.get("policy_sha256") != policy.sha256:
        raise ClaimCoreDecisionError("campaign policy hash differs from the locked policy")
    if int(metadata["total_rows"]) != len(groups):
        raise ClaimCoreDecisionError("campaign cardinality differs from the claim core")
    source_commit = metadata["source_commit"]
    if capability.get("source_commit") != source_commit:
        raise ClaimCoreDecisionError("capability source commit differs from the campaign")

    completed = [row for row in rows if row["state"] == "completed"]
    if len(completed) != len(groups) and not arguments.allow_incomplete:
        raise ClaimCoreDecisionError(
            f"claim core incomplete: {len(completed)}/{len(groups)} groups completed; "
            "no decision record is written from a partial ledger"
        )
    raw_schema = json.loads(
        (repository / "experiments/schema/g4_raw_attempt.schema.json").read_text(encoding="utf-8")
    )
    raw_validator = Draft202012Validator(raw_schema)
    paper1_schema = repository / "experiments/schema/paper1_result.schema.json"

    measured_by_key: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    coordinates_by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    group_dispositions: Counter[str] = Counter()
    measured_dispositions: Counter[str] = Counter()
    energy_valid: Counter[bool] = Counter()
    energy_gaps: list[float] = []
    contamination_events = 0
    validated_groups = 0
    result_files: list[Path] = []
    for group in groups:
        coordinate = group.coordinate
        coordinates_by_key[
            (coordinate["family"], coordinate["intervals"], coordinate["policy"])
        ] = coordinate
    for row in completed:
        group = groups[int(row["ordinal"])]
        if row["coordinate_id"] != group.group_id:
            raise ClaimCoreDecisionError("checkpoint group identity drift")
        run_directory = Path(row["run_directory"])
        result = json.loads((run_directory / "result.json").read_text(encoding="utf-8"))
        command = result.get("command", [])
        if len(command) != 6 or command[5] != capability["capability_sha256"]:
            raise ClaimCoreDecisionError("group executed under a different capability")
        result_files.append(run_directory / "result.json")
        measured = validate_group_evidence(group, result, raw_validator, paper1_schema)
        validated_groups += 1
        group_dispositions[row["disposition"]] += 1
        for record in measured:
            measured_dispositions[record["disposition"]] += 1
            energy = record.get("energy") or {}
            energy_valid[bool(energy.get("gap_valid"))] += 1
            if energy.get("maximum_gap_seconds") is not None:
                energy_gaps.append(float(energy["maximum_gap_seconds"]))
        coordinate = group.coordinate
        measured_by_key[
            (coordinate["family"], coordinate["intervals"], coordinate["policy"])
        ].extend(measured)
    contamination_events = sum(1 for row in rows if row["disposition"] == "contaminated")
    quarantined_history = Counter()
    database = sqlite3.connect(f"file:{campaign / 'checkpoint.sqlite3'}?mode=ro", uri=True)
    for (disposition,) in database.execute(
        "SELECT disposition FROM attempts WHERE state IN ('quarantined', 'interrupted')"
    ):
        quarantined_history[str(disposition)] += 1
    database.close()

    # Content address of the completed evidence: the checkpoint database plus an index over
    # every completed group result, both hashed at decision time.
    results_index = [
        {"path": str(path.relative_to(campaign)), "sha256": sha256_path(path)}
        for path in sorted(result_files)
    ]
    archive = {
        "location": str(campaign),
        "sha256": sha256_path(campaign / "checkpoint.sqlite3"),
        "uri": arguments.archive_uri or f"g4-claim-core://{core.sha256}/{source_commit}",
        "index_sha256": hashlib.sha256(canonical_bytes(results_index)).hexdigest(),
        "note": "local-only build-integration-report checkpoint; not clean-clone portable",
    }
    publication: list[dict[str, Any]] = []
    publication_errors: list[str] = []
    for key in sorted(measured_by_key):
        measured = measured_by_key[key]
        if len(measured) != 140 and not arguments.allow_incomplete:
            raise ClaimCoreDecisionError(
                f"{key} has {len(measured)} measured attempts, expected 140"
            )
        try:
            publication.append(
                build_publication_aggregate(
                    key,
                    measured,
                    source_commit=source_commit,
                    group_coordinate=coordinates_by_key[key],
                    archive=archive,
                )
            )
        except (Paper1ResultError, ClaimCoreDecisionError) as error:
            publication_errors.append(f"{key}: {error}")
            if not arguments.allow_incomplete:
                raise
    h5_rows = build_h5_rows(measured_by_key, coordinates_by_key, core.values)
    h6_rows = build_h6_rows(measured_by_key, coordinates_by_key, core.values)
    decision = g4_decision(h5_rows, h6_rows, policy.values)

    output.mkdir(parents=True, exist_ok=True)
    complete = len(completed) == len(groups) and not publication_errors
    prefix = "" if complete else "PREVIEW-"
    publication_index = []
    for record in publication:
        identity = record["identity"]
        name = (
            f"{prefix}publication-{identity['family']}-N{record['dimensions']['intervals']}-"
            f"{identity['policy']}.json"
        )
        digest = write_json(output / "publication" / name, record)
        publication_index.append(
            {"file": f"publication/{name}", "sha256": digest, "run_id": identity["run_id"]}
        )
    (output / f"{prefix}h5_coordinates.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in h5_rows), encoding="utf-8"
    )
    (output / f"{prefix}h6_coordinates.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in h6_rows), encoding="utf-8"
    )
    summary = {
        "schema_version": "g4-claim-core-decision-1.0.0",
        "preview_only": not complete,
        "campaign": str(campaign),
        "source_commit": source_commit,
        "policy_sha256": policy.sha256,
        "claim_core_sha256": core.sha256,
        "capability_sha256": capability["capability_sha256"],
        "executable_sha256": capability["executable_sha256"],
        "groups_total": len(groups),
        "groups_completed": len(completed),
        "groups_validated": validated_groups,
        "group_dispositions": dict(group_dispositions),
        "measured_attempt_dispositions": dict(sorted(measured_dispositions.items())),
        "measured_attempts_validated": sum(measured_dispositions.values()),
        "energy": {
            "attempts_gap_valid": energy_valid.get(True, 0),
            "attempts_gap_invalid": energy_valid.get(False, 0),
            "maximum_gap_seconds_median": _median(energy_gaps),
            "maximum_gap_seconds_max": max(energy_gaps) if energy_gaps else None,
        },
        "contaminated_group_attempts": contamination_events,
        "non_terminal_attempt_history": dict(quarantined_history),
        "publication_records": publication_index,
        "publication_errors": publication_errors,
        "bootstrap": {
            "seed": policy.values["statistics"]["bootstrap_seed"],
            "samples": policy.values["statistics"]["bootstrap_samples"],
            "confidence": policy.values["statistics"]["confidence"],
            "sustained_coordinates": policy.values["statistics"]["sustained_coordinates"],
            "seed_rule": "policy bootstrap_seed plus the coordinate scale",
        },
        "thresholds": policy.values["decision_thresholds"],
        "pairing_rule": PAIRING_RULE,
        "scope": core.values["scope_statement"],
        "decision": decision,
    }
    digest = write_json(output / f"{prefix}decision.json", summary)
    print(
        json.dumps(
            {
                "decision_file": f"{prefix}decision.json",
                "decision_sha256": digest,
                "preview_only": not complete,
                "G4": decision["decision"],
                "H5": decision["H5"]["decision"],
                "H6": decision["H6"]["decision"],
                "groups_completed": len(completed),
                "groups_total": len(groups),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ClaimCoreDecisionError, G4ContractError, Paper1ResultError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
