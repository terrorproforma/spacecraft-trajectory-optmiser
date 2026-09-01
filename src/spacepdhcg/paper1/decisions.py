"""Preregistered H1-H6 decision records with deterministic paired bootstraps."""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from spacepdhcg.campaign_scope import (
    CampaignScopeError,
    scope_definition,
)

from .evidence import ArchivedRun, write_canonical_json

DECISION_SCHEMA_VERSION: Final = "1.0.0"
SCOPED_DECISION_SCHEMA_VERSION: Final = "1.1.0"
BOOTSTRAP_SAMPLES: Final = 10_000
BOOTSTRAP_SEEDS: Final = {
    "H1": 6101,
    "H2": 6102,
    "H3": 6103,
    "H4": 6104,
    "H5": 6105,
    "H6": 6106,
}
SCIENTIFIC_OUTCOMES: Final = frozenset({"supported", "rejected", "mixed", "unresolved"})
OUTCOMES: Final = SCIENTIFIC_OUTCOMES | {"deferred-not-in-scope"}


class DecisionError(ValueError):
    """Raised when a decision record or comparison violates preregistration."""


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise DecisionError("cannot compute a quantile of no values")
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap(
    pairs: Sequence[tuple[float, float]],
    metric: Callable[[Sequence[tuple[float, float]]], float],
    *,
    seed: int,
) -> tuple[float, float, float]:
    """Return point estimate and deterministic 95% paired bootstrap interval."""

    if not pairs:
        raise DecisionError("paired bootstrap requires at least one pair")
    point = metric(pairs)
    generator = random.Random(seed)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        resample = [pairs[generator.randrange(len(pairs))] for _ in pairs]
        samples.append(metric(resample))
    return point, _quantile(samples, 0.025), _quantile(samples, 0.975)


def _ratio(pairs: Sequence[tuple[float, float]]) -> float:
    ratios = [baseline / candidate for baseline, candidate in pairs if candidate > 0]
    if len(ratios) != len(pairs):
        raise DecisionError("ratio comparison requires positive candidate values")
    return statistics.median(ratios)


def _relative_reduction(pairs: Sequence[tuple[float, float]]) -> float:
    reductions = [
        (baseline - candidate) / baseline for baseline, candidate in pairs if baseline > 0
    ]
    if len(reductions) != len(pairs):
        raise DecisionError("reduction comparison requires positive baseline values")
    return statistics.median(reductions)


def _difference(pairs: Sequence[tuple[float, float]]) -> float:
    return statistics.median(baseline - candidate for baseline, candidate in pairs)


def _identity(run: ArchivedRun) -> Mapping[str, Any]:
    return run.result["identity"]


def _dimensions(run: ArchivedRun) -> Mapping[str, Any]:
    return run.result["dimensions"]


def _timing(run: ArchivedRun) -> Mapping[str, Any]:
    return run.result["timing"]


def _resources(run: ArchivedRun) -> Mapping[str, Any]:
    return run.result["resources"]


def _quality(run: ArchivedRun) -> Mapping[str, Any]:
    return run.result["quality"]


def _coord(run: ArchivedRun) -> tuple[Any, ...]:
    identity, dimensions = _identity(run), _dimensions(run)
    return (
        identity["family"],
        identity["instance_id"],
        dimensions["intervals"],
        dimensions["scenarios"],
        dimensions["gpus"],
        identity.get("quality_tier"),
        identity.get("warm_mode"),
    )


def _scale(run: ArchivedRun) -> int:
    dimensions = _dimensions(run)
    return int(dimensions["intervals"]) * int(dimensions["scenarios"])


def _qualified(run: ArchivedRun) -> bool:
    return run.status == "qualified" and bool(_quality(run)["qualified"])


def _pair_by(
    runs: Sequence[ArchivedRun],
    baseline: Callable[[ArchivedRun], bool],
    candidate: Callable[[ArchivedRun], bool],
    value: Callable[[ArchivedRun], float | None],
) -> tuple[list[tuple[float, float]], list[dict[str, Any]], list[str]]:
    groups: dict[tuple[Any, ...], dict[str, ArchivedRun]] = defaultdict(dict)
    censored: list[str] = []
    for run in runs:
        side = "baseline" if baseline(run) else "candidate" if candidate(run) else None
        if side is None:
            continue
        if not _qualified(run):
            censored.append(run.run_id)
            continue
        coordinate = _coord(run)
        previous = groups[coordinate].get(side)
        current_value = value(run)
        previous_value = value(previous) if previous is not None else None
        if (
            previous is None
            or previous_value is None
            or (current_value is not None and current_value < previous_value)
        ):
            groups[coordinate][side] = run
    pairs: list[tuple[float, float]] = []
    coordinates: list[dict[str, Any]] = []
    for coordinate in sorted(groups, key=str):
        group = groups[coordinate]
        if set(group) != {"baseline", "candidate"}:
            censored.extend(run.run_id for run in group.values())
            continue
        baseline_value, candidate_value = value(group["baseline"]), value(group["candidate"])
        if baseline_value is None or candidate_value is None:
            censored.extend((group["baseline"].run_id, group["candidate"].run_id))
            continue
        pairs.append((float(baseline_value), float(candidate_value)))
        coordinates.append(
            {
                "coordinate": list(coordinate),
                "scale": _scale(group["candidate"]),
                "baseline_run_id": group["baseline"].run_id,
                "candidate_run_id": group["candidate"].run_id,
                "baseline": baseline_value,
                "candidate": candidate_value,
            }
        )
    return pairs, coordinates, sorted(set(censored))


def _record(
    hypothesis: str,
    *,
    outcome: str,
    threshold: Mapping[str, Any],
    metric: str,
    point: float | None,
    low: float | None,
    high: float | None,
    coordinates: Sequence[Mapping[str, Any]],
    censored: Sequence[str],
    notes: Sequence[str],
) -> dict[str, Any]:
    if outcome not in SCIENTIFIC_OUTCOMES:
        raise DecisionError(f"invalid outcome {outcome}")
    run_ids = sorted(
        {
            str(coordinate[key])
            for coordinate in coordinates
            for key in ("baseline_run_id", "candidate_run_id")
            if key in coordinate
        }
        | set(censored)
    )
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "hypothesis": hypothesis,
        "outcome": outcome,
        "input_run_ids": run_ids,
        "comparison_coordinates": list(coordinates),
        "practical_threshold": dict(threshold),
        "method": "95% paired percentile bootstrap over matched instance/repeat medians",
        "bootstrap_seed": BOOTSTRAP_SEEDS[hypothesis],
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "metric": metric,
        "point_estimate": point,
        "confidence_interval_95": [low, high],
        "sustained_scale_rule": (
            "three consecutive increasing coordinates, or every remaining feasible coordinate "
            "when fewer than three remain"
        ),
        "censored_run_ids": sorted(set(censored)),
        "notes": list(notes),
    }


def _sustained(coordinates: Sequence[Mapping[str, Any]], wins: Sequence[bool]) -> bool:
    ordered = sorted(zip(coordinates, wins, strict=True), key=lambda item: item[0]["scale"])
    for index, (_, win) in enumerate(ordered):
        if not win:
            continue
        remaining = ordered[index:]
        required = min(3, len(remaining))
        if required and all(item[1] for item in remaining[:required]):
            return True
    return False


def decide_h1(runs: Sequence[ArchivedRun]) -> dict[str, Any]:
    selected = [run for run in runs if _identity(run)["solver"] == "spacepdhcg-persistent"]
    coordinates, values, censored = [], [], []
    hard_failure = False
    for run in selected:
        if not _qualified(run):
            censored.append(run.run_id)
            continue
        timing, resources = _timing(run), _resources(run)
        total = timing["cqp_total_seconds"]
        numerator = sum(
            float(timing[name] or 0.0)
            for name in ("topology_seconds", "workspace_create_seconds", "h2d_seconds")
        )
        if not total:
            censored.append(run.run_id)
            continue
        overhead = numerator / float(total)
        allocation_count = resources["topology_allocation_count_after_create"]
        hard_failure |= allocation_count != 0
        values.append(overhead)
        coordinates.append(
            {
                "coordinate": list(_coord(run)),
                "scale": _scale(run),
                "candidate_run_id": run.run_id,
                "overhead_fraction": overhead,
                "post_create_topology_allocations": allocation_count,
            }
        )
    if not values:
        return _record(
            "H1",
            outcome="unresolved",
            threshold={"median_max": 0.05, "upper_ci_max": 0.08},
            metric="omega_persist",
            point=None,
            low=None,
            high=None,
            coordinates=coordinates,
            censored=censored,
            notes=["No qualified persistent coordinates."],
        )
    pairs = [(1.0, value) for value in values]
    point, low, high = paired_bootstrap(
        pairs,
        lambda sample: statistics.median(value for _, value in sample),
        seed=BOOTSTRAP_SEEDS["H1"],
    )
    passing = [value <= 0.05 for value in values]
    if hard_failure:
        outcome = "rejected"
    elif point <= 0.05 and high <= 0.08 and _sustained(coordinates, passing):
        outcome = "supported"
    elif all(value > 0.10 for value in values):
        outcome = "rejected"
    elif any(passing):
        outcome = "mixed"
    else:
        outcome = "unresolved"
    return _record(
        "H1",
        outcome=outcome,
        threshold={
            "post_create_topology_allocations": 0,
            "median_max": 0.05,
            "upper_ci_max": 0.08,
            "reject_all_saturated_min": 0.10,
        },
        metric="omega_persist",
        point=point,
        low=low,
        high=high,
        coordinates=coordinates,
        censored=censored,
        notes=[],
    )


def decide_h2(runs: Sequence[ArchivedRun]) -> dict[str, Any]:
    ipms = {"qoco-gpu", "cuclarabel"}
    pairs, coordinates, censored = _pair_by(
        runs,
        lambda run: _identity(run)["solver"] in ipms,
        lambda run: _identity(run)["solver"] == "spacepdhcg-persistent",
        lambda run: _timing(run)["scvx_total_seconds"],
    )
    return _speed_decision(
        "H2",
        pairs,
        coordinates,
        censored,
        threshold=1.20,
        reject_upper_below=1.0,
        metric_name="best qualified GPU IPM / persistent PDHCG",
    )


def _speed_decision(
    hypothesis: str,
    pairs: Sequence[tuple[float, float]],
    coordinates: Sequence[Mapping[str, Any]],
    censored: Sequence[str],
    *,
    threshold: float,
    reject_upper_below: float,
    metric_name: str,
) -> dict[str, Any]:
    if not pairs:
        return _record(
            hypothesis,
            outcome="unresolved",
            threshold={"speedup_min": threshold},
            metric=metric_name,
            point=None,
            low=None,
            high=None,
            coordinates=coordinates,
            censored=censored,
            notes=["No matched qualified comparison."],
        )
    point, low, high = paired_bootstrap(pairs, _ratio, seed=BOOTSTRAP_SEEDS[hypothesis])
    wins = [baseline / candidate >= threshold for baseline, candidate in pairs]
    if point >= threshold and low > 1.0 and _sustained(coordinates, wins):
        outcome = "supported"
    elif high < reject_upper_below:
        outcome = "rejected"
    elif any(wins):
        outcome = "mixed"
    else:
        outcome = "unresolved"
    return _record(
        hypothesis,
        outcome=outcome,
        threshold={"speedup_min": threshold, "lower_ci_strictly_above": 1.0},
        metric=metric_name,
        point=point,
        low=low,
        high=high,
        coordinates=coordinates,
        censored=censored,
        notes=[],
    )


def decide_h3(runs: Sequence[ArchivedRun]) -> dict[str, Any]:
    ipms = {"qoco-gpu", "cuclarabel"}
    declared_ipms = {_identity(run)["solver"] for run in runs if _identity(run)["solver"] in ipms}
    qualified_pdhcg = [
        run
        for run in runs
        if _identity(run)["solver"] == "spacepdhcg-persistent" and _qualified(run)
    ]
    ipm_statuses: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    for run in runs:
        solver = _identity(run)["solver"]
        if solver in declared_ipms:
            ipm_statuses[(_identity(run)["family"], _scale(run))][solver] = run.status
    ipm_oom_scales = {
        coordinate
        for coordinate, statuses in ipm_statuses.items()
        if set(statuses) == declared_ipms and all(status == "oom" for status in statuses.values())
    }
    oom_support = [
        run for run in qualified_pdhcg if (_identity(run)["family"], _scale(run)) in ipm_oom_scales
    ]
    pairs, coordinates, censored = _pair_by(
        runs,
        lambda run: _identity(run)["solver"] in ipms,
        lambda run: _identity(run)["solver"] == "spacepdhcg-persistent",
        lambda run: (
            float(_resources(run)["peak_device_bytes"])
            if _resources(run)["peak_device_bytes"] is not None
            else None
        ),
    )
    point = low = high = None
    if pairs:
        inverse = [(candidate, baseline) for baseline, candidate in pairs]
        point, low, high = paired_bootstrap(inverse, _ratio, seed=BOOTSTRAP_SEEDS["H3"])
    if oom_support or (point is not None and point <= 0.60 and high is not None and high < 0.75):
        outcome = "supported"
    elif not oom_support and point is not None and point >= 1.0:
        outcome = "rejected"
    else:
        outcome = "unresolved"
    notes = (
        ["Qualified PDHCG coordinate retained where every represented GPU IPM is OOM."]
        if oom_support
        else []
    )
    return _record(
        "H3",
        outcome=outcome,
        threshold={"memory_ratio_max": 0.60, "upper_ci_strictly_below": 0.75},
        metric="persistent PDHCG / best qualified GPU IPM peak active bytes",
        point=point,
        low=low,
        high=high,
        coordinates=coordinates,
        censored=censored,
        notes=notes,
    )


def decide_h4(runs: Sequence[ArchivedRun]) -> dict[str, Any]:
    scenario = lambda run: _identity(run)["policy"] == "scenario-aware"  # noqa: E731
    generic = lambda run: _identity(run)["policy"] == "generic-partition"  # noqa: E731
    total_pairs, coordinates, censored = _pair_by(
        runs, generic, scenario, lambda run: _timing(run)["scvx_total_seconds"]
    )
    bytes_pairs, _, _ = _pair_by(
        runs, generic, scenario, lambda run: float(_resources(run)["collective_bytes"] or 0)
    )
    collective_pairs, _, _ = _pair_by(
        runs, generic, scenario, lambda run: _timing(run)["collective_seconds"]
    )
    if not total_pairs or not bytes_pairs or not collective_pairs:
        return _record(
            "H4",
            outcome="unresolved",
            threshold={
                "bytes_reduction": 0.25,
                "collective_time_reduction": 0.20,
                "total_time_reduction": 0.10,
                "load_imbalance_max": 1.15,
            },
            metric="scenario-aware reductions",
            point=None,
            low=None,
            high=None,
            coordinates=coordinates,
            censored=censored,
            notes=["No complete matched scenario-aware/generic comparison."],
        )
    total, low, high = paired_bootstrap(
        total_pairs, _relative_reduction, seed=BOOTSTRAP_SEEDS["H4"]
    )
    bytes_reduction = _relative_reduction(bytes_pairs)
    collective_reduction = _relative_reduction(collective_pairs)
    candidate_ids = {item["candidate_run_id"] for item in coordinates}
    imbalance = max(
        float(_resources(run).get("load_imbalance") or math.inf)
        for run in runs
        if run.run_id in candidate_ids
    )
    wins = [(baseline - candidate) / baseline >= 0.10 for baseline, candidate in total_pairs]
    if (
        bytes_reduction >= 0.25
        and collective_reduction >= 0.20
        and total >= 0.10
        and imbalance <= 1.15
        and low > 0
        and _sustained(coordinates, wins)
    ):
        outcome = "supported"
    elif total <= -0.10:
        outcome = "rejected"
    elif bytes_reduction >= 0.25 or collective_reduction >= 0.20:
        outcome = "mixed"
    else:
        outcome = "unresolved"
    return _record(
        "H4",
        outcome=outcome,
        threshold={
            "bytes_reduction": 0.25,
            "collective_time_reduction": 0.20,
            "total_time_reduction": 0.10,
            "load_imbalance_max": 1.15,
        },
        metric="median total-time relative reduction",
        point=total,
        low=low,
        high=high,
        coordinates=coordinates,
        censored=censored,
        notes=[
            f"bytes_reduction={bytes_reduction}",
            f"collective_time_reduction={collective_reduction}",
        ],
    )


def decide_h5(runs: Sequence[ArchivedRun]) -> dict[str, Any]:
    pairs, coordinates, censored = _pair_by(
        runs,
        lambda run: (
            _identity(run)["solver"] == "spacepdhcg-persistent"
            and _identity(run)["policy"] == "fixed-tight"
        ),
        lambda run: (
            _identity(run)["solver"] == "spacepdhcg-persistent"
            and _identity(run)["policy"] == "adaptive"
        ),
        lambda run: _timing(run)["scvx_total_seconds"],
    )
    if not pairs:
        return _record(
            "H5",
            outcome="unresolved",
            threshold={"time_reduction": 0.15, "failure_delta_max": 0.02, "families_min": 2},
            metric="adaptive time reduction",
            point=None,
            low=None,
            high=None,
            coordinates=coordinates,
            censored=censored,
            notes=["No matched qualified policies."],
        )
    point, low, high = paired_bootstrap(pairs, _relative_reduction, seed=BOOTSTRAP_SEEDS["H5"])
    families = {item["coordinate"][0] for item in coordinates}
    wins = [(baseline - candidate) / baseline >= 0.15 for baseline, candidate in pairs]
    relevant = [
        run
        for run in runs
        if _identity(run)["solver"] == "spacepdhcg-persistent"
        and _identity(run)["policy"] in {"fixed-tight", "adaptive"}
    ]
    totals = defaultdict(int)
    failures = defaultdict(int)
    for run in relevant:
        policy = _identity(run)["policy"]
        totals[policy] += 1
        failures[policy] += not _qualified(run)
    failure_delta = (
        failures["adaptive"] / totals["adaptive"] - failures["fixed-tight"] / totals["fixed-tight"]
        if totals["adaptive"] and totals["fixed-tight"]
        else math.inf
    )
    if (
        point >= 0.15
        and low > 0
        and len(families) >= 2
        and failure_delta <= 0.02
        and _sustained(coordinates, wins)
    ):
        outcome = "supported"
    elif point <= -0.10:
        outcome = "rejected"
    elif any(wins):
        outcome = "mixed"
    else:
        outcome = "unresolved"
    return _record(
        "H5",
        outcome=outcome,
        threshold={"time_reduction": 0.15, "failure_delta_max": 0.02, "families_min": 2},
        metric="fixed-tight to adaptive relative time reduction",
        point=point,
        low=low,
        high=high,
        coordinates=coordinates,
        censored=censored,
        notes=[f"failure_rate_delta={failure_delta}", f"represented_families={len(families)}"],
    )


def decide_h6(runs: Sequence[ArchivedRun]) -> dict[str, Any]:
    pairs, coordinates, censored = _pair_by(
        runs,
        lambda run: _identity(run)["solver"] in {"qoco-gpu", "cuclarabel"},
        lambda run: _identity(run)["solver"] == "hybrid-pdhcg-ipm",
        lambda run: _timing(run)["scvx_total_seconds"],
    )
    if not pairs:
        return _record(
            "H6",
            outcome="unresolved",
            threshold={
                "time_reduction": 0.10,
                "residual_factor_ipm": 2.0,
                "pdhcg_residual_decades": 1.0,
            },
            metric="pure IPM to hybrid time reduction",
            point=None,
            low=None,
            high=None,
            coordinates=coordinates,
            censored=censored,
            notes=["No matched qualified hybrid/IPM pair."],
        )
    point, low, high = paired_bootstrap(pairs, _relative_reduction, seed=BOOTSTRAP_SEEDS["H6"])
    by_coordinate: dict[tuple[Any, ...], dict[str, ArchivedRun]] = defaultdict(dict)
    for run in runs:
        solver = _identity(run)["solver"]
        side = (
            "hybrid"
            if solver == "hybrid-pdhcg-ipm"
            else "pdhcg"
            if solver == "spacepdhcg-persistent"
            else "ipm"
            if solver in {"qoco-gpu", "cuclarabel"}
            else None
        )
        if side is not None:
            by_coordinate[_coord(run)][side] = run

    def final_residual(run: ArchivedRun) -> float:
        quality = _quality(run)
        return max(
            float(quality[name])
            for name in (
                "canonical_primal_residual",
                "canonical_dual_residual",
                "canonical_cone_residual",
                "canonical_gap",
                "dynamics_residual",
                "path_residual",
                "terminal_residual",
            )
            if quality[name] is not None
        )

    residual_conditions: dict[tuple[Any, ...], bool] = {}
    for coordinate, group in by_coordinate.items():
        if "hybrid" not in group or "ipm" not in group or not _qualified(group["hybrid"]):
            residual_conditions[coordinate] = False
            continue
        hybrid_residual = final_residual(group["hybrid"])
        ipm_condition = _qualified(group["ipm"]) and hybrid_residual <= 2 * final_residual(
            group["ipm"]
        )
        pdhcg = group.get("pdhcg")
        pdhcg_condition = pdhcg is not None and (
            not _qualified(pdhcg) or hybrid_residual <= 0.1 * final_residual(pdhcg)
        )
        residual_conditions[coordinate] = ipm_condition and pdhcg_condition
    wins = [
        (baseline - candidate) / baseline >= 0.10
        and residual_conditions.get(tuple(coordinate["coordinate"]), False)
        for (baseline, candidate), coordinate in zip(pairs, coordinates, strict=True)
    ]
    if point >= 0.10 and low > 0 and _sustained(coordinates, wins) and all(wins):
        outcome = "supported"
    elif point <= 0:
        outcome = "rejected"
    elif any(wins):
        outcome = "mixed"
    else:
        outcome = "unresolved"
    return _record(
        "H6",
        outcome=outcome,
        threshold={
            "time_reduction": 0.10,
            "residual_factor_ipm": 2.0,
            "pdhcg_residual_decades": 1.0,
        },
        metric="pure IPM to hybrid relative time reduction",
        point=point,
        low=low,
        high=high,
        coordinates=coordinates,
        censored=censored,
        notes=["Residual and failed-polish qualification is enforced by input eligibility."],
    )


DECIDERS: Final = {
    "H1": decide_h1,
    "H2": decide_h2,
    "H3": decide_h3,
    "H4": decide_h4,
    "H5": decide_h5,
    "H6": decide_h6,
}


def validate_decision(record: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "hypothesis",
        "outcome",
        "input_run_ids",
        "comparison_coordinates",
        "practical_threshold",
        "method",
        "bootstrap_seed",
        "bootstrap_samples",
        "metric",
        "point_estimate",
        "confidence_interval_95",
        "sustained_scale_rule",
        "censored_run_ids",
        "notes",
    }
    version = record.get("schema_version")
    if version == SCOPED_DECISION_SCHEMA_VERSION:
        required.add("campaign_scope_id")
    elif version != DECISION_SCHEMA_VERSION:
        raise DecisionError("unsupported decision schema")
    missing, unknown = sorted(required - set(record)), sorted(set(record) - required)
    if missing or unknown:
        raise DecisionError(f"decision fields invalid; missing={missing}, unknown={unknown}")
    hypothesis = record["hypothesis"]
    if hypothesis not in DECIDERS or record["outcome"] not in OUTCOMES:
        raise DecisionError("decision hypothesis/outcome is invalid")
    if version == SCOPED_DECISION_SCHEMA_VERSION:
        try:
            scope = scope_definition(record["campaign_scope_id"])
        except CampaignScopeError as error:
            raise DecisionError(str(error)) from error
        deferred = hypothesis in scope["deferred_hypotheses"]
        if deferred != (record["outcome"] == "deferred-not-in-scope"):
            raise DecisionError("decision outcome crosses the declared campaign scope")
    elif record["outcome"] == "deferred-not-in-scope":
        raise DecisionError("historical decisions cannot use a scoped deferred outcome")
    if record["bootstrap_seed"] != BOOTSTRAP_SEEDS[hypothesis]:
        raise DecisionError("decision bootstrap seed differs from preregistration")
    if record["bootstrap_samples"] != BOOTSTRAP_SAMPLES:
        raise DecisionError("decision bootstrap sample count must be 10000")
    if not isinstance(record["input_run_ids"], list) or len(record["input_run_ids"]) != len(
        set(record["input_run_ids"])
    ):
        raise DecisionError("decision input run IDs must be a unique array")
    interval = record["confidence_interval_95"]
    if not isinstance(interval, list) or len(interval) != 2:
        raise DecisionError("decision confidence interval must have two entries")


def build_decisions(
    runs: Iterable[ArchivedRun],
    output_directory: str | Path,
    *,
    campaign_scope_id: str | None = None,
) -> dict[str, Any]:
    ordered = tuple(sorted(runs, key=lambda run: run.run_id))
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    scope = None if campaign_scope_id is None else scope_definition(campaign_scope_id)
    records = []
    for hypothesis, decider in DECIDERS.items():
        record = decider(ordered)
        if scope is not None:
            record["schema_version"] = SCOPED_DECISION_SCHEMA_VERSION
            record["campaign_scope_id"] = campaign_scope_id
            if hypothesis in scope["deferred_hypotheses"]:
                record.update(
                    {
                        "outcome": "deferred-not-in-scope",
                        "input_run_ids": [],
                        "comparison_coordinates": [],
                        "point_estimate": None,
                        "confidence_interval_95": [None, None],
                        "censored_run_ids": [],
                        "notes": [
                            f"{hypothesis} requires physical multi-GPU evidence and is "
                            f"deferred by {campaign_scope_id}; no scientific decision was made."
                        ],
                    }
                )
        validate_decision(record)
        write_canonical_json(output / f"{hypothesis.lower()}-decision.json", record)
        records.append(record)
    index = {
        "schema_version": (
            DECISION_SCHEMA_VERSION if campaign_scope_id is None else SCOPED_DECISION_SCHEMA_VERSION
        ),
        "decisions": [
            {
                "hypothesis": record["hypothesis"],
                "outcome": record["outcome"],
                "file": f"{record['hypothesis'].lower()}-decision.json",
                "sha256": hashlib.sha256(
                    (output / f"{record['hypothesis'].lower()}-decision.json").read_bytes()
                ).hexdigest(),
            }
            for record in records
        ],
    }
    if campaign_scope_id is not None:
        index["campaign_scope_id"] = campaign_scope_id
    write_canonical_json(output / "decision-index.json", index)
    return index
