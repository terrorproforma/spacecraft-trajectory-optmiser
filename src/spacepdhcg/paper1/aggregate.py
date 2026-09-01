"""Deterministic source-data and publication rendering for frozen F01-F12/T01-T08."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .decisions import paired_bootstrap
from .evidence import ArchivedRun, canonical_json_bytes, write_canonical_json

FIGURE_SCHEMA_VERSION: Final = "1.0.0"
TIMING_COMPONENTS: Final = (
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
)
FAILURE_MARKERS: Final = {
    "oom": "OOM",
    "timeout": "timeout",
    "numerical": "numerical failure",
    "failed": "failed quality gate",
    "unqualified": "failed quality gate",
    "unsupported": "unsupported",
    "infeasible": "infeasible",
    "unrun": "not run",
}


class AggregationError(ValueError):
    """Raised when frozen selection, units, or aggregation rules are violated."""


@dataclass(frozen=True, slots=True)
class Product:
    product_id: str
    slug: str
    kind: str
    description: str


FIGURES: Final = (
    Product("F01", "architecture_residency", "figure", "Architecture and residency"),
    Product("F02", "fixed_topology", "figure", "Fixed topology and mutable values"),
    Product("F03", "scenario_partition", "figure", "Scenario block-arrow decomposition"),
    Product("F04", "horizon_crossover", "figure", "End-to-end horizon crossover"),
    Product("F05", "memory_crossover", "figure", "Peak memory crossover"),
    Product("F06", "adaptive_accuracy", "figure", "Adaptive-accuracy ablation"),
    Product("F07", "multigpu_scaling", "figure", "Multi-GPU strong and weak scaling"),
    Product("F08", "timing_decomposition", "figure", "End-to-end timing decomposition"),
    Product("F09", "accuracy_time_pareto", "figure", "Accuracy-time Pareto surface"),
    Product("F10", "solver_regime_map", "figure", "Solver regime map"),
    Product("F11", "variational_validation", "figure", "Variational sensitivity validation"),
    Product("F12", "robust_residuals", "figure", "Robust residual anatomy"),
)
TABLES: Final = (
    Product("T01", "hardware_software", "table", "Hardware and software manifest"),
    Product("T02", "problem_dimensions", "table", "Problem dimensions"),
    Product("T03", "correctness", "table", "Correctness"),
    Product("T04", "persistence", "table", "Persistence"),
    Product("T05", "adaptive_policy", "table", "Adaptive policy"),
    Product("T06", "robust_scaling", "table", "Robust scaling"),
    Product("T07", "regime_crossover", "table", "Regime and crossover summary"),
    Product("T08", "negative_mixed", "table", "Negative and mixed results"),
)


def _identity(run: ArchivedRun) -> Mapping[str, Any]:
    return run.result["identity"]


def _dimensions(run: ArchivedRun) -> Mapping[str, Any]:
    return run.result["dimensions"]


def _scale(run: ArchivedRun) -> int:
    dimensions = _dimensions(run)
    return int(dimensions["intervals"]) * int(dimensions["scenarios"])


def _quality(run: ArchivedRun) -> Mapping[str, Any]:
    return run.result["quality"]


def _timing(run: ArchivedRun) -> Mapping[str, Any]:
    return run.result["timing"]


def _resources(run: ArchivedRun) -> Mapping[str, Any]:
    return run.result["resources"]


def _work(run: ArchivedRun) -> Mapping[str, Any]:
    return run.result["work"]


def _aggregation(run: ArchivedRun) -> Mapping[str, Any]:
    return run.result["aggregation"]


def _ordered_runs(runs: Iterable[ArchivedRun]) -> list[ArchivedRun]:
    return sorted(runs, key=lambda item: (item.coordinate, item.run_id))


def _base_source(product: Product, runs: Sequence[ArchivedRun], query: str) -> dict[str, Any]:
    identities = [_identity(run) for run in runs]
    return {
        "schema_version": FIGURE_SCHEMA_VERSION,
        "product_id": product.product_id,
        "slug": product.slug,
        "description": product.description,
        "synthetic": False,
        "manual_coordinates": False,
        "coordinate_origin": "validated archived run and referenced evidence fields",
        "caption_context": {
            "hardware_ids": sorted(
                {
                    str(identity.get("hardware_id"))
                    for identity in identities
                    if identity.get("hardware_id") is not None
                }
            ),
            "precisions": sorted(
                {
                    str(identity.get("precision"))
                    for identity in identities
                    if identity.get("precision") is not None
                }
            ),
            "warm_start_states": sorted(
                {
                    "warm" if identity.get("warm_start") else "cold"
                    for identity in identities
                    if identity.get("warm_start") is not None
                }
            ),
            "quality_tiers": sorted(
                {
                    str(identity.get("quality_tier"))
                    for identity in identities
                    if identity.get("quality_tier") is not None
                }
            ),
            "requested_tolerances": sorted(
                {
                    float(_quality(run)["requested_tolerance"])
                    for run in runs
                    if _quality(run).get("requested_tolerance") is not None
                }
            ),
        },
        "filter_query": query,
        "aggregation": {
            "statistic": "median of measured repeats after declared warm-up",
            "band": ["q1", "q3"],
            "pre_transform": "none",
            "failure_retention": sorted(FAILURE_MARKERS),
        },
        "run_ids": [run.run_id for run in runs],
    }


def _point(run: ArchivedRun, fields: Mapping[str, Any]) -> dict[str, Any]:
    identity = _identity(run)
    dimensions = _dimensions(run)
    aggregation = _aggregation(run)
    return {
        "run_id": run.run_id,
        "family": identity["family"],
        "instance_id": identity["instance_id"],
        "solver": identity["solver"],
        "policy": identity["policy"],
        "quality_tier": identity.get("quality_tier"),
        "warm": identity.get("warm_start"),
        "status": run.status,
        "failure_marker": FAILURE_MARKERS.get(run.status),
        "intervals": dimensions["intervals"],
        "scenarios": dimensions["scenarios"],
        "gpus": dimensions["gpus"],
        "median": aggregation["median"],
        "q1": aggregation["q1"],
        "q3": aggregation["q3"],
        "minimum": aggregation["minimum"],
        "maximum": aggregation["maximum"],
        "repeat_count": aggregation["measured_repeats"],
        "coefficient_of_variation": aggregation["coefficient_of_variation"],
        "censored_count": aggregation["censored_count"],
        **fields,
    }


def _f01(product: Product, runs: Sequence[ArchivedRun]) -> dict[str, Any]:
    source = _base_source(product, [], "diagram; no measured coordinates")
    source.update(
        {
            "units": {},
            "axes": [],
            "blocks": [
                "nonlinear C++ spacecraft model",
                "variational RK4 coefficient fill",
                "fixed CQP topology and mutable values",
                "persistent PDHCG workspace",
                "adaptive forcing/trust acceptance",
                "scenario shards and NCCL reduction",
                "independent nonlinear replay",
                "optional GPU IPM polish",
                "compact host diagnostics",
                "OrbitWeaver continuous-oracle boundary",
            ],
            "implementation_style": "planned",
            "series": [
                "one-time topology",
                "per-iteration values",
                "retained iterates",
                "diagnostics",
            ],
            "data": [],
        }
    )
    return source


def _f02(product: Product, runs: Sequence[ArchivedRun]) -> dict[str, Any]:
    selected = [run for run in runs if _identity(run)["solver"] == "spacepdhcg-persistent"]
    points = []
    for run in selected:
        dimensions, resources = _dimensions(run), _resources(run)
        required = (
            dimensions.get("topology_bytes"),
            dimensions.get("numeric_bytes"),
            resources.get("h2d_bytes"),
            resources.get("topology_allocation_count_after_create"),
            _identity(run).get("repeat"),
        )
        if run.status == "qualified" and any(value is None for value in required):
            raise AggregationError(f"F02 qualified run lacks persistence fields: {run.run_id}")
        points.append(
            _point(
                run,
                {
                    "topology_bytes": dimensions.get("topology_bytes"),
                    "numeric_bytes": dimensions.get("numeric_bytes"),
                    "bytes_h2d": resources.get("h2d_bytes"),
                    "allocation_count": resources.get("topology_allocation_count_after_create"),
                    "workspace_epoch": _identity(run).get("repeat"),
                },
            )
        )
    source = _base_source(product, selected, "solver == spacepdhcg-persistent")
    source.update(
        {
            "axes": [
                {"name": "update number", "scale": "linear", "unit": "count"},
                {"name": "uploaded/allocated bytes", "scale": "linear", "unit": "byte"},
            ],
            "series": ["bytes_h2d", "allocation_count"],
            "units": {"topology_bytes": "byte", "numeric_bytes": "byte", "bytes_h2d": "byte"},
            "data": points,
        }
    )
    return source


def _f03(product: Product, runs: Sequence[ArchivedRun]) -> dict[str, Any]:
    selected = [run for run in runs if _identity(run)["family"] == "P1-F-robust-pd"]
    for run in selected:
        if (
            run.status == "qualified"
            and run.manifest.experiment.get("analytic_collective_bytes") is None
        ):
            raise AggregationError(
                f"F03 qualified robust run lacks analytic communication model: {run.run_id}"
            )
    source = _base_source(product, selected, "family == P1-F-robust-pd")
    source.update(
        {
            "axes": [
                {"name": "scenario count", "scale": "linear", "unit": "count"},
                {"name": "collective bytes/time", "scale": "linear", "unit": "byte/second"},
            ],
            "series": ["analytic communication model", "measured collective"],
            "units": {
                "collective_bytes": "byte",
                "analytic_collective_bytes": "byte",
                "collective_seconds": "second",
            },
            "data": [
                _point(
                    run,
                    {
                        "collective_bytes": _resources(run).get("collective_bytes"),
                        "analytic_collective_bytes": run.manifest.experiment.get(
                            "analytic_collective_bytes"
                        ),
                        "collective_seconds": _timing(run).get("collective_seconds"),
                        "load_imbalance": _resources(run).get("load_imbalance"),
                    },
                )
                for run in selected
            ],
        }
    )
    return source


def _f04(product: Product, runs: Sequence[ArchivedRun]) -> dict[str, Any]:
    families = {"P1-B-hcw", "P1-C-pd3", "P1-D-pd6", "P1-E-low-thrust"}
    selected = [run for run in runs if _identity(run)["family"] in families]
    source = _base_source(product, selected, "family in P1-B..P1-E")
    source.update(
        {
            "axes": [
                {"name": "intervals N", "scale": "log", "unit": "count"},
                {"name": "median T_SCvx", "scale": "log", "unit": "second"},
            ],
            "series": ["solver"],
            "facets": ["family", "quality_tier"],
            "selection_rule": "matched quality; sustained crossover requires three coordinates",
            "units": {"scvx_total_seconds": "second"},
            "data": [
                _point(run, {"scvx_total_seconds": _timing(run)["scvx_total_seconds"]})
                for run in selected
            ],
        }
    )
    return source


def _f05(product: Product, runs: Sequence[ArchivedRun]) -> dict[str, Any]:
    selected = list(runs)
    source = _base_source(product, selected, "all deterministic and robust primary records")
    source.update(
        {
            "axes": [
                {"name": "N or N*S", "scale": "log", "unit": "count"},
                {"name": "peak device allocation", "scale": "log", "unit": "byte"},
            ],
            "series": ["solver", "active", "reserved"],
            "units": {"peak_device_bytes": "byte", "reserved_device_bytes": "byte"},
            "data": [
                _point(
                    run,
                    {
                        "problem_scale": _dimensions(run)["intervals"]
                        * _dimensions(run)["scenarios"],
                        "peak_device_bytes": _resources(run)["peak_device_bytes"],
                        "reserved_device_bytes": _resources(run)["reserved_device_bytes"],
                    },
                )
                for run in selected
            ],
        }
    )
    return source


def _f06(product: Product, runs: Sequence[ArchivedRun]) -> dict[str, Any]:
    policies = {"fixed-tight", "fixed-loose", "adaptive", "adaptive+polish"}
    selected = [run for run in runs if _identity(run)["policy"] in policies]
    source = _base_source(product, selected, "policy in frozen adaptive ablation")
    source.update(
        {
            "axes": [
                {"name": "policy", "scale": "categorical", "unit": "identifier"},
                {"name": "time/work/quality", "scale": "linear", "unit": "panel-specific"},
            ],
            "series": ["family", "problem size"],
            "panels": ["T_SCvx", "matvecs + cone projections", "maximum nonlinear residual"],
            "units": {
                "scvx_total_seconds": "second",
                "inner_work": "operation",
                "final_nonlinear_quality": "dimensionless",
            },
            "data": [
                _point(
                    run,
                    {
                        "scvx_total_seconds": _timing(run)["scvx_total_seconds"],
                        "inner_work": (
                            (_work(run).get("matvecs") or 0)
                            + (_work(run).get("cone_projections") or 0)
                        ),
                        "final_nonlinear_quality": max(
                            float(_quality(run).get(name) or 0.0)
                            for name in (
                                "dynamics_residual",
                                "path_residual",
                                "terminal_residual",
                                "nonanticipativity_residual",
                                "risk_epigraph_residual",
                            )
                        ),
                        "qualified": _quality(run)["qualified"],
                    },
                )
                for run in selected
            ],
        }
    )
    return source


def _f07(product: Product, runs: Sequence[ArchivedRun]) -> dict[str, Any]:
    selected = [
        run
        for run in runs
        if _identity(run)["family"] == "P1-F-robust-pd" and _dimensions(run)["gpus"] >= 1
    ]
    for run in selected:
        scaling_kind = run.manifest.experiment.get("scaling_kind")
        if run.status == "qualified" and scaling_kind not in {"strong", "weak"}:
            raise AggregationError(
                f"F07 qualified robust run lacks strong/weak classification: {run.run_id}"
            )
    by_problem: dict[tuple[Any, ...], dict[int, float]] = defaultdict(dict)
    for run in selected:
        if run.status == "qualified" and _timing(run)["scvx_total_seconds"] is not None:
            key = (
                _identity(run)["instance_id"],
                _dimensions(run)["intervals"],
                _dimensions(run)["scenarios"],
                _identity(run)["policy"],
            )
            by_problem[key][_dimensions(run)["gpus"]] = float(_timing(run)["scvx_total_seconds"])
    data = []
    for run in selected:
        dimensions = _dimensions(run)
        key = (
            _identity(run)["instance_id"],
            dimensions["intervals"],
            dimensions["scenarios"],
            _identity(run)["policy"],
        )
        t1 = by_problem.get(key, {}).get(1)
        total = _timing(run)["scvx_total_seconds"]
        efficiency = (
            t1 / (dimensions["gpus"] * float(total))
            if t1 is not None and total not in (None, 0)
            else None
        )
        data.append(
            _point(
                run,
                {
                    "total_seconds": total,
                    "efficiency": efficiency,
                    "throughput_per_second": _resources(run).get("throughput_per_second"),
                    "communication_fraction": (
                        float(_timing(run)["collective_seconds"]) / float(total)
                        if total not in (None, 0) and _timing(run)["collective_seconds"] is not None
                        else None
                    ),
                    "load_imbalance": _resources(run).get("load_imbalance"),
                    "scaling_kind": run.manifest.experiment.get("scaling_kind"),
                },
            )
        )
    source = _base_source(product, selected, "family == P1-F-robust-pd and gpus >= 1")
    source.update(
        {
            "axes": [
                {"name": "GPU count G", "scale": "linear", "unit": "count"},
                {"name": "time/efficiency/throughput", "scale": "linear", "unit": "panel-specific"},
            ],
            "series": ["scenario-aware", "generic partition"],
            "panels": ["strong scaling", "weak scaling"],
            "units": {
                "total_seconds": "second",
                "efficiency": "ratio",
                "throughput_per_second": "accepted trajectory/second",
            },
            "data": data,
        }
    )
    return source


def _f08(product: Product, runs: Sequence[ArchivedRun]) -> dict[str, Any]:
    selected = list(runs)
    data = []
    for run in selected:
        timing = _timing(run)
        available = [float(timing[name]) for name in TIMING_COMPONENTS if timing[name] is not None]
        total = timing["scvx_total_seconds"]
        if (
            run.status == "qualified"
            and total is not None
            and not math.isclose(sum(available), float(total), rel_tol=1e-8, abs_tol=1e-12)
        ):
            raise AggregationError(f"F08 timing components do not sum for {run.run_id}")
        data.append(_point(run, {name: timing[name] for name in TIMING_COMPONENTS}))
    source = _base_source(product, selected, "all records with complete timing identity")
    source.update(
        {
            "axes": [
                {"name": "cold/first warm/steady", "scale": "categorical", "unit": "class"},
                {"name": "component time", "scale": "linear", "unit": "second"},
            ],
            "series": list(TIMING_COMPONENTS),
            "units": {name: "second" for name in TIMING_COMPONENTS},
            "data": data,
        }
    )
    return source


def _canonical_residual(run: ArchivedRun) -> float | None:
    values = [
        _quality(run).get(name)
        for name in (
            "canonical_primal_residual",
            "canonical_dual_residual",
            "canonical_cone_residual",
            "canonical_gap",
        )
    ]
    if any(value is None for value in values):
        return None
    return max(float(value) for value in values)


def _nonlinear_residual(run: ArchivedRun) -> float | None:
    values = [
        _quality(run).get(name)
        for name in (
            "dynamics_residual",
            "path_residual",
            "terminal_residual",
            "virtual_control_residual",
            "nonanticipativity_residual",
            "risk_epigraph_residual",
        )
    ]
    if any(value is None for value in values):
        return None
    return max(float(value) for value in values)


def _mark_pareto(points: list[dict[str, Any]], residual_field: str) -> None:
    eligible = [
        point
        for point in points
        if point["status"] == "qualified"
        and point["scvx_total_seconds"] not in (None, 0)
        and point[residual_field] not in (None, 0)
    ]
    for point in points:
        matched = [
            other
            for other in eligible
            if (
                other["family"],
                other["quality_tier"],
                other["gpus"],
                other["hardware_id"],
                other["precision"],
            )
            == (
                point["family"],
                point["quality_tier"],
                point["gpus"],
                point["hardware_id"],
                point["precision"],
            )
        ]
        point[f"{residual_field}_pareto"] = point in eligible and not any(
            other is not point
            and float(other["scvx_total_seconds"]) <= float(point["scvx_total_seconds"])
            and float(other[residual_field]) <= float(point[residual_field])
            and (
                float(other["scvx_total_seconds"]) < float(point["scvx_total_seconds"])
                or float(other[residual_field]) < float(point[residual_field])
            )
            for other in matched
        )


def _f09(product: Product, runs: Sequence[ArchivedRun]) -> dict[str, Any]:
    selected = list(runs)
    points = [
        _point(
            run,
            {
                "scvx_total_seconds": _timing(run)["scvx_total_seconds"],
                "canonical_residual": _canonical_residual(run),
                "nonlinear_residual": _nonlinear_residual(run),
                "peak_device_bytes": _resources(run)["peak_device_bytes"],
                "qualified": _quality(run)["qualified"],
                "hardware_id": _identity(run).get("hardware_id"),
                "precision": _identity(run).get("precision"),
            },
        )
        for run in selected
    ]
    _mark_pareto(points, "canonical_residual")
    _mark_pareto(points, "nonlinear_residual")
    source = _base_source(product, selected, "all validated locked-evaluation records")
    source.update(
        {
            "axes": [
                {"name": "median total time", "scale": "log", "unit": "second"},
                {"name": "achieved residual", "scale": "log", "unit": "dimensionless"},
            ],
            "panels": ["canonical CQP residual", "nonlinear trajectory residual"],
            "series": ["solver", "policy"],
            "marker_size": "peak_device_bytes",
            "marker_shape": "family",
            "selection_rule": (
                "Pareto lines contain only nondominated qualified points; all unqualified and "
                "censored points remain visible"
            ),
            "units": {
                "scvx_total_seconds": "second",
                "canonical_residual": "dimensionless",
                "nonlinear_residual": "dimensionless",
                "peak_device_bytes": "byte",
            },
            "data": points,
        }
    )
    return source


def _cone_family(run: ArchivedRun) -> str:
    inventory = _dimensions(run)["cone_inventory"]
    present = sorted(name for name, count in inventory.items() if count)
    return "+".join(present) if present else "QP"


def _regime_cells(runs: Sequence[ArchivedRun]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[ArchivedRun]] = defaultdict(list)
    for run in runs:
        identity, dimensions = _identity(run), _dimensions(run)
        grouped[
            (
                identity["family"],
                dimensions["intervals"] * dimensions["scenarios"],
                identity.get("quality_tier"),
                dimensions["gpus"],
                _cone_family(run),
                identity.get("hardware_id"),
                identity.get("precision"),
                identity.get("warm_start"),
                identity.get("cold_start"),
            )
        ].append(run)
    cells = []
    for coordinate in sorted(grouped, key=str):
        group = grouped[coordinate]
        qualified = [
            run
            for run in group
            if run.status == "qualified" and _timing(run)["scvx_total_seconds"] not in (None, 0)
        ]
        by_candidate: dict[str, list[ArchivedRun]] = defaultdict(list)
        for run in qualified:
            label = f"{_identity(run)['solver']}::{_identity(run)['policy']}"
            by_candidate[label].append(run)
        medians = {
            label: statistics.median(
                float(_timing(run)["scvx_total_seconds"]) for run in candidate_runs
            )
            for label, candidate_runs in by_candidate.items()
        }
        ordered = sorted(medians.items(), key=lambda item: (item[1], item[0]))
        winner = "no qualified solver"
        confidence_interval: list[float | None] = [None, None]
        improvement = None
        if len(ordered) == 1:
            winner = "tie"
        elif len(ordered) >= 2:
            fastest, runner_up = ordered[0], ordered[1]
            improvement = (runner_up[1] - fastest[1]) / runner_up[1]
            fastest_by_instance = {
                _identity(run)["instance_id"]: run for run in by_candidate[fastest[0]]
            }
            runner_by_instance = {
                _identity(run)["instance_id"]: run for run in by_candidate[runner_up[0]]
            }
            common = sorted(fastest_by_instance.keys() & runner_by_instance.keys())
            pairs: list[tuple[float, float]] = []
            for instance in common:
                fastest_repeats = fastest_by_instance[instance].manifest.experiment.get(
                    "measured_repeat_seconds"
                )
                runner_repeats = runner_by_instance[instance].manifest.experiment.get(
                    "measured_repeat_seconds"
                )
                if (
                    not isinstance(fastest_repeats, list)
                    or not isinstance(runner_repeats, list)
                    or len(fastest_repeats) != len(runner_repeats)
                    or len(fastest_repeats) < 5
                ):
                    continue
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) <= 0
                    for value in (*fastest_repeats, *runner_repeats)
                ):
                    raise AggregationError(
                        f"F10 invalid paired repeat evidence for instance {instance}"
                    )
                pairs.extend(
                    (float(baseline), float(candidate))
                    for baseline, candidate in zip(runner_repeats, fastest_repeats, strict=True)
                )
            if pairs:
                seed = int.from_bytes(hashlib.sha256(repr(coordinate).encode()).digest()[:4], "big")
                _, low, high = paired_bootstrap(
                    pairs,
                    lambda sample: statistics.median(
                        (baseline - candidate) / baseline for baseline, candidate in sample
                    ),
                    seed=seed,
                )
                confidence_interval = [low, high]
                if improvement >= 0.10 and low > 0:
                    winner = fastest[0]
                else:
                    winner = "tie"
        cells.append(
            {
                "coordinate": {
                    "family": coordinate[0],
                    "problem_scale": coordinate[1],
                    "quality_tier": coordinate[2],
                    "gpus": coordinate[3],
                    "cone_family": coordinate[4],
                    "hardware_id": coordinate[5],
                    "precision": coordinate[6],
                    "warm_start": coordinate[7],
                    "cold_start": coordinate[8],
                },
                "winner": winner,
                "runner_up_improvement": improvement,
                "paired_confidence_interval_95": confidence_interval,
                "qualified_repeat_count": sum(
                    int(_aggregation(run)["measured_repeats"]) for run in qualified
                ),
                "memory_censored": any(run.status == "oom" for run in group),
                "censored_statuses": sorted(
                    {run.status for run in group if run.status != "qualified"}
                ),
                "run_ids": sorted(run.run_id for run in group),
            }
        )
    return cells


def _f10(product: Product, runs: Sequence[ArchivedRun]) -> dict[str, Any]:
    selected = list(runs)
    missing_tier = [
        run.run_id
        for run in selected
        if run.status == "qualified" and _identity(run).get("quality_tier") is None
    ]
    if missing_tier:
        raise AggregationError(
            "F10 qualified runs require requested/achieved quality tier: " + ", ".join(missing_tier)
        )
    source = _base_source(product, selected, "all locked-evaluation regime coordinates")
    source.update(
        {
            "axes": [
                {"name": "problem scale N*S", "scale": "log", "unit": "count"},
                {
                    "name": "requested/achieved quality tier",
                    "scale": "categorical",
                    "unit": "tier",
                },
            ],
            "facets": [
                "cone_family",
                "gpus",
                "hardware_id",
                "precision",
                "warm_start",
                "cold_start",
            ],
            "series": [
                "CPU IPM",
                "GPU IPM",
                "persistent PDHCG",
                "hybrid",
                "tie",
                "no qualified solver",
            ],
            "selection_rule": (
                "unique winner requires at least 10% lower median end-to-end time and paired "
                "95% interval above zero from at least five archived measured repeat pairs; "
                "unsupported/unqualified excluded, censoring retained"
            ),
            "units": {"problem_scale": "count", "runner_up_improvement": "ratio"},
            "data": _regime_cells(selected),
        }
    )
    return source


def _f11(product: Product, runs: Sequence[ArchivedRun]) -> dict[str, Any]:
    selected = [
        run
        for run in runs
        if _identity(run)["family"] in {"P1-C-pd3", "P1-D-pd6", "P1-E-low-thrust"}
    ]
    data = []
    for run in selected:
        trials = run.manifest.experiment.get("variational_trials", [])
        if not isinstance(trials, list):
            raise AggregationError(f"F11 variational trials must be a list: {run.run_id}")
        for raw in trials:
            if not isinstance(raw, Mapping):
                raise AggregationError(f"F11 variational trial must be an object: {run.run_id}")
            required = {
                "trial",
                "model",
                "maximum_absolute_difference",
                "maximum_relative_difference",
                "analytic_fill_seconds",
                "finite_difference_fill_seconds",
                "quaternion_radial_sensitivity",
                "declared_tolerance",
            }
            if set(raw) != required:
                raise AggregationError(
                    f"F11 variational trial fields differ from contract: {run.run_id}"
                )
            if (
                isinstance(raw["trial"], bool)
                or not isinstance(raw["trial"], int)
                or raw["trial"] < 0
                or raw["model"] not in {"3-DoF", "6-DoF", "low-thrust"}
            ):
                raise AggregationError(f"F11 trial identity is invalid: {run.run_id}")
            numeric_fields = required - {
                "trial",
                "model",
                "quaternion_radial_sensitivity",
            }
            if any(
                isinstance(raw[field], bool)
                or not isinstance(raw[field], (int, float))
                or not math.isfinite(float(raw[field]))
                or float(raw[field]) <= 0
                for field in numeric_fields
            ):
                raise AggregationError(f"F11 trial numeric field is invalid: {run.run_id}")
            radial = raw["quaternion_radial_sensitivity"]
            if raw["model"] == "6-DoF":
                if (
                    isinstance(radial, bool)
                    or not isinstance(radial, (int, float))
                    or not math.isfinite(float(radial))
                    or float(radial) < 0
                ):
                    raise AggregationError(f"F11 6-DoF radial sensitivity is invalid: {run.run_id}")
            elif radial is not None:
                raise AggregationError(
                    f"F11 non-6-DoF radial sensitivity must be null: {run.run_id}"
                )
            data.append({"run_id": run.run_id, **dict(raw)})
    required_models = {"3-DoF", "6-DoF", "low-thrust"}
    qualified_families = {_identity(run)["family"] for run in selected if run.status == "qualified"}
    if qualified_families and {row["model"] for row in data} != required_models:
        raise AggregationError("F11 requires trials for 3-DoF, 6-DoF, and low-thrust")
    source = _base_source(product, selected, "P1-C/P1-D/P1-E variational-validation records")
    source.update(
        {
            "axes": [
                {"name": "admissible state/control trial", "scale": "linear", "unit": "index"},
                {
                    "name": "analytic versus finite-difference error",
                    "scale": "log",
                    "unit": "dimensionless",
                },
            ],
            "panels": ["3-DoF", "6-DoF", "low-thrust"],
            "series": ["maximum absolute difference", "maximum relative difference"],
            "reference_line": "declared_tolerance",
            "companion_metric": "analytic and finite-difference CQP coefficient-fill seconds",
            "units": {
                "maximum_absolute_difference": "dimensionless",
                "maximum_relative_difference": "dimensionless",
                "analytic_fill_seconds": "second",
                "finite_difference_fill_seconds": "second",
                "quaternion_radial_sensitivity": "dimensionless",
            },
            "data": data,
        }
    )
    return source


def _f12(product: Product, runs: Sequence[ArchivedRun]) -> dict[str, Any]:
    selected = [run for run in runs if _identity(run)["family"] == "P1-F-robust-pd"]
    data = []
    required = {
        "risk_mode",
        "outer_iteration",
        "dynamics_residual",
        "path_residual",
        "terminal_residual",
        "virtual_control_residual",
        "nonanticipativity_residual",
        "risk_epigraph_residual",
        "canonical_kkt_residual",
        "accepted",
        "trust_radius",
    }
    for run in selected:
        iterations = run.manifest.experiment.get("robust_iterations", [])
        if not isinstance(iterations, list):
            raise AggregationError(f"F12 robust iterations must be a list: {run.run_id}")
        for raw in iterations:
            if not isinstance(raw, Mapping) or set(raw) != required:
                raise AggregationError(
                    f"F12 robust iteration fields differ from contract: {run.run_id}"
                )
            if (
                raw["risk_mode"] not in {"expected", "worst-case", "CVaR"}
                or isinstance(raw["outer_iteration"], bool)
                or not isinstance(raw["outer_iteration"], int)
                or raw["outer_iteration"] < 0
                or not isinstance(raw["accepted"], bool)
            ):
                raise AggregationError(f"F12 robust iteration identity is invalid: {run.run_id}")
            numeric_fields = required - {"risk_mode", "outer_iteration", "accepted"}
            if any(
                isinstance(raw[field], bool)
                or not isinstance(raw[field], (int, float))
                or not math.isfinite(float(raw[field]))
                or float(raw[field]) <= 0
                for field in numeric_fields
            ):
                raise AggregationError(
                    f"F12 robust iteration numeric field is invalid: {run.run_id}"
                )
            data.append({"run_id": run.run_id, **dict(raw)})
    risk_modes = {row["risk_mode"] for row in data}
    if any(run.status == "qualified" for run in selected) and risk_modes != {
        "expected",
        "worst-case",
        "CVaR",
    }:
        raise AggregationError("F12 requires expected, worst-case, and CVaR iteration evidence")
    source = _base_source(product, selected, "P1-F robust outer-iteration diagnostics")
    source.update(
        {
            "axes": [
                {"name": "outer iteration j", "scale": "linear", "unit": "index"},
                {"name": "residual", "scale": "log", "unit": "dimensionless"},
            ],
            "panels": ["expected", "worst-case", "CVaR"],
            "series": [
                "dynamics",
                "path",
                "terminal",
                "virtual control",
                "non-anticipativity",
                "risk epigraph",
                "canonical KKT",
            ],
            "background": ["accepted/rejected phase", "trust radius"],
            "selection_rule": "diagnostic only; never substitutes for aggregate scaling evidence",
            "units": {"residuals": "dimensionless", "trust_radius": "dimensionless"},
            "data": data,
        }
    )
    return source


FIGURE_BUILDERS: Final[dict[str, Callable[[Product, Sequence[ArchivedRun]], dict[str, Any]]]] = {
    "F01": _f01,
    "F02": _f02,
    "F03": _f03,
    "F04": _f04,
    "F05": _f05,
    "F06": _f06,
    "F07": _f07,
    "F08": _f08,
    "F09": _f09,
    "F10": _f10,
    "F11": _f11,
    "F12": _f12,
}


def _table_rows(
    product_id: str,
    runs: Sequence[ArchivedRun],
    decisions: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[list[Any]]]:
    if product_id == "T01":
        columns = [
            "run_id",
            "hardware_id",
            "cpu",
            "ram_bytes",
            "gpu_model",
            "gpu_count",
            "gpu_memory",
            "interconnect",
            "os",
            "driver",
            "runtime",
            "repository_commit",
            "precision",
        ]
        rows = [
            [
                run.run_id,
                _identity(run).get("hardware_id"),
                run.manifest.host.processor,
                run.manifest.host.memory_bytes,
                run.manifest.host.accelerator_model,
                run.manifest.host.accelerator_count,
                run.manifest.experiment.get("gpu_memory_bytes"),
                run.manifest.host.interconnect,
                run.manifest.host.operating_system,
                run.manifest.host.driver_version,
                run.manifest.host.runtime_version,
                _identity(run)["repository_commit"],
                _identity(run).get("precision"),
            ]
            for run in runs
        ]
    elif product_id == "T02":
        columns = [
            "run_id",
            "family",
            "instance",
            "N",
            "S",
            "nx",
            "nu",
            "variables",
            "scalar_rows",
            "affine_rows",
            "q_nnz",
            "a_nnz",
            "f_nnz",
            "cone_inventory",
            "topology_bytes",
            "numeric_bytes",
        ]
        rows = [
            [
                run.run_id,
                _identity(run)["family"],
                _identity(run)["instance_id"],
                _dimensions(run)["intervals"],
                _dimensions(run)["scenarios"],
                _dimensions(run)["state_dimension"],
                _dimensions(run)["control_dimension"],
                _dimensions(run)["variables"],
                _dimensions(run)["scalar_rows"],
                _dimensions(run)["affine_rows"],
                _dimensions(run)["q_nonzeros"],
                _dimensions(run)["a_nonzeros"],
                _dimensions(run)["f_nonzeros"],
                str(dict(sorted(_dimensions(run)["cone_inventory"].items()))),
                _dimensions(run).get("topology_bytes"),
                _dimensions(run).get("numeric_bytes"),
            ]
            for run in runs
        ]
    elif product_id == "T03":
        columns = [
            "run_id",
            "family",
            "instance",
            "solver",
            "status",
            "objective",
            "objective_gap",
            "rp",
            "rd",
            "rc",
            "rg",
            "rdyn",
            "rpath",
            "rterm",
            "rvc",
            "rna",
            "rrisk",
            "qualified",
        ]
        rows = [
            [
                run.run_id,
                _identity(run)["family"],
                _identity(run)["instance_id"],
                _identity(run)["solver"],
                run.status,
                _quality(run)["objective"],
                _quality(run).get("objective_gap"),
                _quality(run)["canonical_primal_residual"],
                _quality(run)["canonical_dual_residual"],
                _quality(run)["canonical_cone_residual"],
                _quality(run)["canonical_gap"],
                _quality(run)["dynamics_residual"],
                _quality(run)["path_residual"],
                _quality(run)["terminal_residual"],
                _quality(run)["virtual_control_residual"],
                _quality(run)["nonanticipativity_residual"],
                _quality(run)["risk_epigraph_residual"],
                _quality(run)["qualified"],
            ]
            for run in runs
        ]
    elif product_id == "T04":
        selected = [run for run in runs if _identity(run)["solver"] == "spacepdhcg-persistent"]
        columns = [
            "run_id",
            "family",
            "N",
            "setup_seconds",
            "create_seconds",
            "update_seconds",
            "solve_seconds",
            "total_seconds",
            "post_create_topology_allocations",
            "h2d_bytes",
            "d2h_bytes",
            "status",
        ]
        rows = [
            [
                run.run_id,
                _identity(run)["family"],
                _dimensions(run)["intervals"],
                _timing(run)["topology_seconds"],
                _timing(run)["workspace_create_seconds"],
                _timing(run)["update_seconds"],
                _timing(run)["solve_seconds"],
                _timing(run)["cqp_total_seconds"],
                _resources(run)["topology_allocation_count_after_create"],
                _resources(run)["h2d_bytes"],
                _resources(run)["d2h_bytes"],
                run.status,
            ]
            for run in selected
        ]
    elif product_id == "T05":
        policies = {"fixed-tight", "fixed-loose", "adaptive", "adaptive+polish"}
        selected = [run for run in runs if _identity(run)["policy"] in policies]
        columns = [
            "run_id",
            "family",
            "N",
            "policy",
            "total_seconds",
            "outer_iterations",
            "accepted_steps",
            "rejected_steps",
            "resolved_steps",
            "matvecs",
            "cone_projections",
            "objective",
            "achieved_residual",
            "polish_used",
            "qualified",
            "status",
        ]
        rows = [
            [
                run.run_id,
                _identity(run)["family"],
                _dimensions(run)["intervals"],
                _identity(run)["policy"],
                _timing(run)["scvx_total_seconds"],
                _work(run)["outer_iterations"],
                _work(run)["accepted_steps"],
                _work(run)["rejected_steps"],
                _work(run)["resolved_steps"],
                _work(run)["matvecs"],
                _work(run)["cone_projections"],
                _quality(run)["objective"],
                _quality(run).get("achieved_residual"),
                _work(run)["polish_used"],
                _quality(run)["qualified"],
                run.status,
            ]
            for run in selected
        ]
    elif product_id == "T06":
        selected = [run for run in runs if _identity(run)["family"] == "P1-F-robust-pd"]
        columns = [
            "run_id",
            "N",
            "S",
            "G",
            "partition",
            "load_imbalance",
            "collective_count",
            "collective_bytes",
            "collective_seconds",
            "total_seconds",
            "throughput",
            "peak_bytes",
            "qualified",
            "status",
        ]
        rows = [
            [
                run.run_id,
                _dimensions(run)["intervals"],
                _dimensions(run)["scenarios"],
                _dimensions(run)["gpus"],
                _identity(run)["policy"],
                _resources(run).get("load_imbalance"),
                _resources(run)["collective_count"],
                _resources(run)["collective_bytes"],
                _timing(run)["collective_seconds"],
                _timing(run)["scvx_total_seconds"],
                _resources(run).get("throughput_per_second"),
                _resources(run)["peak_device_bytes"],
                _quality(run)["qualified"],
                run.status,
            ]
            for run in selected
        ]
    elif product_id == "T07":
        columns = [
            "source_run_ids",
            "family",
            "quality_tier",
            "first_compute_crossover",
            "first_memory_crossover",
            "winner_below_crossover",
            "winner_above_crossover",
            "censored_range",
            "evidence_count",
            "decision_confidence_qualification_note",
        ]
        grouped: dict[tuple[str, Any], list[dict[str, Any]]] = defaultdict(list)
        for cell in _regime_cells(runs):
            coordinate = cell["coordinate"]
            grouped[(coordinate["family"], coordinate["quality_tier"])].append(cell)

        def first_sustained_compute(cells: list[dict[str, Any]]) -> int | None:
            ordered_cells = sorted(cells, key=lambda cell: cell["coordinate"]["problem_scale"])
            for index, cell in enumerate(ordered_cells):
                if not cell["winner"].startswith("spacepdhcg-persistent"):
                    continue
                remaining = ordered_cells[index:]
                required = min(3, len(remaining))
                if all(
                    item["winner"].startswith("spacepdhcg-persistent")
                    for item in remaining[:required]
                ):
                    return int(cell["coordinate"]["problem_scale"])
            return None

        def first_memory_crossover(family: str, tier: Any) -> int | None:
            if decisions.get("H3", {}).get("outcome") != "supported":
                return None
            family_runs = [
                run
                for run in runs
                if _identity(run)["family"] == family and _identity(run).get("quality_tier") == tier
            ]
            declared_ipms = {
                _identity(run)["solver"]
                for run in family_runs
                if _identity(run)["solver"] in {"qoco-gpu", "cuclarabel"}
            }
            for scale in sorted({_scale(run) for run in family_runs}):
                at_scale = [run for run in family_runs if _scale(run) == scale]
                pdhcg = [
                    run
                    for run in at_scale
                    if _identity(run)["solver"] == "spacepdhcg-persistent"
                    and run.status == "qualified"
                ]
                if not pdhcg:
                    continue
                ipm = [run for run in at_scale if _identity(run)["solver"] in declared_ipms]
                all_oom = (
                    declared_ipms
                    and {_identity(run)["solver"] for run in ipm if run.status == "oom"}
                    == declared_ipms
                )
                pdhcg_memory = [
                    float(_resources(run)["peak_device_bytes"])
                    for run in pdhcg
                    if _resources(run)["peak_device_bytes"] is not None
                ]
                ipm_memory = [
                    float(_resources(run)["peak_device_bytes"])
                    for run in ipm
                    if run.status == "qualified"
                    and _resources(run)["peak_device_bytes"] is not None
                ]
                ratio_pass = (
                    pdhcg_memory
                    and ipm_memory
                    and statistics.median(pdhcg_memory) <= 0.60 * statistics.median(ipm_memory)
                )
                if all_oom or ratio_pass:
                    return scale
            return None

        rows = []
        for (family, tier), cells in sorted(grouped.items(), key=lambda item: str(item[0])):
            cells.sort(key=lambda cell: cell["coordinate"]["problem_scale"])
            first_compute = first_sustained_compute(cells)
            first_memory = first_memory_crossover(family, tier)
            below = [
                cell["winner"]
                for cell in cells
                if first_compute is not None and cell["coordinate"]["problem_scale"] < first_compute
            ]
            above = [
                cell["winner"]
                for cell in cells
                if first_compute is not None
                and cell["coordinate"]["problem_scale"] >= first_compute
            ]
            source_run_ids = sorted({run_id for cell in cells for run_id in cell["run_ids"]})
            censored = sorted(
                {cell["coordinate"]["problem_scale"] for cell in cells if cell["censored_statuses"]}
            )
            h2 = decisions.get("H2", {})
            h3 = decisions.get("H3", {})
            rows.append(
                [
                    "|".join(source_run_ids),
                    family,
                    tier,
                    first_compute,
                    first_memory,
                    below[-1] if below else None,
                    above[0] if above else None,
                    "|".join(str(scale) for scale in censored),
                    len(source_run_ids),
                    f"H2={h2.get('outcome', 'missing')};H3={h3.get('outcome', 'missing')}",
                ]
            )
    elif product_id == "T08":
        columns = [
            "source_run_ids",
            "hypothesis",
            "problem_regime",
            "observed_failure_or_null_result",
            "quality_status",
            "likely_mechanism",
            "supporting_artifact",
            "decision",
        ]
        rows = []
        run_by_id = {run.run_id: run for run in runs}
        for hypothesis in sorted(decisions):
            decision = decisions[hypothesis]
            if decision["outcome"] == "supported":
                continue
            input_ids = sorted(decision["input_run_ids"])
            statuses = sorted(
                {run_by_id[run_id].status for run_id in input_ids if run_id in run_by_id}
            )
            coordinates = decision["comparison_coordinates"]
            regimes = sorted(
                {str(coordinate.get("coordinate", coordinate)) for coordinate in coordinates}
            )
            artifacts = sorted(
                {run_by_id[run_id].archive.uri for run_id in input_ids if run_id in run_by_id}
            )
            rows.append(
                [
                    "|".join(input_ids),
                    hypothesis,
                    "|".join(regimes) or "no matched coordinate",
                    "|".join(statuses) or "unresolved comparison",
                    "qualified-only for performance; censoring retained",
                    "; ".join(decision["notes"])
                    or "mechanism not established by archived evidence",
                    "|".join(artifacts),
                    decision["outcome"],
                ]
            )
        represented = {run_id for row in rows for run_id in str(row[0]).split("|") if run_id}
        for run in runs:
            if run.status == "qualified" or run.run_id in represented:
                continue
            rows.append(
                [
                    run.run_id,
                    "not a primary hypothesis",
                    f"{_identity(run)['family']} at scale {_scale(run)}",
                    run.status,
                    "unqualified/censored",
                    "mechanism not established by archived evidence",
                    run.archive.uri,
                    "not applicable; retained negative result",
                ]
            )
    else:
        raise AggregationError(f"unknown table {product_id}")
    return columns, rows


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    path.write_text(stream.getvalue(), encoding="utf-8", newline="")


def _write_tex(path: Path, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    def escape(value: Any) -> str:
        return str(value if value is not None else "N/A").replace("_", r"\_").replace("%", r"\%")

    lines = [
        r"\begin{tabular}{" + "l" * len(columns) + "}",
        " & ".join(escape(value) for value in columns) + r" \\",
        r"\hline",
        *[" & ".join(escape(value) for value in row) + r" \\" for row in rows],
        r"\end{tabular}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _render_figure(source: Mapping[str, Any], pdf_path: Path, png_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as error:
        raise AggregationError("matplotlib is required to render publication figures") from error

    data = source.get("data", [])
    product_id = source["product_id"]
    if product_id in {"F06", "F11", "F12"}:
        figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), constrained_layout=True)
    elif product_id in {"F07", "F09"}:
        figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
    else:
        figure, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
        axes = [axis]
    figure.suptitle(f"{product_id} — {source['description']}")
    if not data:
        axis = axes[0]
        blocks = source.get("blocks", [])
        axis.axis("off")
        axis.text(
            0.02,
            0.98,
            "\n".join(f"{index + 1}. {block}" for index, block in enumerate(blocks))
            or "No eligible records",
            va="top",
            family="sans-serif",
        )
    elif product_id == "F02":
        axis = axes[0]
        eligible = [
            point
            for point in data
            if point["workspace_epoch"] is not None and point["bytes_h2d"] is not None
        ]
        for solver in sorted({point["solver"] for point in eligible}):
            series = sorted(
                (point for point in eligible if point["solver"] == solver),
                key=lambda point: point["workspace_epoch"],
            )
            axis.plot(
                [point["workspace_epoch"] for point in series],
                [point["bytes_h2d"] for point in series],
                marker="o",
                label=f"{solver}: H2D",
            )
        allocation_axis = axis.twinx()
        allocation_axis.scatter(
            [point["workspace_epoch"] for point in eligible],
            [point["allocation_count"] for point in eligible],
            marker="x",
            color="#c44e52",
            label="post-create topology allocations",
        )
        allocation_axis.set_ylabel("allocation count")
        axis.set(xlabel="workspace epoch", ylabel="uploaded bytes")
        if eligible:
            axis.legend(fontsize="small")
    elif product_id == "F03":
        axis = axes[0]
        eligible = [point for point in data if point["collective_seconds"] is not None]
        axis.plot(
            [point["scenarios"] for point in eligible],
            [point["collective_seconds"] for point in eligible],
            marker="o",
            label="measured collective time",
        )
        bytes_axis = axis.twinx()
        bytes_axis.plot(
            [point["scenarios"] for point in eligible],
            [point["collective_bytes"] for point in eligible],
            marker="s",
            label="measured bytes",
            color="#55a868",
        )
        bytes_axis.plot(
            [point["scenarios"] for point in eligible],
            [point["analytic_collective_bytes"] for point in eligible],
            linestyle="--",
            label="analytic bytes",
            color="#c44e52",
        )
        bytes_axis.set_ylabel("collective bytes")
        axis.set(xlabel="scenario count S", ylabel="collective time (s)")
    elif product_id == "F04":
        axis = axes[0]
        eligible = [
            point
            for point in data
            if point["status"] == "qualified" and point["scvx_total_seconds"] not in (None, 0)
        ]
        for solver in sorted({point["solver"] for point in eligible}):
            series = sorted(
                (point for point in eligible if point["solver"] == solver),
                key=lambda point: point["intervals"],
            )
            axis.plot(
                [point["intervals"] for point in series],
                [point["scvx_total_seconds"] for point in series],
                marker="o",
                label=solver,
            )
        axis.set(xscale="log", yscale="log", xlabel="intervals N", ylabel="median T_SCvx (s)")
        axis.legend(fontsize="small")
    elif product_id == "F05":
        axis = axes[0]
        eligible = [point for point in data if point["peak_device_bytes"] not in (None, 0)]
        for solver in sorted({point["solver"] for point in eligible}):
            series = sorted(
                (point for point in eligible if point["solver"] == solver),
                key=lambda point: point["problem_scale"],
            )
            axis.plot(
                [point["problem_scale"] for point in series],
                [point["peak_device_bytes"] for point in series],
                marker="o",
                label=solver,
            )
        axis.set(xscale="log", yscale="log", xlabel="N times S", ylabel="peak active bytes")
        axis.legend(fontsize="small")
    elif product_id == "F06":
        policies = ["fixed-tight", "fixed-loose", "adaptive", "adaptive+polish"]
        fields = (
            ("scvx_total_seconds", "T_SCvx (s)"),
            ("inner_work", "matvecs + projections"),
            ("final_nonlinear_quality", "max nonlinear residual"),
        )
        for axis, (field, label) in zip(axes, fields, strict=True):
            eligible = [
                point for point in data if point[field] is not None and point["policy"] in policies
            ]
            axis.scatter(
                [policies.index(point["policy"]) for point in eligible],
                [point[field] for point in eligible],
                marker="o",
            )
            axis.set_xticks(range(len(policies)), policies, rotation=25, ha="right")
            axis.set_ylabel(label)
            if field == "final_nonlinear_quality":
                axis.set_yscale("log")
    elif product_id == "F07":
        eligible = [
            point for point in data if point["status"] == "qualified" and point["total_seconds"]
        ]
        for axis, field, label in (
            (axes[0], "total_seconds", "strong-scaling time (s)"),
            (axes[1], "efficiency", "parallel efficiency"),
        ):
            for policy in sorted({point["policy"] for point in eligible}):
                series = sorted(
                    (point for point in eligible if point["policy"] == policy),
                    key=lambda point: point["gpus"],
                )
                axis.plot(
                    [point["gpus"] for point in series],
                    [point[field] for point in series],
                    marker="o",
                    label=policy,
                )
            axis.set(xlabel="GPU count G", ylabel=label)
            axis.legend(fontsize="small")
    elif product_id == "F08":
        axis = axes[0]
        eligible = [point for point in data if point["status"] == "qualified"]
        bottoms = [0.0] * len(eligible)
        labels = [point["run_id"] for point in eligible]
        for component in TIMING_COMPONENTS:
            values = [float(point[component] or 0.0) for point in eligible]
            axis.bar(labels, values, bottom=bottoms, label=component.removesuffix("_seconds"))
            bottoms = [bottom + value for bottom, value in zip(bottoms, values, strict=True)]
        axis.set_ylabel("component time (s)")
        axis.tick_params(axis="x", rotation=45)
        axis.legend(fontsize="x-small", ncols=2)
    elif product_id == "F09":
        for axis, residual_field, label in (
            (axes[0], "canonical_residual", "canonical CQP residual"),
            (axes[1], "nonlinear_residual", "nonlinear trajectory residual"),
        ):
            eligible = [
                point
                for point in data
                if point["scvx_total_seconds"] not in (None, 0)
                and point[residual_field] not in (None, 0)
            ]
            for series_name in sorted(
                {f"{point['solver']}::{point['policy']}" for point in eligible}
            ):
                series = [
                    point
                    for point in eligible
                    if f"{point['solver']}::{point['policy']}" == series_name
                ]
                for qualified in (True, False):
                    subset = [point for point in series if point["qualified"] is qualified]
                    if not subset:
                        continue
                    sizes = [
                        20.0
                        + 40.0
                        * (
                            math.log10(max(float(point["peak_device_bytes"] or 1), 1))
                            / max(
                                math.log10(
                                    max(
                                        float(other["peak_device_bytes"] or 1) for other in eligible
                                    )
                                ),
                                1.0,
                            )
                        )
                        for point in subset
                    ]
                    axis.scatter(
                        [point["scvx_total_seconds"] for point in subset],
                        [point[residual_field] for point in subset],
                        s=sizes,
                        marker="o" if qualified else "x",
                        label=series_name + ("" if qualified else " (unqualified)"),
                    )
                frontier_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
                for point in series:
                    if point[f"{residual_field}_pareto"]:
                        frontier_groups[
                            (
                                point["family"],
                                point["quality_tier"],
                                point["gpus"],
                                point["hardware_id"],
                                point["precision"],
                            )
                        ].append(point)
                for frontier in frontier_groups.values():
                    frontier.sort(key=lambda point: point["scvx_total_seconds"])
                    axis.plot(
                        [point["scvx_total_seconds"] for point in frontier],
                        [point[residual_field] for point in frontier],
                        linewidth=1,
                    )
            axis.set(
                xscale="log",
                yscale="log",
                xlabel="median T_SCvx (s)",
                ylabel=label,
            )
            axis.legend(fontsize="xx-small")
    elif product_id == "F10":
        axis = axes[0]
        tiers = ["coarse", "medium", "tight", "ipm", None]
        winners = sorted({cell["winner"] for cell in data})
        winner_index = {winner: index for index, winner in enumerate(winners)}
        for cell in data:
            coordinate = cell["coordinate"]
            axis.scatter(
                coordinate["problem_scale"],
                tiers.index(coordinate["quality_tier"]),
                marker="x" if cell["memory_censored"] else "s",
                c=f"C{winner_index[cell['winner']] % 10}",
            )
        axis.set_xscale("log")
        axis.set_yticks(range(len(tiers)), [tier or "unspecified" for tier in tiers])
        axis.set(xlabel="problem scale N times S", ylabel="quality tier")
        legend_text = "\n".join(
            f"C{index % 10}: {winner}" for winner, index in winner_index.items()
        )
        axis.text(1.02, 1.0, legend_text, transform=axis.transAxes, va="top", fontsize="x-small")
        censored_counts: dict[str, int] = defaultdict(int)
        for cell in data:
            for status in cell["censored_statuses"]:
                censored_counts[status] += 1
        if censored_counts:
            axis.text(
                0.01,
                0.01,
                "censored cells: "
                + ", ".join(
                    f"{status}={count}" for status, count in sorted(censored_counts.items())
                ),
                transform=axis.transAxes,
                fontsize="x-small",
            )
    elif product_id == "F11":
        models = ["3-DoF", "6-DoF", "low-thrust"]
        for axis, model in zip(axes, models, strict=True):
            series = [point for point in data if point["model"] == model]
            axis.plot(
                [point["trial"] for point in series],
                [point["maximum_absolute_difference"] for point in series],
                marker="o",
                label="absolute",
            )
            axis.plot(
                [point["trial"] for point in series],
                [point["maximum_relative_difference"] for point in series],
                marker="s",
                label="relative",
            )
            if series:
                axis.axhline(
                    float(series[0]["declared_tolerance"]),
                    linestyle="--",
                    label="declared tolerance",
                )
            axis.set(
                title=model,
                xlabel="admissible trial",
                ylabel="variational difference",
                yscale="log",
            )
            axis.legend(fontsize="x-small")
    elif product_id == "F12":
        modes = ["expected", "worst-case", "CVaR"]
        residual_fields = (
            "dynamics_residual",
            "path_residual",
            "terminal_residual",
            "virtual_control_residual",
            "nonanticipativity_residual",
            "risk_epigraph_residual",
            "canonical_kkt_residual",
        )
        for axis, mode in zip(axes, modes, strict=True):
            series = [point for point in data if point["risk_mode"] == mode]
            for field in residual_fields:
                axis.plot(
                    [point["outer_iteration"] for point in series],
                    [point[field] for point in series],
                    marker="o",
                    label=field.removesuffix("_residual"),
                )
            axis.set(
                title=mode,
                xlabel="outer iteration j",
                ylabel="residual",
                yscale="log",
            )
            axis.legend(fontsize="xx-small")
    else:  # pragma: no cover - frozen figure set is exhaustive
        raise AggregationError(f"no renderer for {product_id}")
    retained_failures = [
        point["failure_marker"] for point in data if point.get("failure_marker") is not None
    ]
    if retained_failures:
        counts = {
            marker: retained_failures.count(marker) for marker in sorted(set(retained_failures))
        }
        axes[0].text(
            0.01,
            0.01,
            "Retained censored/failure records: "
            + ", ".join(f"{marker}={count}" for marker, count in counts.items()),
            transform=axes[0].transAxes,
            fontsize="x-small",
            va="bottom",
        )
    context = source["caption_context"]
    figure.text(
        0.5,
        0.002,
        "hardware="
        + (",".join(context["hardware_ids"]) or "diagram/not-applicable")
        + "; precision="
        + (",".join(context["precisions"]) or "not-applicable")
        + "; start="
        + (",".join(context["warm_start_states"]) or "not-applicable")
        + "; quality="
        + (",".join(context["quality_tiers"]) or "not-applicable")
        + "; requested tolerance="
        + (
            ",".join(f"{value:.6g}" for value in context["requested_tolerances"])
            or "not-applicable"
        )
        + "; aggregation=median [Q1,Q3] after declared warm-up",
        ha="center",
        fontsize="xx-small",
    )
    metadata = {
        "Title": f"{source['product_id']} {source['description']}",
        "Author": "SpacePDHCG deterministic Paper 1 builder",
        "Creator": "spacepdhcg-paper1",
        "Producer": "spacepdhcg-paper1",
        "CreationDate": None,
        "ModDate": None,
    }
    figure.savefig(pdf_path, format="pdf", metadata=metadata)
    figure.savefig(png_path, format="png", dpi=160, metadata={"Software": "spacepdhcg-paper1"})
    plt.close(figure)


def build_products(
    runs: Iterable[ArchivedRun],
    output_directory: str | Path,
    *,
    decisions: Mapping[str, Mapping[str, Any]],
    synthetic: bool = False,
) -> dict[str, Any]:
    """Build all and only frozen F01-F12/T01-T08 products."""

    ordered = _ordered_runs(runs)
    if set(decisions) != {f"H{index}" for index in range(1, 7)}:
        raise AggregationError("product build requires complete H1-H6 decision records")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    products: list[dict[str, Any]] = []
    for product in FIGURES:
        source = FIGURE_BUILDERS[product.product_id](product, ordered)
        source["synthetic"] = synthetic
        stem = f"fig{product.product_id[1:]}_{product.slug}"
        source_path = write_canonical_json(output / f"{stem}.json", source)
        pdf_path, png_path = output / f"{stem}.pdf", output / f"{stem}.png"
        _render_figure(source, pdf_path, png_path)
        products.append(
            {
                "product_id": product.product_id,
                "source": source_path.name,
                "pdf": pdf_path.name,
                "png": png_path.name,
                "run_ids": source["run_ids"],
            }
        )
    for product in TABLES:
        columns, rows = _table_rows(product.product_id, ordered, decisions)
        contributing_ids = {run_id for row in rows for run_id in str(row[0]).split("|") if run_id}
        contributing_runs = [run for run in ordered if run.run_id in contributing_ids]
        source = _base_source(
            product,
            contributing_runs,
            f"frozen selection for {product.product_id}",
        )
        source.update(
            {
                "synthetic": synthetic,
                "columns": columns,
                "units": "SI values; explicit field names carry scale",
                "rows": rows,
            }
        )
        stem = f"tab{product.product_id[1:]}_{product.slug}"
        source_path = write_canonical_json(output / f"{stem}.json", source)
        csv_path, tex_path = output / f"{stem}.csv", output / f"{stem}.tex"
        _write_csv(csv_path, columns, rows)
        _write_tex(tex_path, columns, rows)
        products.append(
            {
                "product_id": product.product_id,
                "source": source_path.name,
                "csv": csv_path.name,
                "tex": tex_path.name,
                "run_ids": source["run_ids"],
            }
        )
    manifest = {
        "schema_version": FIGURE_SCHEMA_VERSION,
        "synthetic": synthetic,
        "product_ids": [product.product_id for product in (*FIGURES, *TABLES)],
        "products": products,
        "source_digest": __import__("hashlib")
        .sha256(
            b"".join(
                canonical_json_bytes(
                    __import__("json").loads((output / item["source"]).read_text(encoding="utf-8"))
                )
                for item in products
            )
        )
        .hexdigest(),
    }
    write_canonical_json(output / "build-manifest.json", manifest)
    return manifest
