"""Deterministic source-data and publication rendering for frozen F01-F08/T01-T06."""

from __future__ import annotations

import csv
import io
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

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
)
TABLES: Final = (
    Product("T01", "hardware_software", "table", "Hardware and software manifest"),
    Product("T02", "problem_dimensions", "table", "Problem dimensions"),
    Product("T03", "correctness", "table", "Correctness"),
    Product("T04", "persistence", "table", "Persistence"),
    Product("T05", "adaptive_policy", "table", "Adaptive policy"),
    Product("T06", "robust_scaling", "table", "Robust scaling"),
)


def _identity(run: ArchivedRun) -> Mapping[str, Any]:
    return run.result["identity"]


def _dimensions(run: ArchivedRun) -> Mapping[str, Any]:
    return run.result["dimensions"]


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
    return {
        "schema_version": FIGURE_SCHEMA_VERSION,
        "product_id": product.product_id,
        "slug": product.slug,
        "description": product.description,
        "synthetic": False,
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


FIGURE_BUILDERS: Final[dict[str, Callable[[Product, Sequence[ArchivedRun]], dict[str, Any]]]] = {
    "F01": _f01,
    "F02": _f02,
    "F03": _f03,
    "F04": _f04,
    "F05": _f05,
    "F06": _f06,
    "F07": _f07,
    "F08": _f08,
}


def _table_rows(product_id: str, runs: Sequence[ArchivedRun]) -> tuple[list[str], list[list[Any]]]:
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
    if product_id == "F06":
        figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), constrained_layout=True)
    elif product_id == "F07":
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
    synthetic: bool = False,
) -> dict[str, Any]:
    """Build all and only frozen F01-F08/T01-T06 products."""

    ordered = _ordered_runs(runs)
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
        columns, rows = _table_rows(product.product_id, ordered)
        contributing_ids = {str(row[0]) for row in rows}
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
