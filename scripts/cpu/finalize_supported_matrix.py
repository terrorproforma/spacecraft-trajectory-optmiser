#!/usr/bin/env python3
"""Validate and visualize the complete attempted CPU/reference matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import jsonschema
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

SCHEMA_VERSION = "1.0.0"


def _bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_bytes(value))


def _semantic(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"timing", "resources", "artifacts"}
    }


def _metadata(
    chart_id: str,
    title: str,
    axes: list[dict[str, str]],
    series: list[str],
    caption: str,
    environment: dict[str, Any],
    run_ids: list[str],
    data: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "chart_id": chart_id,
        "title": title,
        "axes": axes,
        "series_legend": series,
        "source_commit": environment["source_commit"],
        "driver_commit": environment["driver_commit"],
        "source_time_range_utc": {
            "start": environment["started_utc"],
            "end": environment["completed_utc"],
        },
        "transformation_aggregation_caption": caption,
        "run_ids": run_ids,
        "data": data,
    }


def _save(figure: Any, output: Path, stem: str, source: dict[str, Any]) -> None:
    _write(output / f"{stem}.json", source)
    metadata = {
        "Title": source["title"],
        "Author": "SpacePDHCG CPU campaign",
        "Creator": "finalize_supported_matrix.py",
        "Producer": "finalize_supported_matrix.py",
        "CreationDate": None,
        "ModDate": None,
    }
    figure.savefig(output / f"{stem}.pdf", metadata=metadata)
    figure.savefig(
        output / f"{stem}.png",
        dpi=160,
        metadata={"Software": "SpacePDHCG CPU campaign"},
    )
    plt.close(figure)


def _render(output: Path, results: list[dict[str, Any]], environment: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    families = sorted({item["family"] for item in results})
    dispositions = sorted({item["disposition"] for item in results})
    coverage = [
        {
            "family": family,
            "disposition": disposition,
            "count": sum(
                item["family"] == family and item["disposition"] == disposition for item in results
            ),
        }
        for family in families
        for disposition in dispositions
    ]
    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    bottoms = [0] * len(families)
    for disposition in dispositions:
        values = [
            next(
                item["count"]
                for item in coverage
                if item["family"] == family and item["disposition"] == disposition
            )
            for family in families
        ]
        axis.bar(families, values, bottom=bottoms, label=disposition)
        bottoms = [left + right for left, right in zip(bottoms, values, strict=True)]
    axis.set(
        title="Complete frozen CPU matrix execution coverage",
        xlabel="problem family",
        ylabel="coordinate count",
    )
    axis.tick_params(axis="x", rotation=35)
    axis.legend()
    _save(
        figure,
        output,
        "diag10_execution_coverage",
        _metadata(
            "D10",
            "Complete frozen CPU matrix execution coverage",
            [
                {"name": "problem family", "unit": "identifier"},
                {"name": "coordinate count", "unit": "count"},
            ],
            dispositions,
            "Exact count over all 16,324 frozen family coordinates; no record is dropped.",
            environment,
            [item["coordinate_id"] for item in results],
            coverage,
        ),
    )

    residual_fields = (
        "canonical_primal_residual",
        "canonical_dual_residual",
        "canonical_cone_residual",
        "dynamics_residual",
        "path_residual",
        "terminal_residual",
        "nonanticipativity_residual",
        "risk_epigraph_residual",
    )
    residual_data = [
        {
            "coordinate_id": item["coordinate_id"],
            "family": item["family"],
            "metric": field,
            "value": item["quality"][field],
        }
        for item in results
        for field in residual_fields
        if item["quality"][field] is not None
    ]
    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for field in residual_fields:
        values = [
            max(float(item["value"]), 1.0e-18) for item in residual_data if item["metric"] == field
        ]
        if values:
            axis.scatter([field] * len(values), values, s=5, alpha=0.35, label=field)
    axis.set(
        title="Independently emitted CPU residual distributions",
        xlabel="residual metric",
        ylabel="residual magnitude (dimensionless)",
        yscale="log",
    )
    axis.tick_params(axis="x", rotation=35)
    _save(
        figure,
        output,
        "diag11_residual_distributions",
        _metadata(
            "D11",
            "Independently emitted CPU residual distributions",
            [
                {"name": "residual metric", "unit": "identifier"},
                {"name": "residual magnitude", "unit": "dimensionless"},
            ],
            list(residual_fields),
            (
                "Every finite emitted residual is shown; exact zeros use a labelled plotting "
                "floor of 1e-18 only for logarithmic display."
            ),
            environment,
            sorted({item["coordinate_id"] for item in residual_data}),
            residual_data,
        ),
    )

    trajectory_families = {"P1-B-hcw", "P1-C-pd3", "P1-D-pd6", "P1-E-low-thrust"}
    trajectory = [
        {
            "coordinate_id": item["coordinate_id"],
            "family": item["family"],
            "intervals": item["dimensions"]["intervals"],
            "dynamics": item["quality"]["dynamics_residual"],
            "path": item["quality"]["path_residual"],
            "terminal": item["quality"]["terminal_residual"],
            "disposition": item["disposition"],
        }
        for item in results
        if item["family"] in trajectory_families
    ]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, metric in zip(axes, ("dynamics", "path", "terminal"), strict=True):
        for family in sorted(trajectory_families):
            selected = [
                item for item in trajectory if item["family"] == family and item[metric] is not None
            ]
            if selected:
                axis.scatter(
                    [item["intervals"] for item in selected],
                    [max(float(item[metric]), 1.0e-18) for item in selected],
                    s=10,
                    label=family,
                )
        axis.set(xscale="log", yscale="log", xlabel="intervals N", ylabel=metric)
    axes[0].legend(fontsize="x-small")
    figure.suptitle("Trajectory dynamics, path, and terminal replay quality")
    _save(
        figure,
        output,
        "diag12_trajectory_quality",
        _metadata(
            "D12",
            "Trajectory dynamics, path, and terminal replay quality",
            [
                {"name": "intervals N", "unit": "count"},
                {"name": "violation", "unit": "dimensionless or model-scaled SI"},
            ],
            sorted(trajectory_families),
            "Raw per-coordinate replay maxima; no cross-family normalization or aggregation.",
            environment,
            [item["coordinate_id"] for item in trajectory],
            trajectory,
        ),
    )

    robust = [
        {
            "coordinate_id": item["coordinate_id"],
            "scenarios": item["dimensions"]["scenarios"],
            "risk_metric": item["parameters"]["risk_metrics"],
            "objective": item["quality"]["objective"],
            "nonanticipativity": item["quality"]["nonanticipativity_residual"],
            "risk_epigraph": item["quality"]["risk_epigraph_residual"],
        }
        for item in results
        if item["family"] == "P1-F-robust-pd"
    ]
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for risk in sorted({item["risk_metric"] for item in robust}):
        selected = [item for item in robust if item["risk_metric"] == risk]
        axis.scatter(
            [item["scenarios"] for item in selected],
            [item["objective"] for item in selected],
            s=12,
            label=risk,
        )
    axis.set(
        title="Robust scenario risk reference",
        xlabel="scenario count S",
        ylabel="risk objective (dimensionless reference cost)",
        xscale="log",
    )
    axis.legend()
    _save(
        figure,
        output,
        "diag13_scenario_risk",
        _metadata(
            "D13",
            "Robust scenario risk reference",
            [
                {"name": "scenario count S", "unit": "count"},
                {"name": "risk objective", "unit": "dimensionless reference cost"},
            ],
            sorted({item["risk_metric"] for item in robust}),
            "Direct expected/worst/CVaR aggregation over every declared scenario; no sampling.",
            environment,
            [item["coordinate_id"] for item in robust],
            robust,
        ),
    )

    paper2 = [
        {
            "coordinate_id": item["coordinate_id"],
            "family": item["family"],
            "disposition": item["disposition"],
            "implementation": item["implementation"]["identifier"],
            "certified": item["quality"]["certified"],
        }
        for item in results
        if item["programme"] == "paper2"
    ]
    paper2_counts = [
        {
            "family": family,
            "disposition": disposition,
            "count": sum(
                item["family"] == family and item["disposition"] == disposition for item in paper2
            ),
        }
        for family in sorted({item["family"] for item in paper2})
        for disposition in dispositions
    ]
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    p2_families = sorted({item["family"] for item in paper2})
    bottoms = [0] * len(p2_families)
    for disposition in dispositions:
        values = [
            next(
                item["count"]
                for item in paper2_counts
                if item["family"] == family and item["disposition"] == disposition
            )
            for family in p2_families
        ]
        axis.bar(p2_families, values, bottom=bottoms, label=disposition)
        bottoms = [left + right for left, right in zip(bottoms, values, strict=True)]
    axis.set(
        title="Lambert/route/master/certification evidence disposition",
        xlabel="Paper 2 family",
        ylabel="coordinate count",
    )
    axis.legend()
    _save(
        figure,
        output,
        "diag14_lambert_route_certification",
        _metadata(
            "D14",
            "Lambert/route/master/certification evidence disposition",
            [
                {"name": "Paper 2 family", "unit": "identifier"},
                {"name": "coordinate count", "unit": "count"},
            ],
            dispositions,
            (
                "Exact retained disposition counts. Component-contract runs remain unqualified "
                "and are not presented as physical Lambert or route solutions."
            ),
            environment,
            [item["coordinate_id"] for item in paper2],
            paper2_counts,
        ),
    )


def _checksums(root: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _file_sha(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != "checksums.json"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    campaign = arguments.campaign.resolve()
    environment = json.loads((campaign / "environment.json").read_text())
    environment["completed_utc"] = (
        __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat()
    )
    _write(campaign / "environment.json", environment)
    schema = json.loads(
        (repository / "experiments/schema/cpu_reference_result.schema.json").read_text()
    )
    validator = jsonschema.Draft202012Validator(schema)
    results = []
    errors = []
    for path in sorted((campaign / "runs").rglob("result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        validation = sorted(validator.iter_errors(result), key=lambda item: list(item.path))
        if validation:
            errors.append(
                {
                    "path": path.relative_to(campaign).as_posix(),
                    "errors": [error.message for error in validation],
                }
            )
        results.append(result)
    index = json.loads((campaign / "coordinate-index.json").read_text())
    ids = [item["coordinate_id"] for item in results]
    missing = sorted(set(index["coordinate_ids"]) - set(ids))
    duplicate_count = len(ids) - len(set(ids))
    if errors or missing or duplicate_count or len(results) != index["count"]:
        raise RuntimeError(
            f"invalid campaign: schema_errors={len(errors)}, missing={len(missing)}, "
            f"duplicates={duplicate_count}, records={len(results)}/{index['count']}"
        )

    products = campaign / "products"
    if products.exists():
        shutil.rmtree(products)
    render_a, render_b = campaign / "_render-a", campaign / "_render-b"
    for path in (render_a, render_b):
        if path.exists():
            shutil.rmtree(path)
        _render(path, results, environment)
    left = _checksums(render_a)
    right = _checksums(render_b)
    if left != right:
        raise RuntimeError("diagnostic source/render bytes are not reproducible")
    shutil.move(render_a, products)
    shutil.rmtree(render_b)

    semantic_records = sorted(
        (_semantic(item) for item in results),
        key=lambda item: item["coordinate_id"],
    )
    source_digest = _sha(semantic_records)
    timing_digest = _sha(
        [
            {"coordinate_id": item["coordinate_id"], "timing": item["timing"]}
            for item in sorted(results, key=lambda item: item["coordinate_id"])
        ]
    )
    dispositions = Counter(item["disposition"] for item in results)
    failures = [
        item
        for item in results
        if item["disposition"] in {"timeout", "oom", "numerical", "infeasible", "failed"}
    ]
    maxima = {}
    for field in (
        "canonical_primal_residual",
        "canonical_dual_residual",
        "canonical_natural_residual",
        "canonical_cone_residual",
        "dynamics_residual",
        "path_residual",
        "terminal_residual",
        "continuous_time_violation",
        "virtual_control_residual",
        "nonanticipativity_residual",
        "risk_epigraph_residual",
    ):
        values = [
            float(item["quality"][field])
            for item in results
            if item["quality"][field] is not None and math.isfinite(float(item["quality"][field]))
        ]
        maxima[field] = max(values, default=None)
    family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in results:
        family_counts[item["family"]][item["disposition"]] += 1
    baseline_dashboard = arguments.baseline / "dashboard-summary.json"
    dashboard = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign.name,
        "source_commit": environment["source_commit"],
        "driver_commit": environment["driver_commit"],
        "gates": {
            "matrix_coverage": "pass",
            "record_schema": "pass",
            "source_reproducibility": "pass",
            "gpu_used": False,
            "paper1_publication_qualification": (
                "partial" if dispositions["executed"] else "censored"
            ),
            "paper2_physical_solution_evidence": "censored",
        },
        "counts": {
            "total": len(results),
            "dispositions": dict(sorted(dispositions.items())),
            "families": {
                family: dict(sorted(counts.items()))
                for family, counts in sorted(family_counts.items())
            },
        },
        "numerical_maxima": maxima,
        "failures": {
            "count": len(failures),
            "by_disposition": dict(
                sorted(Counter(item["disposition"] for item in failures).items())
            ),
        },
        "reproducibility": {
            "semantic_source_sha256": source_digest,
            "timing_observation_sha256": timing_digest,
            "render_checksums_sha256": _sha(left),
            "meaning": (
                "semantic hash excludes timing/resources/artifact locations; timing hash records "
                "the observed run and is not expected to match an independent rerun"
            ),
        },
        "baseline_archive": {
            "path": str(arguments.baseline),
            "dashboard_sha256": (
                _file_sha(baseline_dashboard) if baseline_dashboard.is_file() else None
            ),
        },
        "products": sorted(
            path.relative_to(campaign).as_posix()
            for path in products.iterdir()
            if path.suffix in {".json", ".png", ".pdf"}
        ),
        "g6_products": str(arguments.baseline / "g6-products/products"),
        "remaining_gpu_only_inputs": [
            "persistent CUDA PDHCG and GPU-IPM timing/memory/energy series",
            "physical 2/4/8-GPU robust scaling and collective telemetry",
        ],
        "remaining_cpu_gaps": [
            "native 6-DoF and low-thrust host optimizer dual/natural residual emitter",
            "parameterized physical Paper 2 Lambert/route/master/certification campaign owner",
            "full robust trajectory solve for every P1-F risk/prefix coordinate",
        ],
    }
    _write(campaign / "dashboard-summary.json", dashboard)
    _write(
        campaign / "validation-summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "records": len(results),
            "schema_errors": 0,
            "missing": 0,
            "duplicates": 0,
            "all_failures_retained": True,
            "semantic_source_sha256": source_digest,
        },
    )
    _write(campaign / "checksums.json", _checksums(campaign))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
