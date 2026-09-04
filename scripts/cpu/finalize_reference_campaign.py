#!/usr/bin/env python3
"""Finalize a fail-closed CPU/reference campaign from frozen manifests and test evidence."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import shutil
import socket
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from spacepdhcg.experiments import HostMetadata, RunManifest, write_paper1_result
from spacepdhcg.paper1.evidence import sha256_path, write_canonical_json
from spacepdhcg.paper1.freeze import build_campaign, verify_reproducible_build

SCHEMA_VERSION = "1.0.0"
FROZEN_COMMIT = "e95b902d718ceaf05523e469cbe21945013c2f41"
CPU_TEST_FAMILY = {
    "fixed_cqp_smoke": "P1-A-banded",
    "hcw_rendezvous_smoke": "P1-B-hcw",
    "powered_descent_driver_smoke": "P1-C-pd3",
    "powered_descent_6dof_transcription_smoke": "P1-D-pd6",
}
P2_FIXTURE_GROUPS = {
    "lambert": ("lambert",),
    "trajectory_oracle": ("trajectory_oracle", "low_thrust_oracle"),
    "robust_risk": ("robust", "scenario"),
    "routing_master": (
        "beam_search",
        "time_expanded_graph",
        "restricted_master",
        "column_generation",
    ),
    "certification": ("certification", "continuous_check"),
}


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _matrix_coordinates(
    programme: str,
    matrix: dict[str, Any],
    matrix_digest: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for family in matrix["families"]:
        axes = {
            key: value
            for key, value in family.items()
            if key not in {"id", "name"} and isinstance(value, list)
        }
        names = tuple(axes)
        for values in itertools.product(*(axes[name] for name in names)):
            parameters = dict(zip(names, values, strict=True))
            if programme == "paper1":
                classification = "unrun"
                reason = (
                    "host truth/model fixtures exist, but this commit has no production driver "
                    "that emits every frozen Cartesian coordinate with the required independent "
                    "Paper 1 residual, replay, repeat, and resource fields"
                )
            elif family["id"] in {"P2-D", "P2-E"}:
                classification = "unsupported"
                reason = (
                    "the frozen full mission family lacks a CPU matrix campaign driver at this "
                    "commit; only bounded component fixtures are implemented"
                )
            else:
                classification = "unrun"
                reason = (
                    "bounded CPU component fixtures are implemented, but no frozen-scale Paper 2 "
                    "campaign driver emits this full coordinate"
                )
            identity = {
                "programme": programme,
                "family": family["id"],
                "parameters": parameters,
            }
            records.append(
                {
                    "coordinate_id": hashlib.sha256(_canonical_bytes(identity)).hexdigest()[:24],
                    **identity,
                    "classification": classification,
                    "reason": reason,
                    "matrix_sha256": matrix_digest,
                }
            )
    return records


def _parse_junit(path: Path) -> list[dict[str, Any]]:
    root = ET.parse(path).getroot()
    records: list[dict[str, Any]] = []
    for case in root.iter("testcase"):
        failures = list(case.findall("failure")) + list(case.findall("error"))
        skipped = list(case.findall("skipped"))
        status = "failed" if failures else "skipped" if skipped else "passed"
        records.append(
            {
                "name": case.attrib.get("name", ""),
                "classname": case.attrib.get("classname", ""),
                "duration_seconds": float(case.attrib.get("time", "0") or 0.0),
                "status": status,
                "diagnostic": "\n".join((item.text or "").strip() for item in failures + skipped),
            }
        )
    return sorted(records, key=lambda item: (item["classname"], item["name"]))


def _test_for_family(tests: list[dict[str, Any]], token: str) -> dict[str, Any] | None:
    return next((test for test in tests if token in test["name"]), None)


def _dimensions(repository: Path) -> dict[str, dict[str, Any]]:
    from spacepdhcg.benchmarks.trajectory_banded import (
        TrajectoryBandedConfig,
        TrajectoryBandedFixture,
    )
    from spacepdhcg.models import CWRendezvousConfig, CWRendezvousProblem
    from spacepdhcg.transcription import PoweredDescent3DOFSubproblem, PoweredDescentSCvxConfig

    del repository
    banded = TrajectoryBandedFixture(TrajectoryBandedConfig(intervals=10, seed=17))
    hcw = CWRendezvousProblem(CWRendezvousConfig(intervals=20))
    pd3 = PoweredDescent3DOFSubproblem(config=PoweredDescentSCvxConfig(intervals=20))

    def from_structure(
        intervals: int,
        nx: int,
        nu: int,
        structure: Any,
        cone_inventory: dict[str, int],
    ) -> dict[str, Any]:
        return {
            "intervals": intervals,
            "scenarios": 1,
            "gpus": 0,
            "state_dimension": nx,
            "control_dimension": nu,
            "variables": structure.n_variables,
            "scalar_rows": structure.n_constraints,
            "affine_rows": structure.n_affine_constraints,
            "q_nonzeros": len(structure.quadratic.indices),
            "a_nonzeros": len(structure.constraint.indices),
            "f_nonzeros": (
                0 if structure.affine_cone is None else len(structure.affine_cone.indices)
            ),
            "cone_inventory": cone_inventory,
            "topology_bytes": None,
            "numeric_bytes": None,
        }

    return {
        "P1-A-banded": from_structure(10, 4, 3, banded.structure, {"box": 30}),
        "P1-B-hcw": from_structure(20, 6, 3, hcw.structure, {"box": 60}),
        "P1-C-pd3": from_structure(
            20,
            7,
            4,
            pd3.structure,
            {"soc": len(pd3.structure.affine_cones)},
        ),
        "P1-D-pd6": {
            "intervals": 3,
            "scenarios": 1,
            "gpus": 0,
            "state_dimension": 14,
            "control_dimension": 7,
            "variables": 161,
            "scalar_rows": 160,
            "affine_rows": 133,
            "q_nonzeros": 161,
            "a_nonzeros": 1183,
            "f_nonzeros": 122,
            "cone_inventory": {"native_affine_blocks": 18},
            "topology_bytes": None,
            "numeric_bytes": None,
            "attitude_class": 0.05,
            "rate_class": 0.05,
        },
    }


def _artifact(path: Path) -> dict[str, str]:
    return {"location": path.name, "sha256": sha256_path(path)}


def _empty_quality() -> dict[str, Any]:
    return {
        "qualified": False,
        "objective": None,
        "reference_objective": None,
        "objective_gap": None,
        "canonical_primal_residual": None,
        "canonical_dual_residual": None,
        "canonical_cone_residual": None,
        "canonical_gap": None,
        "native_primal_residual": None,
        "native_dual_residual": None,
        "dynamics_residual": None,
        "path_residual": None,
        "terminal_residual": None,
        "virtual_control_residual": None,
        "nonanticipativity_residual": None,
        "risk_epigraph_residual": None,
        "ct_error_estimate": None,
        "requested_tolerance": None,
        "achieved_residual": None,
        "independent_replay": False,
        "uses_solver_cached_residuals": False,
    }


def _empty_timing() -> dict[str, None]:
    return {
        name: None
        for name in (
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
            "accepted_trajectory_seconds",
        )
    }


def _write_g6_run(
    root: Path,
    family: str,
    dimensions: dict[str, Any],
    test: dict[str, Any] | None,
    environment: dict[str, Any],
) -> None:
    run_id = f"cpu-fixture-{family.lower().replace('-', '_')}"
    directory = root / run_id
    directory.mkdir(parents=True)
    status = "unqualified" if test and test["status"] == "passed" else "failed"
    reason = (
        "CPU fixture passed, but it does not emit the complete frozen Paper 1 residual/replay "
        "and 2-warmup/7-measured timing contract; retained as censored evidence"
        if status == "unqualified"
        else "CPU fixture did not pass or was absent; retained as failed evidence"
    )
    raw = directory / "raw-test.json"
    residual = directory / "residual-unavailable.json"
    replay = directory / "replay-unavailable.json"
    archive = directory / "archive-index.json"
    write_canonical_json(raw, test or {"status": "absent"})
    write_canonical_json(residual, {"available": False, "reason": reason})
    write_canonical_json(replay, {"available": False, "reason": reason, "independent": True})
    write_canonical_json(
        archive,
        {"run_id": run_id, "files": [raw.name, residual.name, replay.name]},
    )
    manifest = RunManifest(
        run_id=run_id,
        timestamp_utc=environment["time_range_utc"]["start"],
        repository={
            "url": "https://github.com/terrorproforma/spacecraft-trajectory-optmiser.git",
            "commit": FROZEN_COMMIT,
            "branch": "sim/cpu-reference-campaign",
            "dirty": False,
        },
        upstream={},
        host=HostMetadata(
            hostname=environment["host"]["hostname"],
            operating_system=environment["host"]["operating_system"],
            architecture=environment["host"]["architecture"],
            processor=environment["host"]["processor"],
            logical_cpu_count=environment["host"]["logical_cpu_count"],
            memory_bytes=environment["host"]["memory_bytes"],
            accelerator_vendor=None,
            accelerator_model=None,
            accelerator_count=0,
            driver_version=None,
            runtime_version=None,
            interconnect=None,
        ),
        experiment={
            "units": {
                "length": "metre",
                "time": "second",
                "mass": "kilogram",
                "angle": "radian",
                "force": "newton",
                "torque": "newton metre",
                "velocity": "metre per second",
                "acceleration": "metre per second squared",
                "energy": "joule",
                "memory": "byte",
            },
            "evidence_class": "cpu-component-fixture-censored",
        },
        problem={"family": family, "instance_id": f"{family}-component-fixture"},
        solver={"name": "clarabel-cpu", "mode": "fixture-only"},
        quality={"qualified": False},
        timing={"scvx_total_seconds": None},
        status="failed",
        artifacts={},
        notes=[reason],
    )
    manifest_path = manifest.write(directory / "run-manifest.json")
    result = {
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "run_id": run_id,
            "repository_commit": FROZEN_COMMIT,
            "family": family,
            "instance_id": f"{family}-component-fixture",
            "solver": "clarabel-cpu",
            "policy": "fixture-only",
            "status": status,
            "hardware_id": environment["host"]["hostname"],
            "precision": "float64",
            "warm_start": False,
            "cold_start": True,
            "failure_class": "evidence",
            "failure_reason": reason,
        },
        "dimensions": dimensions,
        "quality": _empty_quality(),
        "timing": _empty_timing(),
        "work": {
            "outer_iterations": None,
            "inner_iterations": None,
            "matvecs": None,
            "cone_projections": None,
            "factorisations": None,
            "accepted_steps": None,
            "rejected_steps": None,
            "resolved_steps": None,
            "polish_used": None,
            "scaling_refreshes": None,
        },
        "resources": {
            "peak_device_bytes": None,
            "reserved_device_bytes": None,
            "h2d_bytes": None,
            "d2h_bytes": None,
            "collective_bytes": None,
            "collective_count": None,
            "energy_joules": None,
            "topology_allocation_count_after_create": None,
            "load_imbalance": None,
            "throughput_per_second": None,
        },
        "aggregation": {
            "warmup_repeats": 2,
            "measured_repeats": 0,
            "statistic": "median_iqr",
            "median": None,
            "q1": None,
            "q3": None,
            "minimum": None,
            "maximum": None,
            "coefficient_of_variation": None,
            "censored_count": 1,
            "bootstrap_low": None,
            "bootstrap_high": None,
        },
        "artifacts": {
            "manifest": _artifact(manifest_path),
            "raw": _artifact(raw),
            "stdout": _artifact(raw),
            "stderr": _artifact(raw),
            "nsys": None,
            "ncu": None,
            "compute_sanitizer": None,
            "energy_trace": None,
        },
        "notes": [reason, "No GPU API, executable, timing, or energy sampler was invoked."],
    }
    result_path = write_paper1_result(result, directory / "paper1-result.json")
    references = {}
    for name, path in (("residual", residual), ("replay", replay), ("archive", archive)):
        digest = sha256_path(path)
        references[name] = {
            "uri": f"artifact://cpu-reference/{digest}/{path.name}",
            "sha256": digest,
            "internal_index_sha256": sha256_path(archive),
            "media_type": "application/json",
            "local_path": path.name,
        }
    write_canonical_json(
        directory / "evidence-record.json",
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "manifest_sha256": sha256_path(manifest_path),
            "result_sha256": sha256_path(result_path),
            "residual_evidence": references["residual"],
            "replay_evidence": references["replay"],
            "archive": references["archive"],
        },
    )


def _chart_metadata(
    chart_id: str,
    title: str,
    axes: list[dict[str, str]],
    series: list[str],
    caption: str,
    environment: dict[str, Any],
    data: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "chart_id": chart_id,
        "title": title,
        "axes": axes,
        "series_legend": series,
        "source_commit": FROZEN_COMMIT,
        "source_time_range_utc": environment["time_range_utc"],
        "transformation_aggregation_caption": caption,
        "data": data,
    }


def _save_figure(figure: Any, destination: Path, stem: str, metadata: dict[str, Any]) -> None:
    write_canonical_json(destination / f"{stem}.json", metadata)
    pdf_metadata = {
        "Title": metadata["title"],
        "Author": "SpacePDHCG CPU reference campaign",
        "Creator": "finalize_reference_campaign.py",
        "Producer": "finalize_reference_campaign.py",
        "CreationDate": None,
        "ModDate": None,
    }
    figure.savefig(destination / f"{stem}.pdf", metadata=pdf_metadata)
    figure.savefig(
        destination / f"{stem}.png",
        dpi=160,
        metadata={"Software": "SpacePDHCG CPU reference campaign"},
    )
    plt.close(figure)


def _render_diagnostics(
    destination: Path,
    ledger: list[dict[str, Any]],
    tests: list[dict[str, Any]],
    environment: dict[str, Any],
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    counts = Counter((item["programme"], item["classification"]) for item in ledger)
    data = [
        {"programme": programme, "classification": status, "count": count}
        for (programme, status), count in sorted(counts.items())
    ]
    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    statuses = sorted({item["classification"] for item in data})
    programmes = ["paper1", "paper2"]
    bottoms = [0] * len(programmes)
    for status in statuses:
        values = [
            next(
                (
                    item["count"]
                    for item in data
                    if item["programme"] == programme and item["classification"] == status
                ),
                0,
            )
            for programme in programmes
        ]
        axis.bar(programmes, values, bottom=bottoms, label=status)
        bottoms = [left + right for left, right in zip(bottoms, values, strict=True)]
    axis.set(title="Frozen CPU/reference coordinate disposition", ylabel="coordinate count")
    axis.legend()
    _save_figure(
        figure,
        destination,
        "diag01_coordinate_disposition",
        _chart_metadata(
            "D01",
            "Frozen CPU/reference coordinate disposition",
            [
                {"name": "programme", "unit": "identifier"},
                {"name": "coordinate count", "unit": "count"},
            ],
            statuses,
            "Exact Cartesian expansion of frozen family axes; no filtering or dropped failures.",
            environment,
            data,
        ),
    )

    durations = [test["duration_seconds"] for test in tests if test["duration_seconds"] >= 0]
    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    axis.hist(durations, bins=min(20, max(1, len(durations))), label="native CPU fixture")
    axis.set(
        title="Native CPU correctness fixture duration distribution",
        xlabel="CTest wall duration (second)",
        ylabel="test count",
    )
    axis.legend()
    duration_data = [
        {
            "test": test["name"],
            "duration_seconds": test["duration_seconds"],
            "status": test["status"],
        }
        for test in tests
    ]
    _save_figure(
        figure,
        destination,
        "diag02_native_test_durations",
        _chart_metadata(
            "D02",
            "Native CPU correctness fixture duration distribution",
            [
                {"name": "CTest wall duration", "unit": "second"},
                {"name": "test count", "unit": "count"},
            ],
            ["native CPU fixture"],
            (
                "Unweighted histogram of all retained JUnit test durations; "
                "not solver timing evidence."
            ),
            environment,
            duration_data,
        ),
    )

    grouped = []
    for group, tokens in P2_FIXTURE_GROUPS.items():
        selected = [test for test in tests if any(token in test["name"] for token in tokens)]
        grouped.append(
            {
                "group": group,
                "passed": sum(test["status"] == "passed" for test in selected),
                "failed": sum(test["status"] == "failed" for test in selected),
                "skipped": sum(test["status"] == "skipped" for test in selected),
                "tests": [test["name"] for test in selected],
            }
        )
    figure, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    groups = [item["group"] for item in grouped]
    bottoms = [0] * len(groups)
    colors = {"passed": "#55a868", "failed": "#c44e52", "skipped": "#8172b3"}
    for status in ("passed", "failed", "skipped"):
        values = [item[status] for item in grouped]
        axis.bar(groups, values, bottom=bottoms, label=status, color=colors[status])
        bottoms = [left + right for left, right in zip(bottoms, values, strict=True)]
    axis.set(
        title="Lambert, trajectory, risk, routing/master, and certification fixtures",
        xlabel="fixture group",
        ylabel="test count",
    )
    axis.tick_params(axis="x", rotation=20)
    axis.legend()
    _save_figure(
        figure,
        destination,
        "diag03_orbitweaver_fixture_flow",
        _chart_metadata(
            "D03",
            "Lambert, trajectory, risk, routing/master, and certification fixtures",
            [
                {"name": "fixture group", "unit": "identifier"},
                {"name": "test count", "unit": "count"},
            ],
            ["passed", "failed", "skipped"],
            (
                "Exact name-based partition of retained native JUnit fixtures; pass counts show "
                "component correctness only and make no scale/performance claim."
            ),
            environment,
            grouped,
        ),
    )


def _tree_hash(root: Path) -> str:
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_path(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    return _sha(entries)


def _checksums(root: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_path(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != "checksums.json"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ctest-junit", type=Path, required=True)
    parser.add_argument("--native-core-junit", type=Path, required=True)
    parser.add_argument("--ctest-inventory", type=Path, required=True)
    parser.add_argument("--native-log", type=Path, required=True)
    parser.add_argument("--native-core-log", type=Path, required=True)
    parser.add_argument("--python-log", type=Path, required=True)
    parser.add_argument("--start-utc", required=True)
    arguments = parser.parse_args()

    repository = arguments.repository.resolve()
    output = arguments.output.resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("CPU campaign requires CUDA_VISIBLE_DEVICES=-1 or empty")
    if os.popen(f"git -C {repository} rev-parse HEAD").read().strip() != FROZEN_COMMIT:
        raise RuntimeError("campaign worktree is not at the frozen commit")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    paper1_path = repository / "benchmarks/paper1_matrix.json"
    paper2_path = repository / "benchmarks/paper2_matrix.json"
    paper1_digest = sha256_path(paper1_path)
    paper2_digest = sha256_path(paper2_path)
    ledger = _matrix_coordinates("paper1", _load(paper1_path), paper1_digest)
    ledger += _matrix_coordinates("paper2", _load(paper2_path), paper2_digest)
    write_canonical_json(
        output / "coordinate-ledger.json",
        {
            "schema_version": SCHEMA_VERSION,
            "source_commit": FROZEN_COMMIT,
            "matrix_sha256": {"paper1": paper1_digest, "paper2": paper2_digest},
            "records": ledger,
            "classification_counts": dict(
                sorted(Counter(item["classification"] for item in ledger).items())
            ),
        },
    )

    tests = _parse_junit(arguments.ctest_junit) + _parse_junit(arguments.native_core_junit)
    tests.sort(key=lambda item: (item["classname"], item["name"]))
    shutil.copy2(arguments.ctest_junit, output / "native-ctest.xml")
    shutil.copy2(arguments.native_core_junit, output / "native-core-ctest.xml")
    shutil.copy2(arguments.ctest_inventory, output / "native-command-inventory.json")
    shutil.copy2(arguments.native_log, output / "native-tests.log")
    shutil.copy2(arguments.native_core_log, output / "native-core-tests.log")
    shutil.copy2(arguments.python_log, output / "python-tests.log")
    memory_bytes = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
                if line.startswith("MemTotal:")
            )
        )
        * 1024
    )
    now = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    environment = {
        "schema_version": SCHEMA_VERSION,
        "repository_commit": FROZEN_COMMIT,
        "branch": "sim/cpu-reference-campaign",
        "gpu_visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_used": False,
        "host": {
            "hostname": socket.gethostname(),
            "operating_system": platform.platform(),
            "architecture": platform.machine(),
            "processor": platform.processor() or "unknown",
            "logical_cpu_count": os.cpu_count() or 1,
            "memory_bytes": memory_bytes,
            "python": sys.version,
        },
        "time_range_utc": {"start": arguments.start_utc, "end": now},
        "commands": [
            "cmake -S cpp -B build-cpu-campaign -G Ninja -DSPACEPDHCG_BUILD_CUDA=OFF "
            "-DSPACEPDHCG_BUILD_DISTRIBUTED=OFF -DSPACEPDHCG_BUILD_NATIVE_TESTS=ON",
            "cmake --build build-cpu-campaign --parallel 12",
            "ctest --test-dir build-cpu-campaign --parallel 12 --output-junit native-ctest.xml",
            "cmake -S cpp/native -B build-cpu-native-core -G Ninja -DCMAKE_BUILD_TYPE=Release",
            "cmake --build build-cpu-native-core --parallel 12",
            "ctest --test-dir build-cpu-native-core --parallel 8 "
            "--output-junit native-core-ctest.xml",
            "python -m pytest -q",
        ],
    }
    write_canonical_json(output / "environment.json", environment)
    write_canonical_json(
        output / "fixture-results.json",
        {
            "schema_version": SCHEMA_VERSION,
            "tests": tests,
            "status_counts": dict(sorted(Counter(test["status"] for test in tests).items())),
            "duration_seconds": {
                "minimum": min((test["duration_seconds"] for test in tests), default=0.0),
                "median": statistics.median([test["duration_seconds"] for test in tests] or [0.0]),
                "maximum": max((test["duration_seconds"] for test in tests), default=0.0),
            },
        },
    )

    g6_campaign = output / "g6-campaign"
    dimensions = _dimensions(repository)
    for token, family in CPU_TEST_FAMILY.items():
        _write_g6_run(
            g6_campaign,
            family,
            dimensions[family],
            _test_for_family(tests, token),
            environment,
        )
    g6_build = build_campaign(g6_campaign, output / "g6-products", synthetic=False)
    g6_reproducibility = verify_reproducible_build(g6_campaign, synthetic=False)

    render_a, render_b = output / "_render-a", output / "_render-b"
    _render_diagnostics(render_a, ledger, tests, environment)
    _render_diagnostics(render_b, ledger, tests, environment)
    hash_a, hash_b = _tree_hash(render_a), _tree_hash(render_b)
    if hash_a != hash_b:
        raise RuntimeError(f"diagnostic rendering is not reproducible: {hash_a} != {hash_b}")
    shutil.copytree(render_a, output / "diagnostics")
    shutil.rmtree(render_a)
    shutil.rmtree(render_b)

    products = _load(output / "g6-products/products/build-manifest.json")
    mapped = Counter(run_id for product in products["products"] for run_id in product["run_ids"])
    archived_run_ids = sorted(
        path.parent.name for path in g6_campaign.rglob("evidence-record.json")
    )
    omitted = [run_id for run_id in archived_run_ids if mapped[run_id] == 0]
    if omitted:
        raise RuntimeError(f"G6 omitted archived censored runs: {omitted}")

    fixture_counts = Counter(test["status"] for test in tests)
    coordinate_counts = Counter(item["classification"] for item in ledger)
    dashboard = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": "cpu-reference-e95b902",
        "source_commit": FROZEN_COMMIT,
        "gates": {
            "frozen_matrix_integrity": "pass",
            "native_cpu_fixtures": "pass" if not fixture_counts["failed"] else "fail",
            "python_tests": (
                "pass" if "failed" not in arguments.python_log.read_text() else "review"
            ),
            "paper1_full_matrix": "censored",
            "paper2_full_matrix": "censored",
            "g4": "not_changed",
            "g5": "not_authorized",
            "gpu_inputs": "missing_by_design",
            "reproducible_render": "pass",
        },
        "counts": {
            "matrix_coordinates": len(ledger),
            "executed_native_fixtures": len(tests),
            "fixture_statuses": dict(sorted(fixture_counts.items())),
            "coordinate_classifications": dict(sorted(coordinate_counts.items())),
            "g6_archived_censored_runs": len(archived_run_ids),
        },
        "numerical_maxima": {
            "native_fixture_duration_seconds": max(
                (test["duration_seconds"] for test in tests), default=0.0
            ),
            "canonical_residual": None,
            "nonlinear_violation": None,
            "note": (
                "native fixtures at this commit do not emit complete frozen numerical fields; "
                "values remain null instead of being inferred"
            ),
        },
        "timing_distributions": _load(output / "fixture-results.json")["duration_seconds"],
        "charts": sorted(
            path.relative_to(output).as_posix()
            for path in (output / "diagnostics").glob("*")
            if path.suffix in {".json", ".png", ".pdf"}
        ),
        "g6_product_manifest": "g6-products/products/build-manifest.json",
        "remaining_inputs": [
            "complete frozen Paper 1 CPU matrix drivers and independent metric emitters",
            "serialized one-GPU G4 correctness/performance evidence",
            "physical 2/4/8-GPU G5 correctness and scaling evidence",
            "full frozen Paper 2 route/multi-spacecraft/robust campaign drivers",
        ],
        "reproducibility": {
            "diagnostic_hash_first": hash_a,
            "diagnostic_hash_second": hash_b,
            "g6": g6_reproducibility,
        },
    }
    write_canonical_json(output / "dashboard-summary.json", dashboard)
    write_canonical_json(output / "checksums.json", _checksums(output))
    write_canonical_json(
        output / "campaign-summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "source_commit": FROZEN_COMMIT,
            "coordinate_counts": dict(sorted(coordinate_counts.items())),
            "fixture_counts": dict(sorted(fixture_counts.items())),
            "g6": g6_build,
            "reproducibility": dashboard["reproducibility"],
            "dashboard": "dashboard-summary.json",
            "checksums": "checksums.json",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
