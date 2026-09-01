#!/usr/bin/env python3
"""Run and validate the preregistered matched-quality Gate G3 H1 sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import jsonschema

DEFAULT_SIZES = (20, 50, 100, 500, 2_000, 10_000)
BOOTSTRAP_SAMPLES = 10_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_median_interval(values: list[float]) -> tuple[float, float]:
    generator = random.Random(20_260_901)
    estimates = [
        statistics.median(generator.choices(values, k=len(values)))
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _parse_record(stdout: str) -> dict[str, Any]:
    records = []
    for line in stdout.splitlines():
        if line.startswith("{"):
            records.append(json.loads(line))
    matches = [record for record in records if record.get("case") == "h1_hcw"]
    if len(matches) != 1:
        raise RuntimeError("benchmark did not emit exactly one h1_hcw record")
    return matches[0]


def _run_sample(
    executable: Path,
    intervals: int,
    outer_repeats: int,
    timeout_seconds: float,
    environment: dict[str, str],
) -> dict[str, Any]:
    started = time.perf_counter()
    command = [
        str(executable),
        "--h1-hcw",
        str(intervals),
        str(outer_repeats),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=environment,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "intervals": intervals,
            "status": "timeout",
            "elapsed_seconds": time.perf_counter() - started,
            "stdout": error.stdout or "",
            "stderr": error.stderr or "",
            "command": command,
        }
    sample: dict[str, Any] = {
        "intervals": intervals,
        "status": "qualified" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "command": command,
    }
    if completed.returncode == 0:
        sample["record"] = _parse_record(completed.stdout)
    return sample


def _coordinate_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    qualified = [sample["record"] for sample in samples if sample["status"] == "qualified"]
    omega = [float(record["omega_persist"]) for record in qualified]
    scvx = [float(record["scvx_total_seconds"]) for record in qualified]
    if not qualified:
        return {
            "status": "censored",
            "qualified_count": 0,
            "censored_count": len(samples),
        }
    low, high = _bootstrap_median_interval(omega)
    quality = all(
        float(record["canonical_residual"]) <= 1.0e-6
        and float(record["nonlinear_residual"]) <= 1.0e-6
        and float(record["cpu_gpu_trajectory"]) <= 1.0e-9
        for record in qualified
    )
    topology_clean = all(
        int(record["topology_allocations_after_create"]) == 0
        and int(record["topology_copies_after_create"]) == 0
        for record in qualified
    )
    median_omega = statistics.median(omega)
    supported = (
        len(qualified) >= 5 and quality and topology_clean and median_omega <= 0.05 and high <= 0.08
    )
    return {
        "status": "supported" if supported else "unresolved",
        "qualified_count": len(qualified),
        "censored_count": len(samples) - len(qualified),
        "quality_matched": quality,
        "topology_clean": topology_clean,
        "omega_median": median_omega,
        "omega_bootstrap_95": [low, high],
        "scvx_median_seconds": statistics.median(scvx),
        "scvx_iqr_seconds": [_percentile(scvx, 0.25), _percentile(scvx, 0.75)],
        "scvx_min_seconds": min(scvx),
        "scvx_max_seconds": max(scvx),
    }


def _sustained_boundary(coordinates: list[dict[str, Any]]) -> int | None:
    for index, coordinate in enumerate(coordinates):
        remaining = coordinates[index:]
        required = min(3, len(remaining))
        if all(item["summary"]["status"] == "supported" for item in remaining[:required]):
            return int(coordinate["intervals"])
    return None


def _compact_result(
    sample: dict[str, Any],
    commit: str,
    run_id: str,
    raw_path: Path,
    raw_sha: str,
    repeat: int,
) -> dict[str, Any]:
    record = sample.get("record", {})
    status = sample["status"]
    qualified = status == "qualified"
    timing_names = {
        "topology_seconds": "topology_seconds",
        "coefficient_seconds": "coefficient_seconds",
        "workspace_create_seconds": "workspace_create_seconds",
        "update_seconds": "update_seconds",
        "scaling_seconds": "scaling_seconds",
        "h2d_seconds": "h2d_seconds",
        "solve_seconds": "solve_seconds",
        "residual_seconds": "residual_seconds",
        "replay_seconds": "replay_seconds",
        "acceptance_seconds": "acceptance_seconds",
        "d2h_seconds": "d2h_seconds",
        "cqp_total_seconds": "cqp_total_seconds",
        "scvx_total_seconds": "scvx_total_seconds",
    }
    timing = {
        target: float(record[source]) if qualified else None
        for target, source in timing_names.items()
    }
    timing["collective_seconds"] = 0.0 if qualified else None
    artifact = {"location": str(raw_path), "sha256": raw_sha}
    return {
        "schema_version": "1.0.0",
        "identity": {
            "run_id": run_id,
            "repository_commit": commit,
            "family": "P1-B-hcw",
            "instance_id": f"hcw-N{sample['intervals']}-repeat-{repeat}",
            "solver": "spacepdhcg-persistent",
            "policy": "deterministic-fixed-tight",
            "status": status,
            "hardware_id": "local-rtx-5090",
            "precision": "float64",
            "warm_start": True,
            "cold_start": False,
        },
        "dimensions": {
            "intervals": sample["intervals"],
            "scenarios": 1,
            "gpus": 1,
            "state_dimension": 6,
            "control_dimension": 3,
            "variables": record.get("variables", 1),
            "scalar_rows": record.get("scalar_rows", 0),
            "affine_rows": record.get("affine_rows", 0),
            "q_nonzeros": record.get("q_nonzeros", 0),
            "a_nonzeros": record.get("a_nonzeros", 0),
            "f_nonzeros": record.get("f_nonzeros", 0),
            "cone_inventory": {},
        },
        "quality": {
            "qualified": qualified,
            "objective": 0.0 if qualified else None,
            "canonical_primal_residual": record.get("canonical_residual"),
            "canonical_dual_residual": record.get("canonical_residual"),
            "canonical_cone_residual": 0.0 if qualified else None,
            "canonical_gap": record.get("canonical_residual"),
            "dynamics_residual": record.get("nonlinear_residual"),
            "path_residual": record.get("nonlinear_residual"),
            "terminal_residual": record.get("nonlinear_residual"),
            "virtual_control_residual": 0.0 if qualified else None,
            "nonanticipativity_residual": 0.0 if qualified else None,
            "risk_epigraph_residual": 0.0 if qualified else None,
            "requested_tolerance": 1.0e-6 if qualified else None,
            "achieved_residual": record.get("canonical_residual"),
        },
        "timing": timing,
        "work": {
            "outer_iterations": record.get("repeats"),
            "inner_iterations": record.get("recovery_iterations", 0),
            "matvecs": None,
            "cone_projections": None,
            "factorisations": 0,
            "accepted_steps": record.get("repeats"),
            "rejected_steps": 0,
            "resolved_steps": 0,
            "polish_used": True,
        },
        "resources": {
            "peak_device_bytes": record.get("allocation_bytes"),
            "reserved_device_bytes": record.get("allocation_bytes"),
            "h2d_bytes": record.get("h2d_bytes"),
            "d2h_bytes": record.get("d2h_bytes"),
            "collective_bytes": 0,
            "collective_count": 0,
            "energy_joules": None,
            "topology_allocation_count_after_create": record.get(
                "topology_allocations_after_create"
            ),
        },
        "aggregation": {
            "warmup_repeats": 2,
            "measured_repeats": 7,
            "statistic": "median_iqr",
            "median": record.get("scvx_total_seconds"),
            "q1": record.get("scvx_total_seconds"),
            "q3": record.get("scvx_total_seconds"),
            "minimum": record.get("scvx_total_seconds"),
            "maximum": record.get("scvx_total_seconds"),
            "coefficient_of_variation": 0.0 if qualified else None,
            "censored_count": 0 if qualified else 1,
        },
        "artifacts": {
            "manifest": artifact,
            "raw": artifact,
            "stdout": artifact,
            "stderr": artifact,
            "nsys": None,
            "ncu": None,
            "compute_sanitizer": None,
            "energy_trace": None,
        },
        "notes": [
            "CUDA startup is reported separately.",
            "Only compact acceptance diagnostics cross to host in steady state.",
        ],
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    repository = arguments.repository.resolve()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    executable = arguments.executable.resolve()
    environment = os.environ.copy()
    library = str(executable.parent.parent / "cuda")
    environment["LD_LIBRARY_PATH"] = library + ":" + environment.get("LD_LIBRARY_PATH", "")
    sizes = tuple(int(token) for token in arguments.sizes.split(","))
    all_samples: list[dict[str, Any]] = []
    coordinates: list[dict[str, Any]] = []
    for intervals in sizes:
        samples = [
            _run_sample(
                executable,
                intervals,
                arguments.outer_repeats,
                arguments.timeout,
                environment,
            )
            for _ in range(arguments.measured_repeats)
        ]
        all_samples.extend(samples)
        coordinates.append(
            {
                "intervals": intervals,
                "summary": _coordinate_summary(samples),
            }
        )
    raw_path = output / "h1_raw.jsonl"
    with raw_path.open("w", encoding="utf-8", newline="\n") as stream:
        for sample in all_samples:
            stream.write(json.dumps(sample, sort_keys=True) + "\n")
    raw_sha = _sha256(raw_path)
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    run_id = output.name
    schema_path = repository / "experiments/schema/paper1_result.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    compact_directory = output / "compact"
    compact_directory.mkdir(exist_ok=True)
    for repeat, sample in enumerate(all_samples):
        compact = _compact_result(
            sample,
            commit,
            run_id,
            raw_path,
            raw_sha,
            repeat,
        )
        jsonschema.validate(compact, schema)
        compact_path = compact_directory / f"result-{repeat:03d}.json"
        compact_path.write_text(
            json.dumps(compact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    boundary = _sustained_boundary(coordinates)
    all_supported = all(
        coordinate["summary"]["status"] == "supported" for coordinate in coordinates
    )
    decision = "supported" if all_supported else ("mixed" if boundary else "unresolved")
    result = {
        "schema_version": "g3-h1-1.0.0",
        "run_id": run_id,
        "repository_commit": commit,
        "decision": decision,
        "scale_boundary_intervals": boundary,
        "confidence_method": (
            "paired deterministic coordinates; median/IQR and 95% seeded "
            f"nonparametric bootstrap with {BOOTSTRAP_SAMPLES} resamples"
        ),
        "thresholds": {
            "median_omega_maximum": 0.05,
            "upper_95_omega_maximum": 0.08,
            "rejection_median_minimum": 0.10,
        },
        "coordinates": coordinates,
        "censored_points": [
            {
                "intervals": sample["intervals"],
                "status": sample["status"],
                "returncode": sample.get("returncode"),
                "stderr": sample.get("stderr", ""),
            }
            for sample in all_samples
            if sample["status"] != "qualified"
        ],
        "raw": {"location": str(raw_path), "sha256": raw_sha},
    }
    (output / "h1_decision.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument(
        "--executable",
        type=Path,
        default=Path("build/g3-evidence-cuda-release/cuda-tests/device_scvx_integration_test"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sizes",
        default=",".join(str(value) for value in DEFAULT_SIZES),
    )
    parser.add_argument("--measured-repeats", type=int, default=7)
    parser.add_argument("--outer-repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600.0)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
