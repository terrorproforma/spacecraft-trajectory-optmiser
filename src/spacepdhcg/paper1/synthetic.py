"""Clearly labelled synthetic campaign fixtures for G6 tooling demonstrations."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Final

from spacepdhcg.experiments import HostMetadata, RunManifest, write_paper1_result

from .evidence import sha256_path, write_canonical_json
from .freeze import SI_UNITS

SYNTHETIC_COMMIT: Final = "d" * 40
SYNTHETIC_INDEX_DIGEST: Final = "e" * 64


def _timing(total: float | None) -> dict[str, float | None]:
    names = (
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
    if total is None:
        values = {name: None for name in names}
    else:
        weights = (0.02, 0.15, 0.03, 0.05, 0.03, 0.02, 0.5, 0.05, 0.08, 0.03, 0.01, 0.03)
        values = {name: total * weight for name, weight in zip(names, weights, strict=True)}
    return {
        **values,
        "cqp_total_seconds": None if total is None else total * 0.8,
        "scvx_total_seconds": total,
        "accepted_trajectory_seconds": total,
    }


def _artifact(uri: str) -> dict[str, Any]:
    return {"location": uri, "sha256": "a" * 64}


def _result(
    *,
    run_id: str,
    family: str,
    solver: str,
    policy: str,
    status: str,
    intervals: int,
    scenarios: int = 1,
    gpus: int = 1,
    total: float | None,
    peak_bytes: int | None = 1_000_000,
    qualified: bool | None = None,
) -> dict[str, Any]:
    is_qualified = status == "qualified" if qualified is None else qualified
    residual = 1e-7 if is_qualified else None
    quantiles = {
        "median": total,
        "q1": None if total is None else total * 0.95,
        "q3": None if total is None else total * 1.05,
        "minimum": None if total is None else total * 0.9,
        "maximum": None if total is None else total * 1.1,
    }
    result = {
        "schema_version": "1.0.0",
        "identity": {
            "run_id": run_id,
            "repository_commit": SYNTHETIC_COMMIT,
            "family": family,
            "instance_id": f"synthetic-{family.lower()}-n{intervals}-s{scenarios}",
            "solver": solver,
            "policy": policy,
            "status": status,
            "hardware_id": "synthetic-host-not-experiment-evidence",
            "precision": "float64",
            "warm_start": True,
            "cold_start": False,
            "quality_tier": "tight",
            "warm_mode": "primal",
            "repeat": 0,
        },
        "dimensions": {
            "intervals": intervals,
            "scenarios": scenarios,
            "gpus": gpus,
            "state_dimension": 7,
            "control_dimension": 4,
            "variables": intervals * scenarios * 12,
            "scalar_rows": intervals * scenarios * 9,
            "affine_rows": intervals * scenarios * 5,
            "q_nonzeros": intervals * scenarios * 12,
            "a_nonzeros": intervals * scenarios * 40,
            "f_nonzeros": intervals * scenarios * 8,
            "cone_inventory": {"soc": intervals * scenarios},
            "topology_bytes": intervals * scenarios * 800,
            "numeric_bytes": intervals * scenarios * 1200,
        },
        "quality": {
            "qualified": is_qualified,
            "objective": 12.0 if is_qualified else None,
            "reference_objective": 12.0 if is_qualified else None,
            "objective_gap": residual,
            "canonical_primal_residual": residual,
            "canonical_dual_residual": residual,
            "canonical_cone_residual": residual,
            "canonical_gap": residual,
            "native_primal_residual": residual,
            "native_dual_residual": residual,
            "dynamics_residual": residual,
            "path_residual": residual,
            "terminal_residual": residual,
            "virtual_control_residual": residual,
            "nonanticipativity_residual": residual,
            "risk_epigraph_residual": residual,
            "ct_error_estimate": residual,
            "requested_tolerance": 1e-6,
            "achieved_residual": residual,
        },
        "timing": _timing(total),
        "work": {
            "outer_iterations": 5 if total is not None else None,
            "inner_iterations": 100 if total is not None else None,
            "matvecs": 200 if total is not None else None,
            "cone_projections": 100 if total is not None else None,
            "factorisations": 0 if total is not None else None,
            "accepted_steps": 4 if total is not None else None,
            "rejected_steps": 1 if total is not None else None,
            "resolved_steps": 0 if total is not None else None,
            "polish_used": solver == "hybrid-pdhcg-ipm",
            "scaling_refreshes": 1 if total is not None else None,
        },
        "resources": {
            "peak_device_bytes": peak_bytes,
            "reserved_device_bytes": None if peak_bytes is None else int(peak_bytes * 1.1),
            "h2d_bytes": intervals * 8 if total is not None else None,
            "d2h_bytes": 128 if total is not None else None,
            "collective_bytes": scenarios * 64 if scenarios > 1 and total is not None else 0,
            "collective_count": scenarios if scenarios > 1 and total is not None else 0,
            "energy_joules": None if total is None else total * 50,
            "topology_allocation_count_after_create": (
                0 if solver == "spacepdhcg-persistent" and is_qualified else None
            ),
            "load_imbalance": 1.08 if scenarios > 1 and total is not None else 1.0,
            "throughput_per_second": None if not total else 1.0 / total,
        },
        "aggregation": {
            "warmup_repeats": 2,
            "measured_repeats": 7 if is_qualified else 0,
            "statistic": "median_iqr",
            **quantiles,
            "coefficient_of_variation": 0.02 if total is not None else None,
            "censored_count": 0 if is_qualified else 1,
            "bootstrap_low": None,
            "bootstrap_high": None,
        },
        "artifacts": {
            "manifest": _artifact(f"artifact://synthetic/{run_id}/manifest"),
            "raw": _artifact(f"artifact://synthetic/{run_id}/raw"),
            "stdout": _artifact(f"artifact://synthetic/{run_id}/stdout"),
            "stderr": _artifact(f"artifact://synthetic/{run_id}/stderr"),
            "nsys": None,
            "ncu": None,
            "compute_sanitizer": None,
            "energy_trace": None,
        },
        "notes": ["SYNTHETIC FIXTURE ONLY — not experiment evidence and not a scientific result."],
    }
    return result


def _manifest(result: dict[str, Any]) -> RunManifest:
    identity = result["identity"]
    status_map = {
        "qualified": "qualified",
        "infeasible": "infeasible",
        "unrun": "skipped",
    }
    experiment: dict[str, Any] = {
        "synthetic": True,
        "gpu_memory_bytes": 80 * 1024**3,
        "analytic_collective_bytes": result["resources"]["collective_bytes"],
        "scaling_kind": "strong",
    }
    total_seconds = result["timing"]["scvx_total_seconds"]
    if total_seconds is not None:
        experiment["measured_repeat_seconds"] = [
            total_seconds * factor for factor in (0.94, 0.97, 0.99, 1.0, 1.01, 1.03, 1.06)
        ]
    model = {
        "P1-C-pd3": "3-DoF",
        "P1-D-pd6": "6-DoF",
        "P1-E-low-thrust": "low-thrust",
    }.get(identity["family"])
    if model is not None:
        experiment["variational_trials"] = [
            {
                "trial": trial,
                "model": model,
                "maximum_absolute_difference": (trial + 1) * 1e-9,
                "maximum_relative_difference": (trial + 1) * 2e-9,
                "analytic_fill_seconds": (trial + 1) * 1e-4,
                "finite_difference_fill_seconds": (trial + 1) * 8e-4,
                "quaternion_radial_sensitivity": (
                    (trial + 1) * 5e-10 if model == "6-DoF" else None
                ),
                "declared_tolerance": 1e-6,
            }
            for trial in range(3)
        ]
    if identity["family"] == "P1-F-robust-pd":
        experiment["robust_iterations"] = [
            {
                "risk_mode": risk_mode,
                "outer_iteration": iteration,
                "dynamics_residual": 1e-4 / (iteration + 1),
                "path_residual": 2e-4 / (iteration + 1),
                "terminal_residual": 3e-4 / (iteration + 1),
                "virtual_control_residual": 4e-5 / (iteration + 1),
                "nonanticipativity_residual": 5e-5 / (iteration + 1),
                "risk_epigraph_residual": 6e-5 / (iteration + 1),
                "canonical_kkt_residual": 7e-5 / (iteration + 1),
                "accepted": iteration != 1,
                "trust_radius": 1.0 / (iteration + 1),
            }
            for risk_mode in ("expected", "worst-case", "CVaR")
            for iteration in range(3)
        ]
    return RunManifest(
        run_id=identity["run_id"],
        timestamp_utc="2026-09-01T00:00:00Z",
        repository={
            "url": "https://example.invalid/synthetic-only",
            "commit": SYNTHETIC_COMMIT,
            "branch": "synthetic",
            "dirty": False,
        },
        upstream={"synthetic": True},
        host=HostMetadata(
            hostname="synthetic-host",
            operating_system="synthetic",
            architecture="x86_64",
            processor="synthetic-cpu",
            logical_cpu_count=32,
            memory_bytes=128 * 1024**3,
            accelerator_vendor="synthetic",
            accelerator_model="synthetic-gpu",
            accelerator_count=result["dimensions"]["gpus"],
            driver_version="synthetic",
            runtime_version="synthetic",
            interconnect="synthetic",
        ),
        experiment=experiment,
        problem={
            "family": identity["family"],
            "instance_id": identity["instance_id"],
        },
        solver={"name": identity["solver"], "mode": identity["policy"]},
        quality={"qualified": result["quality"]["qualified"]},
        timing={"scvx_total_seconds": result["timing"]["scvx_total_seconds"]},
        status=status_map.get(identity["status"], "failed"),
        artifacts={},
        notes=["SYNTHETIC FIXTURE ONLY"],
    )


def _write_run(root: Path, result: dict[str, Any]) -> None:
    run_id = result["identity"]["run_id"]
    directory = root / run_id
    directory.mkdir(parents=True)
    manifest_path = _manifest(result).write(directory / "run-manifest.json")
    result_path = write_paper1_result(result, directory / "paper1-result.json")
    payloads = {}
    for name in ("residual", "replay", "archive"):
        path = directory / f"{name}.synthetic.txt"
        path.write_text(f"SYNTHETIC {name} payload for {run_id}\n", encoding="utf-8")
        payloads[name] = path
    references = {}
    for name, path in payloads.items():
        digest = sha256_path(path)
        references[name] = {
            "uri": f"artifact://synthetic/{digest}/{name}",
            "sha256": digest,
            "internal_index_sha256": SYNTHETIC_INDEX_DIGEST,
            "media_type": "text/plain",
            "local_path": path.name,
        }
    envelope = {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "manifest_sha256": sha256_path(manifest_path),
        "result_sha256": sha256_path(result_path),
        "residual_evidence": references["residual"],
        "replay_evidence": references["replay"],
        "archive": references["archive"],
    }
    write_canonical_json(directory / "evidence-record.json", envelope)


def generate_synthetic_campaign(destination: str | Path) -> Path:
    """Generate a deterministic fixture matrix with all required censoring classes."""

    root = Path(destination)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    specifications = [
        (
            "syn-positive-001",
            "P1-C-pd3",
            "spacepdhcg-persistent",
            "adaptive",
            "qualified",
            64,
            1,
            1,
            1.0,
            1_000_000,
        ),
        (
            "syn-negative-001",
            "P1-C-pd3",
            "qoco-gpu",
            "fixed-tight",
            "qualified",
            64,
            1,
            1,
            0.8,
            2_000_000,
        ),
        (
            "syn-mixed-001",
            "P1-D-pd6",
            "spacepdhcg-persistent",
            "adaptive",
            "qualified",
            128,
            1,
            1,
            2.0,
            2_000_000,
        ),
        (
            "syn-baseline-002",
            "P1-D-pd6",
            "qoco-gpu",
            "fixed-tight",
            "qualified",
            128,
            1,
            1,
            1.5,
            4_000_000,
        ),
        (
            "syn-robust-aware",
            "P1-F-robust-pd",
            "spacepdhcg-persistent",
            "scenario-aware",
            "qualified",
            64,
            16,
            2,
            3.4,
            8_000_000,
        ),
        (
            "syn-robust-generic",
            "P1-F-robust-pd",
            "spacepdhcg-persistent",
            "generic-partition",
            "qualified",
            64,
            16,
            2,
            3.5,
            8_500_000,
        ),
        (
            "syn-oom-001",
            "P1-E-low-thrust",
            "qoco-gpu",
            "fixed-tight",
            "oom",
            4096,
            1,
            1,
            None,
            None,
        ),
        (
            "syn-timeout-001",
            "P1-E-low-thrust",
            "cuclarabel",
            "fixed-tight",
            "timeout",
            8192,
            1,
            1,
            None,
            20_000_000,
        ),
        (
            "syn-no-crossover",
            "P1-B-hcw",
            "spacepdhcg-persistent",
            "fixed-tight",
            "qualified",
            32,
            1,
            1,
            1.2,
            800_000,
        ),
        (
            "syn-failed-polish",
            "P1-D-pd6",
            "hybrid-pdhcg-ipm",
            "adaptive+polish",
            "failed",
            256,
            1,
            1,
            None,
            6_000_000,
        ),
        (
            "syn-numerical",
            "P1-C-pd3",
            "spacepdhcg-persistent",
            "adaptive",
            "numerical",
            512,
            1,
            1,
            None,
            5_000_000,
        ),
        (
            "syn-unsupported",
            "P1-D-pd6",
            "qoco-gpu",
            "fixed-tight",
            "unsupported",
            512,
            1,
            1,
            None,
            None,
        ),
        (
            "syn-infeasible",
            "P1-C-pd3",
            "clarabel-cpu",
            "fixed-tight",
            "infeasible",
            128,
            1,
            0,
            None,
            None,
        ),
        (
            "syn-unresolved",
            "P1-F-robust-pd",
            "spacepdhcg-persistent",
            "scenario-aware",
            "unrun",
            128,
            32,
            4,
            None,
            None,
        ),
    ]
    for specification in specifications:
        (run_id, family, solver, policy, status, intervals, scenarios, gpus, total, peak) = (
            specification
        )
        result = _result(
            run_id=run_id,
            family=family,
            solver=solver,
            policy=policy,
            status=status,
            intervals=intervals,
            scenarios=scenarios,
            gpus=gpus,
            total=total,
            peak_bytes=peak,
        )
        if run_id == "syn-robust-aware":
            original_collective = result["timing"]["collective_seconds"]
            assert original_collective is not None
            result["timing"]["collective_seconds"] = original_collective * 0.5
            result["timing"]["solve_seconds"] += original_collective * 0.5
            result["resources"]["collective_bytes"] //= 2
        _write_run(root, result)
    config = {
        "schema_version": "1.0.0",
        "campaign_id": "synthetic-demonstration-not-evidence",
        "repository_commit": SYNTHETIC_COMMIT,
        "synthetic": True,
        "units": SI_UNITS,
        "required_coordinates": [
            {"selector": {"run_id": specification[0]}, "minimum_records": 1}
            for specification in specifications
        ],
        "hardware_manifests": [{"path": "synthetic-only", "sha256": "0" * 64}],
        "toolchain_manifests": [{"path": "synthetic-only", "sha256": "0" * 64}],
        "solver_locks": [{"path": "synthetic-only", "sha256": "0" * 64}],
        "claims": {f"H{index}": [f"synthetic-claim-H{index}"] for index in range(1, 7)},
    }
    write_canonical_json(root / "campaign-config.synthetic.json", config)
    (root / "README.md").write_text(
        "# Synthetic G6 demonstration\n\n"
        "**This directory is generated synthetic fixture data. It is not experiment evidence, "
        "does not establish G4/G5, and cannot freeze Paper 1.**\n",
        encoding="utf-8",
    )
    return root
