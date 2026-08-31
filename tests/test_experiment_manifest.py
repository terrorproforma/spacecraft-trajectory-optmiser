from __future__ import annotations

import json

import pytest

from spacepdhcg.experiments import (
    HostMetadata,
    RunManifest,
    make_run_manifest,
)


def _host() -> HostMetadata:
    return HostMetadata(
        hostname="test-host",
        operating_system="test-os",
        architecture="x86_64",
        processor="test-cpu",
        logical_cpu_count=16,
        memory_bytes=64 * 1024**3,
        accelerator_vendor="NVIDIA",
        accelerator_model="test-gpu",
        accelerator_count=2,
        driver_version="test-driver",
        runtime_version="CUDA-test",
        interconnect="NVLink-test",
    )


def test_manifest_round_trip_and_atomic_write(tmp_path) -> None:
    manifest = make_run_manifest(
        repository_commit="a" * 40,
        branch="feat/test",
        host=_host(),
        problem={
            "family": "powered_descent_3dof",
            "instance_id": "pd3-n100-s10",
            "nodes": 100,
            "scenarios": 10,
        },
        solver={
            "name": "pdhcg",
            "mode": "adaptive",
            "requested_tolerance": 1.0e-4,
        },
        status="qualified",
        quality={
            "nonlinear_dynamics_defect_inf": 2.0e-7,
            "path_violation_inf": 0.0,
            "qualified": True,
        },
        timing={
            "setup_seconds": 0.01,
            "solve_seconds": 0.12,
            "total_seconds": 0.13,
        },
        experiment={"suite": "paper1-v0", "replicate": 0},
        upstream={"pdhcg_commit": "b" * 40},
        notes=["synthetic test record"],
    )
    destination = manifest.write(tmp_path / "manifest.json")
    reloaded = RunManifest.read(destination)

    assert reloaded.as_dict() == manifest.as_dict()
    assert json.loads(manifest.to_json())["schema_version"] == "1.0.0"
    assert not (tmp_path / "manifest.json.tmp").exists()


def test_manifest_rejects_non_finite_evidence() -> None:
    manifest = make_run_manifest(
        repository_commit="c" * 40,
        host=_host(),
        problem={"family": "cw_qp", "instance_id": "cw-n50"},
        solver={"name": "clarabel", "mode": "fixed"},
        status="solved",
        quality={"primal_residual": 0.0},
        timing={"total_seconds": 1.0},
    )
    manifest.quality["primal_residual"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        manifest.validate()


def test_manifest_requires_commit_and_problem_identity() -> None:
    payload = {
        "schema_version": "1.0.0",
        "run_id": "run",
        "timestamp_utc": "2026-08-31T00:00:00Z",
        "repository": {"commit": ""},
        "upstream": {},
        "host": _host().__dict__,
        "experiment": {},
        "problem": {"family": "", "instance_id": ""},
        "solver": {"name": "solver", "mode": "mode"},
        "quality": {},
        "timing": {},
        "status": "solved",
        "artifacts": {},
        "notes": [],
    }
    with pytest.raises(ValueError, match="repository.commit"):
        RunManifest.from_dict(payload)
