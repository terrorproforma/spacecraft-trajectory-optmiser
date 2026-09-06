"""CPU-only configuration checks for selecting the explicit CUDA interval path."""

from argparse import Namespace

import pytest

from spacepdhcg.cli import build_parser
from spacepdhcg.gtoc12.cli import _refinement_backend_report, _scvx_settings


@pytest.mark.parametrize(
    "command", ["run", "cluster-fleet", "fleet-master", "retime-returns", "joint-itinerary"]
)
def test_refinement_commands_expose_cuda_backend(command):
    args = ["gtoc12", command, "--run-id", "test", "--output", "unused"]
    if command in {"fleet-master", "retime-returns", "joint-itinerary"}:
        args += ["--source", "unused"]
    parsed = build_parser().parse_args([*args, "--discretisation-backend", "cuda"])
    assert parsed.discretisation_backend == "cuda"


def test_backend_selection_is_not_claimed_as_executed_gpu_work(tmp_path, monkeypatch):
    args = Namespace(scvx_iterations=3, node_days=2.0, workers=1, discretisation_backend="cuda")
    monkeypatch.delenv("SPACEPDHCG_GTOC12_CUDA_LIBRARY", raising=False)
    with pytest.raises(ValueError, match="SPACEPDHCG_GTOC12_CUDA_LIBRARY"):
        _scvx_settings(args)
    library = tmp_path / "library.so"
    library.write_bytes(b"configuration test only")
    monkeypatch.setenv("SPACEPDHCG_GTOC12_CUDA_LIBRARY", str(library))
    settings = _scvx_settings(args)
    assert settings.discretisation_backend == "cuda"
    report = _refinement_backend_report(args)
    assert report["gpu_used"] is None and report["cpu_only"] is None
    assert report["convex_solver_backend"] == "cpu_clarabel"
    args.workers = 2
    with pytest.raises(ValueError, match="workers 1"):
        _scvx_settings(args)
    args.discretisation_backend = "numpy"
    assert _scvx_settings(args).discretisation_backend == "numpy"
    assert _refinement_backend_report(args)["gpu_used"] is False


@pytest.mark.parametrize(
    "name",
    [
        "cmd_run",
        "cmd_cluster_fleet",
        "cmd_fleet_master",
        "cmd_retime_returns",
        "cmd_joint_itinerary",
    ],
)
def test_unsupported_gpu_workers_fail_before_catalogue_or_search(name):
    from spacepdhcg.gtoc12 import cli

    with pytest.raises(ValueError, match="workers 1"):
        getattr(cli, name)(Namespace(discretisation_backend="cuda", workers=2))
