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
    parsed = build_parser().parse_args(
        [
            *args,
            "--discretisation-backend",
            "cuda",
            "--assembly-backend",
            "cuda",
            "--convex-solver",
            "qoco",
            "--qoco-ruiz-iterations",
            "2",
            "--outer-loop-backend",
            "cuda",
        ]
    )
    assert parsed.discretisation_backend == "cuda"
    assert parsed.assembly_backend == "cuda"
    assert parsed.convex_solver_backend == "qoco" and parsed.qoco_ruiz_iterations == 2
    assert parsed.outer_loop_backend == "cuda"
    assert parsed.gpu_execution == "auto"
    assert _refinement_backend_report(parsed)["gpu_execution_selected"] == "graph"


@pytest.mark.parametrize("mode", ["graph", "dispatch"])
def test_execution_command_scope_restores_flags_even_after_failure(monkeypatch, mode):
    import os

    from spacepdhcg.gtoc12.gpu_execution import _GRAPH_SWITCHES, using_gpu_execution

    names = ["SPACEPDHCG_TEST_" + name for name in _GRAPH_SWITCHES]
    for i, name in enumerate(names):
        if i % 2:
            monkeypatch.setenv(name, "caller-value")
        else:
            monkeypatch.delenv(name, raising=False)
    previous = {name: os.environ.get(name) for name in names}
    with pytest.raises(RuntimeError, match="command failed"):
        with using_gpu_execution(Namespace(gpu_execution=mode, outer_loop_backend="cuda")):
            assert all(os.environ[name] == str(int(mode == "graph")) for name in names)
            raise RuntimeError("command failed")
    assert {name: os.environ.get(name) for name in names} == previous


def test_execution_selection_validates_graph_and_preserves_cpu_environment(monkeypatch):
    import os

    from spacepdhcg.gtoc12.gpu_execution import selected_execution, using_gpu_execution

    name = "SPACEPDHCG_TEST_QOCO_IPM_GRAPH"
    monkeypatch.setenv(name, "caller-value")
    with using_gpu_execution(Namespace()) as selected:
        assert selected == "dispatch" and os.environ[name] == "caller-value"
    assert selected_execution(Namespace(outer_loop_backend="cuda")) == "graph"
    dispatch = Namespace(outer_loop_backend="cuda", gpu_execution="dispatch")
    assert selected_execution(dispatch) == "dispatch"
    with pytest.raises(ValueError, match="requires --outer-loop-backend cuda"):
        selected_execution(Namespace(gpu_execution="graph"))
    with pytest.raises(ValueError, match="workers 1"):
        with using_gpu_execution(Namespace(outer_loop_backend="cuda", workers=2)):
            pytest.fail("unsupported graph command started")


def test_qoco_preflight_and_requested_report(tmp_path, monkeypatch):
    args = Namespace(
        discretisation_backend="numpy",
        assembly_backend="numpy",
        convex_solver_backend="qoco",
        scvx_iterations=3,
        node_days=2.0,
        workers=1,
    )
    with pytest.raises(ValueError, match="requires --assembly-backend"):
        _scvx_settings(args)
    args.discretisation_backend = args.assembly_backend = "cuda"
    monkeypatch.delenv("SPACEPDHCG_QOCO_LIBRARY", raising=False)
    with pytest.raises(ValueError, match="SPACEPDHCG_QOCO_LIBRARY"):
        _scvx_settings(args)
    library = tmp_path / "library.so"
    library.write_bytes(b"configuration only")
    monkeypatch.setenv("SPACEPDHCG_QOCO_LIBRARY", str(library))
    monkeypatch.setenv("SPACEPDHCG_GTOC12_CUDA_LIBRARY", str(library))
    args.qoco_ruiz_iterations = 2
    assert _scvx_settings(args).convex_solver_backend == "qoco"
    report = _refinement_backend_report(args)
    assert report["convex_solver_backend"] == "gpu_qoco" and report["gpu_used"] is None
    args.qoco_ruiz_iterations = 101
    with pytest.raises(ValueError, match="in \\[0,100\\]"):
        _scvx_settings(args)


def test_backend_selection_is_not_claimed_as_executed_gpu_work(tmp_path, monkeypatch):
    args = Namespace(scvx_iterations=3, node_days=2.0, workers=1, discretisation_backend="cuda")
    args.assembly_backend = "cuda"
    monkeypatch.delenv("SPACEPDHCG_GTOC12_CUDA_LIBRARY", raising=False)
    with pytest.raises(ValueError, match="SPACEPDHCG_GTOC12_CUDA_LIBRARY"):
        _scvx_settings(args)
    library = tmp_path / "library.so"
    library.write_bytes(b"configuration test only")
    monkeypatch.setenv("SPACEPDHCG_GTOC12_CUDA_LIBRARY", str(library))
    settings = _scvx_settings(args)
    assert settings.discretisation_backend == "cuda"
    assert settings.assembly_backend == "cuda"
    report = _refinement_backend_report(args)
    assert report["gpu_used"] is None and report["cpu_only"] is None
    assert report["convex_solver_backend"] == "cpu_clarabel"
    args.workers = 2
    with pytest.raises(ValueError, match="workers 1"):
        _scvx_settings(args)
    args.discretisation_backend = "numpy"
    with pytest.raises(ValueError, match="requires --discretisation-backend cuda"):
        _scvx_settings(args)
    args.assembly_backend = "numpy"
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
