"""The literature GPU legs must refuse while the G4 measured campaign owns the device."""

from __future__ import annotations

import argparse
import json

import pytest

from spacepdhcg.literature import cli, gpu_preflight

G4_SMI = "827383, [Not Found]\n828904, [Not Found]\n"
G4_COMMANDS = {
    827383: "/x/cuda-tests/device_scvx_integration_test --g4-server 600",
    828904: "/x/cuda-tests/device_scvx_integration_test --g4-session /x/execution-group.json abc",
}


def test_parse_resolves_not_found_names_through_proc_cmdline() -> None:
    processes = gpu_preflight.parse_compute_apps(G4_SMI, command_line_of=G4_COMMANDS.get)
    assert [p.pid for p in processes] == [827383, 828904]
    assert all(p.reported_name == "[Not Found]" for p in processes)
    assert all(p.g4_owner for p in processes)


def test_preflight_refuses_while_g4_session_owns_the_device(monkeypatch) -> None:
    monkeypatch.setattr(gpu_preflight, "_read_command_line", G4_COMMANDS.get)
    result = gpu_preflight.preflight(runner=lambda command: G4_SMI)
    assert not result.ok
    assert result.g4_owned
    assert "G4 measured campaign" in result.reason
    # No override can lift a G4 refusal.
    assert not gpu_preflight.preflight(runner=lambda command: G4_SMI, allow_shared=True).ok
    with pytest.raises(gpu_preflight.GpuPreflightRefused):
        gpu_preflight.require_gpu(runner=lambda command: G4_SMI)


def test_preflight_refuses_other_compute_processes_unless_shared_is_allowed(monkeypatch) -> None:
    monkeypatch.setattr(gpu_preflight, "_read_command_line", {4242: "python train.py"}.get)
    refused = gpu_preflight.preflight(runner=lambda command: "4242, python\n")
    assert not refused.ok and not refused.g4_owned
    assert "allow_shared" in refused.reason
    allowed = gpu_preflight.preflight(runner=lambda command: "4242, python\n", allow_shared=True)
    assert allowed.ok


def test_preflight_ignores_its_own_process_between_targets_of_one_gpu_run(monkeypatch) -> None:
    import os

    own = os.getpid()
    monkeypatch.setattr(gpu_preflight, "_read_command_line", {own: "spacepdhcg literature"}.get)
    result = gpu_preflight.preflight(runner=lambda command: f"{own}, python\n")
    assert result.ok and result.reason == "device free"
    assert [process.pid for process in result.processes] == [own]
    refused = gpu_preflight.preflight(runner=lambda command: f"{own}, python\n4242, python\n")
    assert not refused.ok and "pid 4242" in refused.reason and f"pid {own}" not in refused.reason


def test_preflight_passes_on_an_idle_device_and_checks_the_qoco_library(tmp_path) -> None:
    idle = gpu_preflight.preflight(runner=lambda command: "")
    assert idle.ok and idle.reason == "device free"
    missing = gpu_preflight.preflight(runner=lambda command: "", qoco_library=tmp_path / "no.so")
    assert not missing.ok and "QOCO library" in missing.reason
    library = tmp_path / "libqoco.so"
    library.write_bytes(b"")
    assert gpu_preflight.preflight(runner=lambda command: "", qoco_library=library).ok


def test_preflight_refuses_without_nvidia_smi(monkeypatch) -> None:
    monkeypatch.setattr(gpu_preflight.shutil, "which", lambda name: None)
    result = gpu_preflight.preflight()
    assert not result.ok
    assert "nvidia-smi" in result.reason


def test_gpu_run_command_refuses_before_touching_any_target(monkeypatch, capsys) -> None:
    refused = gpu_preflight.GpuPreflight(
        False, "refused: the G4 measured campaign owns the device", "smi"
    )
    monkeypatch.setattr(gpu_preflight, "preflight", lambda **kwargs: refused)

    def forbidden(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("run_targets must not run while the device is owned")

    from spacepdhcg.literature import report

    monkeypatch.setattr(report, "run_targets", forbidden)
    arguments = argparse.Namespace(
        targets=["acikmese-ploen-2007-pd3"], option=[], allow_shared=False, no_report=True
    )
    code = cli._gpu_run(arguments)
    assert code == cli.GPU_REFUSED_EXIT_CODE
    captured = capsys.readouterr()
    assert json.loads(captured.out.split("\n}\n")[0] + "\n}")["ok"] is False
    assert "refused" in captured.err


def test_gpu_run_command_passes_run_gpu_when_the_device_is_free(monkeypatch) -> None:
    free = gpu_preflight.GpuPreflight(True, "device free", "smi")
    monkeypatch.setattr(gpu_preflight, "preflight", lambda **kwargs: free)
    seen: dict = {}

    def fake_run_targets(targets, *, options):
        seen["targets"] = targets
        seen["options"] = options
        return [{"target_id": targets[0], "status": "reproduced"}]

    from spacepdhcg.literature import report

    monkeypatch.setattr(report, "run_targets", fake_run_targets)
    monkeypatch.setattr(report, "_compact", lambda record: record)
    arguments = argparse.Namespace(
        targets=["chari-2024-pd6-monte-carlo"],
        option=["gpu_batch_sizes=[1]"],
        allow_shared=False,
        no_report=True,
    )
    assert cli._gpu_run(arguments) == 0
    assert seen["targets"] == ["chari-2024-pd6-monte-carlo"]
    assert seen["options"]["*"]["run_gpu"] is True
    assert seen["options"]["*"]["gpu_batch_sizes"] == [1]


def test_pd3_runner_defers_the_gpu_leg_when_preflight_refuses(monkeypatch, tmp_path) -> None:
    from spacepdhcg.literature import pd3_acikmese_ploen as pd3

    refused = gpu_preflight.GpuPreflight(
        False, "refused: the G4 measured campaign owns the device", "smi"
    )
    monkeypatch.setattr(gpu_preflight, "preflight", lambda **kwargs: refused)
    library = tmp_path / "libqoco.so"
    library.write_bytes(b"")
    calls: list[str] = []
    original = pd3.solve_repository_scvx

    def guarded(profile, **kwargs):
        calls.append(kwargs.get("backend", "clarabel"))
        if kwargs.get("backend") == "qoco-gpu":  # pragma: no cover - must not be reached
            raise AssertionError("GPU backend must not be constructed while deferred")
        return original(profile, **kwargs)

    monkeypatch.setattr(pd3, "solve_repository_scvx", guarded)
    from spacepdhcg.literature.registry import load_target_registry

    document = load_target_registry()["acikmese-ploen-2007-pd3"].load_profile()
    record = pd3.run_target(
        document,
        options={
            "run_gpu": True,
            "qoco_library": str(library),
            "run_repository_scvx": True,
            "run_frozen_scvx": False,
            "dt_values": [1.0],
            "scvx_max_iterations": 3,
        },
    )
    leg = record["details"]["repository_scvx"]["qoco-gpu"]
    assert leg["status"] == "deferred"
    assert leg["preflight"]["ok"] is False
    assert record["measured"]["scvx_qoco_gpu_status"] == "deferred"
    assert any("gpu-run" in command for command in record["commands"])
    assert "qoco-gpu" not in calls
