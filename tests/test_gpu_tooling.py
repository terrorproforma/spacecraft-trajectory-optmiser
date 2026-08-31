from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
GPU_SCRIPTS = ROOT / "scripts" / "gpu"


def _load(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def test_environment_version_and_nvidia_csv_parsing() -> None:
    module = _load("spacepdhcg_verify_environment", GPU_SCRIPTS / "verify_environment.py")
    assert module.parse_version("Cuda compilation tools, release 12.6, V12.6.85") == (12, 6, 85)
    assert module.parse_version("cmake version 3.31.5") == (3, 31, 5)
    assert module.parse_version("not a version") is None

    records = module.parse_nvidia_csv(
        "index, name, uuid\n"
        "0, NVIDIA H100 80GB HBM3, GPU-aaaaaaaa\n"
        "1, NVIDIA H100 80GB HBM3, GPU-bbbbbbbb\n"
    )
    assert records == [
        {"index": "0", "name": "NVIDIA H100 80GB HBM3", "uuid": "GPU-aaaaaaaa"},
        {"index": "1", "name": "NVIDIA H100 80GB HBM3", "uuid": "GPU-bbbbbbbb"},
    ]


def test_environment_cpu_only_validation_identifies_a_dirty_repository() -> None:
    module = _load("spacepdhcg_verify_environment_dirty", GPU_SCRIPTS / "verify_environment.py")
    record = {
        "repository": {"commit": "a" * 40, "dirty": True},
        "commands": {
            "cmake": {"stdout": "cmake version 3.30.0"},
            "nvcc": {"stdout": ""},
            "gxx": {"available": True},
            "ninja": {"available": True},
        },
        "gpus": [],
    }
    failures = module.validate(record, allow_no_gpu=True)
    assert failures == ["repository has uncommitted changes"]


def test_evidence_archive_is_byte_reproducible(tmp_path: Path) -> None:
    module = _load("spacepdhcg_archive_run", GPU_SCRIPTS / "archive_run.py")
    source = tmp_path / "run"
    source.mkdir()
    (source / "stdout.log").write_text("deterministic output\n", encoding="utf-8")
    nested = source / "nested"
    nested.mkdir()
    executable = nested / "command.sh"
    executable.write_text("#!/usr/bin/env bash\necho ok\n", encoding="utf-8")
    executable.chmod(0o755)

    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    module.create_reproducible_tar(source, first)
    module.create_reproducible_tar(source, second)
    assert first.read_bytes() == second.read_bytes()
    assert module.sha256(first) == module.sha256(second)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
@pytest.mark.parametrize(
    "script",
    [
        "bootstrap_ubuntu.sh",
        "checkout_pinned_pdhcg.sh",
        "run_first_gate.sh",
    ],
)
def test_gpu_shell_scripts_are_syntactically_valid(script: str) -> None:
    completed = subprocess.run(
        ["bash", "-n", str(GPU_SCRIPTS / script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
