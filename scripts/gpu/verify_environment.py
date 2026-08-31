#!/usr/bin/env python3
"""Capture and validate the local GPU experiment environment.

This script uses only the Python standard library. It never modifies the machine. The generated
JSON is intended to become an immutable input artifact for every GPU experiment campaign.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

MINIMUM_CUDA: Final = (12, 4)
MINIMUM_CMAKE: Final = (3, 24)
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_VERSION = re.compile(r"(?<!\d)(\d+)\.(\d+)(?:\.(\d+))?")


@dataclass(frozen=True, slots=True)
class CommandResult:
    command: tuple[str, ...]
    available: bool
    returncode: int | None
    stdout: str
    stderr: str

    def to_json(self) -> dict[str, Any]:
        return {
            "command": list(self.command),
            "available": self.available,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


def run(command: list[str], *, timeout: float = 30.0) -> CommandResult:
    executable = shutil.which(command[0])
    if executable is None:
        return CommandResult(tuple(command), False, None, "", "executable not found")
    try:
        completed = subprocess.run(
            [executable, *command[1:]],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return CommandResult(tuple(command), True, None, "", str(error))
    return CommandResult(
        tuple(command),
        True,
        completed.returncode,
        completed.stdout.strip(),
        completed.stderr.strip(),
    )


def parse_version(text: str) -> tuple[int, int, int] | None:
    matches = list(_VERSION.finditer(text))
    if not matches:
        return None
    # Tool banners can include a short release followed by the precise build
    # (for example, "release 12.8, V12.8.93"). Prefer the most specific,
    # latest occurrence so the frozen manifest retains the complete version.
    match = max(
        matches,
        key=lambda candidate: (candidate.group(3) is not None, candidate.start()),
    )
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3) or 0),
    )


def parse_nvidia_csv(text: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    header = [column.strip() for column in lines[0].split(",")]
    records: list[dict[str, str]] = []
    for line in lines[1:]:
        values = [column.strip() for column in line.split(",")]
        if len(values) != len(header):
            raise ValueError(f"malformed nvidia-smi CSV row: {line!r}")
        records.append(dict(zip(header, values, strict=True)))
    return records


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def repository_state(root: Path) -> dict[str, Any]:
    commit = run(["git", "-C", str(root), "rev-parse", "HEAD"])
    status = run(["git", "-C", str(root), "status", "--porcelain=v1"])
    branch = run(["git", "-C", str(root), "branch", "--show-current"])
    remote = run(["git", "-C", str(root), "remote", "get-url", "origin"])
    return {
        "commit": commit.stdout if commit.returncode == 0 else None,
        "branch": branch.stdout if branch.returncode == 0 else None,
        "origin": remote.stdout if remote.returncode == 0 else None,
        "dirty": bool(status.stdout) if status.returncode == 0 else None,
        "status_porcelain": status.stdout if status.returncode == 0 else None,
    }


def package_versions() -> dict[str, str]:
    result = run([sys.executable, "-m", "pip", "freeze", "--all"], timeout=60.0)
    versions: dict[str, str] = {}
    if result.returncode != 0:
        return versions
    for line in result.stdout.splitlines():
        if "==" in line:
            name, version = line.split("==", 1)
            versions[name.lower()] = version
    return versions


def filesystem_record(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path.resolve()),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def collect(root: Path) -> dict[str, Any]:
    commands = {
        "nvidia_smi": run(["nvidia-smi"]),
        "nvidia_query": run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total,"
                "compute_cap,pstate,power.limit,ecc.mode.current,mig.mode.current",
                "--format=csv",
            ]
        ),
        "nvidia_topology": run(["nvidia-smi", "topo", "-m"]),
        "nvidia_clock_query": run(
            [
                "nvidia-smi",
                "--query-gpu=index,clocks.current.graphics,clocks.current.memory,"
                "clocks.max.graphics,clocks.max.memory,temperature.gpu,power.draw",
                "--format=csv",
            ]
        ),
        "nvcc": run(["nvcc", "--version"]),
        "cmake": run(["cmake", "--version"]),
        "ninja": run(["ninja", "--version"]),
        "gcc": run(["gcc", "--version"]),
        "gxx": run(["g++", "--version"]),
        "clang": run(["clang++", "--version"]),
        "git": run(["git", "--version"]),
        "python": run([sys.executable, "--version"]),
        "mpirun": run(["mpirun", "--version"]),
        "numactl": run(["numactl", "--hardware"]),
        "lscpu": run(["lscpu"]),
        "uname": run(["uname", "-a"]),
        "ldconfig_nccl": run(["bash", "-lc", "ldconfig -p 2>/dev/null | grep -i nccl || true"]),
    }
    try:
        gpu_records = parse_nvidia_csv(commands["nvidia_query"].stdout)
    except ValueError as error:
        gpu_records = []
        commands["nvidia_query"] = CommandResult(
            commands["nvidia_query"].command,
            commands["nvidia_query"].available,
            commands["nvidia_query"].returncode,
            commands["nvidia_query"].stdout,
            f"{commands['nvidia_query'].stderr}\n{error}".strip(),
        )

    return {
        "schema_version": "1.0.0",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_implementation": platform.python_implementation(),
        },
        "repository": repository_state(root),
        "gpus": gpu_records,
        "commands": {name: result.to_json() for name, result in commands.items()},
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_HOME",
                "CUDA_PATH",
                "CUDACXX",
                "CUDA_VISIBLE_DEVICES",
                "NCCL_DEBUG",
                "NCCL_SOCKET_IFNAME",
                "NCCL_IB_DISABLE",
                "NCCL_P2P_DISABLE",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "PYTHONHASHSEED",
            )
        },
        "os_release": read_text(Path("/etc/os-release")),
        "kernel_cmdline": read_text(Path("/proc/cmdline")),
        "cpu_online": read_text(Path("/sys/devices/system/cpu/online")),
        "filesystem": filesystem_record(root),
        "python_packages": package_versions(),
    }


def validate(record: MappingLike, *, allow_no_gpu: bool) -> list[str]:
    failures: list[str] = []
    repository = record.get("repository", {})
    commit = repository.get("commit")
    if not isinstance(commit, str) or _SHA40.fullmatch(commit) is None:
        failures.append("repository commit could not be resolved")
    if repository.get("dirty") is True:
        failures.append("repository has uncommitted changes")

    commands = record.get("commands", {})
    cmake = commands.get("cmake", {})
    cmake_version = parse_version(str(cmake.get("stdout", "")))
    if cmake_version is None or cmake_version[:2] < MINIMUM_CMAKE:
        failures.append(
            f"CMake {MINIMUM_CMAKE[0]}.{MINIMUM_CMAKE[1]}+ is required"
        )

    nvcc = commands.get("nvcc", {})
    nvcc_version = parse_version(str(nvcc.get("stdout", "")) + " " + str(nvcc.get("stderr", "")))
    gpus = record.get("gpus", [])
    if not allow_no_gpu:
        if not isinstance(gpus, list) or not gpus:
            failures.append("no NVIDIA GPU was detected")
        if nvcc_version is None or nvcc_version[:2] < MINIMUM_CUDA:
            failures.append(
                f"CUDA toolkit {MINIMUM_CUDA[0]}.{MINIMUM_CUDA[1]}+ is required"
            )
    if not commands.get("gxx", {}).get("available", False):
        failures.append("g++ is required")
    if not commands.get("ninja", {}).get("available", False):
        failures.append("Ninja is required")
    return failures


MappingLike = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root (default: inferred from this script)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/gpu/environment.json"),
        help="JSON output path",
    )
    parser.add_argument(
        "--allow-no-gpu",
        action="store_true",
        help="validate the CPU/tooling portion without requiring NVIDIA hardware",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repository.resolve()
    if not (root / ".git").exists():
        raise SystemExit(f"not a Git repository root: {root}")
    record = collect(root)
    failures = validate(record, allow_no_gpu=args.allow_no_gpu)
    record["validation"] = {
        "passed": not failures,
        "allow_no_gpu": args.allow_no_gpu,
        "failures": failures,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(output)
    for failure in failures:
        print(f"ERROR: {failure}", file=sys.stderr)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
