"""Fail-closed Gate G5 topology, launch, and evidence contracts.

This module deliberately separates logical command generation from physical execution.
Logical plans are useful on development hosts but are permanently marked non-executable.
Physical plans require a passing preflight record captured on the target node.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

G5_SCHEMA_VERSION = "1.0.0"
SUPPORTED_GPU_COUNTS = (1, 2, 4, 8)
PARTITIONS = ("scenario_aware", "nonzero_balanced")
FAILURE_MODES = (
    "rank_failure",
    "communicator_error",
    "collective_order",
    "cancellation",
    "checkpoint_restart",
    "topology_mismatch",
    "device_mismatch",
    "timeout",
)
_UPSTREAM_COMMIT = "167c8b72b4b96d2f94d405b8763e485514192b81"
_UPSTREAM_TREE = "62b05e6c1bedd385f6c267af3645ae4aae0421b4"


class PreflightError(RuntimeError):
    """Raised when a physical campaign does not meet its declared prerequisites."""


Runner = Callable[[Sequence[str]], tuple[int, str, str]]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _default_runner(command: Sequence[str]) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        return 127, "", str(error)
    return completed.returncode, completed.stdout, completed.stderr


def _nvcc_executable() -> str:
    discovered = shutil.which("nvcc")
    if discovered:
        return discovered
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-12.8"))
    return str(cuda_home / "bin" / "nvcc")


def _command_capture(
    runner: Runner,
    command: Sequence[str],
) -> dict[str, Any]:
    return_code, stdout, stderr = runner(command)
    return {
        "argv": list(command),
        "return_code": return_code,
        "stdout": stdout,
        "stderr": stderr,
    }


def _first_line(capture: Mapping[str, Any]) -> str | None:
    if capture.get("return_code") != 0:
        return None
    lines = str(capture.get("stdout", "")).strip().splitlines()
    return lines[0] if lines else None


def _parse_optional_int(value: str) -> int | None:
    cleaned = value.strip()
    if cleaned in {"", "N/A", "[N/A]", "Not Supported"}:
        return None
    return int(float(cleaned))


def parse_nvidia_gpu_csv(payload: str) -> list[dict[str, Any]]:
    """Parse the stable, explicitly ordered nvidia-smi GPU query."""

    fields = (
        "index",
        "uuid",
        "model",
        "driver_version",
        "memory_total_mib",
        "memory_free_mib",
        "pci_bus_id",
        "pcie_generation",
        "pcie_width",
        "persistence_mode",
        "ecc_mode",
        "mig_mode",
        "compute_mode",
        "power_limit_w",
        "power_draw_w",
        "sm_clock_mhz",
        "memory_clock_mhz",
    )
    integer_fields = {
        "index",
        "memory_total_mib",
        "memory_free_mib",
        "pcie_generation",
        "pcie_width",
        "sm_clock_mhz",
        "memory_clock_mhz",
    }
    devices: list[dict[str, Any]] = []
    for row in csv.reader(payload.splitlines(), skipinitialspace=True):
        if not row:
            continue
        if len(row) != len(fields):
            raise ValueError(
                f"nvidia-smi returned {len(row)} fields; expected {len(fields)}"
            )
        device: dict[str, Any] = {}
        for field, value in zip(fields, row, strict=True):
            device[field] = (
                _parse_optional_int(value) if field in integer_fields else value.strip()
            )
        devices.append(device)
    return devices


def parse_topology_matrix(payload: str, gpu_count: int) -> dict[str, Any]:
    """Capture matrix links plus per-GPU CPU/NUMA locality from ``nvidia-smi topo -m``."""

    rows: list[dict[str, Any]] = []
    legend: list[str] = []
    in_legend = False
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Legend:"):
            in_legend = True
            continue
        if in_legend:
            legend.append(line)
            continue
        if "CPU Affinity" in line:
            continue
        columns = re.split(r"\s+", line)
        if not columns[0].startswith("GPU"):
            continue
        minimum = 1 + gpu_count + 2
        if len(columns) < minimum:
            raise ValueError(f"malformed nvidia topology row: {raw_line!r}")
        rows.append(
            {
                "gpu": columns[0],
                "links": columns[1 : 1 + gpu_count],
                "cpu_affinity": columns[1 + gpu_count],
                "numa_node": _parse_optional_int(columns[2 + gpu_count]),
                "gpu_numa_id": (
                    _parse_optional_int(columns[3 + gpu_count])
                    if len(columns) > 3 + gpu_count
                    else None
                ),
            }
        )
    if len(rows) != gpu_count:
        raise ValueError(f"topology has {len(rows)} GPU rows; expected {gpu_count}")
    return {"rows": rows, "legend": legend, "raw": payload}


def expand_cpu_list(specification: str) -> list[int]:
    """Expand Linux CPU-list syntax such as ``0-3,8,10-11``."""

    result: list[int] = []
    for part in specification.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            first_text, last_text = token.split("-", maxsplit=1)
            first = int(first_text)
            last = int(last_text)
            if last < first:
                raise ValueError(f"descending CPU range: {token}")
            result.extend(range(first, last + 1))
        else:
            result.append(int(token))
    if len(set(result)) != len(result):
        raise ValueError(f"CPU list contains duplicates: {specification}")
    return result


def compress_cpu_list(cpus: Sequence[int]) -> str:
    values = sorted(set(cpus))
    if not values:
        raise ValueError("CPU binding may not be empty")
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _active_interfaces(ip_payload: str) -> list[dict[str, Any]]:
    parsed = json.loads(ip_payload)
    interfaces: list[dict[str, Any]] = []
    for item in parsed:
        name = item.get("ifname")
        if name == "lo" or item.get("operstate") != "UP":
            continue
        numa_path = Path(f"/sys/class/net/{name}/device/numa_node")
        try:
            numa_value = int(numa_path.read_text(encoding="utf-8").strip())
            numa_node = numa_value if numa_value >= 0 else None
        except (FileNotFoundError, OSError, ValueError):
            numa_node = None
        interfaces.append(
            {
                "name": name,
                "state": item.get("operstate"),
                "mtu": item.get("mtu"),
                "numa_node": numa_node,
            }
        )
    return interfaces


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture_build_pin(build_directory: Path | None) -> dict[str, Any]:
    if build_directory is None:
        return {
            "directory": None,
            "cmake_cache_sha256": None,
            "harness_sha256": None,
            "cmake_flags": {},
        }
    directory = build_directory.resolve()
    cache = directory / "CMakeCache.txt"
    harness = directory / "distributed-tools" / "g5_physical_validation_harness"
    wanted = {
        "CMAKE_BUILD_TYPE",
        "CMAKE_CUDA_ARCHITECTURES",
        "CMAKE_CUDA_COMPILER",
        "SPACEPDHCG_BUILD_CUDA",
        "SPACEPDHCG_BUILD_DISTRIBUTED",
        "SPACEPDHCG_NATIVE_WARNINGS_AS_ERRORS",
    }
    flags: dict[str, str] = {}
    if cache.is_file():
        for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("//") or line.startswith("#") or "=" not in line:
                continue
            key_and_type, value = line.split("=", maxsplit=1)
            key = key_and_type.split(":", maxsplit=1)[0]
            if key in wanted:
                flags[key] = value
    return {
        "directory": str(directory),
        "cmake_cache_sha256": _sha256_file(cache),
        "harness_path": str(harness),
        "harness_sha256": _sha256_file(harness),
        "cmake_flags": flags,
    }


def validate_preflight_record(
    record: Mapping[str, Any],
    *,
    expected_gpu_count: int,
    primary: bool,
    minimum_free_fraction: float = 0.90,
) -> list[str]:
    """Return all fail-closed reasons without mutating a captured record."""

    failures: list[str] = []
    if expected_gpu_count not in SUPPORTED_GPU_COUNTS:
        failures.append(f"unsupported GPU count {expected_gpu_count}")
    devices = list(record.get("gpus", []))
    if len(devices) < expected_gpu_count:
        failures.append(
            f"detected {len(devices)} GPUs but campaign requires at least "
            f"{expected_gpu_count}"
        )
    if len({item.get("uuid") for item in devices}) != len(devices):
        failures.append("GPU UUIDs are missing or duplicated")
    if len({item.get("pci_bus_id") for item in devices}) != len(devices):
        failures.append("GPU PCI bus identifiers are missing or duplicated")
    if primary and len({item.get("model") for item in devices}) > 1:
        failures.append("primary campaign requires a homogeneous GPU model")
    if primary and len({item.get("driver_version") for item in devices}) > 1:
        failures.append("primary campaign requires one NVIDIA driver version")
    total_memory = {item.get("memory_total_mib") for item in devices}
    if primary and len(total_memory) > 1:
        failures.append("primary campaign requires homogeneous GPU memory capacity")
    for field, description in (
        ("ecc_mode", "ECC mode"),
        ("mig_mode", "MIG mode"),
        ("compute_mode", "compute mode"),
        ("power_limit_w", "power limit"),
        ("persistence_mode", "persistence mode"),
    ):
        if primary and len({item.get(field) for item in devices}) > 1:
            failures.append(f"primary campaign requires homogeneous {description}")
    for item in devices:
        index = item.get("index")
        total = item.get("memory_total_mib")
        free = item.get("memory_free_mib")
        if isinstance(total, int) and isinstance(free, int):
            if total <= 0 or free / total < minimum_free_fraction:
                failures.append(
                    f"GPU {index} free-memory fraction is below "
                    f"{minimum_free_fraction:.0%}"
                )
        else:
            failures.append(f"GPU {index} memory telemetry is unavailable")
        if item.get("mig_mode") not in {"Disabled", "N/A", "[N/A]"}:
            failures.append(f"GPU {index} MIG must be disabled")
        if item.get("compute_mode") not in {"Default", "Exclusive_Process"}:
            failures.append(f"GPU {index} compute mode is unsupported")
        if not item.get("cpu_affinity"):
            failures.append(f"GPU {index} CPU affinity is unavailable")
        if item.get("numa_node") is None:
            failures.append(f"GPU {index} NUMA affinity is unavailable")
    repository = record.get("repository", {})
    if not repository.get("commit"):
        failures.append("repository commit is unavailable")
    if repository.get("dirty"):
        failures.append("repository must be clean")
    upstream = record.get("upstream", {})
    if upstream.get("commit") != _UPSTREAM_COMMIT:
        failures.append("pinned upstream PDHCG commit does not match")
    if upstream.get("tree") != _UPSTREAM_TREE:
        failures.append("pinned upstream PDHCG tree does not match")
    if record.get("active_compute_processes"):
        failures.append("exclusive campaign node has active GPU compute processes")
    topology = record.get("topology", {})
    if len(topology.get("rows", [])) != len(devices):
        failures.append("GPU topology matrix is incomplete")
    if not record.get("network", {}).get("interfaces"):
        failures.append("no active non-loopback network interface was captured")
    toolchain = record.get("toolchain", {})
    for name in ("cuda", "nccl", "mpi"):
        if not toolchain.get(name):
            failures.append(f"{name.upper()} version is unavailable")
    build = record.get("build", {})
    if not build.get("cmake_cache_sha256"):
        failures.append("CMake build cache pin is unavailable")
    if not build.get("harness_sha256"):
        failures.append("physical validation harness build pin is unavailable")
    return failures


def capture_preflight(
    repository: str | Path,
    *,
    expected_gpu_count: int,
    primary: bool = True,
    minimum_free_fraction: float = 0.90,
    build_directory: str | Path | None = None,
    runner: Runner = _default_runner,
) -> dict[str, Any]:
    """Capture a physical node and fail closed when campaign prerequisites are unmet."""

    root = Path(repository).resolve()
    queries: dict[str, list[str]] = {
        "gpu_csv": [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,memory.free,"
            "pci.bus_id,pcie.link.gen.current,pcie.link.width.current,persistence_mode,"
            "ecc.mode.current,mig.mode.current,compute_mode,power.limit,power.draw,"
            "clocks.current.sm,clocks.current.memory",
            "--format=csv,noheader,nounits",
        ],
        "topology": ["nvidia-smi", "topo", "-m"],
        "nvlink": ["nvidia-smi", "nvlink", "--status"],
        "gpu_query": ["nvidia-smi", "-q"],
        "compute_apps": [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        "cuda": [_nvcc_executable(), "--version"],
        "mpi": ["mpirun", "--version"],
        "nccl": ["dpkg-query", "-W", "-f=${Version}", "libnccl2"],
        "cpu": ["lscpu", "--json"],
        "numa": ["numactl", "--hardware"],
        "network": ["ip", "-j", "link", "show"],
        "git_commit": ["git", "-C", str(root), "rev-parse", "HEAD"],
        "git_branch": ["git", "-C", str(root), "branch", "--show-current"],
        "git_status": ["git", "-C", str(root), "status", "--porcelain=v1"],
        "upstream_commit": [
            "git",
            "-C",
            str(root / "_upstream" / "pdhcg"),
            "rev-parse",
            "HEAD",
        ],
        "upstream_tree": [
            "git",
            "-C",
            str(root / "_upstream" / "pdhcg"),
            "rev-parse",
            "HEAD^{tree}",
        ],
    }
    captures = {name: _command_capture(runner, command) for name, command in queries.items()}
    fatal_capture_failures = [
        name
        for name in (
            "gpu_csv",
            "topology",
            "cuda",
            "mpi",
            "nccl",
            "cpu",
            "network",
            "git_commit",
            "git_status",
            "upstream_commit",
            "upstream_tree",
        )
        if captures[name]["return_code"] != 0
    ]
    devices: list[dict[str, Any]] = []
    topology: dict[str, Any] = {"rows": [], "legend": [], "raw": ""}
    parse_failures: list[str] = []
    try:
        devices = parse_nvidia_gpu_csv(captures["gpu_csv"]["stdout"])
        topology = parse_topology_matrix(captures["topology"]["stdout"], len(devices))
        for device, locality in zip(devices, topology["rows"], strict=True):
            device["cpu_affinity"] = locality["cpu_affinity"]
            device["numa_node"] = locality["numa_node"]
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        parse_failures.append(f"GPU topology parse failed: {error}")
    try:
        interfaces = _active_interfaces(captures["network"]["stdout"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        interfaces = []
        parse_failures.append(f"network topology parse failed: {error}")
    compute_lines = [
        line
        for line in captures["compute_apps"]["stdout"].splitlines()
        if line.strip() and "No running processes" not in line
    ]
    record: dict[str, Any] = {
        "schema_version": G5_SCHEMA_VERSION,
        "record_type": "g5-physical-preflight",
        "captured_at_utc": _utc_now(),
        "hostname": socket.gethostname(),
        "primary_campaign": primary,
        "expected_gpu_count": expected_gpu_count,
        "minimum_free_fraction": minimum_free_fraction,
        "status": "failed",
        "failures": [],
        "repository": {
            "root": str(root),
            "commit": _first_line(captures["git_commit"]),
            "branch": _first_line(captures["git_branch"]),
            "dirty": bool(captures["git_status"]["stdout"].strip()),
        },
        "upstream": {
            "commit": _first_line(captures["upstream_commit"]),
            "tree": _first_line(captures["upstream_tree"]),
        },
        "toolchain": {
            "driver": devices[0].get("driver_version") if devices else None,
            "cuda": captures["cuda"]["stdout"].strip() or None,
            "nccl": _first_line(captures["nccl"]),
            "mpi": _first_line(captures["mpi"]),
        },
        "build": _capture_build_pin(
            Path(build_directory) if build_directory is not None else None
        ),
        "host": {
            "cpu": captures["cpu"]["stdout"],
            "numa": captures["numa"]["stdout"],
        },
        "gpus": devices,
        "topology": {
            **topology,
            "nvlink_status": captures["nvlink"]["stdout"],
        },
        "network": {"interfaces": interfaces},
        "active_compute_processes": compute_lines,
        "raw_commands": captures,
    }
    failures = [
        *(f"required command failed: {name}" for name in fatal_capture_failures),
        *parse_failures,
        *validate_preflight_record(
            record,
            expected_gpu_count=expected_gpu_count,
            primary=primary,
            minimum_free_fraction=minimum_free_fraction,
        ),
    ]
    record["failures"] = sorted(set(failures))
    record["status"] = "passed" if not failures else "failed"
    record["fingerprint"] = _canonical_digest(
        {
            "repository": record["repository"],
            "upstream": record["upstream"],
            "toolchain": record["toolchain"],
            "gpus": record["gpus"],
            "topology": record["topology"],
            "network": record["network"],
        }
    )
    return record


def logical_topology(gpu_count: int, *, cpus_per_gpu: int = 8) -> dict[str, Any]:
    """Create a clearly labelled logical topology for command-only tests."""

    if gpu_count not in SUPPORTED_GPU_COUNTS:
        raise ValueError(f"unsupported logical GPU count {gpu_count}")
    gpus = []
    rows = []
    for index in range(gpu_count):
        first_cpu = index * cpus_per_gpu
        affinity = f"{first_cpu}-{first_cpu + cpus_per_gpu - 1}"
        gpus.append(
            {
                "index": index,
                "uuid": f"LOGICAL-GPU-{index}",
                "model": "logical-only",
                "pci_bus_id": f"00000000:{index + 1:02x}:00.0",
                "cpu_affinity": affinity,
                "numa_node": index,
            }
        )
        rows.append(
            {
                "gpu": f"GPU{index}",
                "links": ["X" if peer == index else "LOGICAL" for peer in range(gpu_count)],
                "cpu_affinity": affinity,
                "numa_node": index,
                "gpu_numa_id": index,
            }
        )
    return {
        "schema_version": G5_SCHEMA_VERSION,
        "record_type": "g5-logical-topology",
        "logical_only": True,
        "status": "non-executable",
        "hostname": "logical-host",
        "gpus": gpus,
        "topology": {"rows": rows, "legend": ["LOGICAL: not physical evidence"]},
        "network": {"interfaces": [{"name": "logical0", "numa_node": None}]},
    }


def rank_bindings(
    topology_record: Mapping[str, Any],
    gpu_count: int,
    *,
    nic: str | None = None,
) -> list[dict[str, Any]]:
    """Map local rank to PCI-ordered GPU, disjoint CPU cores, NUMA node, and NIC."""

    devices = sorted(
        topology_record.get("gpus", []),
        key=lambda item: str(item.get("pci_bus_id", "")),
    )
    if len(devices) < gpu_count:
        raise PreflightError(
            f"topology provides {len(devices)} GPUs but launch requires {gpu_count}"
        )
    interfaces = topology_record.get("network", {}).get("interfaces", [])
    default_nic = nic or (interfaces[0]["name"] if interfaces else None)
    if not default_nic:
        raise PreflightError("rank-to-NIC mapping cannot be determined")
    selected = devices[:gpu_count]
    affinity_groups: dict[tuple[Any, str], list[int]] = {}
    for rank, device in enumerate(selected):
        key = (device.get("numa_node"), str(device["cpu_affinity"]))
        affinity_groups.setdefault(key, []).append(rank)
    rank_cpus: dict[int, list[int]] = {}
    used_cpus: set[int] = set()
    for (_, affinity), ranks in affinity_groups.items():
        available = [
            cpu for cpu in expand_cpu_list(affinity) if cpu not in used_cpus
        ]
        if len(available) < len(ranks):
            raise PreflightError(
                f"CPU affinity {affinity} cannot provide one core per rank"
            )
        quotient, remainder = divmod(len(available), len(ranks))
        offset = 0
        for group_index, rank in enumerate(ranks):
            count = quotient + (1 if group_index < remainder else 0)
            rank_cpus[rank] = available[offset : offset + count]
            offset += count
        used_cpus.update(available)
    bindings: list[dict[str, Any]] = []
    for rank, device in enumerate(selected):
        local_nics = [
            item["name"]
            for item in interfaces
            if item.get("numa_node") == device.get("numa_node")
        ]
        bindings.append(
            {
                "rank": rank,
                "gpu_index": device["index"],
                "gpu_uuid": device["uuid"],
                "pci_bus_id": device["pci_bus_id"],
                "cpu_set": compress_cpu_list(rank_cpus[rank]),
                "numa_node": device.get("numa_node"),
                "nic": nic or (local_nics[0] if local_nics else default_nic),
            }
        )
    return bindings


@dataclass(frozen=True, slots=True)
class CampaignCoordinate:
    scaling: str
    gpu_count: int
    scenarios: int
    nodes: int
    partition: str
    risk: str
    seed: int

    def validate(self) -> None:
        if self.scaling not in {"strong", "weak"}:
            raise ValueError(f"unsupported scaling mode {self.scaling}")
        if self.gpu_count not in SUPPORTED_GPU_COUNTS:
            raise ValueError(f"unsupported GPU count {self.gpu_count}")
        if self.partition not in PARTITIONS:
            raise ValueError(f"unsupported partition {self.partition}")
        if self.scenarios < self.gpu_count:
            raise ValueError("whole-scenario partition requires at least one scenario per GPU")
        if self.nodes <= 0:
            raise ValueError("nodes must be positive")


def generate_coordinates(config: Mapping[str, Any]) -> list[CampaignCoordinate]:
    """Expand deterministic strong/weak scenario-aware and generic comparisons."""

    coordinates: list[CampaignCoordinate] = []
    gpu_counts = tuple(config["gpu_counts"])
    partitions = tuple(config.get("partitions", PARTITIONS))
    nodes_values = tuple(config["nodes"])
    risks = tuple(config["risks"])
    seeds = tuple(config["seeds"])
    for gpu_count in gpu_counts:
        for scenarios in config["strong"]["scenario_counts"]:
            for nodes in nodes_values:
                for partition in partitions:
                    for risk in risks:
                        for seed in seeds:
                            coordinate = CampaignCoordinate(
                                "strong",
                                gpu_count,
                                scenarios,
                                nodes,
                                partition,
                                risk,
                                seed,
                            )
                            coordinate.validate()
                            coordinates.append(coordinate)
        for scenarios_per_gpu in config["weak"]["scenarios_per_gpu"]:
            for nodes in nodes_values:
                for partition in partitions:
                    for risk in risks:
                        for seed in seeds:
                            coordinate = CampaignCoordinate(
                                "weak",
                                gpu_count,
                                scenarios_per_gpu * gpu_count,
                                nodes,
                                partition,
                                risk,
                                seed,
                            )
                            coordinate.validate()
                            coordinates.append(coordinate)
    return coordinates


def _rankfile(bindings: Sequence[Mapping[str, Any]], hostname: str) -> str:
    return "".join(
        f"rank {binding['rank']}={hostname} slot={binding['cpu_set']}\n"
        for binding in bindings
    )


def make_command_manifest(
    coordinate: CampaignCoordinate,
    *,
    topology_record: Mapping[str, Any],
    repository_commit: str,
    executable: str,
    output_root: str,
    warmups: int,
    repeats: int,
    timeout_seconds: int,
    failure_mode: str | None = None,
    campaign_mode: str = "launch-probe",
) -> dict[str, Any]:
    """Generate one exact, deterministic OpenMPI command and all launch inputs."""

    coordinate.validate()
    logical_only = bool(topology_record.get("logical_only"))
    if not logical_only and topology_record.get("status") != "passed":
        raise PreflightError("physical command generation requires a passing preflight")
    executable_digest = None if logical_only else _sha256_file(Path(executable))
    if not logical_only and executable_digest is None:
        raise PreflightError(f"physical campaign executable does not exist: {executable}")
    if failure_mode is not None and failure_mode not in FAILURE_MODES:
        raise ValueError(f"unsupported failure mode {failure_mode}")
    if failure_mode is not None and repeats != 1:
        raise ValueError("failure-injection manifests require exactly one repeat")
    bindings = rank_bindings(topology_record, coordinate.gpu_count)
    identity = {
        "coordinate": {
            "scaling": coordinate.scaling,
            "gpu_count": coordinate.gpu_count,
            "scenarios": coordinate.scenarios,
            "nodes": coordinate.nodes,
            "partition": coordinate.partition,
            "risk": coordinate.risk,
            "seed": coordinate.seed,
        },
        "repository_commit": repository_commit,
        "failure_mode": failure_mode,
    }
    if campaign_mode != "launch-probe":
        identity["campaign_mode"] = campaign_mode
    if executable_digest is not None:
        identity["executable_sha256"] = executable_digest
    run_id = _canonical_digest(identity)[:20]
    topology_digest = str(
        topology_record.get("fingerprint") or _canonical_digest({"bindings": bindings})
    )
    partition_digest = _canonical_digest(identity["coordinate"])
    run_directory = f"{output_root.rstrip('/')}/{run_id}"
    rankfile_path = f"{run_directory}/rankfile"
    environment = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": ",".join(str(item["gpu_index"]) for item in bindings),
        "NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,COLL,GRAPH,NET",
        "NCCL_SOCKET_IFNAME": str(bindings[0]["nic"]),
        "OMPI_MCA_orte_abort_on_non_zero_status": "1",
    }
    if failure_mode is not None:
        environment["SPACEPDHCG_G5_FAILURE_TEST"] = "1"
    mpi_argv = [
        "mpirun",
        "--np",
        str(coordinate.gpu_count),
        "--rankfile",
        rankfile_path,
        "--bind-to",
        "core",
        "--report-bindings",
        "--display-map",
        "--output-filename",
        f"{run_directory}/rank-output",
    ]
    for key, value in sorted(environment.items()):
        mpi_argv.extend(["-x", f"{key}={value}"])
    program_argv = [
        executable,
        "--campaign-mode",
        campaign_mode,
        "--scaling",
        coordinate.scaling,
        "--partition",
        coordinate.partition,
        "--scenarios",
        str(coordinate.scenarios),
        "--nodes",
        str(coordinate.nodes),
        "--risk",
        coordinate.risk,
        "--seed",
        str(coordinate.seed),
        "--warmups",
        str(warmups),
        "--repeats",
        str(repeats),
        "--evidence-directory",
        run_directory,
        "--topology-fingerprint",
        topology_digest[:16],
        "--partition-fingerprint",
        partition_digest[:16],
    ]
    if failure_mode is not None:
        program_argv.extend(["--test-mode", "--inject", failure_mode])
    argv = [
        "timeout",
        "--signal=TERM",
        "--kill-after=30s",
        f"{timeout_seconds}s",
        *mpi_argv,
        *program_argv,
    ]
    manifest = {
        "schema_version": G5_SCHEMA_VERSION,
        "record_type": "g5-command-manifest",
        "run_id": run_id,
        "generated_at_utc": _utc_now(),
        "logical_only": logical_only,
        "physical_execution_permitted": not logical_only,
        "qualification_permitted": False,
        "repository_commit": repository_commit,
        "executable": {
            "path": executable,
            "sha256": executable_digest,
        },
        "preflight_fingerprint": topology_record.get("fingerprint"),
        "runtime_fingerprints": {
            "topology": topology_digest[:16],
            "partition": partition_digest[:16],
        },
        "coordinate": identity["coordinate"],
        "comparison": {
            "distributed_partitions": list(PARTITIONS),
            "monolithic_reference": {
                "gpu_count": 1,
                "required": True,
                "separate_manifest": True,
            },
        },
        "warmups": warmups,
        "repeats": repeats,
        "timeout_seconds": timeout_seconds,
        "failure_mode": failure_mode,
        "bindings": bindings,
        "rankfile": {
            "path": rankfile_path,
            "content": _rankfile(bindings, str(topology_record["hostname"])),
        },
        "environment": environment,
        "argv": argv,
        "display_command": " ".join(argv),
        "evidence": {
            "directory": run_directory,
            "stdout": f"{run_directory}/launcher.stdout.log",
            "stderr": f"{run_directory}/launcher.stderr.log",
            "gpu_samples": f"{run_directory}/gpu-samples.csv",
            "partial_record": f"{run_directory}/evidence.partial.json",
            "final_record": f"{run_directory}/evidence.json",
        },
    }
    manifest["manifest_sha256"] = _canonical_digest(manifest)
    return manifest


def make_monolithic_reference_manifest(
    coordinate: CampaignCoordinate,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build the matched one-GPU monolithic reference for a distributed coordinate."""

    reference = CampaignCoordinate(
        scaling=coordinate.scaling,
        gpu_count=1,
        scenarios=coordinate.scenarios,
        nodes=coordinate.nodes,
        partition="scenario_aware",
        risk=coordinate.risk,
        seed=coordinate.seed,
    )
    kwargs["campaign_mode"] = "monolithic-reference"
    manifest = make_command_manifest(reference, **kwargs)
    manifest["record_type"] = "g5-monolithic-reference-command"
    manifest["comparison"]["monolithic_reference"]["separate_manifest"] = False
    manifest["comparison"]["monolithic_reference"]["reference_for_gpu_counts"] = list(
        SUPPORTED_GPU_COUNTS
    )
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = _canonical_digest(manifest)
    return manifest


def validate_command_manifest(manifest: Mapping[str, Any]) -> None:
    """Reject logical execution, rank/device mismatch, and unguarded failure injection."""

    gpu_count = manifest["coordinate"]["gpu_count"]
    bindings = manifest["bindings"]
    if len(bindings) != gpu_count:
        raise ValueError("one binding per GPU/rank is required")
    if [item["rank"] for item in bindings] != list(range(gpu_count)):
        raise ValueError("ranks must be contiguous and deterministic")
    if len({item["gpu_uuid"] for item in bindings}) != gpu_count:
        raise ValueError("each rank must own one unique GPU")
    if manifest.get("logical_only") and manifest.get("physical_execution_permitted"):
        raise ValueError("logical manifests may never permit physical execution")
    failure_mode = manifest.get("failure_mode")
    environment = manifest.get("environment", {})
    if failure_mode is not None:
        if not manifest.get("argv") or "--test-mode" not in manifest["argv"]:
            raise ValueError("failure injection requires --test-mode")
        if environment.get("SPACEPDHCG_G5_FAILURE_TEST") != "1":
            raise ValueError("failure injection requires explicit environment opt-in")
    unsigned = dict(manifest)
    recorded_digest = unsigned.pop("manifest_sha256", None)
    if recorded_digest != _canonical_digest(unsigned):
        raise ValueError("command manifest SHA-256 does not match its payload")


def classify_failure(
    *,
    return_code: int | None,
    timed_out: bool,
    stdout: str,
    stderr: str,
) -> dict[str, Any]:
    """Classify launcher and rank failures without discarding partial logs."""

    combined = f"{stdout}\n{stderr}".lower()
    kinds: list[str] = []
    if timed_out:
        kinds.append("timeout")
    patterns = {
        "oom": ("out of memory", "cuda_error_out_of_memory", "ncclunhandledcudaerror"),
        "rank_failure": ("exited on signal", "non-zero exit code", "rank_failure"),
        "nccl_error": ("nccl", "remote process exited", "unhandled system error"),
        "cuda_error": ("cuda error", "cudaerror", "misaligned address"),
        "collective_order": ("collective_order", "mismatched collective"),
        "cancellation": ("cancelled", "cancellation"),
        "topology_mismatch": ("topology_mismatch", "device_mismatch"),
    }
    for kind, markers in patterns.items():
        if any(marker in combined for marker in markers):
            kinds.append(kind)
    if return_code not in {None, 0} and not kinds:
        kinds.append("launcher_failure")
    return {
        "return_code": return_code,
        "timed_out": timed_out,
        "kinds": sorted(set(kinds)),
        "primary_kind": sorted(set(kinds))[0] if kinds else None,
    }


def summarize_gpu_samples(path: str | Path) -> list[dict[str, Any]]:
    """Integrate board-power samples and retain peak memory per GPU."""

    source = Path(path)
    if not source.is_file():
        return []
    samples: dict[str, list[tuple[datetime, float, float]]] = {}
    indices: dict[str, int | None] = {}
    with source.open("r", encoding="utf-8", errors="replace", newline="") as stream:
        for row in csv.DictReader(stream, skipinitialspace=True):
            normalized = {key.strip(): value.strip() for key, value in row.items()}
            uuid = normalized.get("uuid", "")
            timestamp_text = normalized.get("timestamp", "")
            if not uuid or not timestamp_text:
                continue
            try:
                try:
                    timestamp = datetime.strptime(
                        timestamp_text,
                        "%Y/%m/%d %H:%M:%S.%f",
                    ).replace(tzinfo=UTC)
                except ValueError:
                    timestamp = datetime.strptime(
                        timestamp_text,
                        "%Y/%m/%d %H:%M:%S",
                    ).replace(tzinfo=UTC)
                power = float(normalized["power.draw [W]"].split()[0])
                memory = float(normalized["memory.used [MiB]"].split()[0])
                indices[uuid] = int(normalized["index"])
            except (KeyError, TypeError, ValueError):
                continue
            samples.setdefault(uuid, []).append((timestamp, power, memory))
    summaries: list[dict[str, Any]] = []
    for uuid, values in sorted(samples.items()):
        values.sort(key=lambda item: item[0])
        energy = 0.0
        for previous, current in pairwise(values):
            delta = (current[0] - previous[0]).total_seconds()
            if 0.0 <= delta <= 5.0:
                energy += delta * (previous[1] + current[1]) / 2.0
        summaries.append(
            {
                "index": indices.get(uuid),
                "uuid": uuid,
                "sample_count": len(values),
                "energy_joules": energy,
                "peak_device_bytes": int(max(item[2] for item in values) * 1024**2),
            }
        )
    return summaries


def build_partial_evidence(
    manifest: Mapping[str, Any],
    *,
    preflight: Mapping[str, Any],
    return_code: int | None,
    timed_out: bool,
    stdout: str,
    stderr: str,
    rank_telemetry: Sequence[Mapping[str, Any]] = (),
    gpu_samples: Sequence[Mapping[str, Any]] = (),
    launcher_seconds: float | None = None,
) -> dict[str, Any]:
    """Create a schema-stable record even when launch or ranks fail."""

    failure = classify_failure(
        return_code=return_code,
        timed_out=timed_out,
        stdout=stdout,
        stderr=stderr,
    )
    telemetry = list(rank_telemetry)
    complete_ranks = {item.get("rank") for item in telemetry if item.get("complete")}
    expected_ranks = set(range(manifest["coordinate"]["gpu_count"]))
    status = "complete" if return_code == 0 and complete_ranks == expected_ranks else "partial"
    local_compute = [
        item.get("local_compute_seconds")
        for item in telemetry
        if isinstance(item.get("local_compute_seconds"), (int, float))
    ]
    mean_compute = sum(local_compute) / len(local_compute) if local_compute else None
    load_imbalance = (
        max(local_compute) / mean_compute
        if mean_compute is not None and mean_compute > 0
        else None
    )
    collectives = [
        {"rank": item.get("rank"), "collectives": item.get("collectives", [])}
        for item in telemetry
    ]
    return {
        "schema_version": G5_SCHEMA_VERSION,
        "record_type": "g5-physical-evidence",
        "created_at_utc": _utc_now(),
        "run_id": manifest["run_id"],
        "status": status,
        "qualification_claim": False,
        "multi_gpu_scaling_verified": False,
        "manifest_sha256": manifest["manifest_sha256"],
        "preflight_fingerprint": preflight.get("fingerprint"),
        "coordinate": manifest["coordinate"],
        "bindings": manifest["bindings"],
        "failure": failure,
        "rank_telemetry": telemetry,
        "missing_ranks": sorted(expected_ranks - complete_ranks),
        "required_metrics": {
            "collective_bytes_count_frequency": collectives or None,
            "communication_exposed_overlapped_seconds": [
                {
                    "rank": item.get("rank"),
                    "exposed_seconds": item.get("communication_exposed_seconds"),
                    "overlapped_seconds": item.get(
                        "communication_overlapped_seconds"
                    ),
                }
                for item in telemetry
            ]
            or None,
            "local_compute_and_load_imbalance": {
                "per_rank_seconds": local_compute,
                "max_over_mean": load_imbalance,
            }
            if local_compute
            else None,
            "memory_per_gpu_bytes": [
                {"rank": item.get("rank"), **(item.get("memory") or {})}
                for item in telemetry
            ]
            or list(gpu_samples)
            or None,
            "timing_seconds": {
                "launcher": launcher_seconds,
                "per_rank": [
                    {"rank": item.get("rank"), **(item.get("timing") or {})}
                    for item in telemetry
                ],
            }
            if launcher_seconds is not None or telemetry
            else None,
            "energy_joules": [
                {"rank": item.get("rank"), "energy_joules": item.get("energy_joules")}
                for item in telemetry
            ]
            or list(gpu_samples)
            or None,
            "residuals_nonanticipativity_risk": [
                {"rank": item.get("rank"), **(item.get("quality") or {})}
                for item in telemetry
            ]
            or None,
            "nonlinear_quality": [
                {
                    "rank": item.get("rank"),
                    "nonlinear_quality": (item.get("quality") or {}).get(
                        "nonlinear_quality"
                    ),
                }
                for item in telemetry
            ]
            or None,
        },
        "partial_logs": {
            "stdout_captured": bool(stdout),
            "stderr_captured": bool(stderr),
        },
    }


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def verify_openmpi_command_shape(
    manifest: Mapping[str, Any],
    *,
    runner: Runner = _default_runner,
) -> list[str]:
    """Check locally installed OpenMPI exposes every generated launch option.

    This never invokes ``mpirun`` with ranks and therefore cannot launch a GPU process.
    """

    validate_command_manifest(manifest)
    errors: list[str] = []
    executable = shutil.which("mpirun")
    if executable is None:
        return ["mpirun is not installed"]
    return_code, stdout, stderr = runner([executable, "--help", "all"])
    help_text = f"{stdout}\n{stderr}"
    if return_code != 0 and not help_text:
        return ["mpirun --help all failed"]
    for option in (
        "--np",
        "--rankfile",
        "--bind-to",
        "--report-bindings",
        "--display-map",
        "--output-filename",
    ):
        if option not in help_text:
            errors.append(f"installed OpenMPI help does not advertise {option}")
    return errors


def verify_installed_distributed_stack(
    manifest: Mapping[str, Any],
    *,
    runner: Runner = _default_runner,
) -> list[str]:
    """Verify MPI launch flags plus CUDA/NCCL headers and runtime without launching ranks."""

    errors = verify_openmpi_command_shape(manifest, runner=runner)
    checks = (
        ([_nvcc_executable(), "--version"], "CUDA compiler is unavailable"),
        (
            ["dpkg-query", "-W", "-f=${Version}", "libnccl2"],
            "NCCL runtime package is unavailable",
        ),
        (
            ["dpkg-query", "-W", "-f=${Version}", "libnccl-dev"],
            "NCCL development package is unavailable",
        ),
        (["ldconfig", "-p"], "dynamic linker cache is unavailable"),
    )
    outputs: dict[str, str] = {}
    for command, message in checks:
        return_code, stdout, stderr = runner(command)
        if return_code != 0:
            errors.append(message)
        outputs[command[0]] = f"{stdout}\n{stderr}"
    if "libnccl.so" not in outputs.get("ldconfig", ""):
        errors.append("NCCL shared library is absent from the dynamic linker cache")
    return errors


def assert_physical_execution_permitted(
    manifest: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> None:
    """Final guard called immediately before any launcher subprocess."""

    validate_command_manifest(manifest)
    if not manifest.get("physical_execution_permitted"):
        raise PreflightError("logical dry-run manifests are non-executable")
    if preflight.get("status") != "passed":
        raise PreflightError("physical preflight did not pass")
    if manifest.get("preflight_fingerprint") != preflight.get("fingerprint"):
        raise PreflightError("manifest/preflight topology fingerprint mismatch")
    if manifest.get("repository_commit") != preflight.get("repository", {}).get("commit"):
        raise PreflightError("manifest/preflight repository commit mismatch")
    executable = manifest.get("executable", {})
    if not executable.get("sha256"):
        raise PreflightError("physical executable hash is unavailable")
    if _sha256_file(Path(executable.get("path", ""))) != executable.get("sha256"):
        raise PreflightError("physical executable hash changed after planning")
    if manifest.get("failure_mode") and os.environ.get("SPACEPDHCG_G5_FAILURE_TEST") != "1":
        raise PreflightError("failure injection requires explicit process environment opt-in")
