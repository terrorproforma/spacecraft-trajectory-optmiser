#!/usr/bin/env python3
"""Run the frozen G4 ledger through a crash-safe single-GPU scheduler."""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import gzip
import hashlib
import json
import math
import os
import queue
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from spacepdhcg.experiments.g4 import (  # noqa: E402
    ACCEPTED_TIMING_BOUNDARY,
    POLICY_NAMES,
    G4ContractError,
    load_policy,
    sha256_path,
)
from spacepdhcg.experiments.g4_execution_contract import (  # noqa: E402
    AMENDMENT_ID,
    AMENDMENT_RECORD_FIELD,
    CLAIM_CORE_STRATUM,
    EXECUTOR_DEFECT_DISPOSITION,
    REPLAY_DISPOSITION,
    ExecutionGroup,
    LoadedAmendment,
    amended_claim_core_groups,
    amended_schedule_sha256,
    deterministic_replay_eligible,
    deterministic_trace_hash,
    group_censoring,
    iter_claim_core_groups,
    load_claim_core,
    load_claim_core_amendment,
    validate_attempt_record,
)
from spacepdhcg.experiments.g4_scheduler import (  # noqa: E402
    INVALID_EXECUTOR_DEFECT,
    CampaignStore,
    Claim,
    atomic_create,
)
from spacepdhcg.experiments.paper1 import (  # noqa: E402
    Paper1ResultError,
    validate_paper1_result_schema,
)

CAPABILITY_AXES = {
    "family",
    "intervals",
    "policy",
    "quality_tier",
    "conditioning",
    "scaling_mode",
    "warm_start_mode",
    "family_classes",
    "evaluation_seed",
    "repeat",
    "solver_order",
}


GROUP_SAFETY_GRACE_SECONDS = 300
SHARED_GPU_LOCK_FILE = Path("/home/angus/.spacepdhcg-gpu.lock")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def runtime_libraries(executable: Path) -> dict[str, str]:
    """SHA-256 of every SpacePDHCG shared library the executable resolves through ``ldd``."""

    try:
        completed = subprocess.run(
            ["ldd", str(executable)], check=False, capture_output=True, text=True
        )
    except OSError:
        return {}
    libraries: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "spacepdhcg" not in line or "=>" not in line:
            continue
        target = line.split("=>", 1)[1].split("(", 1)[0].strip()
        if target and Path(target).is_file():
            libraries[Path(target).name] = sha256_path(Path(target))
    return libraries


def locked_policy(repository: Path) -> tuple[dict[str, Any], str, str]:
    lock = (repository / "benchmarks/g4_policy.sha256").read_text().split()
    if len(lock) != 2 or lock[1] != "g4_policy.json":
        raise G4ContractError("invalid G4 policy lock")
    loaded = load_policy(repository / "benchmarks/g4_policy.json", expected_sha256=lock[0])
    matrix_sha256 = hashlib.sha256(canonical_bytes(loaded.values["matrix"])).hexdigest()
    return loaded.values, loaded.sha256, matrix_sha256


def load_capabilities(
    path: Path,
    executable: Path,
    policy_sha256: str,
    matrix_sha256: str,
    source_commit: str,
    *,
    require_persistent_group: bool = False,
    amendment: LoadedAmendment | None = None,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise G4ContractError("unsupported G4 executor capability schema")
    if value.get("policy_sha256") != policy_sha256:
        raise G4ContractError("executor capability policy hash mismatch")
    if value.get("source_commit") != source_commit:
        raise G4ContractError("executor capability source commit mismatch")
    if value.get("matrix_sha256") != matrix_sha256:
        raise G4ContractError("executor capability matrix hash mismatch")
    if value.get("executable_sha256") != sha256_path(executable):
        raise G4ContractError("executor capability executable hash mismatch")
    if "runtime_library_sha256" in value and value["runtime_library_sha256"] != runtime_libraries(
        executable
    ):
        raise G4ContractError("executor capability runtime shared-library hash mismatch")
    if set(value.get("axes", {})) != CAPABILITY_AXES:
        raise G4ContractError("executor capability does not audit every frozen axis")
    if any(
        axis.get("status") not in {"applied", "execution_only"} for axis in value["axes"].values()
    ):
        raise G4ContractError("executor cannot apply every roadmap-required axis")
    if value.get("timing_boundary") != ACCEPTED_TIMING_BOUNDARY:
        raise G4ContractError("executor timing boundary is not the frozen common boundary")
    if value.get("independent_replay") is not True:
        raise G4ContractError("executor lacks independent nonlinear replay")
    if require_persistent_group:
        execution = value.get("execution_contract", {})
        if execution != {
            "version": "g4-persistent-group-v1",
            "one_process_per_group": True,
            "persistent_session": True,
            "persistent_workspace": True,
            "separate_attempt_records": True,
            "policy_reset_between_attempts": True,
        }:
            raise G4ContractError("executor lacks the persistent nine-attempt group contract")
        expected_contracts = {
            "applicability": sha256_path(REPOSITORY / "benchmarks/g4_applicability.json"),
            "claim_core": sha256_path(REPOSITORY / "benchmarks/g4_h5_h6_claim_core.json"),
            "execution_group_schema": sha256_path(
                REPOSITORY / "experiments/schema/g4_execution_group.schema.json"
            ),
            "raw_attempt_schema": sha256_path(
                REPOSITORY / "experiments/schema/g4_raw_attempt.schema.json"
            ),
            "paper1_result_schema": sha256_path(
                REPOSITORY / "experiments/schema/paper1_result.schema.json"
            ),
        }
        expected_contracts["claim_core_amendment"] = sha256_path(
            REPOSITORY / "benchmarks/g4_claim_core_amendment_v1_1.json"
        )
        if value.get("contract_hashes") != expected_contracts:
            raise G4ContractError("executor capability authoritative contract hash mismatch")
        if value.get("compiled_source_commit") != source_commit:
            raise G4ContractError(
                "executor was configured at a different commit than the campaign source commit"
            )
        if amendment is not None:
            if expected_contracts.get("claim_core_amendment") != amendment.sha256:
                raise G4ContractError("executor capability pins a different claim-core amendment")
            if amendment.values["amendment_id"] not in value.get("policy_amendments_supported", []):
                raise G4ContractError("executor does not support the requested amendment")
        probe = dict(value.get("session_probe", {}))
        ipm_probe = probe.pop("pure_gpu_ipm_probe", None)
        if probe != {
            "kind": "real_cuda_session",
            "attempt_count": 9,
            "warmup_count": 2,
            "measured_count": 7,
            "same_process": True,
            "same_context": True,
            "same_workspace": True,
            "zero_post_create_topology_allocations": True,
            "zero_post_create_topology_index_copies": True,
        }:
            raise G4ContractError("executor capability lacks a passing real session probe")
        # The capability must have proven a real pure-gpu-ipm session (>= 1 QOCO workspace per
        # attempt, solver dispositions only), and this worker must dlopen the very library that
        # probe used: a worker environment without it would otherwise fail every IPM attempt.
        if (
            not isinstance(ipm_probe, Mapping)
            or ipm_probe.get("policy") != "pure-gpu-ipm"
            or any(
                not isinstance(item, int) or item < 1
                for item in ipm_probe.get("qoco_workspace_creations", [])
            )
            or len(ipm_probe.get("qoco_workspace_creations", [])) != 9
            or any(
                item not in {"qualified", "unqualified"}
                for item in ipm_probe.get("dispositions", [None])
            )
        ):
            raise G4ContractError("executor capability lacks a passing pure-gpu-ipm probe")
        ipm_library = value.get("ipm_library")
        if not isinstance(ipm_library, Mapping) or not ipm_library.get("sha256"):
            raise G4ContractError("executor capability does not pin the IPM library")
        worker_library = os.environ.get("SPACEPDHCG_QOCO_LIBRARY", "")
        if not worker_library or not Path(worker_library).is_file():
            raise G4ContractError(
                "SPACEPDHCG_QOCO_LIBRARY is unset or missing in the worker environment; the "
                "pure-gpu-ipm and hybrid policies cannot run"
            )
        if sha256_path(Path(worker_library)) != ipm_library["sha256"]:
            raise G4ContractError(
                "worker SPACEPDHCG_QOCO_LIBRARY differs from the library the capability probed"
            )
    declared_hash = value.get("capability_sha256")
    payload = {key: item for key, item in value.items() if key != "capability_sha256"}
    if declared_hash != hashlib.sha256(canonical_bytes(payload)).hexdigest():
        raise G4ContractError("executor capability content hash mismatch")
    return value


class NvmlPower:
    """Minimal direct NVML binding; avoids a subprocess per power sample."""

    def __init__(self, library: str | None = None, device_index: int = 0) -> None:
        candidates = (
            library,
            "/usr/lib/wsl/lib/libnvidia-ml.so.1",
            "libnvidia-ml.so.1",
        )
        last_error: OSError | None = None
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                self.library = ctypes.CDLL(candidate)
                break
            except OSError as error:
                last_error = error
        else:
            raise G4ContractError(f"NVML library unavailable: {last_error}")
        self.library.nvmlInit_v2.restype = ctypes.c_int
        self.library.nvmlDeviceGetHandleByIndex_v2.argtypes = [
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.library.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
        self.library.nvmlDeviceGetPowerUsage.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
        ]
        self.library.nvmlDeviceGetPowerUsage.restype = ctypes.c_int
        if self.library.nvmlInit_v2() != 0:
            raise G4ContractError("NVML initialization failed")
        self.handle = ctypes.c_void_p()
        if self.library.nvmlDeviceGetHandleByIndex_v2(device_index, ctypes.byref(self.handle)) != 0:
            raise G4ContractError(f"NVML device {device_index} unavailable")

    def watts(self) -> float:
        milliwatts = ctypes.c_uint()
        status = self.library.nvmlDeviceGetPowerUsage(self.handle, ctypes.byref(milliwatts))
        if status != 0:
            raise RuntimeError(f"NVML power query failed with status {status}")
        return milliwatts.value / 1000.0


class EnergySampler:
    def __init__(
        self,
        power: NvmlPower,
        interval_seconds: float = 0.05,
        cpu_core: int | None = None,
    ) -> None:
        self.power = power
        self.interval_seconds = interval_seconds
        self.cpu_core = cpu_core
        self.samples: list[tuple[float, float]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._first_sample = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        if self.cpu_core is not None and hasattr(os, "sched_setaffinity"):
            try:
                os.sched_setaffinity(threading.get_native_id(), {self.cpu_core})
            except OSError as error:
                self.errors.append(f"sampler affinity: {error}")
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                watts = self.power.watts()
                self.samples.append((time.monotonic(), watts))
            except RuntimeError as error:
                self.errors.append(str(error))
            finally:
                self._first_sample.set()
            self._stop.wait(max(0.0, self.interval_seconds - (time.monotonic() - started)))

    def start(self) -> None:
        self._thread.start()
        if not self._first_sample.wait(timeout=5.0):
            raise G4ContractError("NVML sampler did not produce its boundary sample")

    def finish(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join()
        try:
            watts = self.power.watts()
            self.samples.append((time.monotonic(), watts))
        except RuntimeError as error:
            self.errors.append(str(error))
        gaps = [
            right[0] - left[0] for left, right in zip(self.samples, self.samples[1:], strict=False)
        ]
        joules = sum(
            0.5 * (left[1] + right[1]) * (right[0] - left[0])
            for left, right in zip(self.samples, self.samples[1:], strict=False)
        )
        maximum_gap = max(gaps) if gaps else None
        return {
            "source": "nvml-c-api",
            "scope": "GPU-only",
            "sampling_interval_seconds": self.interval_seconds,
            "sample_count": len(self.samples),
            "maximum_gap_seconds": maximum_gap,
            "gap_valid": maximum_gap is not None and maximum_gap <= 0.1,
            "joules": joules,
            "errors": self.errors,
            "shared_display_gpu": True,
        }


def parse_pmon(text: str) -> list[dict[str, Any]]:
    """Parse ``nvidia-smi pmon`` rows into pid/type/sm/mem/command records."""

    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 10:
            continue
        try:
            pid = int(fields[1])
        except ValueError:
            continue

        def percent(value: str) -> int | None:
            return None if value == "-" else int(value)

        rows.append(
            {
                "pid": pid,
                "type": fields[2],
                "sm_percent": percent(fields[3]),
                "mem_percent": percent(fields[4]),
                "command": " ".join(fields[9:]),
            }
        )
    return rows


def host_compute_activity(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split host compute-only contexts into active (SM or memory busy) and idle."""

    active: list[dict[str, Any]] = []
    idle: list[dict[str, Any]] = []
    for row in rows:
        if row["type"] != "C":
            continue
        busy = bool(row["sm_percent"]) or bool(row["mem_percent"])
        (active if busy else idle).append(row)
    return active, idle


class GpuContaminationMonitor:
    """Detect foreign GPU compute activity before, during, and after one execution group.

    WSL2 ``nvidia-smi`` cannot enumerate GPU processes, so three signals are combined:

    1. ``nvidia-smi`` compute-apps, utilization, memory, and power, recorded verbatim;
    2. ``/dev/dxg`` holders inside the VM that are not this scheduler or its descendants
       (and are not ``nvidia-smi`` itself);
    3. host ``nvidia-smi.exe pmon`` compute-only (``C``) contexts with their SM/memory
       utilization.

    Any foreign VM holder is contaminating whenever it is present. A host compute context is
    contaminating when it reports non-zero SM or memory utilization; an idle host context is
    retained as ``host_idle_compute_contexts`` evidence without censoring the group.
    """

    def __init__(
        self,
        nvidia_smi: str = "nvidia-smi",
        host_nvidia_smi: str | None = "/mnt/c/Windows/System32/nvidia-smi.exe",
        interval_seconds: float = 1.0,
    ) -> None:
        self.nvidia_smi = nvidia_smi
        self.host_nvidia_smi = (
            host_nvidia_smi if host_nvidia_smi and Path(host_nvidia_smi).is_file() else None
        )
        self.interval_seconds = interval_seconds
        self.own_pid = os.getpid()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, Any]] = []
        self._errors: list[str] = []

    @staticmethod
    def _now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _run(self, command: list[str], timeout: float = 20.0) -> str:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=timeout
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{command[0]} exited {completed.returncode}: {completed.stderr.strip()[-300:]}"
            )
        return completed.stdout

    def sample_nvidia_smi(self) -> dict[str, Any]:
        sample: dict[str, Any] = {"at": self._now()}
        try:
            apps = self._run(
                [self.nvidia_smi, "--query-compute-apps=pid,name,used_memory", "--format=csv"]
            )
            sample["compute_apps_csv"] = apps
            sample["compute_apps"] = [
                line.strip() for line in apps.splitlines()[1:] if line.strip()
            ]
            gpu = self._run(
                [
                    self.nvidia_smi,
                    "--query-gpu=utilization.gpu,memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                ]
            )
            utilization, memory, power = (item.strip() for item in gpu.strip().split(","))
            sample["utilization_percent"] = float(utilization)
            sample["memory_used_mib"] = float(memory)
            sample["power_watts"] = float(power)
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
            sample["error"] = str(error)
        return sample

    def _is_own(self, pid: int) -> bool:
        seen: set[int] = set()
        current = pid
        while current > 1 and current not in seen:
            if current == self.own_pid:
                return True
            seen.add(current)
            try:
                status = Path(f"/proc/{current}/status").read_text()
            except OSError:
                return False
            parent = next(
                (line.split()[1] for line in status.splitlines() if line.startswith("PPid:")),
                "0",
            )
            current = int(parent)
        return False

    def dxg_holders(self) -> list[dict[str, Any]]:
        """Return foreign VM processes holding ``/dev/dxg`` (GPU paravirtualization)."""

        holders: list[dict[str, Any]] = []
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            try:
                descriptors = os.listdir(f"/proc/{pid}/fd")
                opened = any(
                    os.readlink(f"/proc/{pid}/fd/{descriptor}") == "/dev/dxg"
                    for descriptor in descriptors
                )
            except OSError:
                continue
            if not opened:
                continue
            try:
                comm = Path(f"/proc/{pid}/comm").read_text().strip()
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
            except OSError:
                continue
            if comm == "nvidia-smi" or self._is_own(pid):
                continue
            holders.append(
                {
                    "pid": pid,
                    "comm": comm,
                    "cmdline": cmdline.decode(errors="replace")[:300],
                    "cuda_disabled": self._cuda_disabled(pid),
                }
            )
        return holders

    @staticmethod
    def _cuda_disabled(pid: int) -> bool:
        """True when the holder's environment hides every CUDA device from it.

        WSL2 processes open ``/dev/dxg`` as soon as the CUDA driver initialises, even when
        ``CUDA_VISIBLE_DEVICES`` is empty and no kernel can ever reach the GPU. Such holders are
        retained as evidence but are not foreign compute. An unreadable environment is treated
        conservatively as CUDA-enabled.
        """

        try:
            environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        except OSError:
            return False
        for item in environment:
            if item.startswith(b"CUDA_VISIBLE_DEVICES="):
                value = item.split(b"=", 1)[1].strip().lower()
                return value in {b"", b"-1", b"none", b"nodevfiles"}
        return False

    def host_compute_contexts(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
        if self.host_nvidia_smi is None:
            return [], [], ""
        text = self._run([self.host_nvidia_smi, "pmon", "-c", "1"])
        active, idle = host_compute_activity(parse_pmon(text))
        return active, idle, text

    def utilization(self) -> dict[str, Any]:
        """Whole-GPU utilization/memory/power; the delta against the boundaries is evidence."""

        try:
            gpu = self._run(
                [
                    self.nvidia_smi,
                    "--query-gpu=utilization.gpu,memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                timeout=10.0,
            )
            utilization, memory, power = (item.strip() for item in gpu.strip().split(","))
            return {
                "utilization_percent": float(utilization),
                "memory_used_mib": float(memory),
                "power_watts": float(power),
            }
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
            return {"utilization_error": str(error)}

    def probe(self) -> dict[str, Any]:
        """One combined foreign-activity observation."""

        probe: dict[str, Any] = {"at": self._now(), "monotonic": time.monotonic()}
        probe.update(self.utilization())
        try:
            holders = self.dxg_holders()
        except OSError as error:
            holders = []
            probe["dxg_error"] = str(error)
        probe["wsl_foreign_processes"] = [
            holder for holder in holders if not holder.get("cuda_disabled")
        ]
        probe["wsl_cuda_disabled_holders"] = [
            holder for holder in holders if holder.get("cuda_disabled")
        ]
        try:
            active, idle, _ = self.host_compute_contexts()
            probe["host_active_compute_processes"] = active
            probe["host_idle_compute_contexts"] = idle
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            probe["host_active_compute_processes"] = []
            probe["host_idle_compute_contexts"] = []
            probe["host_error"] = str(error)
        probe["foreign"] = bool(
            probe["wsl_foreign_processes"] or probe["host_active_compute_processes"]
        )
        return probe

    def boundary_sample(self) -> dict[str, Any]:
        """Full before/after sample: nvidia-smi query plus a process probe."""

        sample = {"nvidia_smi": self.sample_nvidia_smi(), **self.probe()}
        if self.host_nvidia_smi is not None:
            try:
                sample["host_pmon"] = self._run([self.host_nvidia_smi, "pmon", "-c", "1"])
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                sample["host_pmon_error"] = str(error)
        return sample

    def _watch(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self._samples.append(self.probe())
            except Exception as error:  # monitoring must never kill the measured worker
                self._errors.append(str(error))
            self._stop.wait(max(0.0, self.interval_seconds - (time.monotonic() - started)))

    def start(self) -> None:
        self._stop.clear()
        self._samples = []
        self._errors = []
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def foreign_samples(self) -> list[dict[str, Any]]:
        """Timestamped foreign observations for per-attempt attribution (run-and-flag)."""

        return [
            {
                "monotonic": sample["monotonic"],
                "at": sample["at"],
                "utilization_percent": sample.get("utilization_percent"),
                "max_sm_percent": max(
                    (
                        process.get("sm_percent") or 0
                        for process in sample.get("host_active_compute_processes", [])
                    ),
                    default=0,
                ),
                "processes": sorted(
                    {
                        f"host:{process['pid']}:{process['command']}"
                        for process in sample.get("host_active_compute_processes", [])
                    }
                    | {
                        f"wsl:{process['pid']}:{process['comm']}"
                        for process in sample.get("wsl_foreign_processes", [])
                    }
                ),
            }
            for sample in self._samples
            if sample.get("foreign")
        ]

    def sample_times(self) -> list[float]:
        return [sample["monotonic"] for sample in self._samples]

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if not self._samples:
            self._samples.append(self.probe())
        samples = list(self._samples)
        utilizations = [
            float(sample["utilization_percent"])
            for sample in samples
            if sample.get("utilization_percent") is not None
        ]
        wsl: dict[int, dict[str, Any]] = {}
        wsl_disabled: dict[int, dict[str, Any]] = {}
        host_active: dict[int, dict[str, Any]] = {}
        host_idle: dict[int, dict[str, Any]] = {}
        for sample in samples:
            for key, table in (
                ("wsl_foreign_processes", wsl),
                ("wsl_cuda_disabled_holders", wsl_disabled),
            ):
                for process in sample.get(key, []):
                    entry = table.setdefault(
                        process["pid"],
                        {
                            **process,
                            "first_seen": sample["at"],
                            "last_seen": sample["at"],
                            "samples": 0,
                        },
                    )
                    entry["last_seen"] = sample["at"]
                    entry["samples"] += 1
            for process in sample.get("host_active_compute_processes", []):
                entry = host_active.setdefault(
                    process["pid"],
                    {
                        "pid": process["pid"],
                        "command": process["command"],
                        "first_seen": sample["at"],
                        "last_seen": sample["at"],
                        "max_sm_percent": 0,
                        "max_mem_percent": 0,
                        "samples": 0,
                    },
                )
                entry["last_seen"] = sample["at"]
                entry["samples"] += 1
                entry["max_sm_percent"] = max(entry["max_sm_percent"], process["sm_percent"] or 0)
                entry["max_mem_percent"] = max(
                    entry["max_mem_percent"], process["mem_percent"] or 0
                )
            for process in sample.get("host_idle_compute_contexts", []):
                entry = host_idle.setdefault(
                    process["pid"],
                    {"pid": process["pid"], "command": process["command"], "samples": 0},
                )
                entry["samples"] += 1
        errors = list(self._errors) + [
            sample[key]
            for sample in samples
            for key in ("dxg_error", "host_error")
            if key in sample
        ]
        return {
            "foreign_detected": bool(wsl or host_active),
            "wsl_foreign_processes": sorted(wsl.values(), key=lambda item: item["pid"]),
            "wsl_cuda_disabled_holders": sorted(
                wsl_disabled.values(), key=lambda item: item["pid"]
            ),
            "host_active_compute_processes": sorted(
                host_active.values(), key=lambda item: item["pid"]
            ),
            "host_idle_compute_contexts": sorted(host_idle.values(), key=lambda item: item["pid"]),
            "sample_count": len(samples),
            "foreign_sample_count": sum(1 for sample in samples if sample.get("foreign")),
            "utilization_percent": {
                "mean": statistics.fmean(utilizations) if utilizations else None,
                "max": max(utilizations) if utilizations else None,
                "min": min(utilizations) if utilizations else None,
            },
            "interval_seconds": self.interval_seconds,
            "host_monitor_available": self.host_nvidia_smi is not None,
            "errors": errors[:50],
        }

    def wait_until_clear(self, *, poll_seconds: float = 5.0, log: Any = sys.stderr) -> int:
        """Block until two consecutive probes show no foreign compute activity."""

        waited = 0
        clear_streak = 0
        last_report = 0.0
        while True:
            probe = self.probe()
            if not probe["foreign"]:
                clear_streak += 1
                if clear_streak >= 2:
                    return waited
            else:
                clear_streak = 0
                if time.monotonic() - last_report >= 30.0:
                    last_report = time.monotonic()
                    print(
                        json.dumps(
                            {
                                "event": "waiting_for_foreign_gpu_processes",
                                "at": probe["at"],
                                "wsl_foreign_processes": probe["wsl_foreign_processes"],
                                "host_active_compute_processes": probe[
                                    "host_active_compute_processes"
                                ],
                            },
                            sort_keys=True,
                        ),
                        file=log,
                        flush=True,
                    )
            pause = 1.0 if clear_streak else poll_seconds
            time.sleep(pause)
            waited += int(pause)


class SharedGpuLock:
    """Advisory shared GPU lock file (amendment single-gpu-v1.1, run-and-flag evidence).

    The worker holds ``flock`` on ``/home/angus/.spacepdhcg-gpu.lock`` for the whole group and
    writes a JSON payload naming itself. A foreign payload or a held lock is recorded, never
    waited for: the group runs and its attempts are flagged by the GPU monitor instead.
    """

    def __init__(self, path: Path = SHARED_GPU_LOCK_FILE) -> None:
        self.path = path
        self.descriptor: int | None = None

    def acquire(self, payload: dict[str, Any]) -> dict[str, Any]:
        record: dict[str, Any] = {"path": str(self.path), "held_by_other": False}
        try:
            record["existing_payload"] = self.path.read_text(encoding="utf-8", errors="replace")[
                :1000
            ]
        except OSError:
            record["existing_payload"] = None
        try:
            self.descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
            try:
                fcntl.flock(self.descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                record["held_by_other"] = True
                os.close(self.descriptor)
                self.descriptor = None
                return record
            os.ftruncate(self.descriptor, 0)
            os.write(self.descriptor, canonical_bytes(payload) + b"\n")
            os.fsync(self.descriptor)
            record["payload"] = payload
        except OSError as error:
            record["error"] = str(error)
            if self.descriptor is not None:
                os.close(self.descriptor)
                self.descriptor = None
        return record

    def release(self) -> dict[str, Any]:
        record: dict[str, Any] = {"path": str(self.path)}
        try:
            record["payload_at_release"] = self.path.read_text(encoding="utf-8", errors="replace")[
                :1000
            ]
        except OSError:
            record["payload_at_release"] = None
        if self.descriptor is not None:
            try:
                os.ftruncate(self.descriptor, 0)
                fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            except OSError as error:
                record["error"] = str(error)
            finally:
                os.close(self.descriptor)
                self.descriptor = None
        return record


def run_group_process(
    command: list[str],
    environment: dict[str, str],
    timeout_seconds: float,
) -> tuple[str, str, int, bool, list[tuple[float, str]]]:
    """Run one executor process, timestamping every stdout line as it arrives.

    Returns stdout, stderr, returncode, timed_out, and ``[(monotonic, line), ...]`` so every raw
    attempt record can be attributed a wall-clock window for contamination flagging.
    """

    process = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None and process.stderr is not None
    timeline: list[tuple[float, str]] = []
    stderr_chunks: list[str] = []
    stdout_stream, stderr_stream = process.stdout, process.stderr

    def read_stdout() -> None:
        for line in stdout_stream:
            timeline.append((time.monotonic(), line))

    def read_stderr() -> None:
        stderr_chunks.append(stderr_stream.read())

    readers = [
        threading.Thread(target=read_stdout, daemon=True),
        threading.Thread(target=read_stderr, daemon=True),
    ]
    for reader in readers:
        reader.start()
    timed_out = False
    try:
        returncode = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        returncode = process.wait()
        returncode = 124
    for reader in readers:
        reader.join(timeout=30.0)
    stdout = "".join(line for _, line in timeline)
    return stdout, "".join(stderr_chunks), returncode, timed_out, timeline


def attempt_windows(
    timeline: list[tuple[float, str]],
    group_started: float,
) -> dict[tuple[str, int], tuple[float, float]]:
    """Wall-clock window of every emitted raw attempt: previous record (or session ready) to own."""

    windows: dict[tuple[str, int], tuple[float, float]] = {}
    previous = group_started
    for at, line in timeline:
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line.replace("-inf", "-Infinity"))
        except json.JSONDecodeError:
            continue
        case = record.get("case")
        if case == "g4_session_ready":
            previous = at
        elif case == "g4_attempt":
            key = (str(record.get("repeat_kind")), int(record.get("repeat", -1)))
            windows[key] = (previous, at)
            previous = at
    return windows


def flag_contaminated_attempts(
    attempts: list[dict[str, Any]],
    windows: dict[tuple[str, int], tuple[float, float]],
    foreign: list[dict[str, Any]],
    sample_times: list[float],
    *,
    slack_seconds: float,
) -> int:
    """Decision A: flag attempts whose window overlaps a foreign sample; never re-run.

    Replayed attempts were not executed and cannot be contaminated; they carry
    ``contaminated: false`` with an empty window summary.
    """

    flagged = 0
    for attempt in attempts:
        key = (str(attempt.get("repeat_kind")), int(attempt.get("repeat", -1)))
        start, end = windows.get(key, (0.0, 0.0))
        if attempt.get("launched") is not True:
            attempt["contaminated"] = False
            attempt["contamination"] = {
                "window_start_monotonic": start,
                "window_end_monotonic": end,
                "foreign_samples": 0,
                "total_samples": 0,
                "max_foreign_sm_percent": 0,
                "foreign_processes": [],
            }
            continue
        low, high = start - slack_seconds, end + slack_seconds
        hits = [sample for sample in foreign if low <= sample["monotonic"] <= high]
        total = sum(1 for at in sample_times if low <= at <= high)
        processes = sorted({name for sample in hits for name in sample["processes"]})
        attempt["contaminated"] = bool(hits)
        attempt["contamination"] = {
            "window_start_monotonic": start,
            "window_end_monotonic": end,
            "foreign_samples": len(hits),
            "total_samples": total,
            "max_foreign_sm_percent": max((sample["max_sm_percent"] for sample in hits), default=0),
            "foreign_processes": processes,
        }
        flagged += int(bool(hits))
    return flagged


class PersistentExecutor:
    """One long-lived process and CUDA context for sequential independent rows."""

    def __init__(
        self,
        executable: Path,
        environment: dict[str, str],
        row_deadline_seconds: int = 600,
    ) -> None:
        self.executable = executable
        self.environment = environment
        self.row_deadline_seconds = row_deadline_seconds
        self.generation = 0
        self.cuda_startup_seconds = 0.0
        self._stdout: queue.Queue[str | None] = queue.Queue()
        self._stderr: queue.Queue[str] = queue.Queue()
        self._start()

    @staticmethod
    def _read_lines(stream: Any, output: queue.Queue[Any], sentinel: bool) -> None:
        try:
            for line in stream:
                output.put(line)
        finally:
            if sentinel:
                output.put(None)

    def _start(self) -> None:
        self.generation += 1
        self._stdout = queue.Queue()
        self._stderr = queue.Queue()
        self.process = subprocess.Popen(
            [
                str(self.executable),
                "--g4-server",
                str(self.row_deadline_seconds),
            ],
            env=self.environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert self.process.stdout is not None and self.process.stderr is not None
        threading.Thread(
            target=self._read_lines,
            args=(self.process.stdout, self._stdout, True),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._read_lines,
            args=(self.process.stderr, self._stderr, False),
            daemon=True,
        ).start()
        ready = self._stdout.get(timeout=60.0)
        if ready is None:
            raise G4ContractError("persistent executor exited during CUDA startup")
        record = json.loads(ready)
        if record.get("case") != "g4_server_ready":
            raise G4ContractError(f"invalid persistent executor handshake: {record!r}")
        self.cuda_startup_seconds = float(record["cuda_startup_seconds"])

    def _stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5.0)

    def ensure_ready(self) -> None:
        if self.process.poll() is not None:
            self._start()

    def execute(
        self, command: list[str], timeout_seconds: int
    ) -> tuple[str, str, int, bool, int, float]:
        self.ensure_ready()
        assert self.process.stdin is not None
        self.process.stdin.write("\t".join(command[1:]) + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + timeout_seconds + 5.0
        lines: list[str] = []
        timeout = False
        returncode = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                timeout = True
                returncode = 124
                self._stop()
                break
            try:
                line = self._stdout.get(timeout=remaining)
            except queue.Empty:
                timeout = True
                returncode = 124
                self._stop()
                break
            if line is None:
                returncode = self.process.wait()
                break
            try:
                record = json.loads(line.replace("-inf", "-Infinity"))
            except json.JSONDecodeError:
                lines.append(line)
                continue
            if record.get("case") == "g4_server_result":
                if record.get("coordinate_id") != command[18]:
                    raise G4ContractError("persistent executor response identity mismatch")
                returncode = int(record["returncode"])
                break
            lines.append(line)
        stderr: list[str] = []
        while True:
            try:
                stderr.append(self._stderr.get_nowait())
            except queue.Empty:
                break
        generation = self.generation
        startup = self.cuda_startup_seconds
        return "".join(lines), "".join(stderr), returncode, timeout, generation, startup

    def close(self) -> None:
        if self.process.poll() is None and self.process.stdin is not None:
            try:
                self.process.stdin.write("cancel\n")
                self.process.stdin.flush()
                self.process.wait(timeout=5.0)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self._stop()


def archive_stdout(root: Path, stdout: str) -> tuple[str, int]:
    payload = stdout.encode()
    digest = hashlib.sha256(payload).hexdigest()
    target = root / "objects" / "sha256" / digest[:2] / f"{digest}.jsonl.gz"
    if not target.exists():
        compressed = gzip.compress(payload, compresslevel=6, mtime=0)
        try:
            atomic_create(target, compressed)
        except FileExistsError:
            pass
    return digest, len(payload)


def stored_metadata(campaign: Path) -> dict[str, str]:
    """Read a checkpoint's metadata table read-only (no schema creation, no lock)."""

    database = sqlite3.connect(f"file:{campaign / 'checkpoint.sqlite3'}?mode=ro", uri=True)
    try:
        return {
            str(key): str(value)
            for key, value in database.execute("SELECT key, value FROM metadata")
        }
    finally:
        database.close()


def invalidate_completed_groups(
    store: CampaignStore,
    groups: Sequence[ExecutionGroup],
    *,
    policy: str,
    disposition: str,
    reason: str,
    provenance: Mapping[str, Any],
) -> int:
    """Invalidate every completed group of one policy; refuses while a worker owns the GPU lock.

    Records are retained verbatim (see ``CampaignStore.invalidate``); the ledger rows leave the
    completed set so they are never counted, migrated or decided on.
    """

    if policy not in POLICY_NAMES:
        raise G4ContractError(f"unknown policy {policy!r}")
    lock_descriptor = os.open(store.root / "gpu-worker.lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise G4ContractError("campaign worker is still active; pause it first") from error
        rows = list(
            store.database.execute(
                "SELECT ordinal, coordinate_id FROM coordinates WHERE state = 'completed' "
                "ORDER BY ordinal"
            )
        )
        invalidated = 0
        for row in rows:
            group = groups[int(row["ordinal"])]
            if group.group_id != str(row["coordinate_id"]):
                raise G4ContractError("checkpoint coordinate content address drift")
            if group.coordinate["policy"] != policy:
                continue
            store.invalidate(
                int(row["ordinal"]),
                disposition=disposition,
                reason=reason,
                provenance=provenance,
            )
            invalidated += 1
        return invalidated
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)


def migrate_terminal_rows(store: CampaignStore, source: Path) -> dict[str, int]:
    """Import completed evidence only while the source GPU lock is unowned."""

    lock_descriptor = os.open(source / "gpu-worker.lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise G4ContractError("source campaign worker is still active") from error
        database = sqlite3.connect(f"file:{source / 'checkpoint.sqlite3'}?mode=ro", uri=True)
        database.row_factory = sqlite3.Row
        metadata = {
            row["key"]: row["value"] for row in database.execute("SELECT key, value FROM metadata")
        }
        if metadata.get("policy_sha256") != store.policy_sha256:
            raise G4ContractError("source campaign policy hash mismatch")
        if metadata.get("total_rows") != str(store.total):
            raise G4ContractError("source campaign cardinality mismatch")
        imported = 0
        already_present = 0
        rows = database.execute(
            """
            SELECT c.ordinal, c.coordinate_id, c.state, a.attempt_id,
                   a.disposition, a.reason, a.run_directory
            FROM coordinates AS c
            JOIN attempts AS a ON a.attempt_id = c.latest_attempt_id
            WHERE c.state IN ('completed', 'quarantined')
            ORDER BY c.ordinal
            """
        )
        for row in rows:
            run_directory = Path(row["run_directory"])
            changed = store.import_terminal(
                ordinal=int(row["ordinal"]),
                identifier=str(row["coordinate_id"]),
                state=str(row["state"]),
                disposition=str(row["disposition"]),
                reason=str(row["reason"]),
                coordinate_payload=(run_directory / "coordinate.json").read_bytes(),
                result_payload=(run_directory / "result.json").read_bytes(),
                source_campaign=str(source),
                source_attempt_id=str(row["attempt_id"]),
            )
            imported += int(changed)
            already_present += int(not changed)
        database.close()
        return {"imported": imported, "already_present": already_present}
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)


def command_for(
    executable: Path,
    claim: Claim,
    policy_sha256: str,
    matrix_sha256: str,
    capability_sha256: str,
) -> list[str]:
    coordinate = claim.coordinate
    family = coordinate["family"]
    if family == "P1-C-pd3":
        class_arguments = [str(coordinate["dispersion_class"]), "not_applicable"]
    elif family == "P1-D-pd6":
        class_arguments = [str(coordinate["attitude_class"]), str(coordinate["rate_class"])]
    elif family == "P1-E-low-thrust":
        class_arguments = [str(coordinate["trust_class"]), coordinate["transfer_class"]]
    else:
        raise G4ContractError(f"unsupported family {family!r}")
    return [
        str(executable),
        "--g4-sample",
        family,
        str(coordinate["intervals"]),
        coordinate["policy"],
        coordinate["warm_mode"],
        str(coordinate["quality_tolerance"]),
        "100",
        *class_arguments,
        coordinate["quality_tier"],
        coordinate["scaling_mode"],
        policy_sha256,
        str(coordinate["conditioning"]),
        str(coordinate["seed"]),
        coordinate["repeat_kind"],
        str(coordinate["repeat"]),
        str(coordinate["solver_order"]),
        claim.coordinate_id,
        matrix_sha256,
        capability_sha256,
    ]


def command_for_group(
    executable: Path,
    claim: Claim,
    policy_sha256: str,
    matrix_sha256: str,
    capability_sha256: str,
    manifest: Path,
) -> list[str]:
    """Launch exactly one executor process for a complete nine-attempt group."""

    coordinate = claim.coordinate
    if coordinate.get("record_kind") != "execution_group":
        raise G4ContractError("group command requires an execution-group claim")
    return [
        str(executable),
        "--g4-session",
        str(manifest),
        policy_sha256,
        matrix_sha256,
        capability_sha256,
    ]


def parse_records(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.startswith("{"):
            records.append(json.loads(line.replace("-inf", "-Infinity")))
    return records


def validate_success(
    claim: Claim,
    records: list[dict[str, Any]],
    policy_sha256: str,
    matrix_sha256: str,
    capability_sha256: str,
) -> tuple[bool, str]:
    samples = [record for record in records if record.get("case") == "g4_sample"]
    runtimes = [record for record in records if record.get("case") == "g4_runtime"]
    axes = [record for record in records if record.get("case") == "g4_axis_application"]
    if len(samples) != 1 or len(runtimes) != 1 or len(axes) != 1:
        return False, "executor did not emit exactly one sample, runtime, and axis record"
    sample = samples[0]
    requested = runtimes[0].get("requested", {})
    coordinate = claim.coordinate
    expected = {
        "family": coordinate["family"],
        "policy": coordinate["policy"],
        "intervals": coordinate["intervals"],
        "quality_tier": coordinate["quality_tier"],
        "quality_tolerance": coordinate["quality_tolerance"],
        "scaling_mode": coordinate["scaling_mode"],
        "warm_start_mode": coordinate["warm_mode"],
    }
    observed = {
        "family": sample.get("family"),
        "policy": sample.get("policy"),
        "intervals": sample.get("intervals"),
        "quality_tier": requested.get("quality_tier"),
        "quality_tolerance": sample.get("quality_tolerance"),
        "scaling_mode": requested.get("scaling_mode"),
        "warm_start_mode": requested.get("warm_start_mode"),
    }
    if observed != expected:
        return False, f"runtime coordinate mismatch: expected {expected!r}, observed {observed!r}"
    if not isinstance(sample.get("qualified"), bool):
        return False, "sample qualification is missing"
    axis = axes[0]
    axis_expected = {
        "coordinate_id": claim.coordinate_id,
        "policy_sha256": policy_sha256,
        "matrix_sha256": matrix_sha256,
        "capability_sha256": capability_sha256,
        "family": coordinate["family"],
        "intervals": coordinate["intervals"],
        "policy": coordinate["policy"],
        "quality_tier": coordinate["quality_tier"],
        "quality_tolerance": coordinate["quality_tolerance"],
        "conditioning_log10_span": coordinate["conditioning"],
        "scaling_mode": coordinate["scaling_mode"],
        "warm_start_mode": coordinate["warm_mode"],
        "dispersion_class": coordinate.get("dispersion_class", 0.0),
        "attitude_class": coordinate.get("attitude_class", 0.0),
        "rate_class": coordinate.get("rate_class", 0.0),
        "trust_class": coordinate.get("trust_class", 0.0),
        "transfer_class": coordinate.get("transfer_class", "not_applicable"),
        "evaluation_seed": coordinate["seed"],
        "instance": coordinate["instance"],
        "repeat_kind": coordinate["repeat_kind"],
        "repeat": coordinate["repeat"],
        "solver_order": coordinate["solver_order"],
    }
    axis_observed = {key: axis.get(key) for key in axis_expected}
    if axis_observed != axis_expected:
        return (
            False,
            f"applied-axis mismatch: expected {axis_expected!r}, observed {axis_observed!r}",
        )
    for name in ("instance_hash", "problem_hash", "coefficient_hash"):
        value = axis.get(name)
        if not isinstance(value, str) or len(value) != 16:
            return False, f"{name} is missing or invalid"
    factor_min = axis.get("condition_factor_min")
    factor_max = axis.get("condition_factor_max")
    if (
        not isinstance(factor_min, (int, float))
        or not isinstance(factor_max, (int, float))
        or not math.isclose(
            factor_max / factor_min,
            10.0 ** coordinate["conditioning"],
            rel_tol=1.0e-12,
        )
    ):
        return False, "conditioning factors do not realize the requested logarithmic span"
    if axis.get("coefficient_parity_relative", float("inf")) > 5.0e-12:
        return False, "CPU/GPU conditioned coefficient parity failed"
    return True, "strict runtime and sample records validated"


def validate_amendment_records(
    records: list[dict[str, Any]],
    expected: dict[str, Any] | None,
) -> str | None:
    """Amendment single-gpu-v1.1 echo, trace-hash and replay consistency checks.

    ``expected`` is ``{"censoring_stratum", "attempt_deadline_seconds", "inner_iteration_cap"}``
    for the scheduled group, or ``None`` when no amendment is in force (records must then carry
    no amendment fields).
    """

    ordered = sorted(records, key=lambda item: (item["repeat_kind"] != "warmup", item["repeat"]))
    for record in ordered:
        if expected is None:
            if AMENDMENT_RECORD_FIELD in record or record.get("disposition") == REPLAY_DISPOSITION:
                return "raw attempt carries amendment fields without an amendment in force"
            continue
        if record.get(AMENDMENT_RECORD_FIELD) != AMENDMENT_ID:
            return f"raw attempt lacks {AMENDMENT_RECORD_FIELD}={AMENDMENT_ID}"
        echoed = record.get("amendment", {})
        if (
            echoed.get("censoring_stratum") != expected["censoring_stratum"]
            or float(echoed.get("attempt_deadline_seconds", -1))
            != float(expected["attempt_deadline_seconds"])
            or echoed.get("inner_iteration_cap") != expected["inner_iteration_cap"]
            or echoed.get("deterministic_replay") is not True
        ):
            return f"raw attempt amendment echo {echoed!r} differs from the scheduled {expected!r}"
        try:
            recomputed = deterministic_trace_hash(record["disposition"], record["trace"])
        except (KeyError, TypeError, ValueError, G4ContractError) as error:
            return f"raw attempt trace unusable: {error}"
        if recomputed != record.get("trace_hash"):
            return "raw attempt trace_hash differs from the reference recomputation"
    if expected is None:
        return None
    replays = [record for record in ordered if record.get("disposition") == REPLAY_DISPOSITION]
    try:
        eligible = deterministic_replay_eligible(ordered[:3])
    except G4ContractError as error:
        return str(error)
    if replays and not eligible:
        return "executor replayed timeouts although the first three traces were not identical"
    if eligible and len(replays) != 6:
        return "executor executed measured/1..6 although deterministic replay was required"
    for record in replays:
        if record.get("trace_hash") != deterministic_trace_hash(
            REPLAY_DISPOSITION, ordered[2]["trace"]
        ):
            return "replayed attempt does not repeat the measured/0 trace"
    return None


def validate_group_success(
    claim: Claim,
    records: list[dict[str, Any]],
    amendment_expected: dict[str, Any] | None = None,
) -> tuple[bool, str, list[dict[str, Any]]]:
    """Validate distinct raw attempts and every measured Paper 1 result."""

    emitted = [record for record in records if record.get("case") == "g4_attempt"]
    planned = claim.coordinate["attempts"]
    if len(emitted) != len(planned):
        return False, "executor did not emit all nine raw attempt records", emitted
    by_repeat: dict[tuple[str, int], dict[str, Any]] = {}
    for record in emitted:
        key = (str(record.get("repeat_kind")), int(record.get("repeat", -1)))
        if key in by_repeat:
            return False, f"duplicate raw attempt record {key!r}", emitted
        by_repeat[key] = record
    expected_keys = {(item["repeat_kind"], item["repeat"]) for item in planned}
    if set(by_repeat) != expected_keys:
        return False, "raw attempt repeat set differs from the group manifest", emitted
    raw_schema = json.loads(
        (REPOSITORY / "experiments/schema/g4_raw_attempt.schema.json").read_text(encoding="utf-8")
    )
    raw_validator = Draft202012Validator(raw_schema)
    for item in planned:
        key = (item["repeat_kind"], item["repeat"])
        record = by_repeat[key]
        for field in (
            "group_id",
            "family",
            "intervals",
            "policy",
            "instance",
            "seed",
            "repeat_kind",
            "repeat",
            "statistics_eligible",
        ):
            if record.get(field) != item.get(field):
                return False, f"raw attempt field {field} differs from manifest", emitted
        try:
            raw_validator.validate(record)
            validate_attempt_record(record)
            if item["repeat_kind"] == "measured":
                result = record.get("paper1_result")
                if not isinstance(result, dict):
                    raise Paper1ResultError("completed measured attempt omitted paper1_result")
                if result.get("identity", {}).get("record_scope") != "measured_attempt":
                    raise Paper1ResultError(
                        "raw measured evidence must use record_scope=measured_attempt"
                    )
                validate_paper1_result_schema(
                    result,
                    REPOSITORY / "experiments/schema/paper1_result.schema.json",
                )
        except (G4ContractError, Paper1ResultError, ValidationError, ValueError) as error:
            return False, f"strict measured-result validation failed: {error}", emitted
    problem = validate_amendment_records(emitted, amendment_expected)
    if problem is not None:
        return False, f"amendment consistency failed: {problem}", emitted
    return True, "all raw attempts and measured Paper 1 results validated", emitted


def execute_group(
    store: CampaignStore,
    claim: Claim,
    executable: Path,
    _executor: PersistentExecutor,
    power: NvmlPower,
    policy_sha256: str,
    matrix_sha256: str,
    capability_sha256: str,
    timeout_seconds: int,
    sampler_cpu_core: int | None,
    monitor: GpuContaminationMonitor | None = None,
    amendment: LoadedAmendment | None = None,
    group: ExecutionGroup | None = None,
    shared_lock: SharedGpuLock | None = None,
) -> str:
    """Execute warmups and measurements in one persistent process/session/workspace.

    Returns the ledger disposition. Under amendment single-gpu-v1.1 (Decision A, run-and-flag)
    the group always completes: attempts whose wall-clock window overlaps foreign GPU compute
    are flagged ``contaminated`` in place and are never re-run. Without an amendment the
    original single-gpu-v1 quarantine-and-re-run behaviour is retained.
    """

    run_directory = store.root / "runs" / claim.coordinate_id / claim.attempt_id
    manifest = run_directory / "execution-group.json"
    atomic_create(manifest, canonical_bytes(claim.coordinate) + b"\n")
    censoring: dict[str, int] | None = None
    amendment_expected: dict[str, Any] | None = None
    if amendment is not None:
        if group is None:
            raise G4ContractError("amended execution requires the scheduled group")
        censoring = group_censoring(group, amendment.values)
        timeout_seconds = censoring["attempt_deadline_seconds"]
        amendment_expected = {
            "censoring_stratum": group.coordinate.get("censoring_stratum") or CLAIM_CORE_STRATUM,
            "attempt_deadline_seconds": censoring["attempt_deadline_seconds"],
            "inner_iteration_cap": censoring["inner_iteration_cap"],
        }
    group_deadline = timeout_seconds * 9 + 60
    contamination: dict[str, Any] = {
        "monitored": monitor is not None,
        "policy": "run_and_flag" if amendment is not None else "quarantine_and_rerun",
    }
    if shared_lock is not None:
        contamination["lock_file"] = shared_lock.acquire(
            {
                "pid": os.getpid(),
                "campaign": str(store.root),
                "group_id": claim.coordinate_id,
                "attempt_id": claim.attempt_id,
                "started": GpuContaminationMonitor._now(),
            }
        )
    if monitor is not None:
        contamination["before"] = monitor.boundary_sample()
        monitor.start()
    command = command_for_group(
        executable,
        claim,
        policy_sha256,
        matrix_sha256,
        capability_sha256,
        manifest,
    )
    atomic_create(run_directory / "command.json", canonical_bytes(command) + b"\n")
    run_environment = dict(os.environ)
    run_environment.update(
        {
            "SPACEPDHCG_G4_GROUP_ID": claim.coordinate_id,
            "SPACEPDHCG_G4_POLICY_RESET": "independent-with-persistent-workspace",
            "SPACEPDHCG_G4_ATTEMPT_DEADLINE_SECONDS": str(timeout_seconds),
            "SPACEPDHCG_G4_GROUP_DEADLINE_SECONDS": str(group_deadline),
        }
    )
    if amendment is not None and censoring is not None and amendment_expected is not None:
        run_environment.update(
            {
                "SPACEPDHCG_G4_POLICY_AMENDMENT": amendment.values["amendment_id"],
                "SPACEPDHCG_G4_CENSORING_STRATUM": str(amendment_expected["censoring_stratum"]),
                "SPACEPDHCG_G4_INNER_ITERATION_CAP": str(censoring["inner_iteration_cap"]),
                "SPACEPDHCG_G4_DETERMINISTIC_REPLAY": "1",
            }
        )
    started = time.monotonic()
    histories: list[dict[str, Any]] = []
    timeline: list[tuple[float, str]] = []
    for generation in range(2):
        sampler = EnergySampler(power, cpu_core=sampler_cpu_core)
        sampler.start()
        generation_started = time.monotonic()
        # The executor owns the group deadline (nine attempts plus 60 s) and emits explicit
        # unlaunched records when it expires; the outer boundary only guards against a hung
        # process, so it must sit strictly beyond the executor's own deadline.
        stdout, stderr, returncode, process_timed_out, timeline = run_group_process(
            command,
            {**run_environment, "SPACEPDHCG_G4_RESTART_GENERATION": str(generation)},
            group_deadline + GROUP_SAFETY_GRACE_SECONDS,
        )
        energy = sampler.finish()
        partial = [record for record in parse_records(stdout) if record.get("case") == "g4_attempt"]
        histories.append(
            {
                "generation": generation,
                "returncode": returncode,
                "timed_out": process_timed_out,
                "partial_attempts": partial,
                "stdout": stdout,
                "stderr": stderr,
                "energy": energy,
                "started_monotonic": generation_started,
            }
        )
        if not process_timed_out and returncode == 0:
            break
        if generation == 0:
            atomic_create(run_directory / "stdout.restart-0.jsonl", stdout.encode())
            atomic_create(run_directory / "stderr.restart-0.log", stderr.encode())
    elapsed = time.monotonic() - started
    foreign_samples: list[dict[str, Any]] = []
    sample_times: list[float] = []
    if monitor is not None:
        contamination["during"] = monitor.stop()
        foreign_samples = monitor.foreign_samples()
        sample_times = monitor.sample_times()
        contamination["after"] = monitor.boundary_sample()
        contamination["foreign_detected"] = bool(
            contamination["during"]["foreign_detected"]
            or contamination["before"].get("foreign")
            or contamination["after"].get("foreign")
        )
        before_utilization = contamination["before"].get("utilization_percent")
        after_utilization = contamination["after"].get("utilization_percent")
        during_mean = contamination["during"]["utilization_percent"]["mean"]
        contamination["utilization_delta_percent"] = {
            "before": before_utilization,
            "during_mean": during_mean,
            "after": after_utilization,
            "during_minus_before": (
                during_mean - before_utilization
                if during_mean is not None and before_utilization is not None
                else None
            ),
        }
        contamination["rule"] = (
            "run-and-flag (amendment single-gpu-v1.1): an attempt whose wall-clock window "
            "(previous record or session-ready to its own record, +/- one probe interval) "
            "overlaps any sample with a foreign VM /dev/dxg holder with CUDA devices visible or "
            "a host compute-only context with non-zero SM/memory utilization is flagged "
            "contaminated; disposition and quality are retained, timing and energy are invalid "
            "for statistics, and the group is never re-run"
            if amendment is not None
            else "foreign VM /dev/dxg holder with CUDA devices visible, or host compute-only "
            "context with non-zero SM/memory utilization, at any sample from the pre-group "
            "boundary through the post-group boundary censors the whole group as contaminated; "
            "idle host contexts and holders whose CUDA_VISIBLE_DEVICES hides every device are "
            "recorded but do not censor"
        )
    if shared_lock is not None:
        contamination["lock_file_release"] = shared_lock.release()
    atomic_create(run_directory / "stdout.jsonl", stdout.encode())
    atomic_create(run_directory / "stderr.log", stderr.encode())
    if process_timed_out:
        valid, reason, attempts = (
            False,
            "persistent group process exceeded its nine-attempt safety boundary",
            [],
        )
    elif returncode != 0:
        valid, reason, attempts = (
            False,
            f"persistent group process exited {returncode}; no unlaunched row was censored",
            [],
        )
    else:
        try:
            valid, reason, attempts = validate_group_success(
                claim, parse_records(stdout), amendment_expected
            )
        except (G4ContractError, json.JSONDecodeError, ValueError) as error:
            valid, reason, attempts = False, f"invalid group executor records: {error}", []
    contaminated_attempts = 0
    if monitor is not None and amendment is not None:
        windows = attempt_windows(timeline, histories[-1]["started_monotonic"])
        contaminated_attempts = flag_contaminated_attempts(
            attempts,
            windows,
            foreign_samples,
            sample_times,
            slack_seconds=monitor.interval_seconds,
        )
        contamination["contaminated_attempts"] = contaminated_attempts
        contamination["contaminated_measured_attempts"] = sum(
            1
            for item in attempts
            if item.get("contaminated") and item.get("repeat_kind") == "measured"
        )
        contamination["foreign_samples"] = foreign_samples[:2000]
    if monitor is not None:
        atomic_create(
            run_directory / "gpu-contamination.json",
            canonical_bytes(contamination) + b"\n",
        )
    record = {
        "schema_version": "1.0.0",
        "record_kind": "execution_group_result",
        "group_id": claim.coordinate_id,
        "attempt_id": claim.attempt_id,
        "command": command,
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "energy": energy,
        "restart_count": len(histories) - 1,
        "restart_history": [
            {
                "generation": item["generation"],
                "returncode": item["returncode"],
                "timed_out": item["timed_out"],
                "partial_attempt_count": len(item["partial_attempts"]),
            }
            for item in histories
        ],
        "raw_attempts": attempts,
        "reason": reason,
        "gpu_contamination": {
            key: value for key, value in contamination.items() if key != "foreign_samples"
        },
    }
    if amendment is not None and censoring is not None:
        record[AMENDMENT_RECORD_FIELD] = amendment.values["amendment_id"]
        record["policy_amendment_sha256"] = amendment.sha256
        record["censoring"] = {
            "stratum": amendment_expected["censoring_stratum"] if amendment_expected else None,
            **censoring,
        }
        record["deterministic_replay_applied"] = any(
            item.get("disposition") == REPLAY_DISPOSITION for item in attempts
        )
    disposition = "completed_group" if valid else "invalid_evidence"
    defective_attempts = [
        item.get("attempt_id")
        for item in attempts
        if item.get("disposition") == EXECUTOR_DEFECT_DISPOSITION
    ]
    if valid and defective_attempts:
        # Fail closed: the executor (reset boundary, adapter ABI, driver call) failed before a
        # solver outcome on at least one attempt. The records are retained verbatim, but the
        # group is quarantined instead of completed so it never enters any statistic.
        valid = False
        disposition = EXECUTOR_DEFECT_DISPOSITION
        reason = (
            f"executor defect on {len(defective_attempts)} attempt(s) "
            f"({', '.join(str(item).rsplit('/', 1)[-1] for item in defective_attempts)}); "
            "group quarantined, evidence retained"
        )
        record["reason"] = reason
    if amendment is not None:
        if valid and contaminated_attempts:
            reason = (
                f"{reason}; {contaminated_attempts} attempt(s) flagged contaminated by foreign GPU "
                "compute (run-and-flag; timing/energy invalid, disposition/quality retained)"
            )
            record["reason"] = reason
    elif contamination.get("foreign_detected"):
        valid = False
        disposition = "contaminated"
        reason = (
            "foreign GPU compute activity observed during the group; timing and energy are "
            "retained as evidence only and the group is re-run after the foreign activity ends"
        )
        record["reason"] = reason
    store.finish(
        claim,
        disposition=disposition,
        reason=reason,
        record=record,
        valid=valid,
    )
    return disposition


def execute(
    store: CampaignStore,
    claim: Claim,
    executable: Path,
    executor: PersistentExecutor,
    power: NvmlPower,
    policy_sha256: str,
    matrix_sha256: str,
    capability_sha256: str,
    timeout_seconds: int,
    sampler_cpu_core: int | None,
) -> None:
    command = command_for(
        executable,
        claim,
        policy_sha256,
        matrix_sha256,
        capability_sha256,
    )
    run_directory = store.root / "runs" / claim.coordinate_id / claim.attempt_id
    atomic_create(run_directory / "command.json", json.dumps(command).encode() + b"\n")
    executor.ensure_ready()
    sampler = EnergySampler(power, cpu_core=sampler_cpu_core)
    started = time.monotonic()
    sampler.start()
    stdout, stderr, returncode, timeout, generation, startup_seconds = executor.execute(
        command, timeout_seconds
    )
    elapsed = time.monotonic() - started
    energy = sampler.finish()
    stdout_sha256, stdout_bytes = archive_stdout(store.root, stdout)
    atomic_create(run_directory / "stderr.log", stderr.encode())
    try:
        records = parse_records(stdout)
    except (json.JSONDecodeError, ValueError):
        records = []
    if timeout:
        disposition, failure, valid, reason = (
            "timeout",
            "timeout",
            True,
            "persistent executor exceeded the frozen actual-execution timeout",
        )
    elif returncode != 0:
        lowered = stderr.lower()
        if returncode in {137, -9} or "out of memory" in lowered:
            disposition, failure = "oom", "oom"
        elif "unsupported" in lowered:
            disposition, failure = "unsupported", "unsupported"
        else:
            disposition, failure = "numerical", "numerical"
        valid, reason = True, f"persistent executor exited {returncode}"
    else:
        try:
            valid, reason = validate_success(
                claim,
                records,
                policy_sha256,
                matrix_sha256,
                capability_sha256,
            )
        except (G4ContractError, json.JSONDecodeError, ValueError) as error:
            records, valid, reason = [], False, f"invalid executor records: {error}"
        sample = next((record for record in records if record.get("case") == "g4_sample"), {})
        if sample.get("status") == 4:
            disposition, failure = "timeout", "timeout"
            reason = "in-process deadline retained final residual and progress records"
        else:
            disposition = "qualified" if sample.get("qualified") is True else "unqualified"
            failure = "none" if disposition == "qualified" else "max_iterations"
    record = {
        **claim.coordinate,
        "coordinate_id": claim.coordinate_id,
        "attempt_id": claim.attempt_id,
        "command": command,
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "disposition": disposition,
        "failure_class": failure,
        "reason": reason,
        "energy": energy,
        "executor": {
            "mode": "persistent-g4-server",
            "generation": generation,
            "cuda_startup_seconds": startup_seconds,
            "cuda_startup_in_timing_boundary": False,
        },
        "stdout_object_sha256": stdout_sha256,
        "stdout_uncompressed_bytes": stdout_bytes,
        "records": [record for record in records if record.get("case") != "g4_iteration"],
        "progress_record_count": sum(record.get("case") == "g4_iteration" for record in records),
    }
    store.finish(
        claim,
        disposition=disposition,
        reason=reason,
        record=record,
        valid=valid,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("init", "migrate", "status", "run", "invalidate"))
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--capabilities", type=Path)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--source-campaign", type=Path)
    parser.add_argument(
        "--invalidate-policy",
        help="invalidate: only completed groups whose coordinate policy equals this value",
    )
    parser.add_argument(
        "--invalidate-reason",
        help="invalidate: the defect statement recorded on every invalidated attempt",
    )
    parser.add_argument(
        "--invalidate-disposition",
        default=INVALID_EXECUTOR_DEFECT,
        help="invalidate: ledger disposition written on the invalidated attempts",
    )
    parser.add_argument(
        "--fix-commit",
        help="invalidate: the commit carrying the executor fix (provenance only)",
    )
    parser.add_argument(
        "--superseded-by",
        type=Path,
        help="invalidate: the campaign that re-runs the invalidated groups (provenance only)",
    )
    parser.add_argument("--nvml-library")
    parser.add_argument("--sampler-cpu-core", type=int)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument(
        "--host-nvidia-smi",
        default="/mnt/c/Windows/System32/nvidia-smi.exe",
        help="Windows nvidia-smi for host compute-context monitoring under WSL2 (if present)",
    )
    parser.add_argument(
        "--no-contamination-guard",
        action="store_true",
        help="disable foreign-GPU-process monitoring, quarantine, and re-run",
    )
    parser.add_argument(
        "--claim-core",
        action="store_true",
        help="schedule only the hash-pinned 360-group H5/H6 claim core",
    )
    parser.add_argument(
        "--amendment",
        type=Path,
        default=None,
        help=(
            "apply the preregistered claim-core amendment JSON (single-gpu-v1.1): run-and-flag "
            "contamination, deterministic-replay timeouts, 120 s / 200k censoring plus the "
            "censoring-sensitivity stratum; requires --claim-core"
        ),
    )
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    policy, policy_sha256, matrix_sha256 = locked_policy(repository)
    groups = None
    schedule_sha256 = policy_sha256
    amendment: LoadedAmendment | None = None
    extra_metadata: dict[str, str] = {}
    if arguments.amendment is not None and not arguments.claim_core:
        raise G4ContractError("--amendment applies to the claim core only")
    if arguments.claim_core:
        core_lock = (
            (repository / "benchmarks/g4_h5_h6_claim_core.sha256")
            .read_text(encoding="utf-8")
            .split()
        )
        if len(core_lock) != 2 or core_lock[1] != "g4_h5_h6_claim_core.json":
            raise G4ContractError("invalid H5/H6 claim-core lock")
        loaded_core = load_claim_core(
            repository / "benchmarks/g4_h5_h6_claim_core.json",
            expected_sha256=core_lock[0],
        )
        groups = tuple(iter_claim_core_groups(loaded_core.values))
        schedule_sha256 = loaded_core.sha256
        if arguments.amendment is not None:
            amendment_path = arguments.amendment.resolve()
            amendment_lock = (
                amendment_path.with_suffix(".sha256").read_text(encoding="utf-8").split()
            )
            if len(amendment_lock) != 2 or amendment_lock[1] != amendment_path.name:
                raise G4ContractError("invalid claim-core amendment lock")
            amendment = load_claim_core_amendment(
                amendment_path,
                loaded_core.values,
                claim_core_sha256=loaded_core.sha256,
                policy_sha256=policy_sha256,
                expected_sha256=amendment_lock[0],
            )
            groups = amended_claim_core_groups(loaded_core.values, amendment.values)
            schedule_sha256 = amended_schedule_sha256(groups)
            if schedule_sha256 != amendment.values["schedule"]["schedule_sha256"]:
                raise G4ContractError("amended schedule hash drift")
            extra_metadata = {
                AMENDMENT_RECORD_FIELD: amendment.values["amendment_id"],
                "policy_amendment_sha256": amendment.sha256,
                "claim_core_sha256": loaded_core.sha256,
            }
    source_commit = subprocess.run(
        ["git", "-C", repository, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if arguments.action != "status":
        dirty = subprocess.run(
            ["git", "-C", repository, "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if dirty:
            raise G4ContractError("campaign initialization and execution require a clean commit")
    store_source_commit = source_commit
    if arguments.action == "invalidate":
        # Ledger maintenance opens the checkpoint under its own pinned source commit: the
        # fixed executor lives at a later HEAD, and the invalidated rows are re-run by a new
        # campaign initialised there, never by this one.
        store_source_commit = stored_metadata(arguments.campaign.resolve())["source_commit"]
    with CampaignStore(
        arguments.campaign.resolve(),
        policy,
        policy_sha256,
        store_source_commit,
        grouped=True,
        groups=groups,
        schedule_sha256=schedule_sha256,
        extra_metadata=extra_metadata,
    ) as store:
        if arguments.action == "status":
            print(json.dumps(store.status(), sort_keys=True))
            return 0
        if arguments.action == "invalidate":
            if groups is None:
                raise G4ContractError("invalidate is defined for the claim core only")
            if not arguments.invalidate_policy or not arguments.invalidate_reason:
                raise G4ContractError(
                    "invalidate requires --invalidate-policy and --invalidate-reason"
                )
            invalidated = invalidate_completed_groups(
                store,
                groups,
                policy=arguments.invalidate_policy,
                disposition=arguments.invalidate_disposition,
                reason=arguments.invalidate_reason,
                provenance={
                    "defect_source_commit": store_source_commit,
                    "fix_commit": arguments.fix_commit,
                    "superseded_by": (
                        str(arguments.superseded_by.resolve())
                        if arguments.superseded_by is not None
                        else None
                    ),
                    "invalidated_from_commit": source_commit,
                },
            )
            print(json.dumps({**store.status(), "invalidated_now": invalidated}, sort_keys=True))
            return 0
        if arguments.action == "init":
            print(json.dumps(store.status(), sort_keys=True))
            return 0
        if arguments.action == "migrate":
            if arguments.source_campaign is None:
                raise G4ContractError("migrate requires --source-campaign")
            migration = migrate_terminal_rows(store, arguments.source_campaign.resolve())
            print(json.dumps({**store.status(), **migration}, sort_keys=True))
            return 0
        if arguments.executable is None or arguments.capabilities is None:
            raise G4ContractError("run requires --executable and --capabilities")
        executable = arguments.executable.resolve()
        capabilities = load_capabilities(
            arguments.capabilities.resolve(),
            executable,
            policy_sha256,
            matrix_sha256,
            source_commit,
            require_persistent_group=True,
            amendment=amendment,
        )
        capability_sha256 = capabilities["capability_sha256"]
        lock_descriptor = os.open(store.root / "gpu-worker.lock", os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise G4ContractError("another measured GPU worker owns this campaign") from error
        completed = 0
        power = NvmlPower(arguments.nvml_library)
        timeout_seconds = int(policy["matrix"]["timeout_seconds_per_sample"])
        monitor = (
            None
            if arguments.no_contamination_guard
            else GpuContaminationMonitor(arguments.nvidia_smi, arguments.host_nvidia_smi)
        )
        executor = PersistentExecutor(
            executable,
            dict(os.environ),
            row_deadline_seconds=timeout_seconds,
        )
        shared_lock = SharedGpuLock() if amendment is not None else None
        try:
            while arguments.max_runs is None or completed < arguments.max_runs:
                if monitor is not None and amendment is None:
                    # single-gpu-v1 wait-for-idle; amendment single-gpu-v1.1 runs and flags.
                    monitor.wait_until_clear()
                claim = store.claim()
                if claim is None:
                    break
                group = groups[claim.ordinal] if groups is not None else None
                disposition = execute_group(
                    store,
                    claim,
                    executable,
                    executor,
                    power,
                    policy_sha256,
                    matrix_sha256,
                    capability_sha256,
                    timeout_seconds,
                    arguments.sampler_cpu_core,
                    monitor,
                    amendment,
                    group,
                    shared_lock,
                )
                if amendment is not None:
                    result_path = (
                        store.root / "runs" / claim.coordinate_id / claim.attempt_id / "result.json"
                    )
                    summary = json.loads(result_path.read_text(encoding="utf-8"))
                    print(
                        json.dumps(
                            {
                                "event": "group_finished",
                                "at": GpuContaminationMonitor._now(),
                                "ordinal": claim.ordinal,
                                "group_id": claim.coordinate_id,
                                "disposition": disposition,
                                "elapsed_seconds": summary.get("elapsed_seconds"),
                                "censoring": summary.get("censoring"),
                                "attempt_dispositions": sorted(
                                    Counter(
                                        item.get("disposition")
                                        for item in summary.get("raw_attempts", [])
                                    ).items()
                                ),
                                "contaminated_attempts": summary.get("gpu_contamination", {}).get(
                                    "contaminated_attempts"
                                ),
                                "deterministic_replay_applied": summary.get(
                                    "deterministic_replay_applied"
                                ),
                            },
                            sort_keys=True,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                elif disposition == "contaminated" and monitor is not None:
                    print(
                        json.dumps(
                            {
                                "event": "group_contaminated",
                                "ordinal": claim.ordinal,
                                "group_id": claim.coordinate_id,
                                "attempt_id": claim.attempt_id,
                            },
                            sort_keys=True,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                    monitor.wait_until_clear()
                    store.retry_quarantined(claim.ordinal)
                    continue
                completed += 1
        finally:
            executor.close()
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)
        print(json.dumps(store.status(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
