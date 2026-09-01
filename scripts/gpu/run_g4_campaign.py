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
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from spacepdhcg.experiments.g4 import (  # noqa: E402
    ACCEPTED_TIMING_BOUNDARY,
    G4ContractError,
    load_policy,
    sha256_path,
)
from spacepdhcg.experiments.g4_scheduler import (  # noqa: E402
    CampaignStore,
    Claim,
    atomic_create,
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


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


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

    def execute_batch(
        self, commands: list[list[str]], timeout_seconds: int
    ) -> tuple[dict[str, dict[str, Any]], str, str, int, float]:
        """Execute compatible rows concurrently and route identity-tagged records."""

        if not commands:
            raise ValueError("persistent executor batch cannot be empty")
        self.ensure_ready()
        assert self.process.stdin is not None
        responses = {
            command[18]: {
                "stdout": [],
                "returncode": None,
                "timeout": False,
                "complete": False,
                "elapsed_seconds": None,
            }
            for command in commands
        }
        if len(responses) != len(commands):
            raise G4ContractError("batch contains duplicate coordinate identities")
        self.process.stdin.write(f"batch\t{len(commands)}\n")
        for command in commands:
            self.process.stdin.write("\t".join(command[1:]) + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + timeout_seconds + 5.0
        shared_lines: list[str] = []
        batch_complete = False
        while not batch_complete:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                self._stop()
                break
            try:
                line = self._stdout.get(timeout=remaining)
            except queue.Empty:
                self._stop()
                break
            if line is None:
                break
            try:
                record = json.loads(line.replace("-inf", "-Infinity"))
            except json.JSONDecodeError:
                shared_lines.append(line)
                continue
            case = record.get("case")
            identifier = record.get("coordinate_id")
            if case == "g4_server_result" and identifier in responses:
                responses[identifier]["returncode"] = int(record["returncode"])
                responses[identifier]["complete"] = True
                responses[identifier]["elapsed_seconds"] = float(record["elapsed_seconds"])
            elif case == "g4_batch_result":
                batch_complete = True
            elif identifier in responses:
                responses[identifier]["stdout"].append(line)
            else:
                shared_lines.append(line)
        for response in responses.values():
            if not response["complete"] and time.monotonic() >= deadline:
                response["timeout"] = True
                response["returncode"] = 124
            response["stdout"] = "".join(response["stdout"])
        stderr: list[str] = []
        while True:
            try:
                stderr.append(self._stderr.get_nowait())
            except queue.Empty:
                break
        return (
            responses,
            "".join(shared_lines),
            "".join(stderr),
            self.generation,
            self.cuda_startup_seconds,
        )

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
        "valid": valid,
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


def batch_group_key(claim: Claim) -> tuple[str, int, str]:
    policy = claim.coordinate["policy"]
    backend = (
        "qoco"
        if policy == "pure-gpu-ipm"
        else ("hybrid" if policy == "hybrid-pdhcg-ipm" else "pdhcg")
    )
    return claim.coordinate["family"], int(claim.coordinate["intervals"]), backend


def execute_claim_batch(
    store: CampaignStore,
    claims: list[Claim],
    executable: Path,
    executor: PersistentExecutor,
    power: NvmlPower,
    policy_sha256: str,
    matrix_sha256: str,
    capability_sha256: str,
    timeout_seconds: int,
    sampler_cpu_core: int | None,
) -> int:
    """Execute one compatible group and commit each completed row independently."""

    commands = [
        command_for(
            executable,
            claim,
            policy_sha256,
            matrix_sha256,
            capability_sha256,
        )
        for claim in claims
    ]
    for claim, command in zip(claims, commands, strict=True):
        run_directory = store.root / "runs" / claim.coordinate_id / claim.attempt_id
        atomic_create(run_directory / "command.json", json.dumps(command).encode() + b"\n")
    batch_id = hashlib.sha256(
        canonical_bytes(
            [
                {"coordinate_id": claim.coordinate_id, "attempt_id": claim.attempt_id}
                for claim in claims
            ]
        )
    ).hexdigest()
    batch_directory = store.root / "batches" / batch_id
    batch_directory.mkdir(parents=True, exist_ok=False)
    executor.ensure_ready()
    sampler = EnergySampler(power, cpu_core=sampler_cpu_core)
    batch_started = time.monotonic()
    sampler.start()
    responses, shared_stdout, shared_stderr, generation, startup_seconds = executor.execute_batch(
        commands, timeout_seconds
    )
    batch_elapsed = time.monotonic() - batch_started
    energy = sampler.finish()
    shared_stdout_sha256, shared_stdout_bytes = archive_stdout(store.root, shared_stdout)
    stderr_sha256 = hashlib.sha256(shared_stderr.encode()).hexdigest()
    atomic_create(batch_directory / "stderr.log", shared_stderr.encode())
    atomic_create(
        batch_directory / "batch.json",
        canonical_bytes(
            {
                "batch_id": batch_id,
                "coordinate_ids": [claim.coordinate_id for claim in claims],
                "elapsed_seconds": batch_elapsed,
                "energy": energy,
                "energy_attribution": "batch-only; per-row energy intentionally null",
                "shared_stdout_object_sha256": shared_stdout_sha256,
                "shared_stdout_uncompressed_bytes": shared_stdout_bytes,
                "stderr_sha256": stderr_sha256,
                "executor_generation": generation,
            }
        )
        + b"\n",
    )

    completed = 0
    for claim, command in zip(claims, commands, strict=True):
        response = responses[claim.coordinate_id]
        if not response["complete"]:
            continue
        stdout = str(response["stdout"])
        returncode = int(response["returncode"])
        run_directory = store.root / "runs" / claim.coordinate_id / claim.attempt_id
        stdout_sha256, stdout_bytes = archive_stdout(store.root, stdout)
        atomic_create(run_directory / "stderr.log", b"")
        try:
            records = parse_records(stdout)
        except (json.JSONDecodeError, ValueError):
            records = []
        if returncode != 0:
            lowered = shared_stderr.lower()
            if returncode in {137, -9} or "out of memory" in lowered:
                disposition, failure = "oom", "oom"
            elif "unsupported" in lowered:
                disposition, failure = "unsupported", "unsupported"
            else:
                disposition, failure = "numerical", "numerical"
            valid, reason = True, f"persistent batch lane exited {returncode}"
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
            sample = next(
                (record for record in records if record.get("case") == "g4_sample"),
                {},
            )
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
            "elapsed_seconds": response["elapsed_seconds"],
            "disposition": disposition,
            "valid": valid,
            "failure_class": failure,
            "reason": reason,
            "energy": None,
            "batch_energy": {
                "batch_id": batch_id,
                "path": str(batch_directory / "batch.json"),
                "attribution": "none",
            },
            "executor": {
                "mode": "persistent-concurrent-lanes",
                "generation": generation,
                "cuda_startup_seconds": startup_seconds,
                "cuda_startup_in_timing_boundary": False,
                "batch_size": len(claims),
            },
            "stdout_object_sha256": stdout_sha256,
            "stdout_uncompressed_bytes": stdout_bytes,
            "records": [item for item in records if item.get("case") != "g4_iteration"],
            "progress_record_count": sum(item.get("case") == "g4_iteration" for item in records),
        }
        store.finish(
            claim,
            disposition=disposition,
            reason=reason,
            record=record,
            valid=valid,
        )
        completed += 1
    return completed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("init", "migrate", "status", "run"))
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--capabilities", type=Path)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--source-campaign", type=Path)
    parser.add_argument("--nvml-library")
    parser.add_argument("--sampler-cpu-core", type=int)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    policy, policy_sha256, matrix_sha256 = locked_policy(repository)
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
    with CampaignStore(
        arguments.campaign.resolve(),
        policy,
        policy_sha256,
        source_commit,
    ) as store:
        if arguments.action == "status":
            print(json.dumps(store.status(), sort_keys=True))
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
        )
        capability_sha256 = capabilities["capability_sha256"]
        lock_descriptor = os.open(store.root / "gpu-worker.lock", os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise G4ContractError("another measured GPU worker owns this campaign") from error
        completed = 0
        if arguments.batch_size <= 0 or arguments.batch_size > 1024:
            raise G4ContractError("batch size must be in [1, 1024]")
        power = NvmlPower(arguments.nvml_library)
        timeout_seconds = int(policy["matrix"]["timeout_seconds_per_sample"])
        executor = PersistentExecutor(
            executable,
            dict(os.environ),
            row_deadline_seconds=timeout_seconds,
        )
        try:
            while arguments.max_runs is None or completed < arguments.max_runs:
                remaining_limit = (
                    arguments.batch_size
                    if arguments.max_runs is None
                    else min(arguments.batch_size, arguments.max_runs - completed)
                )
                claims = store.claim_batch(remaining_limit)
                if not claims:
                    break
                grouped: dict[tuple[str, int, str], list[Claim]] = {}
                for claim in claims:
                    grouped.setdefault(batch_group_key(claim), []).append(claim)
                progress = 0
                for group_key, group in grouped.items():
                    lane_groups = (
                        [group] if group_key[2] == "pdhcg" else [[claim] for claim in group]
                    )
                    for lane_group in lane_groups:
                        progress += execute_claim_batch(
                            store,
                            lane_group,
                            executable,
                            executor,
                            power,
                            policy_sha256,
                            matrix_sha256,
                            capability_sha256,
                            timeout_seconds,
                            arguments.sampler_cpu_core,
                        )
                completed += progress
                if progress == 0:
                    executor.ensure_ready()
        finally:
            executor.close()
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)
        print(json.dumps(store.status(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
