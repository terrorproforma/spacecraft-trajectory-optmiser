#!/usr/bin/env python3
"""Run the frozen G4 ledger through a crash-safe single-GPU scheduler."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
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


class EnergySampler:
    def __init__(self, nvidia_smi: str, interval_seconds: float = 0.05) -> None:
        self.nvidia_smi = nvidia_smi
        self.interval_seconds = interval_seconds
        self.samples: list[tuple[float, float]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            result = subprocess.run(
                [
                    self.nvidia_smi,
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                    "--id=0",
                ],
                capture_output=True,
                text=True,
            )
            timestamp = time.monotonic()
            if result.returncode == 0:
                try:
                    self.samples.append((timestamp, float(result.stdout.splitlines()[0])))
                except (IndexError, ValueError) as error:
                    self.errors.append(str(error))
            else:
                self.errors.append(result.stderr.strip() or f"nvidia-smi exit {result.returncode}")
            self._stop.wait(max(0.0, self.interval_seconds - (time.monotonic() - started)))

    def start(self) -> None:
        self._thread.start()

    def finish(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join()
        gaps = [
            right[0] - left[0] for left, right in zip(self.samples, self.samples[1:], strict=False)
        ]
        joules = sum(
            0.5 * (left[1] + right[1]) * (right[0] - left[0])
            for left, right in zip(self.samples, self.samples[1:], strict=False)
        )
        maximum_gap = max(gaps) if gaps else None
        return {
            "source": "nvidia-smi",
            "scope": "GPU-only",
            "sampling_interval_seconds": self.interval_seconds,
            "sample_count": len(self.samples),
            "maximum_gap_seconds": maximum_gap,
            "gap_valid": maximum_gap is not None and maximum_gap <= 0.1,
            "joules": joules,
            "errors": self.errors,
            "shared_display_gpu": True,
        }


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
    policy_sha256: str,
    matrix_sha256: str,
    capability_sha256: str,
    timeout_seconds: int,
    nvidia_smi: str,
    environment: dict[str, str],
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
    run_environment = dict(environment)
    run_environment.update(
        {
            "SPACEPDHCG_G4_COORDINATE_ID": claim.coordinate_id,
            "SPACEPDHCG_G4_EVALUATION_SEED": str(claim.coordinate["seed"]),
            "SPACEPDHCG_G4_CONDITIONING_LOG10": str(claim.coordinate["conditioning"]),
            "SPACEPDHCG_G4_SOLVER_ORDER": str(claim.coordinate["solver_order"]),
        }
    )
    sampler = EnergySampler(nvidia_smi)
    started = time.monotonic()
    sampler.start()
    timeout = False
    try:
        process = subprocess.run(
            command,
            env=run_environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        stdout = process.stdout
        stderr = process.stderr
        returncode = process.returncode
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        returncode = 124
        timeout = True
    elapsed = time.monotonic() - started
    energy = sampler.finish()
    atomic_create(run_directory / "stdout.jsonl", stdout.encode())
    atomic_create(run_directory / "stderr.log", stderr.encode())
    if timeout:
        disposition, failure, valid, reason = (
            "timeout",
            "timeout",
            True,
            "launched process exceeded the frozen timeout",
        )
        records: list[dict[str, Any]] = []
    elif returncode != 0:
        lowered = stderr.lower()
        if returncode in {137, -9} or "out of memory" in lowered:
            disposition, failure = "oom", "oom"
        elif "unsupported" in lowered:
            disposition, failure = "unsupported", "unsupported"
        else:
            disposition, failure = "numerical", "numerical"
        valid, reason, records = True, f"launched process exited {returncode}", []
    else:
        try:
            records = parse_records(stdout)
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
        "records": records,
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
    parser.add_argument("action", choices=("init", "status", "run"))
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--capabilities", type=Path)
    parser.add_argument("--max-runs", type=int)
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
        try:
            while arguments.max_runs is None or completed < arguments.max_runs:
                claim = store.claim()
                if claim is None:
                    break
                execute(
                    store,
                    claim,
                    executable,
                    policy_sha256,
                    matrix_sha256,
                    capability_sha256,
                    int(policy["matrix"]["timeout_seconds_per_sample"]),
                    arguments.nvidia_smi,
                    dict(os.environ),
                )
                completed += 1
        finally:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)
        print(json.dumps(store.status(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
