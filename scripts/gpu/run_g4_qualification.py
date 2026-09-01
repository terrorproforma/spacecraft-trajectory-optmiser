#!/usr/bin/env python3
"""Run the Gate G4 pre-primary qualification and retain every negative result."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

FAMILIES = {
    "P1-C-pd3": (20, 0.01),
    "P1-D-pd6": (20, 0.05),
    "P1-E-low-thrust": (100, 0.01),
}
POLICIES = (
    "fixed-tight",
    "fixed-loose",
    "adaptive",
    "adaptive+polish",
    "pure-gpu-ipm",
    "hybrid-pdhcg-ipm",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_json_lines(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def sample_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [record for record in records if record.get("case") == "g4_sample"]
    if len(samples) != 1:
        raise ValueError(f"expected one g4_sample record, received {len(samples)}")
    return samples[0]


def integrate_power(path: Path) -> dict[str, Any]:
    samples: list[tuple[float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2:
            continue
        try:
            timestamp = datetime.strptime(
                fields[0], "%Y/%m/%d %H:%M:%S.%f"
            ).timestamp()
            power = float(fields[1])
        except (ValueError, OverflowError):
            continue
        samples.append((timestamp, power))
    energy = 0.0
    gaps: list[float] = []
    for (left_time, left_power), (right_time, right_power) in pairwise(samples):
        delta = right_time - left_time
        gaps.append(delta)
        energy += 0.5 * (left_power + right_power) * delta
    return {
        "samples": len(samples),
        "energy_joules": energy if len(samples) >= 2 else None,
        "maximum_gap_seconds": max(gaps) if gaps else None,
        "sampling_gap": bool(gaps and max(gaps) > 0.15),
    }


def run_sample(
    executable: Path,
    family: str,
    intervals: int,
    dispersion: float,
    output: Path,
    timeout: int,
) -> dict[str, Any]:
    stem = f"{family}-n{intervals}-adaptive-cold"
    stdout_path = output / "raw" / f"{stem}.stdout.log"
    stderr_path = output / "raw" / f"{stem}.stderr.log"
    power_path = output / "power" / f"{stem}.csv"
    power_stream = power_path.open("w", encoding="utf-8")
    power = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,power.draw,clocks.sm,clocks.mem,temperature.gpu,pstate",
            "--format=csv,noheader,nounits",
            "--loop-ms=50",
        ],
        stdout=power_stream,
        stderr=subprocess.STDOUT,
        text=True,
    )
    command = [
        str(executable),
        "--g4-sample",
        family,
        str(intervals),
        "adaptive",
        "cold",
        "1e-6",
        "1",
        str(dispersion),
    ]
    started = time.perf_counter()
    status = "failed"
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        parsed = sample_record(parse_json_lines(completed.stdout))
        status = "unqualified" if completed.returncode == 0 else "failed"
        parsed["process_returncode"] = completed.returncode
    except subprocess.TimeoutExpired as error:
        stdout_path.write_text(error.stdout or "", encoding="utf-8")
        stderr_path.write_text(error.stderr or "", encoding="utf-8")
        parsed = {"qualified": False, "status": "timeout"}
        status = "timeout"
    finally:
        power.terminate()
        try:
            power.wait(timeout=5)
        except subprocess.TimeoutExpired:
            power.kill()
            power.wait()
        power_stream.close()
    return {
        "family": family,
        "intervals": intervals,
        "dispersion": dispersion,
        "policy": "adaptive",
        "warm_start": "cold",
        "quality_tier": "tight",
        "status": status,
        "elapsed_seconds": time.perf_counter() - started,
        "result": parsed,
        "stdout": str(stdout_path.relative_to(output)),
        "stderr": str(stderr_path.relative_to(output)),
        "power": str(power_path.relative_to(output)),
        "power_summary": integrate_power(power_path),
        "command": command,
    }


def coverage(policy: dict[str, Any], executed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    executed_keys = {
        (
            item["family"],
            item["intervals"],
            item["policy"],
            item["warm_start"],
            item["quality_tier"],
        )
        for item in executed
    }
    rows: list[dict[str, Any]] = []
    for family, family_values in policy["matrix"]["families"].items():
        for intervals in family_values["intervals"]:
            for mode in POLICIES:
                for warm in policy["matrix"]["warm_start_modes"]:
                    for tier in policy["matrix"]["quality_tiers"]:
                        key = (family, intervals, mode, warm, tier)
                        if key in executed_keys:
                            disposition = next(
                                item["status"]
                                for item in executed
                                if (
                                    item["family"],
                                    item["intervals"],
                                    item["policy"],
                                    item["warm_start"],
                                    item["quality_tier"],
                                )
                                == key
                            )
                            reason = "executed qualification sample"
                        elif mode in {"pure-gpu-ipm", "hybrid-pdhcg-ipm"}:
                            disposition = "unsupported"
                            reason = (
                                "no production CQP adapter execution; QOCO smoke execution "
                                "cannot be relabelled as a trajectory comparison"
                            )
                        else:
                            disposition = "censored"
                            reason = (
                                "primary matrix stopped before timing after the first "
                                "nontrivial sample in every family failed matched quality"
                            )
                        rows.append(
                            {
                                "family": family,
                                "intervals": intervals,
                                "policy": mode,
                                "warm_start": warm,
                                "quality_tier": tier,
                                "disposition": disposition,
                                "reason": reason,
                            }
                        )
    return rows


def decision(executed: list[dict[str, Any]]) -> dict[str, Any]:
    qualified = [item for item in executed if item["result"].get("qualified") is True]
    forcing_failures = sum(
        '"forcing_satisfied":0' in Path(item["_stdout_absolute"]).read_text(encoding="utf-8")
        for item in executed
        if Path(item["_stdout_absolute"]).exists()
    )
    return {
        "gate": "G4",
        "decision": "FAIL",
        "g5_authorized": False,
        "H5": {
            "decision": "unresolved",
            "reason": (
                "zero nontrivial family qualification samples passed; no matched-quality "
                "five-repeat pair may enter the preregistered bootstrap"
            ),
        },
        "H6": {
            "decision": "unresolved",
            "reason": (
                "QOCO-GPU executed only its archived CUDA smoke problem; no production "
                "trajectory CQP adapter ran, and the pinned API has no dual warm start"
            ),
        },
        "qualification": {
            "executed_samples": len(executed),
            "qualified_samples": len(qualified),
            "forcing_failures": forcing_failures,
            "median_elapsed_seconds": statistics.median(
                item["elapsed_seconds"] for item in executed
            )
            if executed
            else None,
        },
        "failed_criteria": [
            "matched final nonlinear quality",
            "H5 resolved under preregistered rules",
            "H6 resolved under preregistered rules",
            "production GPU-IPM trajectory execution",
            "five repeats and twenty evaluation instances per coordinate",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    output = arguments.output.resolve()
    for directory in ("raw", "power", "compact"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    policy = json.loads(
        (repository / "benchmarks/g4_policy.json").read_text(encoding="utf-8")
    )
    executed: list[dict[str, Any]] = []
    for family, (intervals, dispersion) in FAMILIES.items():
        item = run_sample(
            arguments.executable.resolve(),
            family,
            intervals,
            dispersion,
            output,
            arguments.timeout,
        )
        item["_stdout_absolute"] = str(output / item["stdout"])
        executed.append(item)
    coverage_rows = coverage(policy, executed)
    coverage_path = output / "compact" / "coverage.jsonl"
    coverage_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in coverage_rows),
        encoding="utf-8",
    )
    result = decision(executed)
    for item in executed:
        item.pop("_stdout_absolute", None)
    manifest = {
        "schema_version": "1.0.0",
        "campaign": "g4-primary-qualification",
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "policy_sha256": sha256(repository / "benchmarks/g4_policy.json"),
        "qoco_lock_sha256": sha256(repository / "third_party/qoco_gpu.lock.json"),
        "environment": {
            "cuda": "12.8.93",
            "driver": "595.97",
            "gpu": "NVIDIA GeForce RTX 5090",
            "wsl": True,
            "energy_scope": "GPU-only; display/shared-machine state not isolated",
        },
        "samples": executed,
        "coverage": {
            "records": len(coverage_rows),
            "path": str(coverage_path.relative_to(output)),
            "sha256": sha256(coverage_path),
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["manifest_sha256"] = sha256(manifest_path)
    result["coverage_sha256"] = sha256(coverage_path)
    (output / "decision.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
