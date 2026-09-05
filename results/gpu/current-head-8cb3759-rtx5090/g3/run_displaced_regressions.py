#!/usr/bin/env python3
"""Run direct, non-campaign displaced G3 regressions through the production owner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts/gpu"))

from run_g4_campaign import (  # noqa: E402
    command_for,
    locked_policy,
    parse_records,
)
from spacepdhcg.experiments.g4 import (  # noqa: E402
    POLICY_NAMES,
    QUALITY_TIERS,
    SCALING_MODES,
    WARM_MODES,
)
from spacepdhcg.experiments.g4_scheduler import (  # noqa: E402
    Claim,
    _family_coordinates,
    coordinate_at,
    coordinate_id,
)


def _ordinal(
    policy: dict[str, Any],
    *,
    family: str,
    intervals: int,
    classes: dict[str, Any],
    policy_name: str,
    quality: str,
    conditioning: float,
    scaling: str,
    warm: str,
    seed: int,
    repeat_kind: str,
    repeat_index: int,
) -> int:
    matrix = policy["matrix"]
    seeds = tuple(policy["tuning_evaluation_split"]["evaluation_seeds"])[
        : matrix["randomised_instances_per_coordinate"]
    ]
    repeats = (
        *(("warmup", index) for index in range(matrix["warmup_repeats"])),
        *(("measured", index) for index in range(matrix["measured_repeats"])),
    )
    family_index = _family_coordinates(policy).index((family, intervals, classes))
    dimensions = (
        tuple(POLICY_NAMES),
        tuple(QUALITY_TIERS),
        tuple(matrix["conditioning_log10_spans"]),
        tuple(SCALING_MODES),
        tuple(WARM_MODES),
        seeds,
        repeats,
    )
    selections = (
        policy_name,
        quality,
        conditioning,
        scaling,
        warm,
        seed,
        (repeat_kind, repeat_index),
    )
    remainder = 0
    for dimension, selected in zip(dimensions, selections, strict=True):
        remainder = remainder * len(dimension) + dimension.index(selected)
    inner_count = 1
    for dimension in dimensions:
        inner_count *= len(dimension)
    return family_index * inner_count + remainder


def _records(stdout: str, case: str) -> list[dict[str, Any]]:
    return [record for record in parse_records(stdout) if record.get("case") == case]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    executable = args.executable.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    policy, policy_sha, matrix_sha = locked_policy(REPOSITORY)
    context = {
        "schema_version": "g3-direct-regression-context-1.0.0",
        "source_commit": "8cb3759b29ea8c7d843322a940a7ebcabfd9ff21",
        "executable": str(executable),
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "policy_sha256": policy_sha,
        "matrix_sha256": matrix_sha,
        "official_g4_capability": False,
        "g4_campaign_launched": False,
        "record_kind": "g3_noncampaign_warmup_regression",
    }
    context_raw = (json.dumps(context, sort_keys=True, separators=(",", ":")) + "\n").encode()
    capability_sha = hashlib.sha256(context_raw).hexdigest()
    (output / "execution-context.json").write_bytes(context_raw)
    environment = dict(os.environ)
    families = (
        ("P1-C-pd3", 20, {"dispersion_class": 0.01}),
        (
            "P1-D-pd6",
            20,
            {"attitude_class": 0.05, "rate_class": 0.05},
        ),
        (
            "P1-E-low-thrust",
            100,
            {"trust_class": 0.25, "transfer_class": "radius_raise"},
        ),
    )
    outcomes: list[dict[str, Any]] = []
    command_log: list[list[str]] = []
    for family, intervals, classes in families:
        for policy_name, quality, timeout_seconds, outer_limit in (
            ("pure-gpu-ipm", "ipm", 600, 100),
            ("fixed-tight", "tight", 150, 1),
        ):
            ordinal = _ordinal(
                policy,
                family=family,
                intervals=intervals,
                classes=classes,
                policy_name=policy_name,
                quality=quality,
                conditioning=0.0,
                scaling="refresh_if_needed",
                warm="primal_dual",
                seed=59,
                repeat_kind="warmup",
                repeat_index=0,
            )
            coordinate = coordinate_at(policy, ordinal)
            identifier = coordinate_id(coordinate)
            claim = Claim(ordinal, identifier, "g3-regression", coordinate)
            command = command_for(
                executable,
                claim,
                policy_sha,
                matrix_sha,
                capability_sha,
            )
            command[7] = str(outer_limit)
            command_log.append(command)
            stem = f"{family}-{policy_name}".lower().replace("+", "-plus-")
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    env=environment,
                )
                returncode: int | None = completed.returncode
                stdout = completed.stdout
                stderr = completed.stderr
                disposition = "returned"
            except subprocess.TimeoutExpired as error:
                returncode = None
                stdout = error.stdout or ""
                stderr = error.stderr or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode(errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode(errors="replace")
                disposition = "timeout"
            (output / f"{stem}.stdout.log").write_text(stdout, encoding="utf-8")
            (output / f"{stem}.stderr.log").write_text(stderr, encoding="utf-8")
            samples = _records(stdout, "g4_sample")
            outers = _records(stdout, "production_outer")
            iterations = _records(stdout, "g4_iteration")
            accepted = int(outers[-1]["accepted"]) if outers else 0
            retained_change = float(outers[-1]["retained_change"]) if outers else 0.0
            qualified = bool(samples[-1]["qualified"]) if samples else False
            if policy_name == "pure-gpu-ipm":
                if returncode != 0 or not qualified or accepted < 1 or retained_change <= 0.0:
                    raise RuntimeError(
                        f"{family} pure QOCO displaced regression failed: "
                        f"returncode={returncode}, qualified={qualified}, "
                        f"accepted={accepted}, retained_change={retained_change}"
                    )
                classification = "qualified-positive"
            else:
                if returncode == 0 and qualified and accepted > 0:
                    classification = "qualified-positive-unexpected"
                else:
                    classification = "honest-negative"
            outcomes.append(
                {
                    "family": family,
                    "policy": policy_name,
                    "ordinal": ordinal,
                    "coordinate_id": identifier,
                    "repeat_kind": "warmup",
                    "campaign_record": False,
                    "disposition": disposition,
                    "returncode": returncode,
                    "qualified": qualified,
                    "accepted_steps": accepted,
                    "retained_change": retained_change,
                    "iteration_records": len(iterations),
                    "classification": classification,
                }
            )
            (output / "partial.json").write_text(
                json.dumps(outcomes, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    (output / "commands.json").write_text(
        json.dumps(command_log, indent=2) + "\n",
        encoding="utf-8",
    )
    result = {
        "schema_version": "g3-displaced-regression-1.0.0",
        "source_commit": "8cb3759b29ea8c7d843322a940a7ebcabfd9ff21",
        "direct_regression_only": True,
        "g4_campaign_launched": False,
        "warmup_records_only": True,
        "execution_context_sha256": capability_sha,
        "outcomes": outcomes,
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
