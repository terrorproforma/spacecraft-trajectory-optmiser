#!/usr/bin/env python3
"""Run a small executable pilot that covers every frozen G4 axis value."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_g4_campaign import (  # noqa: E402
    command_for,
    load_capabilities,
    locked_policy,
    parse_records,
)

from spacepdhcg.experiments.g4 import (  # noqa: E402
    POLICY_NAMES,
    QUALITY_TIERS,
    SCALING_MODES,
    WARM_MODES,
)
from spacepdhcg.experiments.g4_scheduler import Claim, coordinate_id  # noqa: E402


def family_rows(policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    families = policy["matrix"]["families"]
    pd3 = families["P1-C-pd3"]
    for index in range(len(pd3["dispersion_classes"])):
        rows.append(
            {
                "family": "P1-C-pd3",
                "intervals": pd3["intervals"][0],
                "dispersion_class": pd3["dispersion_classes"][
                    index % len(pd3["dispersion_classes"])
                ],
            }
        )
    pd6 = families["P1-D-pd6"]
    for index in range(
        max(len(pd6["attitude_dispersion_radians"]), len(pd6["angular_rate_dispersion"]))
    ):
        rows.append(
            {
                "family": "P1-D-pd6",
                "intervals": pd6["intervals"][0],
                "attitude_class": pd6["attitude_dispersion_radians"][
                    index % len(pd6["attitude_dispersion_radians"])
                ],
                "rate_class": pd6["angular_rate_dispersion"][
                    index % len(pd6["angular_rate_dispersion"])
                ],
            }
        )
    low = families["P1-E-low-thrust"]
    for index in range(max(len(low["trust_radii"]), len(low["transfer_classes"]))):
        rows.append(
            {
                "family": "P1-E-low-thrust",
                "intervals": low["intervals"][0],
                "trust_class": low["trust_radii"][index % len(low["trust_radii"])],
                "transfer_class": low["transfer_classes"][index % len(low["transfer_classes"])],
            }
        )
    return rows


def coordinates(policy: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = policy["matrix"]
    seeds = policy["tuning_evaluation_split"]["evaluation_seeds"]
    result: list[dict[str, Any]] = []
    for index, family in enumerate(family_rows(policy)):
        quality = QUALITY_TIERS[index % len(QUALITY_TIERS)]
        coordinate = {
            "schema_version": 1,
            "ordinal": index,
            **family,
            "policy": POLICY_NAMES[index % len(POLICY_NAMES)],
            "quality_tier": quality,
            "quality_tolerance": policy["quality_tiers"][quality],
            "conditioning": matrix["conditioning_log10_spans"][
                index % len(matrix["conditioning_log10_spans"])
            ],
            "scaling_mode": SCALING_MODES[index % len(SCALING_MODES)],
            "warm_mode": WARM_MODES[index % len(WARM_MODES)],
            "seed": seeds[index % min(len(seeds), 4)],
            "repeat_kind": "warmup" if index % 2 == 0 else "measured",
            "repeat": 0,
            "solver_order": index % len(POLICY_NAMES),
        }
        coordinate["instance"] = f"{coordinate['family']}-seed-{coordinate['seed']}"
        result.append(coordinate)
    replay = dict(result[0])
    replay["ordinal"] = len(result)
    result.append(replay)
    cross_seed = dict(result[0])
    cross_seed["ordinal"] = len(result)
    cross_seed["seed"] = seeds[1]
    cross_seed["instance"] = f"{cross_seed['family']}-seed-{cross_seed['seed']}"
    result.append(cross_seed)
    mode_variant = dict(result[0])
    mode_variant.update(
        {
            "ordinal": len(result),
            "policy": "hybrid-pdhcg-ipm",
            "quality_tier": "medium",
            "quality_tolerance": policy["quality_tiers"]["medium"],
            "scaling_mode": "reuse",
            "warm_mode": "cold",
            "repeat_kind": "measured",
            "solver_order": 5,
        }
    )
    result.append(mode_variant)
    condition_variant = dict(result[0])
    condition_variant.update({"ordinal": len(result), "conditioning": 8.0})
    result.append(condition_variant)
    class_variant = dict(result[0])
    class_variant.update(
        {
            "ordinal": len(result),
            "dispersion_class": policy["matrix"]["families"]["P1-C-pd3"]["dispersion_classes"][1],
        }
    )
    result.append(class_variant)
    return result


def validate_probe(
    claim: Claim,
    records: list[dict[str, Any]],
    policy_sha256: str,
    matrix_sha256: str,
    capability_sha256: str,
) -> tuple[bool, str, dict[str, Any]]:
    probes = [record for record in records if record.get("case") == "g4_axis_probe"]
    if len(probes) != 1:
        return False, "executor did not emit exactly one axis-probe record", {}
    probe = probes[0]
    coordinate = claim.coordinate
    expected = {
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
    if {key: probe.get(key) for key in expected} != expected:
        return False, "axis-probe requested/applied values disagree", probe
    if probe.get("coefficient_parity_relative", math.inf) > 5.0e-12:
        return False, "axis-probe CPU/GPU coefficient parity failed", probe
    if not math.isclose(
        probe["condition_factor_max"] / probe["condition_factor_min"],
        10.0 ** coordinate["conditioning"],
        rel_tol=1.0e-12,
    ):
        return False, "axis-probe conditioning span mismatch", probe
    return True, "strict axis-probe record validated", probe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--capabilities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    executable = arguments.executable.resolve()
    policy, policy_sha256, matrix_sha256 = locked_policy(repository)
    source_commit = subprocess.run(
        ["git", "-C", repository, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    capability = load_capabilities(
        arguments.capabilities.resolve(),
        executable,
        policy_sha256,
        matrix_sha256,
        source_commit,
    )
    capability_sha256 = capability["capability_sha256"]
    environment = dict(os.environ)
    outcomes: list[dict[str, Any]] = []
    for coordinate in coordinates(policy):
        identifier = coordinate_id(coordinate)
        print(
            json.dumps(
                {
                    "event": "pilot_start",
                    "coordinate_id": identifier,
                    "family": coordinate["family"],
                    "policy": coordinate["policy"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        claim = Claim(coordinate["ordinal"], identifier, "pilot", coordinate)
        command = command_for(
            executable,
            claim,
            policy_sha256,
            matrix_sha256,
            capability_sha256,
        )
        command[1] = "--g4-axis-probe"
        command[7] = "1"
        process = subprocess.run(
            command,
            env=environment,
            capture_output=True,
            text=True,
            timeout=arguments.timeout,
        )
        records = parse_records(process.stdout)
        valid, reason, axis = validate_probe(
            claim,
            records,
            policy_sha256,
            matrix_sha256,
            capability_sha256,
        )
        outcomes.append(
            {
                "coordinate": coordinate,
                "coordinate_id": identifier,
                "returncode": process.returncode,
                "valid": valid,
                "reason": reason,
                "instance_hash": axis.get("instance_hash"),
                "problem_hash": axis.get("problem_hash"),
                "coefficient_hash": axis.get("coefficient_hash"),
                "condition_factor_min": axis.get("condition_factor_min"),
                "condition_factor_max": axis.get("condition_factor_max"),
                "coefficient_parity_maximum": axis.get("coefficient_parity_maximum"),
                "coefficient_parity_relative": axis.get("coefficient_parity_relative"),
                "policy_code": axis.get("policy_code"),
                "scaling_code": axis.get("scaling_code"),
                "warm_start_code": axis.get("warm_start_code"),
                "stderr": process.stderr,
            }
        )
        partial_path = arguments.output.with_suffix(arguments.output.suffix + ".partial")
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_text(json.dumps(outcomes, indent=2, sort_keys=True) + "\n")
        if process.returncode != 0 or not valid:
            raise SystemExit(
                f"pilot coordinate {identifier} failed with exit "
                f"{process.returncode}: {reason}; stderr={process.stderr!r}"
            )
    first = outcomes[0]
    replay, cross_seed, mode_variant, condition_variant, class_variant = outcomes[-5:]
    hash_names = ("instance_hash", "problem_hash", "coefficient_hash")
    if any(first[name] != replay[name] for name in hash_names):
        raise SystemExit("same-seed exact replay changed numerical hashes")
    if first["instance_hash"] == cross_seed["instance_hash"]:
        raise SystemExit("different evaluation seeds shared an instance hash")
    if any(first[name] != mode_variant[name] for name in hash_names):
        raise SystemExit("solver/runtime modes changed canonical problem hashes")
    if (
        first["instance_hash"] != condition_variant["instance_hash"]
        or first["coefficient_hash"] == condition_variant["coefficient_hash"]
        or first["problem_hash"] == condition_variant["problem_hash"]
    ):
        raise SystemExit("conditioning did not isolate the intended coefficient transform")
    if first["instance_hash"] == class_variant["instance_hash"]:
        raise SystemExit("family-class change did not alter physical instance")
    policy_codes = {row["coordinate"]["policy"]: row["policy_code"] for row in outcomes}
    scaling_codes = {row["coordinate"]["scaling_mode"]: row["scaling_code"] for row in outcomes}
    warm_codes = {row["coordinate"]["warm_mode"]: row["warm_start_code"] for row in outcomes}
    if len(set(policy_codes.values())) != len(POLICY_NAMES):
        raise SystemExit("policy names did not map to distinct runtime policies")
    if len(set(scaling_codes.values())) != len(SCALING_MODES):
        raise SystemExit("scaling names did not map to distinct runtime modes")
    if len(set(warm_codes.values())) != len(WARM_MODES):
        raise SystemExit("warm-start names did not map to distinct runtime modes")
    report = {
        "schema_version": 1,
        "source_commit": source_commit,
        "policy_sha256": policy_sha256,
        "matrix_sha256": matrix_sha256,
        "capability_sha256": capability_sha256,
        "coverage": {
            "families": sorted({row["coordinate"]["family"] for row in outcomes}),
            "policies": sorted({row["coordinate"]["policy"] for row in outcomes}),
            "quality_tiers": sorted({row["coordinate"]["quality_tier"] for row in outcomes}),
            "conditioning": sorted({row["coordinate"]["conditioning"] for row in outcomes}),
            "scaling_modes": sorted({row["coordinate"]["scaling_mode"] for row in outcomes}),
            "warm_modes": sorted({row["coordinate"]["warm_mode"] for row in outcomes}),
            "seeds": sorted({row["coordinate"]["seed"] for row in outcomes}),
            "solver_orders": sorted({row["coordinate"]["solver_order"] for row in outcomes}),
        },
        "same_seed_exact_replay": True,
        "cross_seed_distinct": True,
        "runtime_modes_problem_invariant": True,
        "conditioning_isolated": True,
        "family_class_distinct": True,
        "policy_codes": policy_codes,
        "scaling_codes": scaling_codes,
        "warm_start_codes": warm_codes,
        "outcomes": outcomes,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["coverage"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
