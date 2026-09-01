#!/usr/bin/env python3
"""Build or audit a Gate G4 qualification campaign.

The default mode is CPU/static: it validates previously captured JSON records,
writes a complete coverage ledger, and applies the preregistered H5/H6 rules.
No solver or GPU workload is launched by this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from spacepdhcg.experiments.g4 import (  # noqa: E402
    DISPOSITIONS,
    G4ContractError,
    coverage_count,
    g4_decision,
    iter_coverage_ledger,
    load_policy,
    qualify_matched_quality,
    runtime_configuration,
    sha256_path,
)


def parse_json_lines(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def sample_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [record for record in records if record.get("case") == "g4_sample"]
    if len(samples) != 1:
        raise G4ContractError(f"expected one g4_sample record, received {len(samples)}")
    return samples[0]


def _locked_hash(repository: Path) -> str:
    line = (repository / "benchmarks/g4_policy.sha256").read_text(encoding="utf-8").strip()
    fields = line.split()
    if len(fields) != 2 or fields[1] != "g4_policy.json":
        raise G4ContractError("invalid benchmarks/g4_policy.sha256 lock")
    return fields[0]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return parse_json_lines(path.read_text(encoding="utf-8"))


def build_runtime_plan(
    repository: Path,
    coordinates: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    loaded = load_policy(
        repository / "benchmarks/g4_policy.json",
        expected_sha256=_locked_hash(repository),
    )
    return [
        runtime_configuration(
            loaded,
            family=str(coordinate["family"]),
            policy_name=str(coordinate["policy"]),
            quality_tier=str(coordinate["quality_tier"]),
            scaling_mode=str(coordinate["scaling_mode"]),
            warm_mode=str(coordinate["warm_mode"]),
        )
        for coordinate in coordinates
    ]


def qualify_records(
    records: Iterable[Mapping[str, Any]], policy: Mapping[str, Any]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in records:
        result = qualify_matched_quality(source, policy)
        disposition = source.get("disposition")
        if disposition not in DISPOSITIONS:
            disposition = "qualified" if result["qualified"] else "unqualified"
        output.append({**source, "disposition": disposition, "matched_quality": result})
    return output


def write_coverage(
    path: Path,
    policy: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
    *,
    supported_policies: Iterable[str],
) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for row in iter_coverage_ledger(
            policy,
            records,
            supported_policies=supported_policies,
        ):
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    expected = coverage_count(policy)
    if count != expected:
        raise G4ContractError(f"coverage row count mismatch: expected {expected}, wrote {count}")
    return count, sha256_path(path)


def decision(
    h5_rows: list[Mapping[str, Any]],
    h6_rows: list[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    return g4_decision(h5_rows, h6_rows, policy)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--supported-policy",
        action="append",
        default=[],
        help="Policy executable in this campaign; omitted policies are explicit unsupported rows.",
    )
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    output = arguments.output.resolve()
    loaded = load_policy(
        repository / "benchmarks/g4_policy.json",
        expected_sha256=_locked_hash(repository),
    )
    policy = loaded.values
    raw_records = _read_jsonl(arguments.input / "samples.jsonl")
    qualified = qualify_records(raw_records, policy)
    output.mkdir(parents=True, exist_ok=True)
    (output / "qualified.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in qualified),
        encoding="utf-8",
    )
    count, ledger_hash = write_coverage(
        output / "coverage.jsonl",
        policy,
        qualified,
        supported_policies=arguments.supported_policy,
    )
    h5_rows = _read_jsonl(arguments.input / "h5_coordinates.jsonl")
    h6_rows = _read_jsonl(arguments.input / "h6_coordinates.jsonl")
    result = decision(h5_rows, h6_rows, policy)
    result.update(
        {
            "policy_sha256": loaded.sha256,
            "coverage_records": count,
            "coverage_sha256": ledger_hash,
            "qualified_records_sha256": sha256_path(output / "qualified.jsonl"),
        }
    )
    (output / "decision.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if result["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
