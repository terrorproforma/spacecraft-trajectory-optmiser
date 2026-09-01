#!/usr/bin/env python3
"""Inspect the hash-pinned H5/H6 claim-core schedule without launching a GPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from spacepdhcg.experiments import (  # noqa: E402
    claim_core_group_at,
    claim_core_invocation_count,
    load_claim_core,
)


def _load(repository: Path):
    lock = (
        (repository / "benchmarks/g4_h5_h6_claim_core.sha256").read_text(encoding="utf-8").split()
    )
    if len(lock) != 2 or lock[1] != "g4_h5_h6_claim_core.json":
        raise ValueError("invalid H5/H6 claim-core lock")
    return load_claim_core(
        repository / "benchmarks/g4_h5_h6_claim_core.json",
        expected_sha256=lock[0],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("status", "group"))
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--index", type=int)
    arguments = parser.parse_args()
    loaded = _load(arguments.repository.resolve())
    if arguments.action == "status":
        print(
            json.dumps(
                {
                    "campaign_id": loaded.values["campaign_id"],
                    "definition_sha256": loaded.sha256,
                    "execution_groups": claim_core_invocation_count(loaded.values) // 9,
                    "warmup_invocations": 720,
                    "measured_invocations": 2520,
                    "total_invocations": 3240,
                    "claims_resolved": ["H5", "H6"],
                    "full_regime_matrix_substitute": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.index is None:
        parser.error("group requires --index")
    group = claim_core_group_at(loaded.values, arguments.index)
    print(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "record_kind": "execution_group",
                "group_id": group.group_id,
                "physical_instance_id": group.physical_instance_id,
                "coordinate": group.coordinate,
                "process_contract": {
                    "processes": 1,
                    "persistent_session": True,
                    "persistent_workspace": True,
                    "policy_reset_between_attempts": True,
                },
                "attempts": list(group.attempts),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
