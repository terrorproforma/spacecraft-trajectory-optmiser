from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator


def schema() -> dict[str, object]:
    path = Path(__file__).parents[1] / "experiments" / "schema" / "g5_implementation.schema.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(value)
    return value


def valid_record() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "gate": "G5-implementation",
        "verification": "one-rank-mpi-nccl-cuda",
        "rank": 0,
        "world_size": 1,
        "device": 0,
        "deterministic": True,
        "overlap": False,
        "rank_status": "healthy",
        "partition": {
            "kind": "scenario_aware",
            "fingerprint": "0123456789abcdef",
            "scenario_owner": [0, 0],
            "predicted_rank_load": [42.0],
            "measured_rank_load": None,
        },
        "collectives": [
            {
                "kind": "shared_arrowhead_sum",
                "count": 2,
                "elements": 16,
                "payload_bytes": 128,
                "wire_bytes_estimate": 0,
                "frequency": 1,
                "purpose": "non-anticipativity shared primal/gradient",
                "collective_seconds": 0.001,
                "exposed_seconds": 0.001,
                "overlapped_seconds": 0.0,
            }
        ],
        "multi_gpu_scaling_verified": False,
        "physical_rank_counts_deferred": [2, 4, 8],
    }


def test_g5_implementation_schema_accepts_one_rank_evidence() -> None:
    Draft202012Validator(schema()).validate(valid_record())


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("multi_gpu_scaling_verified",), True),
        (("physical_rank_counts_deferred",), [2, 4]),
        (("world_size",), 2),
        (("collectives", 0, "frequency"), 0),
        (("collectives", 0, "purpose"), ""),
    ],
)
def test_g5_implementation_schema_rejects_claim_or_telemetry_drift(
    path: tuple[str | int, ...],
    value: object,
) -> None:
    record = copy.deepcopy(valid_record())
    destination: object = record
    for part in path[:-1]:
        destination = destination[part]  # type: ignore[index]
    destination[path[-1]] = value  # type: ignore[index]
    assert list(Draft202012Validator(schema()).iter_errors(record))
