from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from spacepdhcg.experiments.g4 import ACCEPTED_TIMING_BOUNDARY, G4ContractError
from spacepdhcg.experiments.g4_scheduler import Claim

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_g4_campaign",
    ROOT / "scripts/gpu/run_g4_campaign.py",
)
assert SPEC is not None and SPEC.loader is not None
CAMPAIGN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAMPAIGN)


def capability(executable: Path) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": 1,
        "source_commit": "a" * 40,
        "policy_sha256": "b" * 64,
        "matrix_sha256": "c" * 64,
        "executable_sha256": CAMPAIGN.sha256_path(executable),
        "axes": {
            name: {"status": "execution_only" if name in {"repeat", "solver_order"} else "applied"}
            for name in CAMPAIGN.CAPABILITY_AXES
        },
        "timing_boundary": ACCEPTED_TIMING_BOUNDARY,
        "independent_replay": True,
    }
    value["capability_sha256"] = hashlib.sha256(CAMPAIGN.canonical_bytes(value)).hexdigest()
    return value


def test_capability_refuses_every_hash_mismatch(tmp_path: Path) -> None:
    executable = tmp_path / "executor"
    executable.write_bytes(b"executor")
    path = tmp_path / "capability.json"
    value = capability(executable)
    path.write_text(json.dumps(value))
    loaded = CAMPAIGN.load_capabilities(path, executable, "b" * 64, "c" * 64, "a" * 40)
    assert loaded["capability_sha256"] == value["capability_sha256"]

    cases = (
        ("source_commit", "d" * 40, "source commit"),
        ("policy_sha256", "d" * 64, "policy hash"),
        ("matrix_sha256", "d" * 64, "matrix hash"),
        ("executable_sha256", "d" * 64, "executable hash"),
        ("capability_sha256", "d" * 64, "content hash"),
    )
    for field, replacement, message in cases:
        changed = dict(value)
        changed[field] = replacement
        path.write_text(json.dumps(changed))
        with pytest.raises(G4ContractError, match=message):
            CAMPAIGN.load_capabilities(
                path,
                executable,
                "b" * 64,
                "c" * 64,
                "a" * 40,
            )


def test_command_transmits_complete_coordinate_contract(tmp_path: Path) -> None:
    coordinate = {
        "family": "P1-D-pd6",
        "intervals": 50,
        "policy": "hybrid-pdhcg-ipm",
        "warm_mode": "primal_dual",
        "quality_tolerance": 1.0e-8,
        "attitude_class": 0.05,
        "rate_class": 0.01,
        "quality_tier": "ipm",
        "scaling_mode": "always_refresh",
        "conditioning": 8,
        "seed": 71,
        "repeat_kind": "measured",
        "repeat": 6,
        "solver_order": 5,
    }
    claim = Claim(1, "d" * 64, "attempt", coordinate)
    command = CAMPAIGN.command_for(
        tmp_path / "executor",
        claim,
        "a" * 64,
        "b" * 64,
        "c" * 64,
    )
    assert command[13:] == [
        "8",
        "71",
        "measured",
        "6",
        "5",
        "d" * 64,
        "b" * 64,
        "c" * 64,
    ]
