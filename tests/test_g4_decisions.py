from __future__ import annotations

import json
from pathlib import Path

from spacepdhcg.experiments.g4 import decide_h5, decide_h6

ROOT = Path(__file__).resolve().parents[1]


def _policy() -> dict:
    return json.loads((ROOT / "benchmarks/g4_policy.json").read_text(encoding="utf-8"))


def _h5_row(family: str, scale: int, reduction: float, **updates: object) -> dict:
    baseline = [10.0, 10.1, 9.9, 10.2, 9.8, 10.05, 9.95]
    row = {
        "family": family,
        "scale": scale,
        "fixed_tight_seconds": baseline,
        "adaptive_seconds": [value * (1.0 - reduction) for value in baseline],
        "baseline_failures": 0,
        "candidate_failures": 0,
        "attempts": 20,
        "matched_quality": True,
        "objective_equivalent": True,
        "forcing_satisfied": True,
        "censored": False,
    }
    row.update(updates)
    return row


def _h6_row(family: str, scale: int, reduction: float, **updates: object) -> dict:
    baseline = [20.0, 20.2, 19.8, 20.1, 19.9, 20.3, 19.7]
    row = {
        "family": family,
        "scale": scale,
        "ipm_seconds": baseline,
        "hybrid_seconds": [value * (1.0 - reduction) for value in baseline],
        "baseline_failures": 0,
        "candidate_failures": 0,
        "attempts": 20,
        "hybrid_residual": 1e-8,
        "ipm_residual": 8e-9,
        "unpolished_residual": 2e-7,
        "matched_quality": True,
        "conversion_and_polish_included": True,
        "transfer_reliable": True,
        "unpolished_failed_tier": False,
        "censored": False,
    }
    row.update(updates)
    return row


def test_h5_supported_requires_two_sustained_families() -> None:
    rows = [
        _h5_row(family, scale, 0.20)
        for family in ("P1-C-pd3", "P1-D-pd6")
        for scale in (20, 50, 100)
    ]
    result = decide_h5(rows, _policy())
    assert result["decision"] == "supported"
    assert set(result["supported_regions"]) == {"P1-C-pd3", "P1-D-pd6"}


def test_h5_rejected_when_every_family_is_sustained_adverse() -> None:
    rows = [
        _h5_row(family, scale, -0.20)
        for family in ("P1-C-pd3", "P1-D-pd6", "P1-E-low-thrust")
        for scale in (20, 50, 100)
    ]
    assert decide_h5(rows, _policy())["decision"] == "rejected"


def test_h5_mixed_and_unresolved_preserve_nondecisive_evidence() -> None:
    mixed = [
        *[_h5_row("P1-C-pd3", scale, 0.20) for scale in (20, 50, 100)],
        *[_h5_row("P1-D-pd6", scale, 0.02) for scale in (20, 50, 100)],
    ]
    result = decide_h5(mixed, _policy())
    assert result["decision"] == "mixed"
    unresolved = [_h5_row("P1-C-pd3", 20, 0.30, censored=True)]
    result = decide_h5(unresolved, _policy())
    assert result["decision"] == "unresolved"
    assert result["censored_coordinates"] == 1


def test_h5_failure_delta_and_objective_equivalence_block_support() -> None:
    rows = [
        _h5_row(
            family,
            scale,
            0.30,
            candidate_failures=2,
            objective_equivalent=False,
        )
        for family in ("P1-C-pd3", "P1-D-pd6")
        for scale in (20, 50, 100)
    ]
    assert decide_h5(rows, _policy())["decision"] != "supported"


def test_h6_supported_requires_ipm_quality_time_residual_and_scale() -> None:
    rows = [_h6_row("P1-E-low-thrust", scale, 0.15) for scale in (100, 500, 2000)]
    result = decide_h6(rows, _policy())
    assert result["decision"] == "supported"
    assert result["supported_regions"]["P1-E-low-thrust"] == [[100, 500, 2000]]


def test_h6_each_preregistered_gate_blocks_support() -> None:
    mutations = (
        {"hybrid_residual": 3e-8},
        {"unpolished_residual": 5e-8},
        {"conversion_and_polish_included": False},
        {"transfer_reliable": False},
    )
    for mutation in mutations:
        rows = [
            _h6_row("P1-E-low-thrust", scale, 0.15, **mutation)
            for scale in (100, 500, 2000)
        ]
        assert decide_h6(rows, _policy())["decision"] != "supported"
    slow = [_h6_row("P1-E-low-thrust", scale, 0.05) for scale in (100, 500, 2000)]
    assert decide_h6(slow, _policy())["decision"] != "supported"


def test_h6_rejected_mixed_and_unresolved() -> None:
    rejected = [
        _h6_row(family, scale, -0.05)
        for family in ("P1-C-pd3", "P1-D-pd6", "P1-E-low-thrust")
        for scale in (20, 50, 100)
    ]
    assert decide_h6(rejected, _policy())["decision"] == "rejected"
    mixed = [_h6_row("P1-C-pd3", scale, 0.05) for scale in (20, 50, 100)]
    assert decide_h6(mixed, _policy())["decision"] == "mixed"
    unresolved = [_h6_row("P1-C-pd3", 20, 0.20, censored=True)]
    result = decide_h6(unresolved, _policy())
    assert result["decision"] == "unresolved"
    assert result["censored_coordinates"] == 1
