"""No GPU/model calls: tampering tests for frozen-readback classification only."""

from __future__ import annotations

import copy
import ctypes
import json
from unittest.mock import patch

import pytest
from gpu_batch import (
    KIT,
    Candidate,
    Deploy,
    Leg,
    LegResult,
    Policy,
    Result,
    Stats,
    compare,
    decode,
    encode,
)

DATA = decode(json.loads((KIT / "fixtures.json").read_text()))
CASES = DATA["historical"] + DATA["synthetic"]
COMPARISON = json.loads((KIT / "gpu-profile.json").read_text())["comparison"]


def saved_readback():
    return {
        "results": [copy.deepcopy(c["expected"]["result"]) for c in CASES],
        "leg_results": [copy.deepcopy(d) for c in CASES for d in c["expected"]["leg_results"]],
        "collected_by_deploy": [v for c in CASES for v in c["expected"]["collected_by_deploy"]],
    }


def test_abi_field_sizes_match_the_frozen_header_contract():
    assert [
        ctypes.sizeof(c) for c in (Policy, Candidate, Deploy, Leg, Result, LegResult, Stats)
    ] == [80, 24, 24, 128, 48, 72, 48]


def test_readback_comparison_never_loads_a_native_library_or_executes_a_cpu_model():
    with patch.object(ctypes, "CDLL", side_effect=AssertionError("No native execution")):
        assert compare(CASES, saved_readback(), COMPARISON)["passed"]


@pytest.mark.parametrize("field,value", [("failure", 0), ("failed_leg", 0), ("processed_legs", 0)])
def test_a_numeric_tolerance_cannot_convert_rejected_historical_case_to_acceptance(field, value):
    raw = saved_readback()
    assert raw["results"][0][field] != value
    raw["results"][0][field] = value
    assert not compare(CASES, raw, COMPARISON)["passed"]


@pytest.mark.parametrize("name", COMPARISON["exact_result_cases"])
def test_boundary_and_compensated_sum_results_require_exact_values(name):
    raw = saved_readback()
    index = next(i for i, c in enumerate(CASES) if c["id"] == name)
    raw["results"][index]["margin"] += 1e-12
    assert not compare(CASES, raw, COMPARISON)["passed"]


def test_large_mass_relative_tolerance_cannot_hide_lost_cargo_bits():
    raw = saved_readback()
    raw["results"][-1]["collected"] -= 64.0
    assert not compare(CASES, raw, COMPARISON)["passed"]


def test_small_transcendental_roundoff_is_allowed_without_changing_classification():
    raw = saved_readback()
    raw["results"][0]["final_mass"] += 1e-11
    assert compare(CASES, raw, COMPARISON)["passed"]


def test_nan_inflation_and_infinity_sign_are_not_replaced_with_zero():
    raw = saved_readback()
    i = next(
        i
        for i, d in enumerate(raw["leg_results"])
        if isinstance(d["inflation"], float) and d["inflation"] != d["inflation"]
    )
    raw["leg_results"][i]["inflation"] = 0.0
    assert not compare(CASES, raw, COMPARISON)["passed"]
    assert encode(decode({"float64": "-inf"})) == {"float64": "-inf"}
