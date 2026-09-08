"""Behavioral checks for the frozen CPU oracle and native-ready historical cases."""

from __future__ import annotations

import copy
import functools
import math

import pytest
from fixture_oracle import KIT, decode, encode, evaluate, read, sha, source_check

DATA = decode(read(KIT / "fixtures.json"))
CASES = DATA["historical"] + DATA["synthetic"]
SYNTHETIC = {c["id"]: c for c in DATA["synthetic"]}


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_pinned_finish_reproduces_qualification_and_costs_without_mutating_inputs(case):
    original = copy.deepcopy(encode(case))
    expected = copy.deepcopy(case["expected"])
    check = expected.pop("frozen_source_check")
    actual = evaluate(case)
    assert encode(actual) == encode(expected)
    assert source_check(case, actual) == check
    assert encode(case) == original


def test_historical_population_and_failures_are_complete_not_just_successes():
    assert len(DATA["historical"]) == 20
    assert {c["provenance"]["ship"] for c in DATA["historical"]} == {1, 4, 7, 10, 23}
    passing = [c for c in DATA["historical"] if c["expected"]["failure_name"] == "ok"]
    assert len(passing) == 2
    assert all(
        c["provenance"]["ship"] == 4 and c["provenance"]["prefix"] == "flat_deployment_proxy"
        for c in passing
    )
    for c in DATA["historical"]:
        assert c["expected"]["failure_name"] in ("ok", "mass_below_dry_plus_collected")
        assert all(d["stage"] in (1, 4) for d in c["expected"]["leg_results"])
        assert c["provenance"]["archived_dry_margin_kg"] > 0.0


@pytest.mark.parametrize("arm", ["historical", "synthetic"])
def test_native_packing_partitions_preserve_order_and_do_not_drop_failed_cases(arm):
    batch = DATA[arm + "_batch"]
    assert len(batch["candidates"]) == len(DATA[arm])
    di = li = 0
    for case, packed in zip(DATA[arm], batch["candidates"], strict=True):
        assert packed == {
            "deploy_begin": di,
            "deploy_count": len(case["deploys"]),
            "leg_begin": li,
            "leg_count": len(case["legs"]),
            "partial_mass": case["partial_mass"],
        }
        assert encode(batch["deploys"][di : di + len(case["deploys"])]) == encode(case["deploys"])
        assert encode(batch["legs"][li : li + len(case["legs"])]) == encode(case["legs"])
        di += len(case["deploys"])
        li += len(case["legs"])
    assert (di, li) == (len(batch["deploys"]), len(batch["legs"]))


def test_mining_and_authority_precedence_prevent_bad_models_from_hiding_failures():
    for name, code, calls in [
        ("missing_collect_before_bad_leg", 1, 0),
        ("short_stay_before_bad_leg", 2, 0),
        ("nan_deploy_raises_mining_before_authority", 6, 0),
        ("authority_before_nan_inflation", 3, 1),
        ("nan_inflation_after_authority", 4, 1),
        ("certified_cell_still_checks_authority", 3, 1),
    ]:
        e = SYNTHETIC[name]["expected"]
        assert e["result"]["failure"] == code
        assert e["frozen_source_check"]["authority_calls"] == calls
        assert e["result"]["processed_legs"] == 0
    mining = SYNTHETIC["nan_deploy_raises_mining_before_authority"]["expected"]
    assert mining["source_return_kind"] == "ValueError"
    assert mining["leg_results"][0]["pickup"] == 1
    assert mining["leg_results"][0]["gained"] == 0.0


def test_camp_and_reposition_do_not_add_cargo_before_collection():
    for name in ("camp_skips_pickup_bad_dv_and_model", "reposition_does_not_collect_early"):
        case = SYNTHETIC[name]
        first, second = case["expected"]["leg_results"]
        assert first["pickup"] == 0 and first["mass_after"] == case["partial_mass"]
        assert second["pickup"] == 1 and second["gained"] > 0.0
    assert SYNTHETIC["pickup_inside_epoch_tolerance"]["expected"]["leg_results"][0]["pickup"] == 1
    assert SYNTHETIC["pickup_outside_epoch_tolerance"]["expected"]["leg_results"][0]["pickup"] == 0


def test_repeated_pickup_retains_first_dictionary_position_and_distinct_running_mass():
    c = SYNTHETIC["repeated_pickup_changes_mass_once_dictionary"]
    e = c["expected"]
    gained = e["leg_results"][0]["gained"]
    assert e["pickup_insertion_order"] == [0]
    assert e["result"]["collected"] == gained
    assert e["result"]["final_mass"] == c["partial_mass"] + gained + gained


def test_python312_compensated_cargo_sum_cannot_be_replaced_by_plain_running_sum():
    e = SYNTHETIC["cpython312_compensated_sum_pickup_order"]["expected"]
    cargo = e["collected_by_deploy"]
    plain = functools.reduce(lambda a, b: a + b, cargo, 0.0)
    assert e["pickup_insertion_order"] == [0, 2, 1]
    assert len(e["leg_results"]) == 4
    assert e["result"]["collected"] == sum(cargo) == math.fsum(cargo)
    assert e["result"]["collected"] != plain


def test_final_mass_gate_accepts_equality_and_rejects_one_kg_deficit():
    assert SYNTHETIC["exact_dry_plus_cargo_equality"]["expected"]["result"]["margin"] == 0.0
    assert SYNTHETIC["exact_dry_plus_cargo_equality"]["expected"]["failure_name"] == "ok"
    assert SYNTHETIC["mass_below_dry_plus_cargo"]["expected"]["result"]["margin"] == -1.0
    assert SYNTHETIC["mass_below_dry_plus_cargo"]["expected"]["result"]["failure"] == 5


def test_table_and_generic_return_preserve_nonfinite_dv_feature_semantics():
    generic = SYNTHETIC["negative_infinite_dv_generic_return"]["expected"]["leg_results"][0]
    table = SYNTHETIC["negative_infinite_dv_table_return"]["expected"]["leg_results"][0]
    assert generic["stage"] == table["stage"] == 4
    assert generic["inflation"] == table["inflation"]  # Both clamp to the same low ratio branch.
    assert math.isinf(generic["mass_after"]) and generic["mass_after"] > 0


def test_raw_source_inputs_and_header_hashes_remain_frozen():
    for name, digest in read(KIT / "source-sha256.json").items():
        assert sha(KIT / "source" / name) == digest
    for name, digest in read(KIT / "input-provenance.json")["inputs"].items():
        assert sha(KIT / "inputs" / name) == digest
    assert sha(KIT / "gtoc12_completion_c_api.h") == DATA["header_sha256"]
    assert sha(KIT / "features.json") == DATA["features_sha256"]
    assert sha(KIT / "inputs/completion-oracle-v619.json") == DATA["historical_oracle_sha256"]
