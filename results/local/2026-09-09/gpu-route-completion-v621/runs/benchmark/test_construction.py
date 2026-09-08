"""Construction-only checks: no CPU completion, new feature arithmetic or GPU."""

from __future__ import annotations

import ctypes
from unittest.mock import patch

import pytest
from common import KIT, construct, read, sha, signature

import spacepdhcg.gtoc12.search as search_module
from spacepdhcg.gtoc12.gpu_completion import pack_inputs

PLAN = read(KIT / "plan.json")


@pytest.mark.parametrize("group", PLAN["groups"], ids=[g["id"] for g in PLAN["groups"]])
def test_constructed_requests_preserve_archived_epochs_models_and_cargo_without_costing(group):
    with (
        patch.object(ctypes, "CDLL", side_effect=AssertionError("No GPU")),
        patch.object(
            search_module, "propellant_for_delta_v", side_effect=AssertionError("No CPU costing")
        ),
    ):
        search, requests, cases, _ = construct(group)
        before = signature(requests)
        policy, candidates, deployments, legs = pack_inputs(search, requests)
        assert len(requests) == len(candidates) == group["size"]
        assert search.catalogue is None and search.collect_table.catalogue is None
        assert search.collect_table.lambert_evaluations == 0
        assert not search.collect_table.return_sweeps
        assert policy["initial_mass"][0] == 3000.0
        di = li = 0
        for request, case, c in zip(requests, cases, candidates, strict=True):
            assert c["partial_mass"] == case["partial_mass"]
            assert (int(c["deploy_begin"]), int(c["leg_begin"])) == (di, li)
            for saved, d in zip(
                case["deploys"], deployments[di : di + len(request[1])], strict=True
            ):
                assert (d["deploy_epoch"], d["collect_epoch"], d["has_collect"]) == (
                    saved["deploy_epoch"],
                    saved["collect_epoch"],
                    saved["has_collect"],
                )
            for saved, row in zip(case["legs"], legs[li : li + len(request[3])], strict=True):
                for field in ("role", "model", "source_deploy", "departure", "arrival", "dv"):
                    assert row[field] == saved[field]
                if row["role"] != 0:
                    assert row["authority_ratio"] == saved["authority_ratio"]
                if row["model"] == 2:
                    assert row["fit"].tolist() == saved["fit"]
                    assert row["floor"] == saved["floor"]
                    assert row["delta_a_au"] == saved["delta_a_au"]
                    assert row["delta_longitude_rad"] == saved["delta_longitude_rad"]
            di += len(request[1])
            li += len(request[3])
        assert signature(requests) == before
        assert (di, li) == (len(deployments), len(legs))


def test_budgets_and_success_failure_mix_are_prespecified_without_new_unique_solutions():
    assert len(PLAN["groups"]) == 12
    assert sum(g["size"] for g in PLAN["groups"]) * 5 == 14200
    assert PLAN["gpu_evaluate_calls"] == 12 * 5 == 60
    assert len(PLAN["warm_order"]) == 8
    assert PLAN["warm_order"].count("cpu") == PLAN["warm_order"].count("gpu") == 4
    for group in PLAN["groups"]:
        assert 0 < group["expected_accepted"] < group["size"]
        assert group["unique_control_count"] <= 10


def test_all_frozen_source_and_inputs_match_the_final_b_snapshot():
    for name, digest in read(KIT / "source-sha256.json").items():
        assert sha(KIT / "source" / name) == digest
    for name, digest in read(KIT / "input-sha256.json").items():
        assert sha(KIT / "inputs" / name) == digest
    assert (
        sha(KIT / "inputs/host-validation.json")
        == "2e7175e8f818d11987fc240b5511c54b498f5485798501703c03e0906359bc8a"
    )
