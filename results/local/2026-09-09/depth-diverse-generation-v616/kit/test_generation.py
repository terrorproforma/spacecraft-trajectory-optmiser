"""CPU construction and preservation checks using the real saved generated pool."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import common
import domain
import launch
import pytest
import recording
import run


def test_actual_saved_pool_attrition(old_pool, actual_inputs):
    weights, baseline = actual_inputs[1], actual_inputs[-1]
    rows = [domain.row_for(p, i, weights, baseline, 0) for i, p in enumerate(old_pool)]
    evidence = domain.attrition(old_pool, set(), rows)
    assert len(rows) == 198
    assert evidence["actual_selected_indices"] == [0, 11]
    saved = common.read(common.ROOT / "saved-pool-audit.json")
    assert evidence["generated_depth_counts"] == {
        int(k): v for k, v in saved["old_full_pool_depths"].items()
    }
    assert (
        len(evidence["earth_dedup_discarded_indices"])
        == 198 - saved["old_full_pool_distinct_earth_legs"]
    )
    assert evidence["same_earth_depth_groups"]
    assert max(row["raw_proxy_kg"] for row in rows) == pytest.approx(561.6700889801506)
    assert not any(row["proxy_raw_floor_eligible"] for row in rows)
    assert not evidence["closed_raw_and_weighted_proxy_eligible_indices"]
    assert not evidence["useful_discarded_certified_route_claim"]


def test_real_incumbent_weighting_arithmetic_only(actual_inputs):
    from spacepdhcg.gtoc12.search import RoutePlan

    weights, baseline = actual_inputs[1], actual_inputs[-1]
    # An archived plan checks this arithmetic only; it never enters the generated pool.
    p = RoutePlan.from_summary(common.read(common.ROOT / "inputs/ship-23.json")["plan"])
    row = domain.row_for(p, 0, weights, baseline, 123)
    assert row["weighted_proxy_kg"] == pytest.approx(
        sum(weights[a] * m for a, m in p.collected_mass.items())
    )
    assert row["raw_proxy_kg"] != pytest.approx(row["weighted_proxy_kg"])
    assert row["beam_score_including_propellant_penalty"] == 123
    assert row["candidate_fleet_raw_proxy_kg"] == pytest.approx(
        baseline["raw_kg"] - baseline["replaced_ship"]["raw_kg"] + p.total_collected_kg
    )
    assert not row["certified"] and not row["scored_fleet"] and not row["promoted"]


def test_pool_is_saved_in_original_order_before_selection(
    tmp_path, old_pool, actual_inputs, monkeypatch
):
    weights, baseline = actual_inputs[1], actual_inputs[-1]
    original = domain.attrition
    seen = []

    def inspect(plans, keys, rows):
        saved = common.read(tmp_path / "candidate-pool.json")
        assert len(saved) == len(plans) == 198
        assert [r["index"] for r in saved] == list(range(198))
        assert [r["plan"]["deploy_epochs"] for r in saved] == [
            {str(k): v for k, v in p.deploy_epochs.items()} for p in old_pool
        ]
        assert not (tmp_path / "shortlist-attrition.json").exists()
        seen.append(True)
        return original(plans, keys, rows)

    monkeypatch.setattr(domain, "attrition", inspect)
    report = run.persist_pool(tmp_path, old_pool, weights, baseline, lambda p: 0, set())
    assert seen == [True]
    assert report["immutable_pool_sha256"] == common.sha(tmp_path / "candidate-pool.json")
    assert len(common.read(tmp_path / "candidate-pool.json")) > len(
        report["actual_selected_indices"]
    )


def test_certified_earth_seed_and_footprint(actual_inputs):
    _catalogue, _weights, seed, allowed, excluded, baseline = actual_inputs
    assert (seed.target, seed.launch_epoch, seed.tof_days) == (30805, 64403.0, 570.0)
    assert seed.propellant_kg == pytest.approx(419.3243850979434)
    assert seed.certified and len(allowed) == 62
    assert not set(allowed) & excluded
    assert baseline["ship_count"] == 23
    assert baseline["raw_kg"] - baseline["raw_floor_kg"] == pytest.approx(8.359440535673)
    assert baseline["replaced_ship"]["ship"] == 23
    assert baseline["replaced_ship"]["raw_kg"] == pytest.approx(654.09993155373)


def test_finite_settings_do_not_change_physics_gates():
    from spacepdhcg.gtoc12.bundles import ClusterPricingSettings, cluster_search_settings

    expected = cluster_search_settings(ClusterPricingSettings(), 62)
    actual = domain.settings()
    for key in (
        "earth_out_authority_ratio",
        "hop_authority_ratio",
        "earth_return_authority_ratio",
        "hop_inflation",
        "earth_return_inflation",
        "earth_return_tof_model",
        "return_reserve_kg",
        "initial_mass",
        "end_margin_days",
        "collect_dp_step_days",
    ):
        assert getattr(actual, key) == getattr(expected, key)
    assert actual.max_deploys == actual.collect_dp_max_deploys == 10
    assert actual.beam_width == actual.max_per_first == actual.chain_tour_candidates == 24
    assert actual.max_per_deployed_set == 2
    assert actual.first_level_window_days == 0
    assert not actual.harvest_substitution and not actual.randomise
    assert actual.collect_dp and actual.chain_tour_scoring
    assert not actual.collect_dp_inflation_fit


@pytest.mark.parametrize("seconds", [0, -1, 121, float("nan"), float("inf")])
def test_reject_unbounded_wall_budget(seconds):
    with pytest.raises(ValueError, match="Wall budget"):
        domain.settings(seconds)


def test_source_and_inputs_are_exactly_pinned():
    common.activate()
    from spacepdhcg.gtoc12 import search

    assert Path(search.__file__).resolve() == common.ROOT / "source/src/spacepdhcg/gtoc12/search.py"
    assert len(common.read(common.ROOT / "source-sha256.json")) == 190
    assert len(common.read(common.ROOT / "inputs-sha256.json")) == 6
    _, env = common.environment({"SPACEPDHCG_UNREVIEWED": "1", "QOCO_OTHER": "1"})
    assert "SPACEPDHCG_UNREVIEWED" not in env and "QOCO_OTHER" not in env
    common.runtime_check(env)  # Hashes files; never loads them.
    env["SPACEPDHCG_UNREVIEWED"] = "1"
    with pytest.raises(ValueError, match="Unreviewed native switch"):
        common.runtime_check(env)


def test_native_guards_block_refinement(tmp_path):
    from spacepdhcg.gtoc12 import bundles, gpu_scvx, pipeline

    journal = recording.Journal(tmp_path, 120)
    with recording.native_guards(journal):
        for function in (
            bundles.certify_earth_legs,
            bundles.refine_route,
            pipeline.refine_route,
            pipeline.solve_leg,
            gpu_scvx.solve_native,
        ):
            with pytest.raises(RuntimeError, match="forbids all low-thrust"):
                function(None)
    assert not journal.counts


def test_cpu_dp_fallback_is_not_silently_accepted(tmp_path, monkeypatch):
    from spacepdhcg.gtoc12 import gpu_collect_dp

    monkeypatch.setattr(gpu_collect_dp, "cuda_collect_dp", lambda: NotImplemented)
    journal = recording.Journal(tmp_path, 120)
    with recording.native_guards(journal):
        with pytest.raises(RuntimeError, match="fallback forbidden"):
            gpu_collect_dp.cuda_collect_dp()
    assert journal.counts == {"collection_dp_passes_started": 1}


def test_infeasible_gpu_dp_result_is_retained_as_failure(tmp_path, monkeypatch):
    from spacepdhcg.gtoc12 import gpu_collect_dp

    monkeypatch.setattr(gpu_collect_dp, "cuda_collect_dp", lambda: None)
    journal = recording.Journal(tmp_path, 120)
    with recording.native_guards(journal):
        assert gpu_collect_dp.cuda_collect_dp() is None
    assert journal.counts == {"collection_dp_passes_started": 1, "collection_dp_passes_finished": 1}
    events = [
        json.loads(line) for line in (tmp_path / "generation-events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["detail"] == {"feasible": False}


@pytest.mark.parametrize(
    "stage",
    [
        "generation_calls",
        "expansions",
        "completion_attempts",
        "chain_tour_calls",
        "collection_dp_passes",
    ],
)
def test_each_stage_refuses_an_extra_call(tmp_path, stage):
    journal = recording.Journal(tmp_path, 120)
    journal.counts[stage + "_started"] = domain.LIMITS[stage]
    with pytest.raises(recording.GenerationBudget, match="Hard stage budget"):
        journal.begin(stage)
    assert not (tmp_path / "generation-events.jsonl").exists()


def test_completed_plan_survives_later_generation_failure(tmp_path, old_pool):
    class OriginalSearch:
        last_failure = "no_collection"

        def _complete(self, partial):
            return old_pool[0] if partial.good else None

    search = recording.recorded_class(OriginalSearch)()
    search.attach(recording.Journal(tmp_path, 120))
    assert search._complete(SimpleNamespace(deployed=[(30805, 64973)], good=True)) is old_pool[0]
    assert search._complete(SimpleNamespace(deployed=[(30805, 64973)], good=False)) is None
    lines = (tmp_path / "completed-plans.jsonl").read_text().splitlines()
    assert (
        len(lines) == 1
        and json.loads(lines[0])["total_collected_kg"] == old_pool[0].total_collected_kg
    )
    events = [
        json.loads(line) for line in (tmp_path / "generation-events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["detail"]["failure"] == "no_collection"
    assert search.journal.counts["completion_attempts_finished"] == 2


def test_supervisor_is_print_only_without_execute(tmp_path, capsys):
    args = SimpleNamespace(execute=False, output=tmp_path / "never_created", wall_seconds=120)
    assert launch.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["limits"]["full_route_refinements"] == 0
    assert report["limits"]["generation_calls"] == 1
    assert not report["launched"] and not args.output.exists()


def test_supervisor_refuses_existing_output_before_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ready_check", lambda: None)
    args = SimpleNamespace(execute=True, output=tmp_path, wall_seconds=120)
    with pytest.raises(FileExistsError, match="must be new"):
        launch.main(args)
    assert not (common.ROOT / "launch.json").exists()


def test_owned_child_timeout_is_reaped_and_partial_report_preserved(tmp_path):
    record = {}
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    try:
        code = launch.wait_owned_child(child, record, timeout=0.05, grace=0.2)
        assert code == 124 and child.poll() is not None and record["timed_out"]
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    common.write(
        tmp_path / "report.json",
        {"status": "generating", "counts": {"generation_calls_started": 1}},
    )
    original_hash = common.sha(tmp_path / "report.json")
    launch.retain_timeout_state(tmp_path, record)
    assert common.sha(tmp_path / "report.json") == original_hash
    retained = common.read(tmp_path / "supervisor-timeout.json")
    assert not retained["counts_complete"] and not retained["automatic_retry"]
