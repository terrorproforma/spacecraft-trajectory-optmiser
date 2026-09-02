"""Targets that depend on pinned external artifacts (skip with an explicit reason when absent)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from spacepdhcg.literature import external_sources, gtoc, gtopx, tops
from spacepdhcg.literature import pd6_monte_carlo as mc

ROOT = Path(__file__).resolve().parents[1]


def _require(*artifact_ids: str) -> None:
    try:
        external_sources.require(artifact_ids)
    except external_sources.ArtifactUnavailable as error:
        pytest.skip(f"pinned artifact not cached: {error}")


# ------------------------------------------------------------------------------- GTOPX
def test_gtopx_best_known_vectors_reproduce_to_printed_precision() -> None:
    _require(
        "gtopx.source",
        "gtopx.solution.cassini1",
        "gtopx.solution.rosetta",
        "gtopx.solution.messenger_reduced",
        "gtopx.solution.gtoc1",
    )
    if shutil.which("g++") is None and shutil.which("clang++") is None:
        pytest.skip("no C++ compiler available to build the GTOPX evaluator")
    rows = gtopx.verify_best_known(("cassini1", "rosetta", "messenger_reduced", "gtoc1"))
    assert rows["cassini1"]["reproduced_exactly"]
    assert rows["gtoc1"]["reproduced_exactly"]
    assert rows["messenger_reduced"]["reproduced_exactly"]
    # Rosetta's official file prints six decimals; the evaluator agrees to that precision.
    assert rows["rosetta"]["reproduced_to_printed_precision"]
    assert rows["rosetta"]["printed_decimals"] == 6
    for row in rows.values():
        assert row["constraints_satisfied"]


def test_gtopx_solution_parser_keeps_all_digits() -> None:
    _require("gtopx.solution.gtoc1")
    solution = gtopx.load_best_known("gtoc1")
    assert solution.objective_text == "-1581950.131840605288744"
    assert len(solution.vector) == 8
    assert solution.vector_text[0] == "6810.405216554911021"


# ------------------------------------------------------------------------------- TOPS
def test_tops_ingest_and_frozen_selection() -> None:
    _require("tops.twobody", "tops.mee", "tops.cr3bp", "tops.solar_sail")
    problems = tops.ingest()
    assert len(problems) == 34
    families = {p.family for p in problems}
    assert families == {"two_body_cartesian", "modified_equinoctial", "cr3bp", "solar_sail"}
    selection = tops.select_by_metadata(problems)
    committed = json.loads(
        (ROOT / "benchmarks" / "literature" / "tops_selection.json").read_text(encoding="utf-8")
    )
    profile = json.loads(
        (ROOT / "benchmarks" / "literature" / "profiles" / "esa-tops-2026.json").read_text(
            encoding="utf-8"
        )
    )
    assert selection["selected"] == committed["selected"] == profile["frozen_selection"]
    assert committed["revision"] == "24fe8849b403af376773f09b64b5132e5591b94e"


def test_tops_dionysus_entry_matches_tafazzol_taheri_scaling() -> None:
    _require("tops.mee")
    problems = {p.key: p for p in tops.ingest() if p.family == "modified_equinoctial"}
    dionysus = problems["modified_equinoctial:P0"]
    assert "Dionysus" in dionysus.info
    assert abs(dionysus.tof_bounds[0] - 60.79091977865148) < 1e-9
    assert abs(dionysus.max_thrust - 0.013490919161437792) < 1e-12


# ------------------------------------------------------------------------------- GTOC
def test_gtoc9_official_examples_validate_under_reimplemented_rules() -> None:
    _require("gtoc9.debris", "gtoc9.example1", "gtoc9.example2")
    debris = gtoc.load_gtoc9_debris()
    assert len(debris) == 123
    example1 = gtoc.evaluate_gtoc9_mission(external_sources.fetch("gtoc9.example1"), debris)
    assert example1.valid, example1.violations
    assert example1.debris_removed == [3, 23, 51]
    assert example1.max_rendezvous_position_error_m < 100.0
    assert example1.max_propagation_position_error_m < 100.0
    assert (
        abs(
            example1.mission_cost_min_meur
            - (45.0 + 2.0e-6 * (example1.initial_mass_kg - 2000.0) ** 2)
        )
        < 1e-9
    )
    example2 = gtoc.evaluate_gtoc9_mission(external_sources.fetch("gtoc9.example2"), debris)
    assert example2.valid, example2.violations
    assert example2.debris_removed == [38, 46, 103, 114]


def test_gtoc9_evaluator_detects_violations() -> None:
    _require("gtoc9.debris", "gtoc9.example1")
    debris = gtoc.load_gtoc9_debris()
    events = gtoc.parse_gtoc9_submission(external_sources.fetch("gtoc9.example1"))
    events[2].m += 1.0  # break the Tsiolkovsky mass update
    corrupted = ROOT / "results" / "literature" / "_tmp_corrupted.txt"
    corrupted.parent.mkdir(parents=True, exist_ok=True)
    try:
        corrupted.write_text(
            "\n".join(" ".join(map(str, [e.t, *e.r, *e.v, e.m, *e.dv, e.event_id])) for e in events)
        )
        evaluation = gtoc.evaluate_gtoc9_mission(corrupted, debris)
    finally:
        corrupted.unlink(missing_ok=True)
    assert not evaluation.valid
    assert any("rule 13" in v or "rule 17" in v for v in evaluation.violations)


def test_gtoc_reduced_subsets_are_frozen_from_metadata() -> None:
    _require("gtoc9.debris", "gtoc12.asteroids", "gtoc5.problem_data")
    committed = json.loads(
        (ROOT / "benchmarks" / "literature" / "gtoc_reduced_subsets.json").read_text(
            encoding="utf-8"
        )
    )
    regenerated = gtoc.build_reduced_subsets()
    assert committed == regenerated
    assert len(committed["gtoc9"]["debris_ids"]) == 15
    assert len(committed["gtoc12"]["asteroid_ids"]) == 500
    assert committed["gtoc5"]["beletskij_id"] == 1


@pytest.mark.skipif(os.name != "posix", reason="pinned GTOC12 verifier is a Linux binary")
def test_gtoc12_official_verifier_accepts_bundled_example() -> None:
    _require("gtoc12.verification_program")
    _, _, example = gtoc.extract_gtoc12_verifier()
    verification = gtoc.run_gtoc12_verifier(example)
    assert verification.accepted
    assert verification.ships == 1
    assert verification.mined_asteroids == 0


# ------------------------------------------------------------------------------- Chari samples
def test_chari_samples_are_seeded_and_within_published_distribution() -> None:
    document = mc.load_samples()
    for size, batch in document["batches"].items():
        positions = np.asarray(batch["positions"])
        assert positions.shape == (int(size), 3)
        regenerated = mc.sample_initial_positions(int(size), batch["seed"])
        assert np.array_equal(positions, regenerated)
        for column, (low, high) in enumerate(mc.PUBLISHED_DISTRIBUTION):
            assert np.all(positions[:, column] >= low) and np.all(positions[:, column] < high)
