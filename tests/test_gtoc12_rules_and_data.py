"""GTOC12 rules encoding, pinned-data manifest and bonus/fleet formulas."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.data import PINS_PATH, RULES_PATH, data_available, load_pins

ROOT = Path(__file__).resolve().parents[1]


def test_rules_json_matches_constants_module() -> None:
    payload = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    assert payload == C.rules_payload()


def test_official_constants_transcribed() -> None:
    assert C.MU_SUN_KM3_S2 == 1.32712440018e11
    assert C.AU_KM == 1.49597870691e8
    assert C.G0_M_S2 == 9.80665
    assert C.MISSION_START_MJD == 64328.0 and C.MISSION_END_MJD == 69807.0
    assert C.ISP_S == 4000.0 and C.THRUST_MAX_N == 0.6
    assert C.DRY_MASS_KG == 500.0 and C.MAX_INITIAL_MASS_KG == 3000.0
    assert C.MINER_MASS_KG == 40.0 and C.MAX_MINERS_PER_SHIP == 20
    assert C.MINING_RATE_KG_PER_YEAR == 10.0 and C.MIN_MINING_STAY_YEARS == 1.0
    assert C.MAX_VINF_EARTH_KM_S == 6.0 and C.MIN_SUN_DISTANCE_AU == 0.3
    assert C.TOLERANCE_POSITION_KM == 1000.0
    assert C.TOLERANCE_VELOCITY_KM_S == pytest.approx(1.0e-3)
    assert C.TOLERANCE_MASS_KG == 1.0e-3
    assert C.EARTH.gravitational_parameter_km3_s2 == 3.98600435436e5
    assert C.MARS.minimum_pericentre_radius_km == 3689.0
    assert set(C.PLANETS) == {-2, -3, -4}


def test_bonus_coefficient_formula() -> None:
    assert C.bonus_coefficient(0.0) == 1.0
    # first row of the archived bonus_coefficients.txt: B = 0.85907431733049799 at 194.9805 kg
    assert C.bonus_coefficient(194.98050000000012) == pytest.approx(0.85907431733049799, abs=1e-14)
    # slow power-law decay towards the 1/3 floor, strictly decreasing in already-mined mass
    assert 1.0 / 3.0 < C.bonus_coefficient(1.0e12) < 0.4
    assert C.bonus_coefficient(100.0) > C.bonus_coefficient(200.0) > C.bonus_coefficient(1.0e4)
    with pytest.raises(ValueError):
        C.bonus_coefficient(-1.0)


@pytest.mark.parametrize(
    ("mean_mass", "table_value"),
    [(100.0, 2), (300.0, 6), (500.0, 14), (700.0, 32), (900.0, 73), (1000.0, 100)],
)
def test_maximum_ship_count_reproduces_table_1(mean_mass: float, table_value: int) -> None:
    assert math.floor(C.maximum_ship_count(mean_mass)) == table_value


def test_maximum_collected_mass() -> None:
    assert C.maximum_collected_mass(365.25) == pytest.approx(10.0)
    assert C.maximum_collected_mass(0.0) == 0.0


def test_pins_manifest_shape() -> None:
    pins = load_pins()
    names = {entry["name"] for entry in pins["files"]}
    assert {
        "GTOC12_Problem.pdf",
        "GTOC12_Submission_Format.pdf",
        "GTOC12_Asteroids_Data.txt",
        "bonus_coefficients.txt",
        "GTOC12_Verification_Program.zip",
        "GTOC12_JPL_merged_solution_36sc.txt",
        "39_mass_optimal.txt",
        "37_mass_optimal_self_cleaning.txt",
    } <= names
    for entry in pins["files"]:
        assert len(entry["sha256"]) == 64
        assert entry["bytes"] > 0
        assert entry["urls"]
    assert PINS_PATH.is_file()
    # large datasets are ignored, never committed
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "benchmarks/gtoc12/data/" in ignored


def test_data_available_is_boolean() -> None:
    assert data_available() in (True, False)
