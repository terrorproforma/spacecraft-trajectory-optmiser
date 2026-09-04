"""Earth-return pricing (TOF model, SCvx sweep override), harvest-window deploy ranking and the
memory accounting of the pricing worker."""

from __future__ import annotations

import numpy as np
import pytest

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.data import data_available, load_catalogue
from spacepdhcg.gtoc12.screening import (
    RETURN_INFLATION_P65,
    RETURN_INFLATION_TOF_DAYS,
    return_inflation_model,
    thrust_authority_km_s,
)
from spacepdhcg.gtoc12.search import EARTH_ID, PlannedLeg, RoutePlan

requires_data = pytest.mark.skipif(not data_available(), reason="pinned GTOC12 data not fetched")


@pytest.fixture(scope="module")
def catalogue():
    if not data_available():
        pytest.skip("pinned GTOC12 data not fetched")
    return load_catalogue()


# -- return inflation model ---------------------------------------------------------------


def test_return_inflation_model_prices_the_short_return_dear_and_the_long_one_cheap() -> None:
    """The archive's certified returns cost 1.30x Lambert at 420 d and 0.96x at 540 d; the
    model must reproduce that ordering (a flat factor cannot) and its ratio correction."""

    short = float(return_inflation_model(420.0, 0.33))
    long = float(return_inflation_model(540.0, 0.33))
    assert short == pytest.approx(1.383, abs=1e-6)  # the p65 of the 405-435 d bin
    assert long == pytest.approx(0.977, abs=1e-6)
    tofs = np.arange(420.0, 581.0, 15.0)
    values = return_inflation_model(tofs, 0.33)
    assert np.all(np.diff(values) <= 1e-12)  # monotone from 420 to 578 days
    # a return flown near the authority limit costs more than a slow one at the same TOF
    assert return_inflation_model(420.0, 0.45) > return_inflation_model(420.0, 0.25)
    # the correction is clamped, the model floored, and the table is well-formed
    assert float(return_inflation_model(578.0, 0.0)) == pytest.approx(0.85, abs=1e-6)  # floor
    assert float(return_inflation_model(578.0, 0.33)) == pytest.approx(0.885, abs=1e-6)
    assert len(RETURN_INFLATION_TOF_DAYS) == len(RETURN_INFLATION_P65)
    assert list(RETURN_INFLATION_TOF_DAYS) == sorted(RETURN_INFLATION_TOF_DAYS)
    # vectorised over a (departures, tofs) table as the DP uses it
    table = return_inflation_model(np.tile(tofs, (3, 1)), np.full((3, tofs.shape[0]), 0.3))
    assert table.shape == (3, tofs.shape[0])


@requires_data
def test_collect_table_and_beam_price_the_return_with_the_tof_model(catalogue) -> None:
    from spacepdhcg.gtoc12.collectdp import CollectDPSettings, CollectPairTable
    from spacepdhcg.gtoc12.search import RouteSearch, SearchSettings

    flat = CollectPairTable(catalogue, CollectDPSettings(return_tof_model=False))
    model = CollectPairTable(catalogue, CollectDPSettings())
    dv = np.asarray([6.2, 6.2])
    tofs = np.asarray([420.0, 540.0])
    mass = 1400.0
    assert np.allclose(flat.return_inflation(dv, mass, tofs), 1.6)
    ratio = dv / thrust_authority_km_s(mass, tofs, 1.0)
    expected = return_inflation_model(tofs, ratio)
    assert np.allclose(model.return_inflation(dv, mass, tofs), expected)
    # the same Lambert ΔV costs less propellant over the long return under the model
    propellant = model.return_propellant(dv, mass, tofs)
    assert propellant[1] < propellant[0]
    assert np.isclose(
        flat.return_propellant(dv, mass, tofs)[0], flat.return_propellant(dv, mass, tofs)[1]
    )
    # the beam's forward pass uses the same model for the return leg it writes into the plan
    search = RouteSearch(catalogue, catalogue.ids[:20], SearchSettings())
    assert search.return_inflation_for(6.2, mass, 540.0) == pytest.approx(
        float(return_inflation_model(540.0, float(ratio[1])))
    )
    off = RouteSearch(catalogue, catalogue.ids[:20], SearchSettings(earth_return_tof_model=False))
    assert off.return_inflation_for(6.2, mass, 540.0) == 1.6


# -- SCvx return sweep --------------------------------------------------------------------


def _plan_with_return(departure: float, tof: float) -> RoutePlan:
    legs = (
        PlannedLeg(
            EARTH_ID,
            7,
            C.MISSION_START_MJD + 60.0,
            C.MISSION_START_MJD + 560.0,
            7.0,
            1.0,
            "earth_out",
        ),
        PlannedLeg(7, 7, C.MISSION_START_MJD + 560.0, departure, 0.0, 1.0, "camp"),
        PlannedLeg(7, EARTH_ID, departure, departure + tof, 6.0, 1.0, "earth_return"),
    )
    return RoutePlan(
        legs,
        {7: C.MISSION_START_MJD + 560.0},
        {7: departure},
        {7: 100.0},
        300.0,
        700.0,
    )


def test_sweep_grid_is_lattice_aligned_and_stays_inside_the_window() -> None:
    from spacepdhcg.gtoc12.returnsweep import DEFAULT_RETURN_TOFS, return_leg_of, sweep_grid

    end = C.MISSION_END_MJD - 2.0
    departure = C.MISSION_START_MJD + 15.0 * round((end - 430.0 - C.MISSION_START_MJD) / 15.0)
    plan = _plan_with_return(departure, 420.0)
    assert return_leg_of(plan) is not None and return_leg_of(plan).to_id == EARTH_ID
    grid = sweep_grid(plan, back_steps=2, forward_steps=4)
    assert grid is not None
    departures, tofs = grid
    assert tuple(tofs) == DEFAULT_RETURN_TOFS
    assert np.allclose(
        (departures - C.MISSION_START_MJD) / 15.0,
        np.round((departures - C.MISSION_START_MJD) / 15.0),
    )
    # departures whose shortest TOF already misses the window are dropped
    assert np.all(departures + tofs.min() <= end + 1e-9)
    assert departures.min() == pytest.approx(departure - 30.0)
    with pytest.raises(ValueError):
        sweep_grid(plan, tofs=(420.0, 433.0))
    no_return = RoutePlan((plan.legs[0],), {7: 0.0}, {}, {}, 0.0, 0.0)
    assert sweep_grid(no_return) is None


@requires_data
def test_retimer_prices_a_swept_return_from_the_measurement(catalogue) -> None:
    """Swept cells: SCvx's feasibility replaces the authority check and the measured inflation
    replaces the model; unswept cells nearby inherit the nearest measurement; far cells keep the
    model; a return calibrated from a certified leg stores the residual against the model."""

    from spacepdhcg.gtoc12.retiming import RETURN_SWEEP_REACH, Retimer, RetimeSettings
    from spacepdhcg.gtoc12.returnsweep import ReturnSweep
    from spacepdhcg.gtoc12.search import SearchSettings

    retimer = Retimer(catalogue, SearchSettings(), RetimeSettings(step_days=15.0))
    asteroid = int(catalogue.ids[0])
    tofs = retimer._tofs("earth_return")
    lattice = retimer.lattice
    dv_table, _ = retimer.leg_table(asteroid, EARTH_ID, "earth_return")
    k = lattice.count - 40  # a departure ~600 days before the window end
    departures = lattice.epochs[[k, k + 1]]
    swept_tofs = np.asarray([420.0, 540.0])
    certified = np.asarray([[True, True], [False, True]])
    lambert = np.asarray(
        [
            [dv_table[k, int((420 - tofs[0]) / 15)], dv_table[k, int((540 - tofs[0]) / 15)]],
            [np.nan, dv_table[k + 1, int((540 - tofs[0]) / 15)]],
        ]
    )
    measured = np.where(certified, lambert * np.asarray([[1.3, 0.95], [1.0, 0.95]]), np.inf)
    sweep = ReturnSweep(
        asteroid,
        1400.0,
        departures,
        swept_tofs,
        np.ones((2, 2), dtype=bool),
        certified,
        measured,
        np.where(certified, 200.0, np.inf),
    )
    retimer.set_return_sweep(sweep)
    override = retimer._return_override(asteroid)
    assert override is not None
    inflation, ok = override
    t420, t540 = int((420 - tofs[0]) / 15), int((540 - tofs[0]) / 15)
    assert inflation[k, t420] == pytest.approx(1.3) and ok[k, t420]
    assert inflation[k, t540] == pytest.approx(0.95) and ok[k, t540]
    assert not ok[k + 1, t420] and np.isnan(inflation[k + 1, t420])  # SCvx refused it
    # a neighbour one step away inherits the nearest measurement; a cell beyond the sweep's
    # reach is infeasible (strict: the return is only re-timed onto what SCvx certified)
    assert inflation[k, t420 + 1] == pytest.approx(1.3) and ok[k, t420 + 1]
    far = k - RETURN_SWEEP_REACH - 3
    assert np.isnan(inflation[far, t420]) and not ok[far, t420]
    # a re-flight refusal retires the cell and the cells that inherited it
    assert retimer.refuse_return(asteroid, departures[0], 420.0)
    inflation, ok = retimer._return_override(asteroid)
    assert not ok[k, t420] and not ok[k, t420 + 1] and ok[k, t540]
    assert not retimer.refuse_return(asteroid, departures[0], 420.0)  # already refused
    # the DP's grid: swept cells are priced by the measurement, not by the authority ratio
    # (a refused cell is infeasible even though the proxy would accept it)
    assert retimer.return_sweeps[asteroid] is sweep
    # calibration under the TOF model stores the residual against the model at the flown TOF
    retimer.calibrate(asteroid, EARTH_ID, 1.383 * 1.1, authority_ratio=0.33, tof_days=420.0)
    residual = retimer.inflations[(asteroid, EARTH_ID)]
    assert residual == pytest.approx(1.1 * retimer.settings.calibration_margin, rel=1e-6)
    ratio = 6.0 / float(thrust_authority_km_s(1400.0, 540.0, 1.0))
    assert float(
        retimer.leg_inflation("earth_return", asteroid, EARTH_ID, 6.0, 1400.0, 540.0)
    ) == pytest.approx(float(return_inflation_model(540.0, ratio)) * residual)
    # without the model the calibration is the absolute factor as before
    flat = Retimer(catalogue, SearchSettings(earth_return_tof_model=False), RetimeSettings())
    flat.calibrate(asteroid, EARTH_ID, 1.2, authority_ratio=0.33, tof_days=420.0)
    assert flat.inflations[(asteroid, EARTH_ID)] == pytest.approx(
        1.2 * flat.settings.calibration_margin
    )


# -- archive-wide campaign -----------------------------------------------------------------


def test_select_tasks_takes_stand_alone_primaries_best_first_and_once_per_asteroid_set(
    tmp_path,
) -> None:
    import json

    from spacepdhcg.gtoc12.archive import discover_archives
    from spacepdhcg.gtoc12.returncampaign import ReturnCampaignSettings, select_tasks

    def archive(run: str, slot: int, kg: float, asteroids: list[int], **plan_extra) -> None:
        plan = {
            "asteroids": asteroids,
            "legs": [{"role": "earth_out"}, {"role": "earth_return"}],
            "foreign_deploy_epochs": {},
            "orphaned": False,
        }
        plan.update(plan_extra)
        directory = tmp_path / run / f"ship_{slot:02d}"
        directory.mkdir(parents=True)
        (directory / "route_summary.json").write_text(
            json.dumps({"certified": True, "total_collected_kg": kg, "plan": plan})
        )

    archive("run_a", 1, 603.7, [1, 2, 3])
    archive("run_a", 2, 480.0, [4, 5])
    archive("run_a", 3, 590.0, [6, 7], foreign_deploy_epochs={"9": 60000.0})  # cooperative
    archive("run_a", 4, 610.0, [8], orphaned=True)  # leaves a miner behind
    archive("run_b", 1, 603.7, [1, 2, 3])  # the same route archived by a later run
    archive("run_b", 2, 440.0, [10, 11])  # below the floor
    groups = discover_archives([tmp_path / "run_a", tmp_path / "run_b"])
    tasks = select_tasks(groups, ReturnCampaignSettings(min_collected_kg=450.0))
    assert [(t.slot, t.collected_kg) for t in tasks] == [(1, 603.7), (2, 480.0)]
    assert tasks[0].asteroids == (1, 2, 3) and tasks[0].name.endswith("/ship_01")
    assert select_tasks(groups, ReturnCampaignSettings(top=1))[0].collected_kg == 603.7


# -- memory accounting --------------------------------------------------------------------


def test_phase_memory_attributes_the_high_water_mark_to_the_phase_that_grew_it() -> None:
    from spacepdhcg.gtoc12.memory import (
        PhaseMemory,
        bound_heap_growth,
        release_heap,
    )
    from spacepdhcg.gtoc12.memory import current_rss_mb as _current_rss_mb
    from spacepdhcg.gtoc12.memory import peak_rss_mb as _peak_rss_mb

    memory = PhaseMemory()
    memory.mark("idle")
    if np.isnan(_peak_rss_mb()) or np.isnan(_current_rss_mb()):
        pytest.skip("no resource usage on this platform")
    # enough to lift the process high-water mark by >= 50 MB whatever ran before in this process
    headroom_mb = max(_peak_rss_mb() - _current_rss_mb(), 0.0) + 60.0
    block = np.ones(int(headroom_mb * 1024 * 1024) // 8)  # touched -> resident
    memory.mark("allocate")
    del block
    memory.mark("free")
    phases = {r["phase"]: r for r in memory.records}
    assert set(phases) == {"start", "idle", "allocate", "free"}
    assert phases["allocate"]["peak_growth_mb"] >= 40.0
    assert phases["free"]["peak_growth_mb"] == pytest.approx(0.0, abs=1.0)
    assert memory.hottest() == "allocate"
    assert all(r["elapsed_seconds"] >= 0.0 for r in memory.records)
    # the heap helpers are safe to call anywhere (True on glibc, False elsewhere)
    assert release_heap() in (True, False)
    assert bound_heap_growth() in (True, False)
