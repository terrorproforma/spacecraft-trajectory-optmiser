"""Independent GPU propagation against Kepler and CPU DOP853, including failures."""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.ephemeris import propagate_kepler
from spacepdhcg.gtoc12.gpu_verifier import ARC, LEG, SAMPLE, GpuVerifier
from spacepdhcg.gtoc12.solution import make_burn_arc
from spacepdhcg.gtoc12.verifier import (
    LagrangeThrust,
    _thrust_dynamics,
    propagate_burn,
    propagate_coast,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="requires serialized CUDA tests"
)


def fixture(n=1, days=365.25, eccentricity=0.0):
    legs = np.zeros(n, dtype=LEG)
    r = C.AU_KM * (1 - eccentricity)
    legs["initial"] = [r, 0, 0, 0, np.sqrt(C.MU_SUN_KM3_S2 * (1 + eccentricity) / r), 0, 2500]
    legs["duration_s"] = days * C.DAY_S
    return legs, np.empty(0, dtype=ARC), np.empty(0, dtype=SAMPLE)


def test_retained_session_reuses_grows_and_closes():
    from spacepdhcg.gtoc12.gpu_verifier import GpuVerifierSession

    with GpuVerifierSession() as session:
        first = session.propagate(*fixture(2))
        handle = session._gpu
        second = session.propagate(*fixture(1))
        assert session._gpu is handle
        np.testing.assert_array_equal(first[:1], second)
        grown = session.propagate(*fixture(17))
        assert session._gpu is not handle
        np.testing.assert_array_equal(grown[0], first[0])
        with ThreadPoolExecutor(1) as pool:
            with pytest.raises(RuntimeError, match="creating thread"):
                pool.submit(session.propagate, *fixture(1)).result()
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.propagate(*fixture(1))


@pytest.mark.parametrize("days,eccentricity", [(0, 0), (365.25, 0), (5479, 0), (250, 0.65)])
def test_coast_matches_exact_kepler(days, eccentricity):
    legs, arcs, samples = fixture(17, days, eccentricity)
    with GpuVerifier(17, 0, 0) as gpu:
        out = gpu.propagate(legs, arcs, samples)
        assert not out["status"].any()
    r, v = propagate_kepler(legs["initial"][:, :3], legs["initial"][:, 3:6], legs["duration_s"])
    assert np.max(np.linalg.norm(out["final_state"][:, :3] - r, axis=1)) < 0.01
    assert np.max(np.linalg.norm(out["final_state"][:, 3:6] - v, axis=1)) < 1e-8
    np.testing.assert_array_equal(out["final_state"][:, 6], legs["initial"][:, 6])
    _, _, expected_min = propagate_coast(
        0, legs["initial"][0, :3], legs["initial"][0, 3:6], 2500, days
    )
    assert np.max(abs(out["minimum_radius_km"] - expected_min)) < 0.01


@pytest.mark.parametrize("count", [2, 3, 4, 37])
def test_cubic_burn_and_coast_gaps_match_cpu(count):
    legs, _, _ = fixture(33, 45)
    times = np.linspace(3, 37, count) * C.DAY_S
    thrust = np.column_stack(
        (
            0.15 + 0.12 * np.sin(times / (20 * C.DAY_S)),
            0.08 * np.cos(times / (9 * C.DAY_S)),
            np.linspace(-0.02, 0.04, count),
        )
    )
    samples = np.zeros(count, dtype=SAMPLE)
    samples["seconds"], samples["thrust"] = times, thrust
    arcs = np.array([(0, count)], dtype=ARC)
    legs["arc_count"] = 1
    with GpuVerifier(len(legs), 1, count) as gpu:
        out = gpu.propagate(legs, arcs, samples)
        assert not out["status"].any(), out
        np.testing.assert_array_equal(out, gpu.propagate(legs, arcs, samples))
    r, v, min1 = propagate_coast(0, legs["initial"][0, :3], legs["initial"][0, 3:6], 2500, 3)
    # A much tighter, bounded-step CPU oracle resolves every cubic stencil.
    # Ordinary CPU DOP853 can skip these changes despite its local tolerance.
    oracle = solve_ivp(
        _thrust_dynamics(LagrangeThrust(times - times[0], thrust)),
        (0, times[-1] - times[0]),
        np.r_[r, v, 2500],
        method="DOP853",
        rtol=2.3e-14,
        atol=[1e-9] * 3 + [1e-12] * 3 + [1e-11],
        max_step=C.DAY_S / 16,
    )
    assert oracle.success
    exact = oracle.y[:, -1]
    er, ev, _ = propagate_coast(37, exact[:3], exact[3:6], exact[6], 45)
    assert np.max(np.linalg.norm(out["final_state"][:, :3] - er, axis=1)) < 0.001
    assert np.max(np.linalg.norm(out["final_state"][:, 3:6] - ev, axis=1)) < 1e-9
    assert np.max(abs(out["final_state"][:, 6] - exact[6])) < 1e-7
    r, v, m, min2 = propagate_burn(3, r, v, 2500, make_burn_arc(times / C.DAY_S, thrust))
    r, v, min3 = propagate_coast(37, r, v, m, 45)
    # Retain ordinary production-verifier parity separately from the tighter oracle.
    assert np.max(np.linalg.norm(out["final_state"][:, :3] - r, axis=1)) < 0.2
    assert np.max(np.linalg.norm(out["final_state"][:, 3:6] - v, axis=1)) < 2e-7
    assert np.max(abs(out["final_state"][:, 6] - m)) < 1e-5
    assert np.max(abs(out["minimum_radius_km"] - min(min1, min2, min3))) < 0.002


def test_constant_burn_mass_and_bad_leg_isolation():
    legs, _, _ = fixture(7, 9)
    samples = np.array([(0, [0.3, 0.4, 0]), (9 * C.DAY_S, [0.3, 0.4, 0])], dtype=SAMPLE)
    arcs = np.array([(0, 2)], dtype=ARC)
    legs["arc_count"] = 1
    legs[1]["initial"][0] = np.nan
    legs[2]["initial"][6] = -1
    legs[3]["arc_offset"] = 100
    legs[4]["duration_s"] = -1
    with GpuVerifier(7, 1, 2) as gpu:
        out = gpu.propagate(legs, arcs, samples)
        np.testing.assert_array_equal(out["status"], [0, 1, 1, 1, 1, 0, 0])
        assert np.isnan(out["final_state"][1:5]).all()
        assert (
            np.max(
                abs(
                    out["final_state"][[0, 5, 6], 6]
                    - (2500 - 0.5 * 9 * C.DAY_S / (C.ISP_S * C.G0_M_S2))
                )
            )
            < 1e-9
        )
        limited = gpu.propagate(legs[:1], arcs, samples, max_steps=1)
        assert limited["status"][0] == 3 and np.isnan(limited["final_state"]).all()
        samples[1]["seconds"] = 0
        assert gpu.propagate(legs[:1], arcs, samples)["status"][0] == 1


def test_capacity_thread_and_closed_workspace():
    legs, arcs, samples = fixture(1, 0)
    gpu = GpuVerifier(1, 0, 0)
    with gpu:
        with pytest.raises(RuntimeError, match="CUDA status"):
            gpu.propagate(np.repeat(legs, 2), arcs, samples)
        with ThreadPoolExecutor(1) as pool:
            with pytest.raises(RuntimeError, match="creating thread"):
                pool.submit(gpu.propagate, legs, arcs, samples).result()
        assert gpu.propagate(legs, arcs, samples)["status"][0] == 0
    gpu.close()
    with pytest.raises(RuntimeError, match="closed"):
        gpu.propagate(legs, arcs, samples)


@pytest.mark.parametrize("vinf,mass_drop", [(0, 0), (7, 0), (0, 1)])
def test_mission_rules_and_no_cpu_propagation(monkeypatch, vinf, mass_drop):
    from test_gtoc12_verifier import _coasting_earth_ship, _empty_catalogue

    from spacepdhcg.gtoc12 import verifier

    solution = _coasting_earth_ship(vinf, mass_drop)
    expected = verifier.Gtoc12Verifier(_empty_catalogue()).verify(solution)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU numerical propagation was invoked")

    monkeypatch.setattr(verifier, "propagate_burn", forbidden)
    monkeypatch.setattr(verifier, "propagate_coast", forbidden)
    actual = verifier.Gtoc12Verifier(_empty_catalogue(), propagation_backend="cuda").verify(
        solution
    )
    assert actual.ok == expected.ok
    assert [(v.code, v.ship_id) for v in actual.violations] == [
        (v.code, v.ship_id) for v in expected.violations
    ]
    assert actual.total_mass_kg == expected.total_mass_kg
    assert abs(actual.max_position_error_km - expected.max_position_error_km) < 0.01


def test_cuda_options_do_not_silently_fallback():
    from spacepdhcg.gtoc12.verifier import Gtoc12Verifier

    for options in [
        {"propagation_backend": "bogus"},
        {"propagation_backend": "cuda", "history": {}},
        {"propagation_backend": "cuda", "rtol": 1e-10},
    ]:
        with pytest.raises(ValueError):
            Gtoc12Verifier(None, **options)


@pytest.mark.parametrize(
    "name,ships,asteroids,total",
    [
        ("39_mass_optimal.txt", 39, 356, 28975.140269),
        ("37_mass_optimal_self_cleaning.txt", 37, 338, 27045.268330),
        ("GTOC12_JPL_merged_solution_36sc.txt", 36, 320, 26062.646065),
    ],
)
def test_cuda_archived_missions_reproduce_reference_scores(name, ships, asteroids, total):
    from spacepdhcg.gtoc12.data import (
        data_available,
        load_bonus_table,
        load_catalogue,
        verified_path,
    )
    from spacepdhcg.gtoc12.verifier import Gtoc12Verifier

    if not data_available():
        pytest.skip("pinned GTOC12 data not fetched")
    report = Gtoc12Verifier(
        load_catalogue(), bonus=load_bonus_table(), propagation_backend="cuda"
    ).verify_file(verified_path(name))
    assert report.ok, report.violations[:5]
    assert report.ship_count == ships and report.mined_asteroid_count == asteroids
    assert report.total_mass_kg == pytest.approx(total, abs=1e-5)
