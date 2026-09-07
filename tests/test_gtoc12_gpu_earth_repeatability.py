"""Real 500-day Earth departure that exposed failures missed by coast fixtures."""

import os

import numpy as np
import pytest

from spacepdhcg.gtoc12.low_thrust import LegBoundary, ScvxSettings, certify_leg, solve_leg


@pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1",
    reason="requires serialized GPU native runtime",
)
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("origin", [False, True])
def test_repeated_real_earth_departure_retains_physics_and_fuel(monkeypatch, graph, origin):
    # Frozen official-catalogue boundary: Earth at MJD 64328 -> asteroid 57530
    # at MJD 64828. Constants make this regression independent of data downloads.
    boundary = LegBoundary(
        64328.0,
        np.array([-25267390.158699147, 144918560.83232275, -11774.074663434592]),
        np.array([-29.830354405015772, -5.218793306246871, -0.0003853654898849385]),
        64828.0,
        np.array([171651373.51726958, -363709202.4912605, -19475.760097566756]),
        np.array([15.012470659244006, 7.97938096478592, -0.2193023430910643]),
        3000.0,
        free_departure_vinf=True,
    )
    # Isolate the production defaults from any surrounding benchmark flags.
    for key in list(os.environ):
        if key.startswith("SPACEPDHCG_TEST_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_STATE_ORIGIN", str(int(origin)))
    if graph:
        for flag in (
            "GTOC12_OUTER_GRAPH",
            "QOCO_DEVICE_VALIDATION",
            "QOCO_NATIVE_NUMERIC_REPLAY",
            "QOCO_NATIVE_REPLAY",
            "QOCO_IPM_GRAPH",
        ):
            monkeypatch.setenv("SPACEPDHCG_TEST_" + flag, "1")
    settings = ScvxSettings(
        node_days=2.0,
        max_iterations=40,
        discretisation_backend="cuda",
        assembly_backend="cuda",
        convex_solver_backend="qoco",
        outer_loop_backend="cuda",
        seed_backend="cuda",
    )
    for repeat in range(4):
        solution = solve_leg(boundary, settings)
        certificate = certify_leg(solution)
        assert solution.converged, (repeat, solution.status, solution.diagnostic)
        assert certificate.within_tolerance, (repeat, certificate)
        assert abs(solution.final_mass_kg - 2657.3638394) <= 2e-5
        assert solution.outer_transfer_bytes["trajectory_upload_bytes"] == 0
