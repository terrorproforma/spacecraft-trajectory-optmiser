"""GPU bound classification must preserve independently replayed trajectory accuracy."""

from types import SimpleNamespace

import pytest
from test_gtoc12_gpu_discretisation import synthetic_boundary
from test_gtoc12_gpu_scvx import GPU, settings

from spacepdhcg.gtoc12.gpu_execution import using_gpu_execution
from spacepdhcg.gtoc12.low_thrust import certify_leg, solve_leg


@GPU
@pytest.mark.parametrize("origin", [0, 1])
@pytest.mark.parametrize("hold", ["zoh", "lagrange"])
@pytest.mark.parametrize("ruiz,preserve", [(0, 0), (2, 1), (5, 0)])
def test_device_bound_types_match_fresh_reference(monkeypatch, origin, hold, ruiz, preserve):
    monkeypatch.setenv("SPACEPDHCG_TEST_QOCO_DEVICE_INITIALIZATION", "1")
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_QOCO_POOL", "0")
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_STATE_ORIGIN", str(origin))
    monkeypatch.setenv("SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE", str(preserve))
    config = settings(max_iterations=30, time_limit_s=30, qoco_ruiz_iterations=ruiz, hold=hold)
    policy = SimpleNamespace(gpu_execution="graph", outer_loop_backend="cuda", workers=1)
    with using_gpu_execution(policy):
        results, certificates = [], []
        for enabled in (0, 1, None):
            if enabled is None:
                monkeypatch.delenv("SPACEPDHCG_TEST_QOCO_DEVICE_BOUND_TYPES", raising=False)
            else:
                monkeypatch.setenv("SPACEPDHCG_TEST_QOCO_DEVICE_BOUND_TYPES", str(enabled))
            result = solve_leg(synthetic_boundary(), config)
            certificate = certify_leg(result)
            assert result.converged and certificate.within_tolerance
            assert result.iterations == len(result.solver_reports) == len(result.history)
            results.append(result)
            certificates.append(certificate)
        for index in (1, 2):
            assert abs(certificates[0].final_mass_kg - certificates[index].final_mass_kg) < 1e-5
            before, after = (results[i].solver_reports[0] for i in (0, index))
            assert after["adapter_d2h_bytes"] < before["adapter_d2h_bytes"]
            assert after["adapter_d2h_count"] == before["adapter_d2h_count"] - 3
            assert after["device_numeric_updates"] == before["device_numeric_updates"]
