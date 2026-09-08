"""Native refinement must not invoke CPU numerical certification implicitly."""

import threading
from types import SimpleNamespace

import pytest

from spacepdhcg.gtoc12 import pipeline
from spacepdhcg.gtoc12.low_thrust import LegCertificate, ScvxSettings


@pytest.mark.parametrize(
    "outer,requested,selected",
    [
        ("python", "auto", "cpu"),
        ("cuda", "auto", "cuda"),
        ("cuda", "cpu", "cpu"),
        ("python", "cuda", "cuda"),
    ],
)
def test_backend_selection(outer, requested, selected):
    assert (
        ScvxSettings(
            outer_loop_backend=outer, certification_backend=requested
        ).selected_certification_backend()
        == selected
    )


def test_invalid_backend():
    with pytest.raises(ValueError, match="certification_backend"):
        ScvxSettings(certification_backend="typo").selected_certification_backend()


def driver(monkeypatch, backend="auto"):
    request = SimpleNamespace(deterministic_id=17)
    registry = pipeline.LegRegistry()
    registry.register(request, None)
    solution = SimpleNamespace(
        status="converged",
        diagnostic="",
        propellant_kg=1.0,
        boundary=SimpleNamespace(duration_days=10.0),
        delta_v_km_s=1.0,
        final_mass_kg=100.0,
        max_defect=0.0,
    )
    monkeypatch.setattr(pipeline, "solve_leg", lambda *args: solution)
    monkeypatch.setattr(pipeline, "clamp_thrust", lambda *args: None)
    instance = pipeline.Gtoc12ScvxDriver(
        None, None, registry, ScvxSettings(outer_loop_backend="cuda", certification_backend=backend)
    )
    return instance, request, registry.records[17]


def test_native_driver_reuses_cuda_certificate_and_rejects_failure(monkeypatch):
    from spacepdhcg.gtoc12 import gpu_verifier

    instance, request, record = driver(monkeypatch)
    counters = dict(created=0, closed=0, calls=0)

    class Workspace:
        def __init__(self):
            counters["created"] += 1

        def close(self):
            counters["closed"] += 1

    def certify(solutions, *, workspace):
        counters["calls"] += 1
        assert len(solutions) == 1 and isinstance(workspace, Workspace)
        if counters["calls"] == 3:
            raise RuntimeError("propagation failed")
        return [LegCertificate(0.0, 0.0, 100.0, 1.0, 0.0, 0.0)]

    def forbidden(*args):
        raise AssertionError("CPU propagation is forbidden")

    monkeypatch.setattr(gpu_verifier, "GpuVerifierSession", Workspace)
    monkeypatch.setattr(gpu_verifier, "certify_legs_cuda", certify)
    monkeypatch.setattr(pipeline, "certify_leg", forbidden)
    for _ in range(2):
        assert instance.solve(request, threading.Event()).status == pipeline.G3Status.CONVERGED
        assert record.certification_backend == "cuda"
    result = instance.solve(request, threading.Event())
    assert result.status == pipeline.G3Status.NUMERICAL_FAILURE
    assert "propagation failed" in result.diagnostic
    assert record.certificate is None and record.certification_backend is None
    instance.close()
    instance.close()
    assert counters == dict(created=1, closed=1, calls=3)


def test_explicit_cpu_ablation(monkeypatch):
    instance, request, record = driver(monkeypatch, "cpu")
    certificate = LegCertificate(0.0, 0.0, 100.0, 1.0, 0.0, 0.0)
    monkeypatch.setattr(pipeline, "certify_leg", lambda solution: certificate)
    assert instance.solve(request, threading.Event()).status == pipeline.G3Status.CONVERGED
    assert record.certificate is certificate and record.certification_backend == "cpu"
    assert instance._certificate_workspace is None
    instance.close()
