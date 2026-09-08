"""Optional winner ABI compatibility, using fake native calls and no CUDA device."""

import ctypes as ct
from types import SimpleNamespace

import numpy as np
import pytest

from spacepdhcg.gtoc12 import gpu_joint
from spacepdhcg.gtoc12.jointopt import JointItinerary
from spacepdhcg.gtoc12.retiming import Visit

SELECTION_FLAG = "SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SELECTION"


def _fake_native(monkeypatch, *, has_best):
    """Write deterministic C-ABI outputs so the real Python routing is exercised."""
    events = []
    count, n = 3, 3
    values = np.zeros(count, gpu_joint.RESULT)
    values["mass_count"] = n - 1
    values["objective"] = [11.0, 17.0, 17.0]
    values["weighted"] = values["objective"]
    values["collected"] = 20.0
    values["spare"] = 5.0
    values["propellant"] = 100.0
    values["final_mass"] = 525.0
    masses = np.tile([3000.0, 2800.0], (count, 1))
    inflations = np.ones((count, n - 1))
    proxies = np.tile([1.5, 2.0], (count, 1))
    payload = np.tile([0.0, 20.0, 0.0], (count, 1))

    def write(pointer, array):
        ct.memmove(pointer, array.ctypes.data, array.nbytes)

    def create(device, capacity, visits, output):
        assert (device, capacity, visits) == (0, count, n)
        ct.cast(output, ct.POINTER(ct.c_void_p))[0] = ct.c_void_p(123)
        return 0

    def destroy(output):
        ct.cast(output, ct.POINTER(ct.c_void_p))[0] = ct.c_void_p()
        return 0

    def evaluate(handle, rows, *arrays):
        assert handle.value == 123 and rows == count
        if arrays[7] is None:
            events.append("native_preflight")
            preflight = np.zeros(count, gpu_joint.RESULT)
            preflight["failure"] = 12
            write(arrays[6], preflight)
        else:
            events.append("native_full_results")
            for pointer, array in zip(
                arrays[6:], (values, masses, inflations, proxies, payload), strict=True
            ):
                write(pointer, array)
        return 0

    def best(handle, rows, *arrays):
        assert handle.value == 123 and rows == count and arrays[6] == 10.0
        events.append("native_best")
        selected = np.zeros(1, gpu_joint.SELECTION)
        selected["index"] = 1
        selected["value"] = values[1]
        write(arrays[7], selected)
        for pointer, array in zip(arrays[8:], (masses, inflations, proxies, payload), strict=True):
            write(pointer, array[1:2])
        return 0

    library = SimpleNamespace(
        spacepdhcg_gtoc12_joint_create=create,
        spacepdhcg_gtoc12_joint_evaluate_host=evaluate,
        spacepdhcg_gtoc12_joint_destroy=destroy,
    )
    if has_best:
        library.spacepdhcg_gtoc12_joint_best_host = best

    def no_geometry(*args, **kwargs):
        pytest.fail("compatibility tests must not compute geometry")

    backend = SimpleNamespace(
        library=library, device_id=0, _owned=lambda: None, telemetry={}, paired_hops=no_geometry
    )
    native = gpu_joint.GpuJoint(backend, count, n)
    visits = [
        Visit(0, False, False, "earth_out"),
        Visit(1, True, True, "earth_return"),
        Visit(0, False, False, ""),
    ]
    arrivals = np.tile([64500.0, 65000.0, 66500.0], (count, 1))
    departures = np.tile([64500.0, 65800.0, 66500.0], (count, 1))
    joint = SimpleNamespace(
        key=JointItinerary.key,
        _lambert={
            JointItinerary.key(0, 1, 64500.0, 65000.0): 1.5,
            JointItinerary.key(1, 0, 65800.0, 66500.0): 2.0,
        },
        measured={},
        evaluations=0,
        lambert_evaluations=0,
    )

    def metadata(*args):
        events.append("metadata")
        return (
            np.zeros(n, gpu_joint.VISIT),
            np.zeros(n - 1, gpu_joint.STAGE),
            np.zeros(1, gpu_joint.POLICY),
        )

    monkeypatch.setattr(gpu_joint, "_metadata", metadata)
    return native, joint, visits, arrivals, departures, events


@pytest.mark.parametrize("minimum", [None, 10.0])
@pytest.mark.parametrize(
    "has_best, override, use_best",
    [
        (False, None, False),
        (False, "0", False),
        (True, None, True),
        (True, "0", False),
        (True, "1", True),
    ],
)
def test_selection_default_and_explicit_flags(monkeypatch, has_best, override, use_best, minimum):
    if override is None:
        monkeypatch.delenv(SELECTION_FLAG, raising=False)
    else:
        monkeypatch.setenv(SELECTION_FLAG, override)
    native, joint, visits, arr, dep, events = _fake_native(monkeypatch, has_best=has_best)
    try:
        output = native.run(joint, visits, arr, dep, minimum_objective=minimum)
        expected_call = "native_best" if use_best and minimum is not None else "native_full_results"
        assert events == ["metadata", "native_preflight", expected_call]
        assert joint.evaluations == len(arr) and joint.lambert_evaluations == 0
        if minimum is None:
            assert [value.objective for value in output] == [11.0, 17.0, 17.0]
            assert all(value.feasible for value in output)
        else:
            assert output[0] == 1  # both native-best and host selection keep the first tie
            assert output[1].feasible and output[1].objective == 17.0
            assert output[1].masses == [3000.0, 2800.0]
    finally:
        native.close()


@pytest.mark.parametrize("minimum", [None, 10.0])
def test_explicit_missing_selection_fails_before_any_work(monkeypatch, minimum):
    monkeypatch.setenv(SELECTION_FLAG, "1")
    native, joint, visits, arr, dep, events = _fake_native(monkeypatch, has_best=False)
    before = dict(joint._lambert)
    try:
        with pytest.raises(RuntimeError, match=r"DEVICE_SELECTION=1.*gtoc12_joint_best_host"):
            native.run(joint, visits, arr, dep, minimum_objective=minimum)
        assert events == []  # neither metadata nor either native evaluation call ran
        assert joint._lambert == before and joint.evaluations == joint.lambert_evaluations == 0
        assert native.gpu.telemetry == {}
    finally:
        native.close()
