"""GPU winner reduction preserves selection and downloads only one detail row."""
import math
import os

import numpy as np
import pytest
import test_gtoc12_gpu_joint as oracle

from spacepdhcg.gtoc12.gpu_joint import COST, SELECTION, _metadata, evaluate_joint
from spacepdhcg.gtoc12.lambert import using_lambert_backend

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="requires serialized CUDA"
)


@pytest.fixture
def gpu(monkeypatch):
    monkeypatch.setenv(oracle.JOINT_FLAG, "1")
    with using_lambert_backend("cuda", maximum_batch_size=97) as backend:
        yield backend


@pytest.mark.parametrize("count", [1, 127, 128, 129, 257, 4097])
def test_device_selection_matches_full_download(monkeypatch, gpu, count):
    joint, visits, arr, dep, costs = oracle._synthetic()
    arrivals = np.tile(arr, (count, 1))
    departures = np.tile(dep, (count, 1))
    # The last candidate wins, spanning multiple reduction tiles and exercising
    # compaction from a nonzero row. Two copies create a first-in-order tie.
    first = max(0, count - 2)
    arrivals[first:, 1] -= 1
    departures[first:, 1] -= 1
    oracle._cache(joint, visits, arrivals, departures, costs)
    outputs, downloaded = [], []
    for mode in (0, 1):
        monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SELECTION", str(mode))
        before = gpu.telemetry.get("joint_result_download_bytes", 0)
        with monkeypatch.context() as guard:
            if mode:
                def forbidden(*args, **kwargs):
                    raise AssertionError("winner reduction must remain on CUDA")
                guard.setattr(np, "argmax", forbidden)
            outputs.append(evaluate_joint(
                joint, visits, arrivals, departures, minimum_objective=-math.inf
            ))
        downloaded.append(gpu.telemetry["joint_result_download_bytes"] - before)
    assert outputs[0][0] == outputs[1][0] == first
    oracle._same(outputs[1][1], outputs[0][1], atol=0, rtol=0)
    assert downloaded == [count * (32 * len(visits) + 40), 32 * len(visits) + 48]
    # Reuse must overwrite the compacted first row before evaluating again.
    assert evaluate_joint(
        joint, visits, arrivals, departures, minimum_objective=outputs[1][1].objective
    ) == (None, None)
    for minimum in (math.inf, math.nan):
        assert evaluate_joint(
            joint, visits, arrivals, departures, minimum_objective=minimum
        ) == (None, None)


def test_native_selection_empty_and_error_contract(monkeypatch, gpu):
    joint, visits, arr, dep, costs = oracle._synthetic(camp_only=True)
    oracle._cache(joint, visits, [arr], [dep], costs)
    evaluate_joint(joint, visits, [arr], [dep])
    native = gpu.joint_workspace
    output = np.zeros(1, SELECTION)
    assert SELECTION.itemsize == 72 and SELECTION.fields["value"][1] == 8
    assert native.best(native.handle, 0, *([None] * 6), 0., output.ctypes.data,
                       None, None, None, None) == 0
    assert output["index"][0] == -1 and output["invalid_stay"][0] == 0
    output["index"] = 41
    assert native.best(native.handle, 1, *([None] * 6), 0., output.ctypes.data,
                       None, None, None, None) == 1
    assert output["index"][0] == 41


def test_nonwinning_invalid_stay_survives_device_reduction(monkeypatch, gpu):
    joint, visits, arr, dep, costs = oracle._synthetic(camp_only=True)
    oracle._cache(joint, visits, [arr], [dep], costs)
    evaluate_joint(joint, visits, [arr], [dep])
    native = gpu.joint_workspace
    metadata, stages, policy = _metadata(joint, visits)
    # Extreme finite epochs trigger the same invalid-stay error as the full API.
    policy["mission_start"] = -1e308
    policy["latest_arrival"] = 1e308
    policy["minimum_stay"] = 0
    metadata["dwell_limit"] = np.inf
    stages["tof_min"] = -1e308
    stages["tof_max"] = np.inf
    stages["ratio_limit"] = np.inf
    arrivals = np.asarray([arr, [0, -1e308, 1e308]], dtype=np.float64)
    departures = np.asarray([dep, [0, 1e308, 1e308]], dtype=np.float64)
    packed = np.zeros((2, 2), COST)
    packed["lambert"] = costs
    output = np.zeros(1, SELECTION)
    inputs = (policy, metadata, stages, arrivals, departures, packed)
    assert native.best(native.handle, 2, *(a.ctypes.data for a in inputs), -math.inf,
                       output.ctypes.data, None, None, None, None) == 0
    assert output["index"][0] == 0 and output["invalid_stay"][0] == 1
