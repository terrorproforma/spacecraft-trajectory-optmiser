"""Recovery is a permutation, never an infeasibility filter."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from spacepdhcg.gtoc12.lambert import using_lambert_backend
from spacepdhcg.gtoc12.refinement_queue import recovery_order


def plan(body, arrival=65000.0):
    return SimpleNamespace(
        legs=[
            SimpleNamespace(from_id=0, to_id=body, departure_epoch=64000.0, arrival_epoch=arrival)
        ]
    )


def test_prefixes_preserve_duplicates_and_exact_epochs():
    plans = [
        plan(1),
        plan(1),
        plan(2),
        plan(1),
        plan(3),
        plan(2),
        plan(1, np.nextafter(65000.0, np.inf)),
    ]
    assert recovery_order(plans, 3) == [4, 6, 3, 5]
    assert recovery_order(plans, 0) == [0, 2, 4, 6, 1, 3, 5]
    assert recovery_order([], 0) == []
    assert recovery_order(plans, len(plans)) == []
    with pytest.raises(ValueError):
        recovery_order(plans, -1)
    with pytest.raises(ValueError):
        recovery_order([plan(1, np.nan)], 0)


@pytest.mark.skipif(os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="requires CUDA")
@pytest.mark.parametrize("count,initial", [(1, 0), (7, 3), (257, 5), (1025, 17)])
def test_gpu_order_matches_reference_across_blocks(count, initial):
    rng = np.random.default_rng(739)
    plans = [plan(int(i)) for i in rng.integers(0, 101, count)]
    expected = recovery_order(plans, initial)
    with using_lambert_backend("cuda") as gpu:
        for current in (plans, list(reversed(plans)), plans):
            actual = recovery_order(current, initial, gpu)
            assert actual == recovery_order(current, initial)
            assert sorted(actual) == list(range(initial, count))
        assert actual == expected
        assert gpu.telemetry["refinement_recovery_order_calls"] == 3
        assert gpu.telemetry["refinement_recovery_download_bytes"] == 3 * 4 * (count - initial)
