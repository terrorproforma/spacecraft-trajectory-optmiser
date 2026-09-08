"""Generated heterogeneous CUDA rows against the independent scalar seed builder."""

import ctypes as ct
import math
from copy import deepcopy

import numpy as np
import pytest
import test_gtoc12_gpu_joint as oracle

from spacepdhcg.gtoc12.gpu_joint import _decode_evaluation, evaluate_joint
from spacepdhcg.gtoc12.gpu_joint_insertions import _layouts, insertion_batch, insertions
from spacepdhcg.gtoc12.jointopt import JointItinerary

catalogue_and_bonus = oracle.catalogue_and_bonus
incumbent = oracle.incumbent
gpu = oracle.gpu
pytestmark = oracle.requires_gpu


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("slack", [0.0, 800.0])
def test_generated_heterogeneous_rows(monkeypatch, gpu, incumbent, warm, slack):
    joint, visits, arr, dep = incumbent
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_JOINT_RESIDENT_GEOMETRY", "1")
    camp = next((j for j, v in enumerate(visits) if v.deploy and v.collect), None)
    if camp is None:
        assert (
            insertions(joint, visits, arr, dep, [1, 2])
            == joint.insertions(visits, arr, dep, [1, 2])
            == []
        )
        return
    dep = dep.copy()
    dep[camp] = arr[camp] + 365.25 + 5.0 + slack
    present = {v.body for v in visits}
    candidates = [a for a in range(1, 30) if a not in present][:2]
    all_layouts = list(_layouts(visits, candidates, camp))
    # Includes both asteroids, deploy extremes, final Earth-return insertion,
    # different measured-record spans, and a partial CUDA block.
    chosen = [all_layouts[i] for i in sorted({0, 1, len(all_layouts) // 2, len(all_layouts) - 1})]
    seeds = [
        list(JointItinerary._insertion_seeds(visits, arr, dep, camp, i, k, a))
        for _, i, k, a in chosen
    ]
    if warm:
        for group in seeds:
            for expanded, a, d in group:
                oracle._cache(joint, expanded, a, d, [0.0] * (len(expanded) - 1))
    before = deepcopy(joint)
    outputs = insertion_batch(joint, visits, arr, dep, chosen, camp)
    values, enabled, out_a, out_d, mass, inflation, proxy, payload = outputs
    valid_count = 0
    for layout, group in enumerate(seeds):
        active = np.flatnonzero(enabled[4 * layout : 4 * layout + 4]) + 4 * layout
        assert len(active) == len(group)
        for row, (expanded, a, d) in zip(active, group, strict=True):
            valid_count += 1
            assert expanded == chosen[layout][0]
            np.testing.assert_array_equal(out_a[row], a)
            np.testing.assert_array_equal(out_d[row], d)
            expected = evaluate_joint(before, expanded, a[None], d[None])[0]
            actual = _decode_evaluation(
                joint,
                expanded,
                out_a[row],
                out_d[row],
                values[row],
                mass[row],
                inflation[row],
                proxy[row],
                payload[row],
            )
            oracle._same(actual, expected, atol=0, rtol=0)
        for row in set(range(4 * layout, 4 * layout + 4)) - set(active):
            assert values[row]["failure"] == 18
    assert joint.evaluations == before.evaluations
    assert valid_count == sum(len(s) for s in seeds)
    # Workspace reuse must not retain old layout data or sparse-record offsets.
    again = insertion_batch(joint, visits, arr, dep, chosen[::-1], camp)
    for layout_index in range(len(chosen)):
        np.testing.assert_array_equal(
            again[0][4 * layout_index : 4 * layout_index + 4],
            values.reshape(-1, 4)[-1 - layout_index],
        )


def test_full_insertion_ranking_and_no_scalar_generation(monkeypatch, gpu, incumbent):
    joint, visits, arr, dep = incumbent
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_JOINT_RESIDENT_GEOMETRY", "1")
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_INSERTIONS", "0")
    present = {v.body for v in visits}
    asteroid = next(a for a in range(1, 30) if a not in present)
    candidates = [visits[1].body, asteroid, asteroid]
    camp = next((j for j, v in enumerate(visits) if v.deploy and v.collect), None)
    if camp is None:
        assert insertions(joint, visits, arr, dep, candidates) == []
        return
    # Cheap known geometry admits real feasible rows, exercising plans and stable
    # duplicate/tie ordering rather than merely comparing empty result lists.
    for expanded, i, k, a in _layouts(visits, candidates, camp):
        for _, sa, sd in JointItinerary._insertion_seeds(visits, arr, dep, camp, i, k, a):
            oracle._cache(joint, expanded, sa, sd, [0.0] * (len(expanded) - 1))
    expected = joint.insertions(visits, arr, dep, candidates)
    assert expected
    with monkeypatch.context() as guard:

        def forbidden(*args, **kwargs):
            raise AssertionError(
                "native insertion must not call scalar evaluation or seed generation"
            )

        guard.setattr(JointItinerary, "_insertion_seeds", staticmethod(forbidden))
        guard.setattr(JointItinerary, "evaluate", forbidden)
        actual = insertions(joint, visits, arr, dep, candidates, layouts_per_batch=7)
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected, strict=True):
        assert a[0] == e[0] and a[4] == e[4]
        np.testing.assert_array_equal(a[1], e[1])
        np.testing.assert_array_equal(a[2], e[2])
        oracle._same(a[3], e[3], atol=0, rtol=0)
    assert all(math.isfinite(a[3].objective) for a in actual)


def test_native_input_gates_preserve_outputs(monkeypatch, gpu, incumbent):
    joint, visits, arr, dep = incumbent
    camp = next((j for j, v in enumerate(visits) if v.deploy and v.collect), None)
    if camp is None:
        assert insertions(joint, visits, arr, dep, [1]) == []
        return
    layouts = list(_layouts(visits, [1, 2, 3], camp))[:3]
    native = gpu.library.spacepdhcg_gtoc12_joint_insertions_host
    checked = []

    def guarded(*args):
        # Called while all wrapper-owned input/output arrays are still alive.
        status = native(*args)
        assert status == 0
        values = ct.string_at(args[13], len(layouts) * 4 * 64)
        stats = ct.string_at(args[21], 24)
        for field in ("camp", "slots", "offsets", "records", "policy"):
            bad = list(args)
            restore = None
            if field == "camp":
                bad[2] = -1
            elif field == "records":
                bad[11] = -1
            else:
                pointer, index, invalid = {
                    "slots": (args[6], 0, -1),
                    "offsets": (args[12], len(layouts), -1),
                    "policy": (args[3] + 120, 0, 1),
                }[field]
                buf = ct.cast(pointer, ct.POINTER(ct.c_int32))
                restore = buf, index, buf[index]
                buf[index] = invalid
            try:
                assert native(*bad) == (4 if field == "policy" else 1)
                assert ct.string_at(args[13], len(values)) == values
                assert ct.string_at(args[21], 24) == stats
                checked.append(field)
            finally:
                if restore is not None:
                    buf, index, previous = restore
                    buf[index] = previous
        return status

    monkeypatch.setattr(gpu.library, "spacepdhcg_gtoc12_joint_insertions_host", guarded)
    # The guard proxies the pre-bound ABI since argtypes on a Python callable
    # are merely metadata. Configure the actual native symbol explicitly.
    native.argtypes = (
        [ct.c_void_p, ct.c_int32, ct.c_int32]
        + [ct.c_void_p] * 8
        + [ct.c_int32]
        + [ct.c_void_p] * 10
    )
    native.restype = ct.c_int
    insertion_batch(joint, visits, arr, dep, layouts, camp)
    assert len(checked) == 5
