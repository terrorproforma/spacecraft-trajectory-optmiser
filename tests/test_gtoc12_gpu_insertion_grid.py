"""Unequal CUDA split epochs, legacy midpoint parity and independent row evaluation."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import test_gtoc12_gpu_joint as oracle

from spacepdhcg.gtoc12.gpu_joint import GEOMETRY_STATS, RESULT, _decode_evaluation, evaluate_joint
from spacepdhcg.gtoc12.gpu_joint_insertions import _layouts, insertion_batch
from spacepdhcg.gtoc12.gpu_joint_layouts import PreparedInsertions
from spacepdhcg.gtoc12.jointopt import insertion_pivot

catalogue_and_bonus = oracle.catalogue_and_bonus
incumbent = oracle.incumbent
gpu = oracle.gpu
pytestmark = oracle.requires_gpu


@pytest.mark.parametrize("points", [3, 5, 9])
@pytest.mark.parametrize("warm", [False, True])
def test_grid_epochs_and_all_enabled_results(gpu, incumbent, points, warm):
    joint, visits, arr, dep = incumbent
    candidates = [a for a in range(1, 30) if a not in {v.body for v in visits}][:2]
    pivot = insertion_pivot(visits)
    layouts = list(_layouts(visits, candidates, pivot))
    prepared = PreparedInsertions(joint, visits, arr, dep, candidates, 2, split_points=points)
    for first in (0, len(layouts) - 1):
        values, enabled, aa, dd, mm, ii, pp, cc, *_ = prepared.run(first, 1)
        expanded, i, k, _ = layouts[first]
        legacy = insertion_batch(deepcopy(joint), visits, arr, dep, [layouts[first]], pivot)
        center = 4 * ((points // 2) * points + points // 2)
        for index, actual in enumerate((values, enabled, aa, dd)):
            np.testing.assert_array_equal(actual[center : center + 4], legacy[index])
        for row in np.flatnonzero(enabled):
            seed = row % 4
            gd, gc = divmod(row // 4, points)
            for j, f in ((i + 1, (gd + 1) / (points + 1)), (k + 2, (gc + 1) / (points + 1))):
                assert aa[row, j] == dd[row, j]
                interval = aa[row, j + 1] - dd[row, j - 1]
                # Independent geometry of the split interval, rather than the
                # CUDA generator's midpoint-offset arithmetic.
                assert aa[row, j] == pytest.approx(dd[row, j - 1] + f * interval, abs=2e-11, rel=0)
            keep = [j for j in range(len(expanded)) if j not in (i + 1, k + 2)]
            np.testing.assert_array_equal(aa[row, keep], legacy[2][seed, keep])
            np.testing.assert_array_equal(dd[row, keep], legacy[3][seed, keep])
            if warm:
                oracle._cache(joint, expanded, aa[row], dd[row], [0.0] * (len(expanded) - 1))
        if warm:
            # Recompile the now populated cache and repeat exactly the same grid.
            prepared = PreparedInsertions(
                joint, visits, arr, dep, candidates, 2, split_points=points
            )
            values, enabled, aa, dd, mm, ii, pp, cc, *_ = prepared.run(first, 1)
        active = np.flatnonzero(enabled)
        expected = evaluate_joint(deepcopy(joint), expanded, aa[active], dd[active])
        for row, ev in zip(active, expected, strict=True):
            actual = _decode_evaluation(
                joint, expanded, aa[row], dd[row], values[row], mm[row], ii[row], pp[row], cc[row]
            )
            oracle._same(actual, ev, atol=0, rtol=0)
        assert np.all(values["failure"][enabled == 0] == 18)


def test_grid_native_invalid_inputs_preserve_outputs(gpu, incumbent):
    joint, visits, arr, dep = incumbent
    p = PreparedInsertions(joint, visits, arr, dep, [1, 2], 2, split_points=3)
    result = np.zeros(1, RESULT)
    result["failure"] = 71
    enabled = np.array([7], np.uint8)
    stats = np.zeros(1, GEOMETRY_STATS)
    stats["computed_hops"] = 99
    raw = gpu.library.spacepdhcg_gtoc12_joint_prepared_insertion_grid_host
    for first, count, points in (
        (0, 1, 0),
        (0, 1, 2),
        (0, 1, 11),
        (-1, 1, 3),
        (p.total, 1, 3),
        (0, p.native.capacity // 36 + 1, 3),
    ):
        assert (
            raw(
                p.native.handle,
                first,
                count,
                points,
                result.ctypes.data,
                enabled.ctypes.data,
                None,
                None,
                None,
                None,
                None,
                None,
                stats.ctypes.data,
                None,
                None,
            )
            == 1
        )
        assert result["failure"][0] == 71 and enabled[0] == 7 and stats["computed_hops"][0] == 99
    for points in (0, 2, 11, True, 3.0):
        with pytest.raises(ValueError, match="split_points"):
            PreparedInsertions(joint, visits, arr, dep, [1], split_points=points)


def test_joint_settings_select_grid_and_preserve_survivor_ranking(monkeypatch, gpu, incumbent):
    from spacepdhcg.gtoc12.gpu_joint_layouts import insertions

    joint, visits, arr, dep = incumbent
    candidates = [a for a in range(1, 30) if a not in {v.body for v in visits}][:1]
    p = PreparedInsertions(joint, visits, arr, dep, candidates, 5, split_points=3)
    for first in range(0, p.total, 5):
        output = p.run(first, min(5, p.total - first))
        for row in np.flatnonzero(output[1]):
            expanded, _ = p.layout(first + row // p.rows_per_layout)
            oracle._cache(
                joint, expanded, output[2][row], output[3][row], [0.0] * (len(expanded) - 1)
            )
    expected = insertions(
        deepcopy(joint), visits, arr, dep, candidates, layouts_per_batch=7, split_points=3
    )
    assert expected
    joint.settings = replace(joint.settings, insert_split_points=3)
    actual = joint.insertions(visits, arr, dep, candidates)
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected, strict=True):
        assert a[0] == e[0] and a[4] == e[4]
        np.testing.assert_array_equal(a[1], e[1])
        np.testing.assert_array_equal(a[2], e[2])
        oracle._same(a[3], e[3], atol=0, rtol=0)
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_LAYOUTS", "0")
    with pytest.raises(RuntimeError, match="device layouts"):
        joint.insertions(visits, arr, dep, candidates)
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_JOINT_BATCH", "0")
    with pytest.raises(RuntimeError, match="CUDA joint backend"):
        joint.insertions(visits, arr, dep, candidates)
