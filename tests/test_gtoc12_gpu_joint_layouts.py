"""GPU-built layouts against the established host metadata and insertion pipeline."""

import ctypes as ct
from copy import deepcopy

import numpy as np
import pytest
import test_gtoc12_gpu_joint as oracle

from spacepdhcg.gtoc12.gpu_joint import GEOMETRY_STATS, RESULT, _metadata
from spacepdhcg.gtoc12.gpu_joint_insertions import _layouts, insertion_batch
from spacepdhcg.gtoc12.gpu_joint_layouts import PreparedInsertions, insertions
from spacepdhcg.gtoc12.jointopt import JointItinerary, insertion_pivot

catalogue_and_bonus = oracle.catalogue_and_bonus
incumbent = oracle.incumbent
gpu = oracle.gpu
pytestmark = oracle.requires_gpu


@pytest.mark.parametrize("warm", [False, True])
def test_device_layout_metadata_and_every_candidate(monkeypatch, gpu, incumbent, warm):
    joint, visits, arr, dep = incumbent
    present = {v.body for v in visits}
    candidates = [a for a in range(1, 30) if a not in present][:2]
    camp = insertion_pivot(visits)
    assert camp is not None and camp >= 2
    layouts = list(_layouts(visits, candidates, camp))
    if warm:
        for expanded, i, k, a in layouts:
            for _, sa, sd in JointItinerary._insertion_seeds(visits, arr, dep, camp, i, k, a):
                oracle._cache(joint, expanded, sa, sd, [0.0] * (len(expanded) - 1))
    prepared = PreparedInsertions(joint, visits, arr, dep, candidates, layouts_per_batch=13)
    expected_joint = deepcopy(joint)
    for first in range(0, len(layouts), 13):
        block = layouts[first : first + 13]
        actual = prepared.run(first, len(block), inspect=True)
        # The old operator may reuse the workspace, but it must not overwrite
        # the immutable compact source used by the following prepared batch.
        expected = insertion_batch(expected_joint, visits, arr, dep, block, camp)
        value, enabled, *_ = actual
        np.testing.assert_array_equal(enabled, expected[1])
        np.testing.assert_array_equal(value, expected[0])
        for index, (expanded, _, _, _) in enumerate(block):
            metadata, stages, _ = _metadata(joint, expanded)
            for field in metadata.dtype.names:
                np.testing.assert_array_equal(actual[8][index][field], metadata[field])
            np.testing.assert_array_equal(prepared.source["stages"][actual[9][index]], stages)
            assert prepared.layout(first + index)[0] == expanded
        # Every enabled epoch and valid detail row must agree exactly. Inactive
        # rows intentionally have no numerical-detail contract.
        active = np.flatnonzero(enabled)
        np.testing.assert_array_equal(actual[2][active], expected[2][active])
        np.testing.assert_array_equal(actual[3][active], expected[3][active])
        detail = np.flatnonzero(enabled & np.isin(value["failure"], [0, 15]))
        for k in (4, 5, 6, 7):
            np.testing.assert_array_equal(actual[k][detail], expected[k][detail])


def test_layout_generation_does_not_enumerate_host_routes(monkeypatch, gpu, incumbent):
    joint, visits, arr, dep = incumbent
    camp = insertion_pivot(visits)
    assert camp is not None and camp >= 2
    candidates = [a for a in range(1, 30) if a not in {v.body for v in visits}][:2]
    import spacepdhcg.gtoc12.gpu_joint_insertions as old

    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_LAYOUTS", "0")
    expected = old.insertions(deepcopy(joint), visits, arr, dep, candidates)
    with monkeypatch.context() as guard:

        def forbidden(*a, **kw):
            raise AssertionError("host layout enumeration is forbidden")

        guard.setattr(old, "_layouts", forbidden)
        guard.setattr(JointItinerary, "_insertion_seeds", staticmethod(forbidden))
        actual = insertions(joint, visits, arr, dep, candidates)
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected, strict=True):
        assert a[0] == e[0] and a[4] == e[4]
        np.testing.assert_array_equal(a[1], e[1])
        np.testing.assert_array_equal(a[2], e[2])
        oracle._same(a[3], e[3], atol=0, rtol=0)


def test_prepared_source_snapshot_replacement_and_native_gates(gpu, incumbent):
    joint, visits, arr, dep = incumbent
    original_visits = list(visits)
    arr = arr.copy()
    dep = dep.copy()
    candidates = [a for a in range(1, 30) if a not in {v.body for v in visits}][:2]
    prepared = PreparedInsertions(joint, visits, arr, dep, candidates, layouts_per_batch=13)
    if not prepared.total:
        with pytest.raises(ValueError, match="empty"):
            prepared.run(0, 1)
        return
    original = prepared.run(0, 2)
    original_layout = prepared.layout(0)
    arr[:] += 100
    dep[:] += 100
    visits[1] = visits[0]
    candidates.reverse()
    after = prepared.run(0, 2)
    for k in (0, 1, 2, 3):
        np.testing.assert_array_equal(after[k], original[k])
    assert prepared.layout(0) == original_layout
    source = prepared.source
    before = source["weights"][0]
    source["weights"][0] = np.nan
    try:
        assert (
            prepared.prepare(
                prepared.native.handle,
                len(source["candidates"]),
                source["camp"],
                *(source[k].ctypes.data for k in ("policy", "metadata", "arr", "dep", "weights")),
                source["dwell"],
                source["stages"].ctypes.data,
                ct.addressof(source["elements"]),
                source["records"].ctypes.data,
                len(source["records"]),
            )
            == 1
        )
    finally:
        source["weights"][0] = before
    np.testing.assert_array_equal(prepared.run(0, 2)[0], original[0])
    result = np.zeros(4, RESULT)
    result["failure"] = 71
    enabled = np.full(4, 7, np.uint8)
    stats = np.zeros(1, GEOMETRY_STATS)
    stats["computed_hops"] = 99
    for first, count in ((-1, 1), (prepared.total, 1), (0, prepared.native.capacity // 4 + 1)):
        assert (
            prepared.evaluate(
                prepared.native.handle,
                first,
                count,
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
        assert (
            np.all(result["failure"] == 71)
            and np.all(enabled == 7)
            and stats["computed_hops"][0] == 99
        )
    # A second Python owner must not silently reuse the first owner's source.
    newer = PreparedInsertions(joint, original_visits, arr, dep, candidates, layouts_per_batch=13)
    assert newer.native is prepared.native
    with pytest.raises(RuntimeError, match="replaced"):
        prepared.run(0, 1)
