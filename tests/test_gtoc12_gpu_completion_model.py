"""Catalogue-derived CUDA metadata: compare full numeric inputs and all forward gates."""

from __future__ import annotations

import ctypes as ct
import dataclasses
import hashlib
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_gtoc12_completion_costs import fixture
from test_gtoc12_gpu_completion import request

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.completion_capture import CompletionRecorder
from spacepdhcg.gtoc12.gpu_completion import (
    LEG as EXPANDED_LEG,
)
from spacepdhcg.gtoc12.gpu_completion import (
    LEG_RESULT,
    POLICY,
    RESULT,
    STATS,
    GpuCompletion,
    pack_inputs,
)
from spacepdhcg.gtoc12.gpu_completion_model import (
    CANDIDATE,
    DEPLOY,
    GRID,
    LEG,
    MODEL,
    ORBIT,
    ORBIT_NAMES,
    CatalogueFingerprint,
    NativeModelOwner,
    model_sources,
    pack_requests,
)


def real_geometry_fixture(*, fit=True, swept=True):
    search, partial, tour = fixture(fit=fit, swept=swept)
    size = 12
    cat = SimpleNamespace(
        ids=np.arange(1, size + 1, dtype=np.int64),
        semi_major_axis_km=np.linspace(2.5, 2.6, size) * C.AU_KM,
        epoch_mjd=np.full(size, 60000.0),
        mean_anomaly_rad=np.linspace(-20.0, 20.0, size),
        ascending_node_rad=np.linspace(0.0, 2.0, size),
        argument_of_perihelion_rad=np.linspace(-2.0, 0.0, size),
    )
    search.catalogue = search.collect_table.catalogue = cat
    search.collect_table._geometry.clear()
    return search, partial, tour


def buffers(n=300, nd=1000, nl=2000):
    return np.zeros(1, POLICY), np.zeros(n, CANDIDATE), np.zeros(nd, DEPLOY), np.zeros(nl, LEG)


def test_compact_abi_and_bulk_topology():
    assert [a.itemsize for a in (MODEL, ORBIT, GRID, CANDIDATE, DEPLOY, LEG)] == [
        176,
        40,
        16,
        32,
        32,
        48,
    ]
    search, partial, tour = real_geometry_fixture()
    rows = [request(search, partial, tour), request(search, partial, tour, False)]
    _, cs, ds, ls = pack_requests(search, rows, buffers())
    assert cs["use_table"].tolist() == [1, 0]
    assert ds["body_id"].tolist() == [11, 12, 11, 12]
    assert ls["candidate"].tolist() == [0] * len(rows[0][3]) + [1] * len(rows[1][3])
    assert np.array_equal(ls["dv"], pack_inputs(search, rows)[3]["dv"])
    empty = pack_requests(search, [], buffers())
    assert [len(a) for a in empty] == [1, 0, 0, 0]


def test_compact_signature_tracks_live_inputs_without_pair_geometry(monkeypatch):
    search, partial, tour = real_geometry_fixture()
    rows = [request(search, partial, tour)]
    first = model_sources(search, rows)[0]
    assert model_sources(search, rows)[0] == first
    search.catalogue.mean_anomaly_rad[0] += 0.01
    changed = model_sources(search, rows)[0]
    assert changed != first
    search.collect_table._return_overrides[11][0][0, 0] += 0.01
    assert model_sources(search, rows)[0] != changed
    assert not search.collect_table._geometry
    assert search.collect_table.lambert_evaluations == 0


def test_compact_rejects_synthetic_geometry_and_invalid_epoch():
    search, partial, tour = fixture()
    rows = [request(search, partial, tour)]
    with pytest.raises(ValueError, match="real catalogue"):
        model_sources(search, rows)
    search, partial, tour = real_geometry_fixture()
    rows = [request(search, partial, tour)]
    rows[0][1][11] = np.float64(rows[0][1][11])
    with pytest.raises(ValueError, match="finite exact Python float epochs"):
        pack_requests(search, rows, buffers())


def test_cached_return_row_overflow_fixture_reaches_metadata_validation():
    search, partial, tour = real_geometry_fixture()
    row = request(search, partial, tour)
    search.collect_table.settings = dataclasses.replace(
        search.collect_table.settings, step_days=0.5
    )
    row[3][-1] = dataclasses.replace(row[3][-1], departure_epoch=float(np.finfo(float).max))
    pack_requests(search, [row], buffers())
    model_sources(search, [row])
    with np.errstate(over="ignore"), pytest.raises(OverflowError):
        pack_inputs(search, [row])


def freeze_catalogue(search):
    for name in ("ids", *ORBIT_NAMES):
        value = getattr(search.catalogue, name)
        setattr(search.catalogue, name, np.frombuffer(value.tobytes(), dtype=value.dtype))


@pytest.mark.parametrize("immutable", [False, True])
def test_cached_catalogue_prefix_keeps_exact_signature_and_tracks_changes(immutable):
    search, partial, tour = real_geometry_fixture()
    if immutable:
        freeze_catalogue(search)
    rows = [request(search, partial, tour)]
    cache = CatalogueFingerprint()
    first = model_sources(search, rows, cache)[0]
    assert first == model_sources(search, rows)[0] == model_sources(search, rows, cache)[0]
    assert (cache.hashes, cache.reuses) == ((1, 1) if immutable else (2, 0))
    # Mutable grids remain live even when the large catalogue prefix is cached.
    search.collect_table._return_overrides[11][0][0, 0] += 0.01
    grid_changed = model_sources(search, rows, cache)[0]
    assert grid_changed != first and grid_changed == model_sources(search, rows)[0]
    changed = search.catalogue.mean_anomaly_rad.copy()
    changed[0] += 0.01
    search.catalogue.mean_anomaly_rad = changed
    catalogue_changed = model_sources(search, rows, cache)[0]
    assert catalogue_changed != grid_changed
    changed[0] += 0.01
    assert model_sources(search, rows, cache)[0] != catalogue_changed


def test_readonly_view_cannot_mask_mutation_of_writable_backing_storage():
    search, partial, tour = real_geometry_fixture()
    parent = search.catalogue.mean_anomaly_rad
    view = parent.view()
    view.setflags(write=False)
    search.catalogue.mean_anomaly_rad = view
    cache = CatalogueFingerprint()
    rows = [request(search, partial, tour)]
    first = model_sources(search, rows, cache)[0]
    parent[0] += 0.01
    assert model_sources(search, rows, cache)[0] != first
    assert cache.reuses == 0


def test_immutable_replacement_and_policy_changes_refresh_prefix(monkeypatch):
    search, partial, tour = real_geometry_fixture()
    freeze_catalogue(search)
    rows = [request(search, partial, tour)]
    cache = CatalogueFingerprint()
    first = model_sources(search, rows, cache)[0]
    freeze_catalogue(search)
    assert model_sources(search, rows, cache)[0] == first
    assert cache.hashes == 2
    search.settings = dataclasses.replace(search.settings, hop_inflation=1.7)
    assert model_sources(search, rows, cache)[0] != first
    assert cache.hashes == 3
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_CACHE_CATALOGUE_DIGEST", "0")
    assert model_sources(search, rows, cache)[0] == model_sources(search, rows)[0]
    assert cache.hashes == 4 and cache.reuses == 0


def test_alternating_policies_reuse_catalogue_and_bound_retained_state():
    search, partial, tour = real_geometry_fixture()
    freeze_catalogue(search)
    cached = CatalogueFingerprint()
    table, plain = [request(search, partial, tour)], [request(search, partial, tour, False)]
    expected = [model_sources(search, rows)[0] for rows in (table, plain)]
    for _ in range(5):
        for rows, signature in zip((table, plain), expected, strict=True):
            assert model_sources(search, rows, cached)[0] == signature
    assert cached.hashes == 2 and cached.reuses == 8
    for inflation in np.linspace(1.0, 2.0, 20):
        search.settings = dataclasses.replace(search.settings, hop_inflation=float(inflation))
        assert model_sources(search, table, cached)[0] == model_sources(search, table)[0]
        assert len(cached.prefixes) <= 4


requires_gpu = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1",
    reason="explicit serialized CUDA test only",
)


class StandaloneOwner:
    def __init__(self):
        self.library = ct.CDLL(os.environ["SPACEPDHCG_COMPLETION_TEST_LIBRARY"])
        self.device_id = 0
        self.telemetry = {}

    def _owned(self):
        pass


def assert_records_close(actual, expected, *, atol=2e-11):
    for field in actual.dtype.names:
        if actual.dtype[field].kind in "iu":
            np.testing.assert_array_equal(actual[field], expected[field])
        else:
            np.testing.assert_allclose(
                actual[field], expected[field], rtol=2e-14, atol=atol, equal_nan=True
            )


def save_comparison(model, size, compact, full, actual, expected, expanded):
    destination = os.environ.get("SPACEPDHCG_COMPLETION_TEST_READBACKS")
    if destination is None:
        return
    target = Path(destination) / f"{model}-{size}.npz"
    arrays = {"expanded": expanded}
    for prefix, values in (
        ("compact", compact),
        ("full", full),
        ("actual", actual),
        ("expected", expected),
    ):
        arrays.update({f"{prefix}_{i}": a for i, a in enumerate(values)})
    with target.open("xb") as stream:
        np.savez(stream, **arrays)


@requires_gpu
@pytest.mark.parametrize("model", ["fit", "flat", "ratio"])
@pytest.mark.parametrize("immutable", [False, True])
def test_compact_native_metadata_gates_reuse_and_invalid_batch(model, immutable, monkeypatch):
    search, partial, tour = real_geometry_fixture(fit=model == "fit")
    if immutable:
        freeze_catalogue(search)
    if model == "ratio":
        search.settings = dataclasses.replace(search.settings, hop_inflation_slope=0.3)
    rows = []
    for i in range(259):
        p, t = deepcopy(partial), deepcopy(tour)
        p.mass = 500.0 + (i % 100) * 10.0
        # New epochs and payloads in every group, including tie-to-even row
        # boundaries and a nearest-TOF tie. No request identity cache is involved.
        shift = float((i % 13) * 7.5)
        t.hops = [(a, b, dep + shift, tof, dv) for a, b, dep, tof, dv in t.hops]
        t.collect_epochs = {a: epoch + shift for a, epoch in t.collect_epochs.items()}
        t.return_departure += shift
        t.return_tof = 465.0 if i % 2 else 450.0
        row = request(search, p, t, i % 3 != 0)
        if i % 17 == 0:
            row[2].pop(11)
        elif i % 19 == 0:
            row[2][11] = row[1][11]
        elif i % 23 == 0:
            row[3][-1] = dataclasses.replace(row[3][-1], delta_v_proxy_km_s=100.0)
        rows.append(row)
    # Distinguish adjacent cells; a uniform override cannot detect bad rounding.
    grid, ok = search.collect_table._return_overrides[11]
    grid[:] = 0.8 + np.arange(grid.size).reshape(grid.shape) * 1e-5
    ok[::3, ::2] = False
    owner = GpuCompletion(StandaloneOwner(), 300, 1000, 2000)
    native = NativeModelOwner(owner)
    owner.native_model = native
    try:
        for selected in (rows, rows[:3]):
            native.configure(search, selected)
            compact = pack_requests(search, selected, native.inputs)
            full = pack_inputs(search, selected)
            n, nd, nl = len(selected), len(compact[2]), len(compact[3])
            expected = (
                np.zeros(n, RESULT),
                np.zeros(nl, LEG_RESULT),
                np.zeros(nd),
                np.zeros(1, STATS),
            )
            owner._check(
                owner.evaluate(
                    owner.handle, n, nd, nl, *(a.ctypes.data for a in (*full, *expected))
                )
            )
            actual = tuple(np.zeros_like(a) for a in expected)
            expanded = np.zeros(nl, EXPANDED_LEG)
            native.run(compact, actual, expanded=expanded)
            save_comparison(model, n, compact, full, actual, expected, expanded)
            assert_records_close(expanded, full[3], atol=1e-12)
            assert_records_close(actual[0], expected[0])
            assert_records_close(actual[1], expected[1])
            np.testing.assert_array_equal(actual[2], expected[2])
        assert native.rebuilds == 1
        assert native.catalogue_fingerprint.reuses == int(immutable)
        # An updated grid must create a replacement model and release its old one.
        search.collect_table._return_overrides[11][0][0, 0] += 0.01
        native.configure(search, rows[:3])
        assert native.rebuilds == 2
        malformed = [a.copy() for a in compact]
        malformed[3]["candidate"][-1] = 99
        untouched = tuple(a.copy() for a in actual)
        with pytest.raises(RuntimeError, match="native status 1"):
            native.run(malformed, actual)
        for a, b in zip(actual, untouched, strict=True):
            assert a.tobytes() == b.tobytes()
        # Finite epochs with an overflowing row quotient must reject before
        # touching outputs, rather than falling back to generic return prices.
        huge = [a.copy() for a in compact]
        huge[3]["departure"][-1] = np.finfo(float).max
        old_settings = search.collect_table.settings
        search.collect_table.settings = dataclasses.replace(old_settings, step_days=0.5)
        native.configure(search, rows[:3])
        with pytest.raises(RuntimeError, match="native status 1"):
            native.run(huge, actual)
        for a, b in zip(actual, untouched, strict=True):
            assert a.tobytes() == b.tobytes()
        search.collect_table.settings = old_settings
        # Exercise production readout: failed proxy routes reach the recorder
        # before None is returned. All retained readbacks survive workspace reuse.
        captured = []

        def consume(envelope):
            captured.append(envelope)
            return "captured_for_refinement"

        producer = hashlib.sha256(
            open(os.environ["SPACEPDHCG_COMPLETION_TEST_LIBRARY"], "rb").read()
        ).hexdigest()
        search.completion_capture = CompletionRecorder(producer, consume)
        monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_COMPLETION_NATIVE_MODEL", "1")
        returned = owner.run(search, rows)
        assert any(
            plan is None and reason == "mass_below_dry_plus_collected" for plan, reason in returned
        )
        assert any(e.observation.failure == "mass_below_dry_plus_collected" for e in captured)
        assert all(not e.observation.blocker(e.request) for e in captured)
        preserved = [e.raw_result + e.raw_legs + e.raw_collected + e.plan_summary for e in captured]
        first_count = len(captured)
        owner.run(search, rows[:3])
        assert preserved == [
            e.raw_result + e.raw_legs + e.raw_collected + e.plan_summary
            for e in captured[:first_count]
        ]
        assert search.collect_table.lambert_evaluations == 0
    finally:
        owner.close()
        owner.close()
