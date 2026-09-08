"""Compact CUDA completion requests with retained catalogue/model snapshots.

This explicit experimental mode derives geometry from the catalogue, not custom
entries injected into CollectPairTable._geometry. It retains CUDA metadata across
new routes. Source arrays are hashed on every batch so changing a catalogue or a
certified grid cannot silently reuse stale prices. That host cost is timed too.
"""

from __future__ import annotations

import ctypes as ct
import hashlib
import math
import os

import numpy as np

from . import constants as C
from .gpu_completion import POLICY, ROLES, _compatible, _layout, counts
from .screening import exhaust_velocity_km_s

ENVIRONMENT = "SPACEPDHCG_TEST_GTOC12_COMPLETION_NATIVE_MODEL"
MODEL = np.dtype(
    [
        (x, np.int32)
        for x in (
            "abi_version",
            "hop_model",
            "table_hop_model",
            "return_model",
            "table_return_model",
            "reserved0",
            "reserved1",
            "reserved2",
        )
    ]
    + [
        (x, np.float64)
        for x in (
            "hop_flat",
            "hop_floor",
            "hop_slope",
            "return_flat",
            "table_return_flat",
            "fit_floor",
        )
    ]
    + [("fit", np.float64, (5,)), ("authority_ratio", np.float64, (5,))]
    + [("epoch0", np.float64), ("step_days", np.float64)],
    align=True,
)
ORBIT_NAMES = (
    "semi_major_axis_km",
    "epoch_mjd",
    "mean_anomaly_rad",
    "ascending_node_rad",
    "argument_of_perihelion_rad",
)
ORBIT = _layout((), ORBIT_NAMES)
GRID = _layout(("body_id", "rows", "cell_begin", "reserved"), ())
CANDIDATE = np.dtype(
    [(x, np.int32) for x in ("deploy_begin", "deploy_count", "leg_begin", "leg_count")]
    + [("partial_mass", np.float64), ("use_table", np.int32), ("reserved", np.int32)],
    align=True,
)
DEPLOY = np.dtype(
    [("deploy_epoch", np.float64), ("collect_epoch", np.float64)]
    + [(x, np.int32) for x in ("has_collect", "body_id", "reserved0", "reserved1")],
    align=True,
)
LEG = _layout(
    ("role", "from_id", "to_id", "candidate"),
    ("departure", "arrival", "dv", "input_inflation"),
)


def enabled():
    value = os.environ.get(ENVIRONMENT, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{ENVIRONMENT} must be 0 or 1")
    return value == "1"


def model_sources(search, requests):
    """Describe and fingerprint immutable native inputs; never compute pair geometry."""
    use_table = any(row[4] for row in requests)
    _compatible(search, use_table)
    table = search.collect_table if use_table else None
    catalogue = table.catalogue if table is not None else search.catalogue
    if catalogue is None:
        raise ValueError("native completion geometry requires a real catalogue")
    arrays = [np.asarray(catalogue.ids)] + [np.asarray(getattr(catalogue, x)) for x in ORBIT_NAMES]
    n = len(arrays[0])
    if not n or n > np.iinfo(np.int32).max or any(a.shape != (n,) for a in arrays):
        raise ValueError("invalid native completion catalogue dimensions")
    if arrays[0].dtype != np.int64 or any(a.dtype != np.float64 for a in arrays[1:]):
        raise ValueError("native completion requires int64 IDs and FP64 catalogue arrays")
    s = search.settings
    p = np.zeros(1, MODEL)
    p["abi_version"] = 1
    p["hop_model"] = int(s.hop_inflation_slope is not None)
    p["table_hop_model"] = p["hop_model"]
    p["hop_flat"], p["hop_floor"] = s.hop_inflation, s.hop_inflation_floor
    p["hop_slope"] = 0.0 if s.hop_inflation_slope is None else s.hop_inflation_slope
    p["return_model"] = 3 if s.earth_return_tof_model else 0
    p["return_flat"] = s.earth_return_inflation
    p["epoch0"], p["step_days"] = 0.0, 1.0
    p["authority_ratio"][0] = [search.limits(role)[1] for role in ROLES]
    grids = []
    tofs = np.asarray([1.0])
    if table is not None:
        p["table_return_model"] = 5 if table.settings.return_tof_model else 0
        p["table_return_flat"] = table.settings.return_inflation
        p["epoch0"], p["step_days"] = table.epochs[0], table.settings.step_days
        tofs = np.asarray(table.return_tofs, dtype=np.float64)
        fit = table.settings.inflation_fit
        if fit is not None:
            p["table_hop_model"], p["fit_floor"], p["fit"] = 2, fit.floor, fit.coefficients
        # Resolve only requested source tables, just as the old packer does. This
        # can invoke the existing sweep-to-grid preparation once; its work is
        # included in packing time and Lambert telemetry, never called geometry-free.
        bodies = sorted(
            {
                leg.from_id
                for row in requests
                if row[4]
                for leg in row[3]
                if leg.role == "earth_return"
            }
        )
        for body in bodies:
            table.return_override(body)
        for body in sorted(table._return_overrides):
            if body not in table.return_sweeps:
                continue
            inflation, ok = table._return_overrides[body]
            inflation, ok = np.asarray(inflation), np.asarray(ok)
            if (
                inflation.ndim != 2
                or inflation.shape[1] != len(tofs)
                or ok.shape != inflation.shape
            ):
                raise ValueError("invalid native completion return-grid shape")
            if inflation.dtype != np.float64 or ok.dtype != np.bool_:
                raise ValueError("native completion requires FP64 return cells and bool masks")
            grids.append((int(body), inflation, ok))
    digest = hashlib.sha256(p.tobytes())
    for a in [*arrays, tofs]:
        digest.update(str((a.dtype.str, a.shape)).encode())
        digest.update(np.ascontiguousarray(a).view(np.uint8))
    for body, values, ok in grids:
        digest.update(str((body, values.shape)).encode())
        digest.update(np.ascontiguousarray(values).view(np.uint8))
        digest.update(np.ascontiguousarray(ok).view(np.uint8))
    return digest.hexdigest(), p, arrays, tofs, grids


def pack_requests(search, requests, buffers):
    """Pack topology and epochs in bulk; CUDA resolves every model and pair."""
    n, nd, nl = counts(requests)
    p, candidates, deploys, legs = (
        a[:size] for a, size in zip(buffers, (1, n, nd, nl), strict=True)
    )
    p[0] = (
        1,
        1,
        0,
        0,
        search.settings.initial_mass,
        C.DRY_MASS_KG,
        C.MINER_MASS_KG,
        C.THRUST_MAX_N,
        exhaust_velocity_km_s(),
        C.MINING_RATE_KG_PER_YEAR,
        C.YEAR_DAYS,
        C.MIN_MINING_STAY_YEARS * C.YEAR_DAYS,
    )
    cs, ds, ls = [], [], []
    for index, (partial, deploy, collect, forward, use_table) in enumerate(requests):
        epochs = [*deploy.values(), *collect.values()]
        epochs.extend(t for leg in forward for t in (leg.departure_epoch, leg.arrival_epoch))
        if any(type(t) is not float or not math.isfinite(t) for t in epochs):
            raise ValueError("CUDA completion metadata requires finite exact Python float epochs")
        cs.append((len(ds), len(deploy), len(ls), len(forward), partial.mass, int(use_table), 0))
        ds.extend(
            (epoch, collect.get(body, 0.0), int(body in collect), body, 0, 0)
            for body, epoch in deploy.items()
        )
        for leg in forward:
            if leg.role not in ROLES:
                raise ValueError(f"CUDA completion does not support role {leg.role!r}")
            if leg.role in ("collect_hop", "earth_return") and leg.from_id not in deploy:
                raise ValueError(
                    "CUDA completion requires each collection source in the deploy dictionary"
                )
            for body in (leg.from_id, leg.to_id):
                if (
                    not isinstance(body, (int, np.integer))
                    or not 0 <= body <= np.iinfo(np.int32).max
                ):
                    raise ValueError("native completion body IDs must fit nonnegative int32")
            ls.append(
                (
                    ROLES[leg.role],
                    leg.from_id,
                    leg.to_id,
                    index,
                    leg.departure_epoch,
                    leg.arrival_epoch,
                    leg.delta_v_proxy_km_s,
                    leg.inflation,
                )
            )
    candidates[:] = cs
    deploys[:] = ds
    legs[:] = ls
    return p, candidates, deploys, legs


class NativeModelOwner:
    def __init__(self, completion):
        self.owner = completion
        self.handle = ct.c_void_p()
        self.signature = None
        library = completion.gpu.library
        try:
            self.create = library.spacepdhcg_gtoc12_completion_model_create
            self.destroy = library.spacepdhcg_gtoc12_completion_model_destroy
            self.evaluate = library.spacepdhcg_gtoc12_completion_evaluate_compact_host
        except AttributeError as error:
            raise RuntimeError(
                "native completion model requires the compact completion ABI"
            ) from error
        self.create.argtypes = [
            ct.c_int32,
            ct.c_void_p,
            ct.c_int32,
            ct.c_void_p,
            ct.c_int32,
            ct.c_void_p,
            ct.c_int32,
            ct.c_void_p,
            ct.c_int32,
            ct.c_void_p,
            ct.c_void_p,
            ct.POINTER(ct.c_void_p),
        ]
        self.destroy.argtypes = [ct.POINTER(ct.c_void_p)]
        self.evaluate.argtypes = [ct.c_void_p, ct.c_void_p] + [ct.c_int32] * 3 + [ct.c_void_p] * 9
        for f in (self.create, self.destroy, self.evaluate):
            f.restype = ct.c_int
        n, nd, nl = completion.capacities
        self.inputs = (
            np.zeros(1, POLICY),
            np.zeros(n, CANDIDATE),
            np.zeros(nd, DEPLOY),
            np.zeros(nl, LEG),
        )
        self.rebuilds = 0

    def configure(self, search, requests):
        key, policy, arrays, tofs, grids = model_sources(search, requests)
        if key == self.signature:
            return
        ids, *elements = arrays
        if not np.array_equal(ids, np.arange(1, len(ids) + 1)):
            raise ValueError("native completion catalogue IDs must be in identity order")
        orbits = np.empty(len(ids), ORBIT)
        for name, values in zip(ORBIT_NAMES, elements, strict=True):
            orbits[name] = values
        records = np.zeros(len(grids), GRID)
        values, masks, offset = [], [], 0
        for i, (body, inflation, ok) in enumerate(grids):
            if offset + inflation.size > np.iinfo(np.int32).max:
                raise ValueError("native completion return grids exceed int32")
            records[i] = (body, inflation.shape[0], offset, 0)
            values.append(inflation.ravel())
            masks.append(ok.ravel())
            offset += inflation.size
        cell_values = np.concatenate(values) if values else np.empty(0, np.float64)
        cell_ok = np.concatenate(masks).astype(np.uint8) if masks else np.empty(0, np.uint8)
        tofs = np.ascontiguousarray(tofs)
        replacement = ct.c_void_p()
        self.owner._check(
            self.create(
                self.owner.gpu.device_id,
                policy.ctypes.data,
                len(orbits),
                orbits.ctypes.data,
                len(tofs),
                tofs.ctypes.data,
                len(records),
                records.ctypes.data,
                len(cell_values),
                cell_values.ctypes.data,
                cell_ok.ctypes.data,
                ct.byref(replacement),
            )
        )
        try:
            self.close()
        except BaseException:
            self.destroy(ct.byref(replacement))
            raise
        self.handle, self.signature = replacement, key
        self.rebuilds += 1

    def close(self):
        if self.handle.value:
            self.owner.gpu._owned()
            self.owner._check(self.destroy(ct.byref(self.handle)))
        self.signature = None

    def run(self, inputs, outputs, *, expanded=None):
        _, candidates, deploys, legs = inputs
        self.owner._check(
            self.evaluate(
                self.owner.handle,
                self.handle,
                len(candidates),
                len(deploys),
                len(legs),
                *(a.ctypes.data for a in (*inputs, *outputs)),
                None if expanded is None else expanded.ctypes.data,
            )
        )
