"""Retained CUDA dynamic programme for a fixed-order GTOC12 timing lattice."""

import ctypes as ct
from itertools import pairwise

import numpy as np

from . import constants as C
from .screening import exhaust_velocity_km_s

STAGE = np.dtype(
    [
        (name, "i4")
        for name in [
            "cell_offset",
            "tof_offset",
            "tofs",
            "camp_min",
            "camp_max",
            "pinned_next",
            "model",
            "reserved",
        ]
    ]
    + [
        (name, "f8")
        for name in [
            "arrival_rate",
            "departure_rate",
            "earliest_collect",
            "mass",
            "ratio_limit",
            "flat",
            "floor",
            "slope",
            "calibration",
        ]
    ],
    align=True,
)

SWEEP_CELL = np.dtype(
    [
        ("departure", "i4"),
        ("tof", "i4"),
        ("delta_v", "f8"),
        ("certified", "i4"),
        ("reserved", "i4"),
    ],
    align=True,
)


class GpuRetime:
    """One cached immutable set of tables; masses, prices and policies update per call."""

    def __init__(self, library, device, lambert):
        self.device = device
        self.lambert = lambert
        self.last_path = None
        self.handle = ct.c_void_p()
        self.key = None
        self.refs = None
        self.create = library.spacepdhcg_gtoc12_retime_create
        self.create.argtypes = [ct.c_int32] * 5 + [ct.c_void_p] * 7 + [ct.POINTER(ct.c_void_p)]
        self.create.restype = ct.c_int
        self.create_elements = library.spacepdhcg_gtoc12_retime_create_elements
        self.create_elements.argtypes = (
            [ct.c_int32] * 5 + [ct.c_void_p] * 6 + [ct.POINTER(ct.c_void_p)]
        )
        self.create_elements.restype = ct.c_int
        self.set_sweep = library.spacepdhcg_gtoc12_retime_set_sweep
        self.set_sweep.argtypes = [
            ct.c_void_p,
            ct.c_int32,
            ct.c_int32,
            ct.c_int32,
            ct.c_void_p,
            ct.c_int32,
        ]
        self.set_sweep.restype = ct.c_int
        self.read_sweep = library.spacepdhcg_gtoc12_retime_read_sweep
        self.read_sweep.argtypes = [ct.c_void_p, ct.c_int32, ct.c_int32, ct.c_void_p, ct.c_void_p]
        self.read_sweep.restype = ct.c_int
        self.evaluate = library.spacepdhcg_gtoc12_retime_swept_path_host
        self.evaluate.argtypes = [
            ct.c_void_p,
            ct.c_void_p,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.c_void_p,
            ct.c_void_p,
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_int32),
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
        ]
        self.evaluate.restype = ct.c_int
        self.set_graph = library.spacepdhcg_gtoc12_retime_set_graph
        self.set_graph.argtypes = [ct.c_void_p, ct.c_int32]
        self.set_graph.restype = ct.c_int
        self.graph_stats = library.spacepdhcg_gtoc12_retime_graph_stats
        self.graph_stats.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint64), ct.POINTER(ct.c_uint64)]
        self.graph_stats.restype = ct.c_int
        self.graph_enabled = True
        self.destroy = library.spacepdhcg_gtoc12_retime_destroy
        self.destroy.argtypes = [ct.POINTER(ct.c_void_p)]
        self.destroy.restype = ct.c_int
        self.calls = self.uploads = 0
        self.resident_builds = self.resident_cells = 0
        self.sweep_updates = self.sweep_samples = 0
        self.sweep_keys = {}

    @staticmethod
    def _check(code):
        if code:
            raise RuntimeError(f"CUDA retiming failed (native status {code})")

    def close(self):
        if self.handle.value:
            self._check(self.destroy(ct.byref(self.handle)))
        self.key = self.refs = None
        self.last_path = None
        self.sweep_keys.clear()
        self.graph_enabled = True

    def path_values(self, retimer, visits, arrivals, departures, include_sweep=False):
        if self.last_path is None:
            return None
        owner, revision, order, a, d, values = self.last_path
        if (
            owner is not retimer
            or revision != (retimer._cache_revision, retimer._sweep_revision)
            or order != visits
            or a != arrivals
            or d != departures
        ):
            return None
        return values if include_sweep else values[0]

    def solve(self, retimer, visits, masses, price):
        from .retiming import Retimer

        self.last_path = None
        if len(visits) == 1:
            return [0], [0], 0.0
        s, lat = retimer.settings, retimer.lattice
        rate = C.MINING_RATE_KG_PER_YEAR / C.YEAR_DAYS
        collected = {v.body for v in visits if v.collect}
        params = np.zeros(len(visits) - 1, dtype=STAGE)
        tables, tofs_all, keys, refs = [], [], [], [retimer]
        cell_offset = tof_offset = 0
        resident = (
            self.lambert.resident_retime_tables
            and not retimer._tables
            and getattr(retimer.leg_table, "__func__", None) is Retimer.leg_table
            and getattr(retimer._return_override, "__func__", None) is Retimer._return_override
        )
        for j, (visit, nxt) in enumerate(pairwise(visits)):
            p = params[j]
            weight = 1.0 if retimer.weights is None else retimer.weights.get(visit.body, 1.0)
            if visit.body == 0:
                camp_min = camp_max = 0
            elif visit.deploy and visit.collect:
                camp_min = int(np.ceil(C.MIN_MINING_STAY_YEARS * C.YEAR_DAYS / s.step_days))
                camp_max = int(s.long_camp_max_days // s.step_days)
            else:
                camp_min, camp_max = 0, int(s.camp_max_days // s.step_days)
            if camp_min > camp_max:
                return None
            pin = -1 if nxt.pinned_arrival is None else lat.exact_index(nxt.pinned_arrival)
            if pin is None:
                return None
            arrival_rate = 0.0
            if visit.deploy:
                arrival_rate = (
                    weight * rate if visit.body in collected else s.orphan_credit * weight * rate
                )
            earliest = -np.inf
            if visit.collect and visit.foreign_deploy_epoch is not None:
                earliest = visit.foreign_deploy_epoch + C.MIN_MINING_STAY_YEARS * C.YEAR_DAYS
            dv, feasible = (
                (None, None)
                if resident
                else retimer.leg_table(visit.body, nxt.body, visit.role_out)
            )
            tofs = retimer._tofs(visit.role_out)
            override = (
                retimer._return_override(visit.body)
                if not resident and visit.role_out == "earth_return"
                else None
            )
            refs.extend((dv, feasible, override))
            keys.append(
                (
                    visit.body,
                    nxt.body,
                    visit.role_out,
                    id(dv),
                    id(feasible),
                    id(override),
                    tofs.tobytes(),
                )
            )
            tables.append((dv, feasible, override))
            tofs_all.append(tofs)
            flat, ratio = retimer._limits(visit.role_out, visit.body, nxt.body)
            model = (
                2
                if retimer._return_model(visit.role_out)
                else 1
                if retimer._ratio_model(visit.role_out)
                else 0
            )
            p["cell_offset"], p["tof_offset"], p["tofs"] = cell_offset, tof_offset, len(tofs)
            p["camp_min"], p["camp_max"], p["pinned_next"], p["model"] = (
                camp_min,
                camp_max,
                pin,
                model,
            )
            p["arrival_rate"], p["departure_rate"] = (
                arrival_rate,
                weight * rate if visit.collect else 0.0,
            )
            p["earliest_collect"], p["mass"], p["ratio_limit"] = earliest, masses[j], ratio
            p["flat"], p["floor"], p["slope"] = (
                flat,
                s.hop_inflation_floor,
                s.hop_inflation_slope or 0.0,
            )
            p["calibration"] = retimer.inflations.get((visit.body, nxt.body), 1.0)
            cell_offset += lat.count * len(tofs)
            tof_offset += len(tofs)
        key = (id(retimer), retimer._cache_revision, resident, lat.epochs.tobytes(), tuple(keys))
        if self.key != key:
            self.close()
            if max(lat.count, len(params), cell_offset, tof_offset) > np.iinfo(np.int32).max:
                raise ValueError("CUDA retiming table exceeds int32 capacity")
            epochs = np.ascontiguousarray(lat.epochs, dtype=np.float64)
            tofs = np.concatenate(tofs_all).astype(np.float64, copy=False)
            shifts = np.asarray([round(t / s.step_days) for t in tofs], dtype=np.int32)
            if resident:
                from .gpu_lambert import HopElements, body_elements

                elements = (HopElements * len(params))(
                    *[
                        HopElements(
                            body_elements(retimer.catalogue, v.body),
                            body_elements(retimer.catalogue, nxt.body),
                            C.MU_SUN_KM3_S2,
                            C.MAX_VINF_EARTH_KM_S if v.body == 0 else 0.0,
                            C.MAX_VINF_EARTH_KM_S if nxt.body == 0 else 0.0,
                        )
                        for v, nxt in pairwise(visits)
                    ]
                )
                self.lambert._prepare(256)
                self._check(
                    self.create_elements(
                        self.device,
                        lat.count,
                        len(params),
                        cell_offset,
                        tof_offset,
                        epochs.ctypes.data,
                        tofs.ctypes.data,
                        shifts.ctypes.data,
                        params.ctypes.data,
                        ct.addressof(elements),
                        self.lambert.handle,
                        ct.byref(self.handle),
                    )
                )
                self.resident_builds += 1
                self.resident_cells += cell_offset
                retimer.lambert_evaluations += 2 * cell_offset
                # Count the same branch requests and bounded kernel batches as
                # the former host-table builder, without synchronizing counters.
                for stage in params:
                    remaining = lat.count * int(stage["tofs"])
                    while remaining:
                        count = min(remaining, self.lambert.capacity)
                        self.lambert._record(2 * count)
                        remaining -= count
                self.lambert.telemetry["completed_element_hops"] = (
                    self.lambert.telemetry.get("completed_element_hops", 0) + cell_offset
                )
            else:
                self._create_host_tables(
                    epochs, tables, tofs, shifts, len(params), cell_offset, tof_offset
                )
            self.key, self.refs = key, refs
            self.uploads += int(not resident)
        if resident:
            for j, visit in enumerate(visits[:-1]):
                if visit.role_out == "earth_return":
                    self._update_sweep(retimer, visit.body, params[j], tofs_all[j])
        if self.graph_enabled != self.lambert.retime_cuda_graph:
            self._check(self.set_graph(self.handle, int(self.lambert.retime_cuda_graph)))
            self.graph_enabled = self.lambert.retime_cuda_graph
        arrivals = np.empty(len(visits), dtype=np.int32)
        departures = np.empty(len(visits), dtype=np.int32)
        path_dv = np.empty(len(params), dtype=np.float64)
        path_swept = np.empty(len(params), dtype=np.float64)
        path_ok = np.empty(len(params), dtype=np.uint8)
        objective, feasible_result = ct.c_double(), ct.c_int32()
        self._check(
            self.evaluate(
                self.handle,
                params.ctypes.data,
                price,
                C.THRUST_MAX_N,
                exhaust_velocity_km_s(),
                arrivals.ctypes.data,
                departures.ctypes.data,
                ct.byref(objective),
                ct.byref(feasible_result),
                path_dv.ctypes.data,
                path_swept.ctypes.data,
                path_ok.ctypes.data,
            )
        )
        self.calls += 1
        if feasible_result.value:
            self.last_path = (
                retimer,
                (retimer._cache_revision, retimer._sweep_revision),
                visits[:],
                lat.epochs[arrivals].tolist(),
                lat.epochs[departures].tolist(),
                (path_dv, path_swept, path_ok),
            )
            return arrivals.tolist(), departures.tolist(), objective.value
        return None

    def _update_sweep(self, retimer, body, stage, tofs):
        from .retiming import RETURN_SWEEP_REACH

        sweep = retimer.return_sweeps.get(body)
        cells = []
        if sweep is not None:
            for i, departure in enumerate(sweep.departures):
                k = retimer.lattice.exact_index(float(departure))
                if k is None:
                    continue
                for j, tof in enumerate(sweep.tofs):
                    t = round((float(tof) - tofs[0]) / retimer.settings.step_days)
                    if (
                        not (0 <= t < len(tofs))
                        or abs(tofs[t] - tof) > 1e-6
                        or not sweep.attempted[i, j]
                    ):
                        continue
                    cells.append(
                        (k, t, float(sweep.delta_v_km_s[i, j]), int(bool(sweep.certified[i, j])), 0)
                    )
        samples = np.asarray(cells, dtype=SWEEP_CELL)
        key = samples.tobytes()
        offset = int(stage["cell_offset"])
        if self.sweep_keys.get(offset, b"") == key:
            return
        self._check(
            self.set_sweep(
                self.handle,
                offset,
                len(tofs),
                len(samples),
                samples.ctypes.data,
                RETURN_SWEEP_REACH,
            )
        )
        self.sweep_keys[offset] = key
        self.sweep_updates += 1
        self.sweep_samples += len(samples)

    def _create_host_tables(self, epochs, tables, tofs, shifts, stages, cells, nt):
        dv = np.concatenate([t[0].ravel() for t in tables]).astype(np.float64, copy=False)
        ok = np.concatenate([t[1].ravel() for t in tables]).astype(np.uint8)
        swept = np.concatenate(
            [np.full(t[0].size, np.nan) if t[2] is None else t[2][0].ravel() for t in tables]
        ).astype(np.float64, copy=False)
        swept_ok = np.concatenate(
            [
                np.ones(t[0].size, dtype=np.uint8) if t[2] is None else t[2][1].ravel()
                for t in tables
            ]
        ).astype(np.uint8)
        self._check(
            self.create(
                self.device,
                len(epochs),
                stages,
                cells,
                nt,
                *(x.ctypes.data for x in [epochs, dv, ok, swept, swept_ok, tofs, shifts]),
                ct.byref(self.handle),
            )
        )
