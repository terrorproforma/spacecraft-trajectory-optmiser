"""Multi-block CUDA collection-tour DP with compact final backtracking output."""

from __future__ import annotations

import ctypes as ct
import os

import numpy as np

from . import constants as C


class Policy(ct.Structure):
    _fields_ = (
        [
            (n, ct.c_int32)
            for n in ["k", "n", "nt", "nr", "camp", "hop_model", "return_model", "reserved"]
        ]
        + [
            (n, ct.c_double)
            for n in [
                "thrust",
                "exhaust",
                "hop_ratio",
                "hop_flat",
                "hop_floor",
                "hop_slope",
                "return_ratio",
                "return_flat",
                "fit_floor",
            ]
        ]
        + [("fit_coefficients", ct.c_double * 5)]
    )


class Inputs(ct.Structure):
    _fields_ = [
        (name, ct.c_void_p)
        for name in [
            "dv",
            "returns",
            "tofs",
            "return_tofs",
            "mined",
            "geometry_a",
            "geometry_l",
            "penalty",
            "override_inflation",
            "steps",
            "banned",
        ]
    ]


class Result(ct.Structure):
    _fields_ = (
        [
            (n, ct.c_int32)
            for n in [
                "feasible",
                "states",
                "hops",
                "terminal_j",
                "terminal_t",
                "terminal_r",
                "reposition",
                "reserved",
            ]
        ]
        + [("objective", ct.c_double), ("penalty", ct.c_double)]
        + [(n, ct.c_int32 * 16) for n in ["collected_at", "source", "target", "departure", "tof"]]
        + [("hop_propellant", ct.c_double * 16)]
        + [("hop_dv", ct.c_double * 16), ("return_dv", ct.c_double)]
    )


class PlanInputs(ct.Structure):
    _fields_ = [
        ("abi_version", ct.c_int32),
        ("reserved", ct.c_int32),
        ("epochs", ct.c_void_p),
        ("deploy_epochs", ct.c_void_p),
        ("weights", ct.c_void_p),
        ("minimum_stay", ct.c_double),
        ("mining_rate", ct.c_double),
        ("year_days", ct.c_double),
        ("floor_mass", ct.c_double),
    ]


class PlanResult(ct.Structure):
    _fields_ = [
        ("tour", Result),
        ("burn_per_hop", ct.c_double),
        ("first_objective", ct.c_double),
        ("passes", ct.c_int32),
        ("has_first_objective", ct.c_int32),
    ]


class GpuCollectDP:
    def __init__(
        self,
        gpu,
        key,
        table,
        ids,
        camp,
        t0,
        epochs,
        mined,
        banned,
        phase_penalty,
        previous=None,
        mining=None,
    ):
        self.gpu, self.key, self.table, self.ids = gpu, key, table, ids
        self.t0, self.epochs = t0, epochs
        self.handle = ct.c_void_p()
        try:
            self.create = getattr(
                gpu.library,
                "spacepdhcg_collect_create_plan" if mining else "spacepdhcg_collect_create",
            )
            self.solve_native = gpu.library.spacepdhcg_collect_solve_v2
            self.destroy = gpu.library.spacepdhcg_collect_destroy
        except AttributeError as error:
            raise RuntimeError("CUDA collection DP requires a rebuilt native library") from error
        self.create.argtypes = (
            [ct.POINTER(Policy), ct.POINTER(Inputs)]
            + ([ct.POINTER(PlanInputs)] if mining else [])
            + [ct.POINTER(ct.c_void_p)]
        )
        self.solve_native.argtypes = [
            ct.c_void_p,
            ct.c_void_p,
            ct.c_double,
            ct.c_double,
            ct.POINTER(Result),
        ]
        self.destroy.argtypes = [ct.c_void_p]
        for f in [self.create, self.solve_native, self.destroy]:
            f.restype = ct.c_int
        s = table.settings
        from .collectdp import CollectPairTable

        resident = isinstance(table, CollectPairTable) and gpu.collect_tables_resident
        k, n, nt, nr = len(ids), len(epochs), len(table.tofs), len(table.return_tofs)
        if not 1 <= k <= 16:
            raise ValueError("CUDA collection DP supports 1-16 asteroids")
        fit = s.inflation_fit
        p = Policy(
            k,
            n,
            nt,
            nr,
            camp,
            2 if fit else int(s.hop_inflation_slope is not None),
            int(s.return_tof_model),
            0,
            C.THRUST_MAX_N,
            C.ISP_S * C.G0_M_S2 * 1e-3,
            s.hop_authority_ratio,
            s.hop_inflation,
            s.hop_inflation_floor,
            s.hop_inflation_slope or 0.0,
            s.return_authority_ratio,
            s.return_inflation,
            fit.floor if fit else 1.0,
            (ct.c_double * 5)(*(fit.coefficients if fit else [0] * 5)),
        )
        data = dict(
            dv=np.empty(0) if resident else np.full((k, k, n, nt), np.inf),
            returns=np.empty(0) if resident else np.empty((k, n, nr)),
            tofs=np.ascontiguousarray(table.tofs, dtype=np.float64),
            return_tofs=np.ascontiguousarray(table.return_tofs, dtype=np.float64),
            mined=np.empty(0) if mining else np.ascontiguousarray(mined),
            geometry_a=np.zeros((k, k)),
            geometry_l=np.zeros((k, k, n)),
            penalty=np.zeros((k, k, n)),
            override_inflation=np.full((k, n, nr), np.nan),
            steps=np.ascontiguousarray(table.tof_steps, dtype=np.int32),
            banned=np.zeros((k, k), dtype=np.int32),
        )
        retained, pair_handles, return_handles = [], [None] * (k * k), []
        for j, source in enumerate(ids):
            if resident:
                native = table._resident_table(
                    source, 0, table.return_tofs, C.MISSION_END_MJD - s.end_margin_days
                )
                retained.append(native)
                return_handles.append(native.handle.value)
            else:
                data["returns"][j] = table.earth_return(source)[t0:]
            override = table.return_override(source)
            if override is not None:
                inflation, ok = override
                data["override_inflation"][j] = np.where(ok[t0:], inflation[t0:], np.inf)
            for target_index, target in enumerate(ids):
                refused = source == target or (source, target) in banned
                data["banned"][j, target_index] = refused
                if refused:
                    continue
                if resident:
                    native = table._resident_table(source, target, table.tofs, table.epochs[-1])
                    retained.append(native)
                    pair_handles[j * k + target_index] = native.handle.value
                else:
                    data["dv"][j, target_index] = table.hop(source, target)[t0:]
                if fit:
                    a, longitude = table.pair_geometry(source, target, epochs)
                    data["geometry_a"][j, target_index] = a
                    data["geometry_l"][j, target_index] = longitude
                if phase_penalty is not None:
                    data["penalty"][j, target_index] = phase_penalty(j, target_index)
        self.data = data
        packed = Inputs(
            *(
                None if name == "mined" and mining else data[name].ctypes.data
                for name, _ in Inputs._fields_
            )
        )
        self.plan_inputs = None
        if mining:
            deployed, weights = mining
            self.plan_data = [
                np.ascontiguousarray(epochs, dtype=np.float64),
                np.asarray([deployed[a] for a in ids], dtype=np.float64),
                np.asarray([weights.get(a, 1.0) for a in ids], dtype=np.float64),
            ]
            self.plan_inputs = PlanInputs(
                1,
                0,
                *(a.ctypes.data for a in self.plan_data),
                C.MIN_MINING_STAY_YEARS * C.YEAR_DAYS,
                C.MINING_RATE_KG_PER_YEAR,
                C.YEAR_DAYS,
                C.DRY_MASS_KG + 1.0,
            )
        metadata = [ct.byref(self.plan_inputs)] if mining else []
        self.policy = p
        pair_array = (ct.c_void_p * (k * k))(*pair_handles)
        return_array = (ct.c_void_p * k)(*return_handles)
        update = getattr(
            gpu.library,
            (
                "spacepdhcg_collect_update_plan_tables"
                if resident
                else "spacepdhcg_collect_update_plan"
            )
            if mining
            else ("spacepdhcg_collect_update_tables" if resident else "spacepdhcg_collect_update"),
            None,
        )
        reuse = os.environ.get("SPACEPDHCG_TEST_GTOC12_REUSE_COLLECT_DP", "1") != "0"
        if previous is not None and previous.handle.value and reuse and update is not None:
            update.argtypes = (
                [ct.c_void_p, ct.POINTER(Policy), ct.POINTER(Inputs)]
                + ([ct.POINTER(PlanInputs)] if mining else [])
                + ([ct.c_void_p, ct.c_void_p, ct.c_int32] if resident else [])
            )
            update.restype = ct.c_int
            args = [previous.handle, ct.byref(p), ct.byref(packed), *metadata]
            if resident:
                args += [pair_array, return_array, t0]
            status = update(*args)
            if status == 0:
                self.handle = ct.c_void_p(previous.handle.value)
                previous.handle = ct.c_void_p()
                gpu.telemetry["collect_dp_rebinds"] = gpu.telemetry.get("collect_dp_rebinds", 0) + 1
                return
            if status != 4:
                if status == 2:
                    previous.close()
                self.check(status)
        if previous is not None:
            previous.close()
        if resident:
            create_tables = getattr(
                gpu.library,
                "spacepdhcg_collect_create_plan_tables"
                if mining
                else "spacepdhcg_collect_create_tables",
            )
            create_tables.argtypes = [
                ct.POINTER(Policy),
                ct.POINTER(Inputs),
                *([ct.POINTER(PlanInputs)] if mining else []),
                ct.c_void_p,
                ct.c_void_p,
                ct.c_int32,
                ct.POINTER(ct.c_void_p),
            ]
            create_tables.restype = ct.c_int
            self.check(
                create_tables(
                    ct.byref(p),
                    ct.byref(packed),
                    *metadata,
                    pair_array,
                    return_array,
                    t0,
                    ct.byref(self.handle),
                )
            )
            gpu.telemetry["collect_resident_dp_builds"] = (
                gpu.telemetry.get("collect_resident_dp_builds", 0) + 1
            )
        else:
            self.check(self.create(ct.byref(p), ct.byref(packed), *metadata, ct.byref(self.handle)))
        gpu.telemetry["collect_dp_allocations"] = gpu.telemetry.get("collect_dp_allocations", 0) + 1

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"CUDA collection DP failed (native status {code})")

    def close(self):
        if self.handle.value:
            self.check(self.destroy(self.handle))
            self.handle = ct.c_void_p()

    def solve(self, mass_by_subset, camp_mass, price, weights, deployed, burn, phase_penalty):
        masses = np.ascontiguousarray(mass_by_subset, dtype=np.float64)
        result = Result()
        self.check(
            self.solve_native(self.handle, masses.ctypes.data, camp_mass, price, ct.byref(result))
        )
        return self._tour(result, price, weights, deployed, burn, phase_penalty)

    def solve_plan(self, camp_mass, price, weights, deployed, burn, phase_penalty):
        call = self.gpu.library.spacepdhcg_collect_solve_plan
        call.argtypes = [ct.c_void_p, ct.c_double, ct.c_double, ct.c_double, ct.POINTER(PlanResult)]
        call.restype = ct.c_int
        output = PlanResult()
        burn = float(burn) if burn is not None else np.nan
        if not np.isfinite(burn):
            burn = np.nan
        self.check(call(self.handle, camp_mass, price, burn, ct.byref(output)))
        self.gpu.telemetry["completed_collection_dp_passes"] = (
            self.gpu.telemetry.get("completed_collection_dp_passes", 0) + output.passes
        )
        self.gpu.telemetry["native_collection_plans"] = (
            self.gpu.telemetry.get("native_collection_plans", 0) + 1
        )
        self.gpu.telemetry["collection_plan_result_download_bytes"] = self.gpu.telemetry.get(
            "collection_plan_result_download_bytes", 0
        ) + ct.sizeof(PlanResult)
        result = self._tour(
            output.tour, price, weights, deployed, output.burn_per_hop, phase_penalty
        )
        if result is not None and output.has_first_objective:
            result.diagnostics["pass1_objective_kg"] = output.first_objective
        return result

    def _tour(self, result, price, weights, deployed, burn, phase_penalty):
        from .collectdp import CollectTour

        if result.reserved:
            raise RuntimeError("CUDA collection DP returned an invalid backpointer")
        if not result.feasible:
            return None
        ids, epochs, table = self.ids, self.epochs, self.table
        collect = {
            body: float(epochs[result.collected_at[j]])
            for j, body in enumerate(ids)
            if result.collected_at[j] >= 0
        }
        order = tuple(sorted(collect, key=collect.get))
        hops, costs, phases = [], [], []
        for h in reversed(range(result.hops)):
            j, target_index, t, index = (
                result.source[h],
                result.target[h],
                result.departure[h],
                result.tof[h],
            )
            hops.append(
                (
                    ids[j],
                    ids[target_index],
                    float(epochs[t]),
                    float(table.tofs[index]),
                    result.hop_dv[h],
                )
            )
            costs.append(result.hop_propellant[h])
            if phase_penalty is not None:
                phases.append(
                    float(table.phase_deg(ids[j], ids[target_index], epochs[t : t + 1])[0])
                )
        collected = sum(C.maximum_collected_mass(collect[a] - deployed[a]) for a in ids)
        weighted = sum(
            weights.get(a, 1.0) * C.maximum_collected_mass(collect[a] - deployed[a]) for a in ids
        )
        j, t, h = result.terminal_j, result.terminal_t, result.terminal_r
        return CollectTour(
            order,
            collect,
            hops,
            bool(result.reposition),
            result.objective,
            collected,
            (weighted - result.objective) / price - result.penalty,
            float(epochs[t]),
            float(table.return_tofs[h]),
            result.return_dv,
            result.states,
            dict(
                lattice_start=float(epochs[0]),
                asteroids=len(ids),
                propellant_weight=price,
                burn_per_hop_kg=burn,
            ),
            costs,
            phases,
            result.penalty,
        )


def cuda_collect_plan(
    table, ids, camp, t0, epochs, weights, banned, price, camp_mass, burn, deployed
):
    from .lambert import _GPU_BACKEND

    gpu = _GPU_BACKEND.get()
    if (
        gpu is None
        or not gpu.collect_dp_cuda
        or os.environ.get("SPACEPDHCG_TEST_GTOC12_NATIVE_COLLECT_PLAN", "1") == "0"
    ):
        return NotImplemented
    gpu._owned()
    phase_penalty = None
    if table.settings.harvest_phase is not None and table.settings.phase_weight > 0.0:

        def phase_penalty(j, target):
            return table.phase_penalty(ids[j], ids[target], epochs)

    gpu.collect_dp_workspace = GpuCollectDP(
        gpu,
        (table, object()),
        table,
        ids,
        camp,
        t0,
        epochs,
        None,
        banned,
        phase_penalty,
        previous=gpu.collect_dp_workspace,
        mining=(deployed, weights),
    )
    gpu.telemetry["collection_plan_metadata_upload_bytes"] = (
        gpu.telemetry.get("collection_plan_metadata_upload_bytes", 0)
        + (len(epochs) + 2 * len(ids)) * 8
    )
    return gpu.collect_dp_workspace.solve_plan(
        camp_mass, price, weights, deployed, burn, phase_penalty
    )


def cuda_collect_dp(
    table,
    ids,
    camp,
    t0,
    epochs,
    mined,
    mass_by_subset,
    weights,
    banned,
    price,
    camp_mass,
    burn,
    fraction,
    deployed,
    phase_penalty,
):
    from .lambert import _GPU_BACKEND

    gpu = _GPU_BACKEND.get()
    if gpu is None or not gpu.collect_dp_cuda:
        return NotImplemented
    gpu._owned()
    key = (table, fraction)
    if (
        gpu.collect_dp_workspace is None
        or not gpu.collect_dp_workspace.handle.value
        or gpu.collect_dp_workspace.key != key
    ):
        gpu.collect_dp_workspace = GpuCollectDP(
            gpu,
            key,
            table,
            ids,
            camp,
            t0,
            epochs,
            mined,
            banned,
            phase_penalty,
            previous=gpu.collect_dp_workspace,
        )
    result = gpu.collect_dp_workspace.solve(
        mass_by_subset, camp_mass, price, weights, deployed, burn, phase_penalty
    )
    gpu.telemetry["completed_collection_dp_passes"] = (
        gpu.telemetry.get("completed_collection_dp_passes", 0) + 1
    )
    return result
