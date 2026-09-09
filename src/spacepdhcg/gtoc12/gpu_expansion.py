"""CUDA expansion pricing/ranking; materialize only the beam-admission prefix."""

from __future__ import annotations

import ctypes as ct
import os
import sys
import time

import numpy as np

from . import constants as C

ENVIRONMENT = "SPACEPDHCG_TEST_GTOC12_GPU_EXPANSION"
ADMISSION_ENVIRONMENT = "SPACEPDHCG_TEST_GTOC12_GPU_ADMISSION"
ADMISSION = np.dtype(
    [(x, "i4") for x in ("abi", "limit", "max_set", "max_first")]
    + [(x, "f8") for x in ("dry_mass", "reserve_fraction", "return_reserve", "authority")],
    align=True,
)
POLICY = np.dtype(
    [(x, "i4") for x in ("abi", "ratio", "compensated", "reserved")]
    + [
        (x, "f8")
        for x in (
            "thrust",
            "day",
            "ve",
            "authority",
            "inflation",
            "floor",
            "slope",
            "miner",
            "initial",
            "arrival_horizon",
            "mining_horizon",
            "mining_rate",
            "year",
            "propellant_weight",
            "time_weight",
            "lookahead_weight",
        )
    ],
    align=True,
)
PARENT = np.dtype(
    [("begin", "i4"), ("count", "i4")]
    + [(x, "f8") for x in ("mass", "epoch", "launch", "propellant", "lookahead")],
    align=True,
)
DEPLOY = np.dtype([("body", "i8")] + [(x, "f8") for x in ("epoch", "weight", "price")], align=True)
OPTION = np.dtype(
    [("parent", "i4"), ("allowed", "i4"), ("target", "i8")]
    + [(x, "f8") for x in ("departure", "tof", "dv", "lookahead", "weight", "price", "bonus")],
    align=True,
)
RESULT = np.dtype(
    [("parent", "i4"), ("valid", "i4"), ("target", "i8")]
    + [
        (x, "f8")
        for x in (
            "departure",
            "arrival",
            "dv",
            "inflation",
            "propellant",
            "mass",
            "score",
            "lookahead",
            "hop_propellant",
        )
    ],
    align=True,
)


class GpuExpansion:
    def __init__(self, gpu):
        self.gpu = gpu
        self.handle = ct.c_void_p()
        self.capacity = (0, 0, 0)
        lib = gpu.library
        self.create = lib.spacepdhcg_gtoc12_expansion_create
        self.create.argtypes = [ct.c_int32] * 4 + [ct.POINTER(ct.c_void_p)]
        self.destroy = lib.spacepdhcg_gtoc12_expansion_destroy
        self.destroy.argtypes = [ct.POINTER(ct.c_void_p)]
        self.rank = lib.spacepdhcg_gtoc12_expansion_rank
        self.rank.argtypes = (
            [ct.c_void_p, ct.c_void_p] + [ct.c_int32, ct.c_void_p] * 3 + [ct.POINTER(ct.c_int32)]
        )
        self.read = lib.spacepdhcg_gtoc12_expansion_read
        self.read.argtypes = [ct.c_void_p, ct.c_int32, ct.c_int32, ct.c_void_p]
        self.admit = lib.spacepdhcg_gtoc12_expansion_admit
        self.admit.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int32, ct.c_void_p, ct.c_void_p]
        for function in (self.create, self.destroy, self.rank, self.read, self.admit):
            function.restype = ct.c_int

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError(f"CUDA expansion failed (native status {status})")

    def close(self):
        if self.handle.value:
            self.gpu._owned()
            self.check(self.destroy(ct.byref(self.handle)))

    def evaluate(self, policy, parents, deploys, options):
        self.gpu._owned()
        counts = len(parents), len(deploys), len(options)
        if not self.handle.value or any(
            n > cap for n, cap in zip(counts, self.capacity, strict=True)
        ):
            capacity = tuple(
                max(1, n, cap * 2) for n, cap in zip(counts, self.capacity, strict=True)
            )
            if max(capacity) > np.iinfo(np.int32).max:
                raise ValueError("CUDA expansion capacity exceeds int32")
            replacement = ct.c_void_p()
            self.check(self.create(self.gpu.device_id, *capacity, ct.byref(replacement)))
            try:
                self.close()
            except BaseException:
                self.destroy(ct.byref(replacement))
                raise
            self.handle, self.capacity = replacement, capacity
        valid = ct.c_int32()
        self.check(
            self.rank(
                self.handle,
                policy.ctypes.data,
                len(parents),
                parents.ctypes.data,
                len(deploys),
                deploys.ctypes.data,
                len(options),
                options.ctypes.data,
                ct.byref(valid),
            )
        )
        return valid.value

    def rows(self, count, batch=128):
        output = np.empty(min(count, batch), RESULT)
        for offset in range(0, count, batch):
            n = min(count - offset, len(output))
            self.check(self.read(self.handle, offset, n, output.ctypes.data))
            stats = self.gpu.telemetry
            stats["expansion_download_bytes"] = (
                stats.get("expansion_download_bytes", 0) + n * RESULT.itemsize
            )
            yield from output[:n].tolist()

    def admission(self, search, current, count):
        from .gpu_options import GpuResidentOptions
        from .search import RouteSearch

        if os.environ.get(ADMISSION_ENVIRONMENT, "1") == "0":
            return None
        for name in (
            "_filter",
            "_reserve",
            "_return_feasible",
            "_return_options",
            "limits",
            "_select_ranked",
        ):
            actual = getattr(search, name)
            if getattr(actual, "__func__", actual) is not getattr(RouteSearch, name):
                return None
        s = search.settings
        # Nonstandard negative policy settings retain their established Python behavior.
        if (
            min(
                s.max_per_first,
                s.max_per_deployed_set,
                s.beam_width,
                s.chain_tour_candidates,
                s.reserve_fraction,
                s.return_reserve_kg,
                s.earth_return_authority_ratio,
            )
            < 0
        ):
            return None
        if not count:
            return 0
        tables = []
        for parent in current:
            first = parent.deployed[0][0]
            if first not in search._return_cache:
                search._return_cache[first] = search._return_options(
                    first, C.MISSION_END_MJD - s.end_margin_days
                )
            table = search._return_cache[first]
            if not isinstance(table, GpuResidentOptions) or table.gpu is not self.gpu:
                return None
            table.owned()
            tables.append(table)
        limit = s.beam_width
        if (
            s.chain_tour_scoring
            and s.collect_dp
            and len(current[0].deployed) + 1 >= s.chain_tour_min_deploys
        ):
            limit = max(limit, s.chain_tour_candidates)
        policy = np.array(
            [
                (
                    1,
                    limit,
                    s.max_per_deployed_set,
                    s.max_per_first,
                    C.DRY_MASS_KG,
                    s.reserve_fraction,
                    s.return_reserve_kg,
                    s.earth_return_authority_ratio,
                )
            ],
            ADMISSION,
        )
        handles = (ct.c_void_p * len(tables))(*(t.handle.value for t in tables))
        selected = ct.c_int32()
        start = time.perf_counter()
        self.check(
            self.admit(self.handle, policy.ctypes.data, len(tables), handles, ct.byref(selected))
        )
        stats = self.gpu.telemetry
        for name, value in (
            ("admission_depths", 1),
            ("admission_input_children", count),
            ("admission_selected", selected.value),
            ("admission_seconds", time.perf_counter() - start),
        ):
            stats[name] = stats.get(name, 0) + value
        return selected.value


def pack(search, current):
    s = search.settings
    policy = np.zeros(1, POLICY)
    policy[0] = (
        1,
        int(s.hop_inflation_slope is not None),
        int(sys.version_info >= (3, 12)),
        0,
        C.THRUST_MAX_N,
        C.DAY_S,
        C.ISP_S * C.G0_M_S2 * 1e-3,
        s.hop_authority_ratio,
        s.hop_inflation,
        s.hop_inflation_floor,
        s.hop_inflation_slope or 0.0,
        C.MINER_MASS_KG,
        s.initial_mass,
        C.MISSION_END_MJD - 3 * C.YEAR_DAYS,
        C.MISSION_END_MJD - 2 * C.YEAR_DAYS,
        C.MINING_RATE_KG_PER_YEAR,
        C.YEAR_DAYS,
        s.propellant_weight,
        s.time_weight,
        s.collect_lookahead_weight,
    )
    parents = np.zeros(len(current), PARENT)
    deployment = []
    blocks = []
    for index, parent in enumerate(current):
        parents[index] = (
            len(deployment),
            len(parent.deployed),
            parent.mass,
            parent.epoch,
            parent.legs[0].departure_epoch,
            parent.hop_propellant,
            parent.lookahead_kg,
        )
        deployment.extend(
            (a, e, search.weights.get(a, 1.0), search.asteroid_prices.get(a, 0.0))
            for a, e in parent.deployed
        )
        visited = {a for a, _ in parent.deployed}
        for wait in s.deploy_wait_days:
            departure = parent.epoch + float(wait)
            hops = search.hops_from(parent.location, departure)
            targets = np.asarray(hops["target_ids"], dtype=np.int64)
            tofs = np.asarray(hops["tofs_days"], dtype=np.float64)
            block = np.zeros((len(targets), len(tofs)), OPTION)
            block["parent"] = index
            block["target"] = targets[:, None]
            block["departure"] = departure
            block["tof"] = tofs[None, :]
            block["dv"] = hops["total_delta_v"]
            block["allowed"] = (
                np.asarray(hops["feasible"], bool)
                & np.asarray(
                    [(parent.location, int(a)) not in search.banned_pairs for a in targets],
                    dtype=bool,
                )[:, None]
            )
            block["weight"] = np.asarray([search.weights.get(int(a), 1.0) for a in targets])[
                :, None
            ]
            block["price"] = np.asarray([search.asteroid_prices.get(int(a), 0.0) for a in targets])[
                :, None
            ]
            if s.collect_lookahead_weight > 0:
                block["lookahead"] = search.collect_lookahead(
                    parent.location, targets, departure, parent.mass
                )[:, None]
            if search.clusters is not None and s.cluster_bonus_kg > 0:
                remaining = max(s.max_deploys - len(parent.deployed) - 1, 0)
                block["bonus"] = np.asarray(
                    [
                        s.cluster_bonus_kg
                        * min(
                            search.clusters.unvisited_potential(int(a), visited | {int(a)}),
                            remaining,
                        )
                        / max(s.max_deploys, 1)
                        for a in targets
                    ]
                )[:, None]
            blocks.append(block.ravel())
    return (
        policy,
        parents,
        np.asarray(deployment, DEPLOY),
        (np.concatenate(blocks) if blocks else np.empty(0, OPTION)),
    )


def expand_and_select(search, current):
    from .lambert import _GPU_BACKEND
    from .search import PlannedLeg, RouteSearch, _Partial

    gpu = _GPU_BACKEND.get()
    if gpu is None or os.environ.get(ENVIRONMENT, "1") == "0" or not current:
        return None
    # Custom heuristic overrides keep their established semantics. The native
    # pool implements these production methods, not arbitrary Python callbacks.
    for name in (
        "_expand",
        "_select",
        "_ordered",
        "_feasible",
        "_propellant",
        "hop_inflation_for",
        "_price_of",
        "_cluster_potential_kg",
    ):
        actual = getattr(search, name)
        expected = getattr(RouteSearch, name)
        if getattr(actual, "__func__", actual) is not expected:
            return None
    # NumPy scalars/custom numeric objects use Python's generic sum path, which
    # is distinct from the exact-float compensated sum implemented on CUDA.
    if any(
        type(v) is not float
        for table in (search.weights, search.asteroid_prices)
        for v in table.values()
    ):
        return None
    if gpu.expansion_workspace is None:
        gpu.expansion_workspace = GpuExpansion(gpu)
    workspace = gpu.expansion_workspace
    start = time.perf_counter()
    inputs = pack(search, current)
    count = workspace.evaluate(*inputs)
    stats = gpu.telemetry
    for key, n in (
        ("expansion_depths", 1),
        ("expansion_input_slots", len(inputs[3])),
        ("expansion_valid_children", count),
        ("expansion_upload_bytes", sum(a.nbytes for a in inputs)),
        ("expansion_pack_rank_seconds", time.perf_counter() - start),
    ):
        stats[key] = stats.get(key, 0) + n
    selected = workspace.admission(search, current, count)
    if selected is not None:
        count = selected

    def candidates():
        for row in workspace.rows(count):
            (
                parent_index,
                _,
                target,
                departure,
                arrival,
                dv,
                inflation,
                _,
                mass,
                score,
                lookahead,
                propellant,
            ) = row
            parent = current[parent_index]
            legs = list(parent.legs)
            if departure > parent.epoch:
                legs.append(
                    PlannedLeg(
                        parent.location, parent.location, parent.epoch, departure, 0.0, 1.0, "camp"
                    )
                )
            legs.append(
                PlannedLeg(parent.location, target, departure, arrival, dv, inflation, "deploy_hop")
            )
            stats["expansion_materialized_children"] = (
                stats.get("expansion_materialized_children", 0) + 1
            )
            yield _Partial(
                legs,
                target,
                arrival,
                mass,
                [*parent.deployed, (target, arrival)],
                propellant,
                score=score,
                lookahead_kg=lookahead,
                chain_burn=parent.chain_burn,
            )

    if selected is not None:
        return search._select_ranked(candidates(), len(current[0].deployed) + 1, admitted=True)
    return search._select_ranked(candidates(), len(current[0].deployed) + 1)
