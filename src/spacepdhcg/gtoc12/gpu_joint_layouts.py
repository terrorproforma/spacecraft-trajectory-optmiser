"""Compile shared edge inputs once; construct every insertion layout on CUDA."""

import ctypes as ct
import math
from dataclasses import replace
from itertools import pairwise

import numpy as np

from . import constants as C
from .gpu_joint import (
    CACHED_COST,
    GEOMETRY_STATS,
    RESULT,
    STAGE,
    VISIT,
    GpuJoint,
    _decode_evaluation,
    _metadata,
    evaluate_joint,
)
from .gpu_lambert import HopElements, body_elements
from .jointopt import JointItinerary, insertion_pivot
from .retiming import Visit


def compile_source(joint, visits, arrivals, departures, candidates):
    """O(candidates * visits) shared edges, rather than all O(C*N*N) layouts."""
    camp = insertion_pivot(visits)
    present = {v.body for v in visits}
    candidates = [int(a) for a in candidates if a not in present]
    if camp is None or camp < 2 or not candidates:
        return None
    if (
        getattr(joint.evaluate, "__func__", None) is not JointItinerary.evaluate
        or joint._insertion_seeds is not JointItinerary._insertion_seeds
    ):
        raise ValueError("CUDA layouts do not support overridden evaluation/seed generation")
    if (
        evaluate_joint(joint, visits, np.empty((0, len(visits))), np.empty((0, len(visits))))
        is None
    ):
        raise RuntimeError("CUDA joint backend is unavailable")
    metadata, _, policy = _metadata(joint, visits)
    arr, dep = (np.ascontiguousarray(a, dtype=np.float64) for a in (arrivals, departures))
    if arr.shape != (len(visits),) or dep.shape != arr.shape:
        raise ValueError("one base arrival and departure per visit required")
    descriptors = [(a.body, b.body, a.role_out) for a, b in pairwise(visits)]
    for asteroid in candidates:
        for i in range(1, camp):
            descriptors.extend(
                [
                    (visits[i].body, asteroid, visits[i].role_out),
                    (asteroid, visits[i + 1].body, "deploy_hop"),
                ]
            )
        for k in range(camp, len(visits) - 1):
            descriptors.extend(
                [
                    (visits[k].body, asteroid, "collect_hop"),
                    (
                        asteroid,
                        visits[k + 1].body,
                        "earth_return" if visits[k + 1].body == 0 else "collect_hop",
                    ),
                ]
            )
    orbits = {body: body_elements(joint.catalogue, body) for body in present | set(candidates)}
    stages = np.zeros(len(descriptors), STAGE)
    elements = (HopElements * len(descriptors))()
    stage_cache, pair_edges = {}, {}
    for edge, (a, b, role) in enumerate(descriptors):
        key = a, b, role
        if key not in stage_cache:
            # Reuse the authoritative stage-policy packer on one edge. The
            # temporary visits are descriptors, never trial epoch schedules.
            stage_cache[key] = _metadata(
                joint, [Visit(a, False, False, role), Visit(b, False, False, "")]
            )[1][0].copy()
        stages[edge] = stage_cache[key]
        elements[edge] = HopElements(
            orbits[a],
            orbits[b],
            C.MU_SUN_KM3_S2,
            C.MAX_VINF_EARTH_KM_S if a == 0 else 0.0,
            C.MAX_VINF_EARTH_KM_S if b == 0 else 0.0,
        )
        pair_edges.setdefault((a, b), []).append(edge)
    entries = []
    for key in joint._lambert.keys() | joint.measured.keys():
        a, b, departure, arrival = key
        if math.isfinite(departure) and math.isfinite(arrival):
            entries.extend((edge, departure, arrival, key) for edge in pair_edges.get((a, b), ()))
    entries.sort(key=lambda x: x[:3])
    records = np.zeros(len(entries), CACHED_COST)
    for row, (edge, departure, arrival, key) in zip(records, entries, strict=True):
        row["leg"], row["departure"], row["arrival"] = edge, departure, arrival
        row["cached"] = key in joint._lambert
        row["value"]["lambert"] = joint._lambert.get(key, math.nan)
        measured = joint.measured.get(key)
        if measured is not None:
            row["value"]["measured"] = 1
            row["value"]["measured_delta_v"] = measured.delta_v_km_s
            row["value"]["measured_mass"] = measured.mass_before_kg
    weights = np.asarray(
        [1.0 if joint.weights is None else joint.weights.get(a, 1.0) for a in candidates],
        dtype=np.float64,
    )
    return dict(
        metadata=metadata,
        policy=policy,
        arr=arr,
        dep=dep,
        weights=weights,
        stages=stages,
        elements=elements,
        records=records,
        candidates=candidates,
        camp=camp,
        descriptors=descriptors,
        dwell=joint.dwell_limit(Visit(candidates[0], True, False, "deploy_hop")),
        layouts=len(candidates) * (camp - 1) * (len(visits) - 1 - camp),
    )


class PreparedInsertions:
    def __init__(self, joint, visits, arrivals, departures, candidates, layouts_per_batch=4096):
        from .lambert import _GPU_BACKEND

        if not isinstance(layouts_per_batch, int) or layouts_per_batch < 1:
            raise ValueError("layouts_per_batch must be a positive integer")
        self.joint, self.visits = joint, tuple(visits)
        self.source = s = compile_source(joint, visits, arrivals, departures, candidates)
        self.total = 0 if s is None else s["layouts"]
        self.batch_size = layouts_per_batch
        if s is None:
            return
        self._candidates, self._camp = tuple(s["candidates"]), s["camp"]
        self.gpu = gpu = _GPU_BACKEND.get()
        gpu._owned()
        self.n = len(visits) + 2
        capacity = 4 * min(layouts_per_batch, self.total)
        current = gpu.joint_workspace
        if current is None or current.capacity < capacity or current.visits != self.n:
            if current is not None:
                current.close()
            gpu.joint_workspace = current = GpuJoint(gpu, capacity, self.n)
        self.native = current
        self.prepare = gpu.library.spacepdhcg_gtoc12_joint_prepare_insertions_host
        self.prepare.argtypes = (
            [ct.c_void_p, ct.c_int32, ct.c_int32]
            + [ct.c_void_p] * 5
            + [ct.c_double]
            + [ct.c_void_p] * 3
            + [ct.c_int32]
        )
        self.prepare.restype = ct.c_int
        self.evaluate = gpu.library.spacepdhcg_gtoc12_joint_prepared_insertions_host
        self.evaluate.argtypes = [ct.c_void_p, ct.c_int64, ct.c_int32] + [ct.c_void_p] * 11
        self.evaluate.restype = ct.c_int
        current._check(
            self.prepare(
                current.handle,
                len(s["candidates"]),
                s["camp"],
                *(s[k].ctypes.data for k in ("policy", "metadata", "arr", "dep", "weights")),
                s["dwell"],
                s["stages"].ctypes.data,
                ct.addressof(s["elements"]),
                s["records"].ctypes.data,
                len(s["records"]),
            )
        )
        self._token = object()
        current._prepared_insertion_token = self._token
        for key, value in {
            "joint_layout_preparations": 1,
            "joint_layout_source_edges": len(s["stages"]),
            "joint_layout_source_upload_bytes": sum(
                s[k].nbytes
                for k in ("policy", "metadata", "arr", "dep", "weights", "stages", "records")
            )
            + ct.sizeof(s["elements"]),
        }.items():
            gpu.telemetry[key] = gpu.telemetry.get(key, 0) + value

    def run(self, first, count, *, inspect=False):
        if not self.total:
            raise ValueError("empty prepared insertion neighbourhood")
        self.gpu._owned()
        if self.native._prepared_insertion_token is not self._token:
            raise RuntimeError("prepared insertion source has been replaced")
        if first < 0 or count < 1 or count > self.batch_size or first + count > self.total:
            raise ValueError("invalid prepared insertion slice")
        n, rows = self.n, 4 * count
        result, enabled, stats = (
            np.empty(rows, RESULT),
            np.empty(rows, np.uint8),
            np.zeros(1, GEOMETRY_STATS),
        )
        masses, inflations, proxies = (np.empty((rows, n - 1)) for _ in range(3))
        payload, arr, dep = (np.empty((rows, n)) for _ in range(3))
        metadata = np.empty((count, n), VISIT) if inspect else None
        edges = np.empty((count, n - 1), np.int32) if inspect else None
        self.native._check(
            self.evaluate(
                self.native.handle,
                first,
                count,
                *(
                    a.ctypes.data
                    for a in (
                        result,
                        enabled,
                        masses,
                        inflations,
                        proxies,
                        payload,
                        arr,
                        dep,
                        stats,
                    )
                ),
                metadata.ctypes.data if inspect else None,
                edges.ctypes.data if inspect else None,
            )
        )
        evaluations = int(np.count_nonzero(enabled))
        computed = int(stats["computed_hops"][0])
        self.joint.evaluations += evaluations
        self.joint.lambert_evaluations += 2 * computed
        if computed:
            self.gpu._record(2 * computed)
        updates = dict(
            completed_joint_batches=1,
            completed_joint_insertion_batches=1,
            completed_joint_insertion_layouts=count,
            completed_joint_evaluations=evaluations,
            joint_insertion_skipped_seeds=rows - evaluations,
            completed_element_hops=computed,
            joint_geometry_stats_download_bytes=stats.nbytes,
            joint_insertion_epoch_download_bytes=arr.nbytes + dep.nbytes,
            joint_result_download_bytes=sum(
                a.nbytes for a in (result, enabled, masses, inflations, proxies, payload)
            ),
        )
        updates.update({"joint_geometry_" + k: int(stats[k][0]) for k in stats.dtype.names})
        for key, value in updates.items():
            self.gpu.telemetry[key] = self.gpu.telemetry.get(key, 0) + value
        if np.any(result["failure"] == 17):
            raise ValueError("mining stay must be finite and nonnegative")
        return result, enabled, arr, dep, masses, inflations, proxies, payload, metadata, edges

    def layout(self, index):
        """Reconstruct only a requested survivor for the host refinement boundary."""
        if not 0 <= index < self.total:
            raise ValueError("invalid insertion layout index")
        D = self._camp - 1
        K = len(self.visits) - 1 - self._camp
        asteroid = self._candidates[index // (D * K)]
        i = 1 + index // K % D
        k = self._camp + index % K
        expanded = list(self.visits)
        last = expanded[k + 1].body == 0
        expanded[k] = replace(expanded[k], role_out="collect_hop")
        expanded.insert(
            k + 1, Visit(asteroid, False, True, "earth_return" if last else "collect_hop")
        )
        expanded.insert(i + 1, Visit(asteroid, True, False, "deploy_hop"))
        return expanded, asteroid


def insertions(joint, visits, arrivals, departures, candidates, *, layouts_per_batch=4096):
    prepared = PreparedInsertions(
        joint, visits, arrivals, departures, candidates, layouts_per_batch
    )
    results = []
    for first in range(0, prepared.total, layouts_per_batch):
        count = min(layouts_per_batch, prepared.total - first)
        value, enabled, arr, dep, mass, inflation, proxy, payload, _, _ = prepared.run(first, count)
        for row in np.flatnonzero(enabled & (value["failure"] == 0)):
            expanded, asteroid = prepared.layout(first + row // 4)
            ev = _decode_evaluation(
                joint,
                expanded,
                arr[row],
                dep[row],
                value[row],
                mass[row],
                inflation[row],
                proxy[row],
                payload[row],
            )
            results.append((expanded, arr[row].copy(), dep[row].copy(), ev, asteroid))
    results.sort(key=lambda r: (-r[3].objective, r[4]))
    return results
