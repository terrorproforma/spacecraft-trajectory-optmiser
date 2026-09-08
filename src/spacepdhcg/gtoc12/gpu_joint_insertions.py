"""CUDA-generated insertion schedules across heterogeneous visit layouts.

Python prepares immutable route metadata. All trial epoch arithmetic, preflight,
geometry and surrogate mass passes run in native batches. Feasible surrogates
still require the ordinary trajectory refinement and independent certification.
"""

import ctypes as ct
import os
from dataclasses import replace

import numpy as np

from .gpu_joint import (
    CACHED_COST,
    GEOMETRY_STATS,
    RESULT,
    STAGE,
    VISIT,
    GpuJoint,
    _decode_evaluation,
    _geometry_inputs,
    _metadata,
    evaluate_joint,
)


def _layouts(visits, candidates, camp):
    """Structural metadata only: no candidate epoch vectors are built here."""
    from .retiming import Visit

    present = {v.body for v in visits}
    for asteroid in candidates:
        if asteroid in present:
            continue
        for i in range(1, camp):
            for k in range(camp, len(visits) - 1):
                expanded = list(visits)
                last = expanded[k + 1].body == 0
                expanded[k] = replace(expanded[k], role_out="collect_hop")
                expanded.insert(
                    k + 1, Visit(asteroid, False, True, "earth_return" if last else "collect_hop")
                )
                expanded.insert(i + 1, Visit(asteroid, True, False, "deploy_hop"))
                yield expanded, i, k, asteroid


def insertion_batch(joint, visits, arrivals, departures, layouts, camp):
    """Native batch plus raw outputs for parity/telemetry; preserves every row."""
    from .gpu_lambert import HopElements
    from .jointopt import JointItinerary
    from .lambert import _GPU_BACKEND

    if (
        getattr(joint.evaluate, "__func__", None) is not JointItinerary.evaluate
        or joint._insertion_seeds is not JointItinerary._insertion_seeds
    ):
        raise ValueError("CUDA insertions do not support overridden evaluation/seed generation")
    # Reuse the evaluator's strict model compatibility gate without launching work.
    evaluate_joint(joint, visits, np.empty((0, len(visits))), np.empty((0, len(visits))))
    gpu = _GPU_BACKEND.get()
    gpu._owned()
    fn = getattr(gpu.library, "spacepdhcg_gtoc12_joint_insertions_host", None)
    if fn is None:
        raise RuntimeError("CUDA insertions require spacepdhcg_gtoc12_joint_insertions_host")
    fn.argtypes = (
        [ct.c_void_p, ct.c_int32, ct.c_int32]
        + [ct.c_void_p] * 8
        + [ct.c_int32]
        + [ct.c_void_p] * 10
    )
    fn.restype = ct.c_int
    n, count = len(visits) + 2, 4 * len(layouts)
    if not layouts:
        raise ValueError("insertion_batch requires at least one layout")
    arr, dep = (np.ascontiguousarray(a, dtype=np.float64) for a in (arrivals, departures))
    if arr.shape != (len(visits),) or dep.shape != arr.shape:
        raise ValueError("one base arrival and departure per visit required")
    current = gpu.joint_workspace
    if current is None or current.capacity < count or current.visits != n:
        if current is not None:
            current.close()
        gpu.joint_workspace = current = GpuJoint(gpu, count, n)
    metadata, stages = np.empty((len(layouts), n), VISIT), np.empty((len(layouts), n - 1), STAGE)
    elements = (HopElements * (len(layouts) * (n - 1)))()
    offsets, records, slots = [0], [], []
    policy = None
    for layout_index, (expanded, i, k, _) in enumerate(layouts):
        metadata[layout_index], stages[layout_index], packed_policy = _metadata(joint, expanded)
        if policy is None:
            policy = packed_policy
        elif policy.tobytes() != packed_policy.tobytes():
            raise ValueError("insertion policy must be identical across layouts")
        hops, costs = _geometry_inputs(joint, expanded)
        ct.memmove(
            ct.addressof(elements) + layout_index * ct.sizeof(hops),
            ct.addressof(hops),
            ct.sizeof(hops),
        )
        records.append(costs)
        offsets.append(offsets[-1] + len(costs))
        slots.append((i, k))
    records = np.concatenate(records).astype(CACHED_COST, copy=False)
    offsets, slots = np.asarray(offsets, dtype=np.int32), np.asarray(slots, dtype=np.int32)
    result, enabled, stats = (
        np.empty(count, RESULT),
        np.empty(count, np.uint8),
        np.zeros(1, GEOMETRY_STATS),
    )
    masses, inflations, proxies = (np.empty((count, n - 1)) for _ in range(3))
    payload, out_arr, out_dep = (np.empty((count, n)) for _ in range(3))
    current._check(
        fn(
            current.handle,
            len(layouts),
            camp,
            *(a.ctypes.data for a in (policy, metadata, stages, slots, arr, dep)),
            ct.addressof(elements),
            records.ctypes.data,
            len(records),
            *(
                a.ctypes.data
                for a in (
                    offsets,
                    result,
                    enabled,
                    masses,
                    inflations,
                    proxies,
                    payload,
                    out_arr,
                    out_dep,
                    stats,
                )
            ),
        )
    )
    evaluations = int(np.count_nonzero(enabled))
    computed = int(stats["computed_hops"][0])
    joint.evaluations += evaluations
    joint.lambert_evaluations += 2 * computed
    if computed:
        gpu._record(2 * computed)
    updates = {
        "completed_joint_batches": 1,
        "completed_joint_insertion_batches": 1,
        "completed_joint_insertion_layouts": len(layouts),
        "completed_joint_evaluations": evaluations,
        "joint_insertion_skipped_seeds": count - evaluations,
        "completed_element_hops": computed,
        "joint_geometry_stats_download_bytes": stats.nbytes,
        "joint_insertion_epoch_upload_bytes": arr.nbytes + dep.nbytes,
        "joint_insertion_epoch_download_bytes": out_arr.nbytes + out_dep.nbytes,
        "joint_result_download_bytes": sum(
            a.nbytes for a in (result, enabled, masses, inflations, proxies, payload)
        ),
    }
    updates.update({"joint_geometry_" + field: int(stats[field][0]) for field in stats.dtype.names})
    for key, value in updates.items():
        gpu.telemetry[key] = gpu.telemetry.get(key, 0) + value
    if np.any(result["failure"] == 17):
        raise ValueError("mining stay must be finite and nonnegative")
    return result, enabled, out_arr, out_dep, masses, inflations, proxies, payload


def insertions(joint, visits, arrivals, departures, candidates, *, layouts_per_batch=None):
    """Bounded native batches, stable ranking; no scalar trial evaluator calls."""
    from itertools import islice

    from .lambert import _GPU_BACKEND

    gpu = _GPU_BACKEND.get()
    requested = os.environ.get("SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_LAYOUTS")
    available = gpu is not None and hasattr(
        gpu.library, "spacepdhcg_gtoc12_joint_prepare_insertions_host"
    )
    if requested == "1" or (requested is None and available):
        if not available:
            raise RuntimeError("CUDA insertion layouts require the prepared insertion API")
        from .gpu_joint_layouts import insertions as prepared_insertions

        return prepared_insertions(
            joint,
            visits,
            arrivals,
            departures,
            candidates,
            layouts_per_batch=4096 if layouts_per_batch is None else layouts_per_batch,
        )
    if layouts_per_batch is None:
        layouts_per_batch = 256

    camp = next(
        (j for j in range(1, len(visits) - 1) if visits[j].deploy and visits[j].collect), None
    )
    if camp is None:
        return []
    if layouts_per_batch < 1:
        raise ValueError("layouts_per_batch must be positive")
    pending = iter(_layouts(visits, candidates, camp))
    results = []
    while layouts := list(islice(pending, layouts_per_batch)):
        value, enabled, arr, dep, mass, inflation, proxy, payload = insertion_batch(
            joint, visits, arrivals, departures, layouts, camp
        )
        for row in np.flatnonzero(enabled & (value["failure"] == 0)):
            expanded, _, _, asteroid = layouts[row // 4]
            evaluation = _decode_evaluation(
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
            results.append((expanded, arr[row].copy(), dep[row].copy(), evaluation, asteroid))
    results.sort(key=lambda item: (-item[3].objective, item[4]))
    return results
