"""Bounded route recovery ordering; local solver failure never excludes a route."""

import ctypes as ct

import numpy as np


def recovery_order(plans, initial, gpu=None):
    """Prefer a new first-three-leg prefix, then retain every remaining rank.

    Prefixes are scheduling hints only: identical prefixes can have different
    propagated masses and later legs. The original score order breaks ties.
    The explicit CPU screening mode provides a reference implementation.
    """
    count = len(plans)
    if not 0 <= initial <= count:
        raise ValueError("initial refinement count must be within candidate count")
    if count == initial:
        return []
    prefixes = np.zeros((count, 13), dtype=np.float64)
    for i, plan in enumerate(plans):
        legs = plan.legs[:3]
        prefixes[i, 0] = len(legs)
        for j, leg in enumerate(legs):
            prefixes[i, 1 + 4 * j : 5 + 4 * j] = (
                leg.from_id,
                leg.to_id,
                leg.departure_epoch,
                leg.arrival_epoch,
            )
    if not np.isfinite(prefixes).all():
        raise ValueError("route recovery requires finite prefix values")
    if gpu is None:
        seen = set()
        novel, remaining = [], []
        for i, prefix in enumerate(prefixes):
            key = tuple(prefix)
            if i >= initial:
                (remaining if key in seen else novel).append(i)
            seen.add(key)
        return novel + remaining
    gpu._owned()
    try:
        native = gpu.library.spacepdhcg_route_recovery_order
    except AttributeError as error:
        raise RuntimeError("CUDA route recovery requires a rebuilt native library") from error
    native.argtypes = [ct.c_int32, ct.c_void_p, ct.c_int32, ct.c_int32, ct.c_void_p]
    native.restype = ct.c_int
    order = np.empty(count - initial, dtype=np.int32)
    gpu._check(native(gpu.device_id, prefixes.ctypes.data, count, initial, order.ctypes.data))
    for key, value in {
        "refinement_recovery_order_calls": 1,
        "refinement_recovery_upload_bytes": prefixes.nbytes,
        "refinement_recovery_download_bytes": order.nbytes,
    }.items():
        gpu.telemetry[key] = gpu.telemetry.get(key, 0) + value
    return order.tolist()
