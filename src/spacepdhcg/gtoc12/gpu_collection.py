"""CUDA collection option pricing and deterministic selection."""

import ctypes as ct

import numpy as np

from . import constants as C


class Query(ct.Structure):
    _fields_ = [
        ("mode", ct.c_int32),
        ("ratio_inflation", ct.c_int32),
        *[
            (name, ct.c_double)
            for name in (
                "mass",
                "epoch",
                "max_span",
                "authority_ratio",
                "inflation",
                "floor",
                "slope",
                "wait_penalty",
                "penalty_scale",
                "thrust",
                "day_seconds",
                "year_days",
                "mining_rate",
                "exhaust_velocity",
            )
        ],
    ]


class Result(ct.Structure):
    _fields_ = [("index", ct.c_int32), ("status", ct.c_int32), ("cost", ct.c_double)]


class GpuCollection:
    def __init__(self, library, device):
        self.device = device
        self.handle = ct.c_void_p()
        self.capacity = 0
        self.create = library.spacepdhcg_gtoc12_collection_create
        self.create.argtypes = [ct.c_int32, ct.c_int32, ct.POINTER(ct.c_void_p)]
        self.create.restype = ct.c_int
        self.evaluate = library.spacepdhcg_gtoc12_collection_host
        self.evaluate.argtypes = [
            ct.c_void_p,
            ct.c_void_p,
            ct.c_int32,
            ct.POINTER(Query),
            ct.POINTER(Result),
        ]
        self.evaluate.restype = ct.c_int
        self.destroy = library.spacepdhcg_gtoc12_collection_destroy
        self.destroy.argtypes = [ct.POINTER(ct.c_void_p)]
        self.destroy.restype = ct.c_int

    @staticmethod
    def _check(status):
        if status:
            raise RuntimeError(f"CUDA collection selection failed (native status {status})")

    def close(self):
        if self.handle.value:
            self._check(self.destroy(ct.byref(self.handle)))

    def select(
        self, options, mass, epoch, settings, penalty_scale=1.0, max_span=np.inf, *, first=False
    ):
        from .gpu_options import GpuResidentOptions

        resident = isinstance(options, GpuResidentOptions)
        if resident:
            options.owned()
            if options.gpu.device_id != self.device:
                raise ValueError("CUDA option table belongs to another device")
        rows = None if resident else np.ascontiguousarray(options, dtype=np.float64).reshape(-1, 3)
        count = len(options) if resident else len(rows)
        if count > 2**31 - 1:
            raise ValueError("collection option count exceeds int32")
        if not self.handle.value or count > self.capacity:
            self.close()
            self.capacity = max(256, count)
            self._check(self.create(self.capacity, self.device, ct.byref(self.handle)))
        query = Query(
            int(first),
            int(settings.hop_inflation_slope is not None),
            mass,
            epoch,
            max_span,
            settings.earth_return_authority_ratio if first else settings.hop_authority_ratio,
            settings.hop_inflation,
            settings.hop_inflation_floor,
            settings.hop_inflation_slope or 0.0,
            settings.wait_penalty,
            penalty_scale,
            C.THRUST_MAX_N,
            C.DAY_S,
            C.YEAR_DAYS,
            C.MINING_RATE_KG_PER_YEAR,
            C.ISP_S * C.G0_M_S2 * 1e-3,
        )
        result = Result()
        if resident:
            native = options.gpu.library.spacepdhcg_gtoc12_collection_resident
            native.argtypes = [
                ct.c_void_p,
                ct.c_void_p,
                ct.POINTER(Query),
                ct.POINTER(Result),
                ct.c_void_p,
            ]
            native.restype = ct.c_int
            winner = np.empty(3, dtype=np.float64)
            self._check(
                native(
                    self.handle,
                    options.handle,
                    ct.byref(query),
                    ct.byref(result),
                    winner.ctypes.data,
                )
            )
            if result.status or not -1 <= result.index < count:
                raise RuntimeError("CUDA collection query or option is invalid")
            return result.cost, None if result.index == -1 else tuple(float(x) for x in winner)
        self._check(
            self.evaluate(self.handle, rows.ctypes.data, count, ct.byref(query), ct.byref(result))
        )
        if result.status or not -1 <= result.index < count:
            raise RuntimeError("CUDA collection query or option is invalid")
        return result.cost, None if result.index == -1 else tuple(
            float(x) for x in rows[result.index]
        )
