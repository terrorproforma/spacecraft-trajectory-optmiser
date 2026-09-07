"""Retained CUDA Lambert workspace used by candidate screening; no CPU fallback."""

from __future__ import annotations

import ctypes as ct
import os
import threading
from pathlib import Path

import numpy as np


class Device(ct.Structure):
    _fields_ = [("type", ct.c_int32), ("id", ct.c_int32)]


class Stream(ct.Structure):
    _fields_ = [("device", Device), ("native_handle", ct.c_size_t)]


class Config(ct.Structure):
    _fields_ = [
        ("abi_version", ct.c_uint32),
        ("device_id", ct.c_uint32),
        ("maximum_batch_size", ct.c_size_t),
        ("revolutions", ct.c_uint32),
        ("scan_samples", ct.c_uint32),
    ]


class Elements(ct.Structure):
    _fields_ = [
        (key, ct.c_double)
        for key in ["epoch", "a", "e", "inclination", "node", "perihelion", "mean"]
    ]


class HopElements(ct.Structure):
    _fields_ = [
        ("departure", Elements),
        ("arrival", Elements),
        ("mu", ct.c_double),
        ("departure_allowance", ct.c_double),
        ("arrival_allowance", ct.c_double),
    ]


def body_elements(catalogue, body):
    from . import constants as C

    if body == 0:
        earth = C.EARTH
        return Elements(
            earth.epoch_mjd,
            earth.semi_major_axis_km,
            earth.eccentricity,
            *np.deg2rad(
                [
                    earth.inclination_deg,
                    earth.ascending_node_deg,
                    earth.argument_of_perihelion_deg,
                    earth.mean_anomaly_deg,
                ]
            ),
        )
    i = int(catalogue.index_of(body))
    return Elements(
        *(
            float(getattr(catalogue, field)[i])
            for field in [
                "epoch_mjd",
                "semi_major_axis_km",
                "eccentricity",
                "inclination_rad",
                "ascending_node_rad",
                "argument_of_perihelion_rad",
                "mean_anomaly_rad",
            ]
        )
    )


REQUEST = np.dtype(
    [
        ("id", "u8"),
        ("r1", "f8", (3,)),
        ("r2", "f8", (3,)),
        ("tof", "f8"),
        ("mu", "f8"),
        ("tolerance", "f8"),
        ("iterations", "u4"),
        ("revolutions", "u4"),
        ("short", "i4"),
        ("long", "i4"),
    ],
    align=True,
)
RESULT = np.dtype(
    [
        ("id", "u8"),
        ("input", "u4"),
        ("family", "u4"),
        ("revolutions", "u4"),
        ("long", "i4"),
        ("branch", "i4"),
        ("status", "i4"),
        ("v1", "f8", (3,)),
        ("v2", "f8", (3,)),
        ("z", "f8"),
        ("angle", "f8"),
        ("iterations", "u4"),
        ("residual", "f8"),
    ],
    align=True,
)


HOP_REQUEST = np.dtype(
    [
        ("lambert", REQUEST),
        ("v1_body", "f8", (3,)),
        ("v2_body", "f8", (3,)),
        ("departure_allowance", "f8"),
        ("arrival_allowance", "f8"),
    ],
    align=True,
)
HOP_RESULT = np.dtype(
    [
        ("v1", "f8", (3,)),
        ("v2", "f8", (3,)),
        ("dep", "f8"),
        ("arr", "f8"),
        ("feasible", "i4"),
        ("long", "i4"),
    ],
    align=True,
)


class GpuLambert:
    """One thread/device, bounded batches; retains the most recently used scan grid."""

    def __init__(self, maximum_batch_size=16384, device_id=0):
        if not isinstance(maximum_batch_size, int) or not 1 <= maximum_batch_size <= 2**31 - 1:
            raise ValueError("maximum_batch_size must be a positive int32")
        if not isinstance(device_id, int) or not 0 <= device_id <= 2**31 - 1:
            raise ValueError("device_id must be a nonnegative int32")
        path = os.environ.get("SPACEPDHCG_GTOC12_CUDA_LIBRARY")
        if not path:
            raise RuntimeError("CUDA Lambert requires SPACEPDHCG_GTOC12_CUDA_LIBRARY")
        self.library = ct.CDLL(str(Path(path).resolve(strict=True)))
        self.create = self.library.spacepdhcg_orbitweaver_lambert_workspace_create
        self.create.argtypes = [ct.POINTER(Config), Stream, ct.POINTER(ct.c_void_p)]
        self.create.restype = ct.c_int
        self.evaluate = self.library.spacepdhcg_orbitweaver_lambert_screening_host
        self.evaluate.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t]
        self.evaluate.restype = ct.c_int
        self.evaluate_hops = self.library.spacepdhcg_orbitweaver_hop_screening_host
        self.evaluate_hops.argtypes = self.evaluate.argtypes
        self.evaluate_hops.restype = ct.c_int
        self.finish = self.library.spacepdhcg_orbitweaver_lambert_workspace_finish
        self.finish.argtypes = [ct.c_void_p]
        self.finish.restype = ct.c_int
        self.destroy = self.library.spacepdhcg_orbitweaver_lambert_workspace_destroy
        self.destroy.argtypes = [ct.POINTER(ct.c_void_p)]
        self.destroy.restype = ct.c_int
        self.capacity, self.device_id = maximum_batch_size, device_id
        self.stream = Stream(Device(2, device_id), 0)
        self.handle = ct.c_void_p()
        self.scan_samples = None
        self.owner = threading.get_ident()
        self.closed = False
        self.resident_retime_tables = True
        self.requests = np.zeros(maximum_batch_size, dtype=REQUEST)
        self.results = np.zeros((maximum_batch_size, 2), dtype=RESULT)
        self.hops = np.zeros(maximum_batch_size, dtype=HOP_REQUEST)
        self.batches = self.evaluations = 0
        self.neighbour_workspace = None
        self.neighbour_key = None
        self.collection_workspace = None
        self.retime_workspace = None
        self.telemetry = {
            "backend": "cuda",
            "completed_batches": 0,
            "completed_branch_requests": 0,
            "gpu_used": False,
        }

    @staticmethod
    def _check(status):
        if status:
            raise RuntimeError(f"CUDA Lambert failed (native status {status})")

    def _owned(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError("CUDA Lambert must be used on its creating thread")
        if self.closed:
            raise RuntimeError("CUDA Lambert workspace is closed")

    def close(self):
        if self.closed:
            return
        self._owned()
        if self.retime_workspace is not None:
            self.retime_workspace.close()
            self.retime_workspace = None
        if self.collection_workspace is not None:
            self.collection_workspace.close()
            self.collection_workspace = None
        if self.neighbour_workspace is not None:
            self.neighbour_workspace.close()
            self.neighbour_workspace = None
        if self.handle.value:
            self._check(self.destroy(ct.byref(self.handle)))
        self.closed = True

    def __enter__(self):
        return self

    def neighbours(self, catalogue, pool, source, epoch, settings):
        from .gpu_neighbours import GpuNeighbours

        self._owned()
        if not len(pool):
            return np.empty(0, dtype=np.int64)
        # Retain one immutable snapshot. Pool/TOF changes replace it; nested
        # searches reacquire by key instead of keeping pointers to closed buffers.
        key = (id(catalogue), np.asarray(pool, dtype=np.int64).tobytes(), tuple(settings.hop_tofs))
        if self.neighbour_workspace is None or self.neighbour_key != key:
            if self.neighbour_workspace is not None:
                self.neighbour_workspace.close()
                self.neighbour_workspace = None
            self.neighbour_workspace = GpuNeighbours(
                self.library, catalogue, pool, settings.hop_tofs, self.device_id
            )
            self.neighbour_key = key
        result = self.neighbour_workspace.candidates(source, epoch, settings)
        self.telemetry["completed_neighbour_queries"] = (
            self.telemetry.get("completed_neighbour_queries", 0) + 1
        )
        self.telemetry["gpu_used"] = True
        return result

    def __exit__(self, *args):
        self.close()

    def select_collection(self, options, *args, **kwargs):
        from .gpu_collection import GpuCollection

        self._owned()
        if self.collection_workspace is None:
            self.collection_workspace = GpuCollection(self.library, self.device_id)
        result = self.collection_workspace.select(options, *args, **kwargs)
        for key, count in [
            ("completed_collection_queries", 1),
            ("completed_collection_options", len(options)),
        ]:
            self.telemetry[key] = self.telemetry.get(key, 0) + count
        self.telemetry["gpu_used"] = True
        return result

    def _prepare(self, scan_samples):
        if not isinstance(scan_samples, int) or not 16 <= scan_samples < 2**32 - 1:
            raise ValueError("scan_samples outside supported uint32 range")
        if not self.handle.value or self.scan_samples != scan_samples:
            if self.handle.value:
                self._check(self.destroy(ct.byref(self.handle)))
            config = Config(1, self.device_id, self.capacity, 0, scan_samples)
            self._check(self.create(ct.byref(config), self.stream, ct.byref(self.handle)))
            self.scan_samples = scan_samples

    def retime_dp(self, *args):
        from .gpu_retime import GpuRetime

        self._owned()
        if self.retime_workspace is None:
            self.retime_workspace = GpuRetime(self.library, self.device_id, self)
        result = self.retime_workspace.solve(*args)
        self.telemetry["completed_retime_dp_calls"] = self.retime_workspace.calls
        self.telemetry["retime_table_uploads"] = self.retime_workspace.uploads
        self.telemetry["retime_resident_builds"] = self.retime_workspace.resident_builds
        self.telemetry["retime_resident_cells"] = self.retime_workspace.resident_cells
        if self.retime_workspace.calls:
            self.telemetry["gpu_used"] = True
        return result

    def _record(self, branches):
        from .lambert import record_completed_branches

        record_completed_branches(branches)
        self.batches += 1
        self.evaluations += branches
        self.telemetry.update(
            completed_batches=self.batches,
            completed_branch_requests=self.evaluations,
            gpu_used=True,
        )

    def leg_table(self, catalogue, from_body, to_body, departures, tofs):
        """Build the ephemerides and screening requests on CUDA from orbital elements."""
        from . import constants as C

        self._owned()
        departures, tofs = (np.asarray(x, dtype=np.float64) for x in (departures, tofs))
        if departures.ndim != 1 or tofs.ndim != 1:
            raise ValueError("Departure epochs and flight durations must be vectors")
        shape = (len(departures), len(tofs))
        n = shape[0] * shape[1]
        output = np.empty(n, dtype=HOP_RESULT)
        query = HopElements(
            body_elements(catalogue, from_body),
            body_elements(catalogue, to_body),
            C.MU_SUN_KM3_S2,
            C.MAX_VINF_EARTH_KM_S if from_body == 0 else 0.0,
            C.MAX_VINF_EARTH_KM_S if to_body == 0 else 0.0,
        )
        evaluate = self.library.spacepdhcg_orbitweaver_hop_elements_host
        evaluate.argtypes = [
            ct.c_void_p,
            ct.POINTER(HopElements),
            ct.c_void_p,
            ct.c_size_t,
            ct.c_void_p,
            ct.c_size_t,
        ]
        evaluate.restype = ct.c_int
        if n:
            self._prepare(256)
        for start in range(0, n, self.capacity):
            end = min(n, start + self.capacity)
            indices = np.arange(start, end)
            times = np.column_stack((departures[indices // shape[1]], tofs[indices % shape[1]]))
            self._check(
                evaluate(
                    self.handle,
                    ct.byref(query),
                    times.ctypes.data,
                    end - start,
                    output[start:end].ctypes.data,
                    end - start,
                )
            )
            self._record(2 * (end - start))
        self.telemetry["completed_element_hops"] = (
            self.telemetry.get("completed_element_hops", 0) + n
        )
        dv = output["dep"] + output["arr"]
        feasible = output["feasible"].astype(bool) & np.isfinite(dv)
        return np.where(feasible, dv, np.inf).reshape(shape), feasible.reshape(shape)

    def screen_hops(
        self,
        r1,
        v1_body,
        r2,
        v2_body,
        departure_epoch,
        tof_days,
        *,
        departure_allowance_km_s=0.0,
        arrival_allowance_km_s=0.0,
        scan_samples=256,
    ):
        from . import constants as C
        from .screening import LambertHop

        self._owned()
        r1, r2 = (np.atleast_2d(np.asarray(v, dtype=np.float64)) for v in (r1, r2))
        if r1.ndim != 2 or r1.shape[1] != 3 or r2.shape != r1.shape:
            raise ValueError("Lambert positions must have matching (n,3) shapes")
        n = len(r1)
        v1_body, v2_body = (
            np.broadcast_to(np.asarray(v, dtype=np.float64), (n, 3)) for v in (v1_body, v2_body)
        )
        tof = np.broadcast_to(np.asarray(tof_days, dtype=np.float64), (n,))
        for allowance in (departure_allowance_km_s, arrival_allowance_km_s):
            if not np.isfinite(allowance) or allowance < 0:
                raise ValueError("velocity allowances must be finite and nonnegative")
        output = np.empty(n, dtype=HOP_RESULT)
        if n:
            self._prepare(scan_samples)
        for start in range(0, n, self.capacity):
            end = min(n, start + self.capacity)
            hops = self.hops[: end - start]
            q = hops["lambert"]
            q["id"] = np.arange(start, end)
            q["r1"], q["r2"] = r1[start:end], r2[start:end]
            q["tof"] = tof[start:end] * C.DAY_S
            q["mu"], q["tolerance"], q["iterations"] = C.MU_SUN_KM3_S2, 1e-8, 256
            hops["v1_body"], hops["v2_body"] = v1_body[start:end], v2_body[start:end]
            hops["departure_allowance"] = departure_allowance_km_s
            hops["arrival_allowance"] = arrival_allowance_km_s
            self._check(
                self.evaluate_hops(
                    self.handle,
                    hops.ctypes.data,
                    end - start,
                    output[start:end].ctypes.data,
                    end - start,
                )
            )
            self._record(2 * (end - start))
        return LambertHop(
            np.asarray(departure_epoch, dtype=np.float64),
            np.asarray(tof_days, dtype=np.float64),
            output["dep"],
            output["arr"],
            output["v1"],
            output["v2"],
            output["feasible"].astype(bool),
        )

    def solve(
        self,
        r1,
        r2,
        tof,
        mu,
        *,
        long_way=False,
        time_tolerance=1e-8,
        scan_samples=8192,
        maximum_iterations=256,
    ):
        from .lambert import LambertBatchResult

        self._owned()
        for value, minimum in [(scan_samples, 16), (maximum_iterations, 1)]:
            if not isinstance(value, int) or not minimum <= value < 2**32 - 1:
                raise ValueError("scan_samples/maximum_iterations outside supported uint32 range")
        if not np.isfinite(mu) or mu <= 0 or not np.isfinite(time_tolerance) or time_tolerance <= 0:
            raise ValueError("mu and time_tolerance must be finite and positive")
        r1, r2 = (np.atleast_2d(np.asarray(v, dtype=np.float64)) for v in (r1, r2))
        if r1.ndim != 2 or r1.shape[1] != 3 or r2.shape != r1.shape:
            raise ValueError("Lambert positions must have matching (n,3) shapes")
        n = len(r1)
        tof = np.broadcast_to(np.asarray(tof, dtype=np.float64), (n,))
        directions = np.broadcast_to(np.asarray(long_way, dtype=bool), (n,))
        v1, v2 = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
        z, angle, residual = (np.full(n, np.nan) for _ in range(3))
        feasible = np.zeros(n, dtype=bool)
        if n:
            self._prepare(scan_samples)
        for start in range(0, n, self.capacity):
            end = min(n, start + self.capacity)
            count = end - start
            request = self.requests[:count]
            request["id"] = np.arange(start, end)
            request["r1"], request["r2"] = r1[start:end], r2[start:end]
            request["tof"], request["mu"], request["tolerance"] = tof[start:end], mu, time_tolerance
            request["iterations"] = maximum_iterations
            request["short"], request["long"] = ~directions[start:end], directions[start:end]
            self._check(
                self.evaluate(
                    self.handle, request.ctypes.data, count, self.results.ctypes.data, 2 * count
                )
            )
            self._record(count)
            result = self.results[np.arange(count), directions[start:end].astype(int)]
            valid = result["status"] == 0
            feasible[start:end] = valid
            for output, field in [
                (v1, "v1"),
                (v2, "v2"),
                (z, "z"),
                (angle, "angle"),
                (residual, "residual"),
            ]:
                output[start:end] = np.where(
                    valid[:, None] if output.ndim == 2 else valid, result[field], np.nan
                )
        return LambertBatchResult(v1, v2, z, angle, residual, feasible)
