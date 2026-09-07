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
        self.requests = np.zeros(maximum_batch_size, dtype=REQUEST)
        self.results = np.zeros((maximum_batch_size, 2), dtype=RESULT)
        self.batches = self.evaluations = 0
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
        if self.handle.value:
            self._check(self.destroy(ct.byref(self.handle)))
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

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
        if n and self.scan_samples != scan_samples:
            if self.handle.value:
                self._check(self.destroy(ct.byref(self.handle)))
            config = Config(1, self.device_id, self.capacity, 0, scan_samples)
            self._check(self.create(ct.byref(config), self.stream, ct.byref(self.handle)))
            self.scan_samples = scan_samples
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
            self.batches += 1
            self.evaluations += count
            self.telemetry.update(
                completed_batches=self.batches,
                completed_branch_requests=self.evaluations,
                gpu_used=True,
            )
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
