"""Batched CUDA orbital states with immutable elements and retained buffers."""

from __future__ import annotations

import ctypes as ct
import operator
import os
import threading
from pathlib import Path

import numpy as np

from .constants import MU_SUN_KM3_S2
from .gpu_lambert import Elements, body_elements

REQUEST = np.dtype([("body", "i4"), ("reserved", "i4"), ("epoch", "f8")], align=True)
RESULT = np.dtype(
    [("position", "f8", (3,)), ("velocity", "f8", (3,)), ("status", "i4"), ("reserved", "i4")],
    align=True,
)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error


class GpuEphemeris:
    """One thread/device, one immutable element upload, bounded repeated batches.

    The host bridge returns states for existing LegBoundary consumers. Native
    callers can use the companion launch_device API without either transfer.
    """

    def __init__(self, catalogue, body_ids, capacity, *, device_id=0):
        capacity = _integer(capacity, "capacity")
        device_id = _integer(device_id, "device_id")
        ids = tuple(dict.fromkeys(_integer(body, "body ID") for body in body_ids))
        if not 1 <= capacity <= 2**31 - 1 or not 0 <= device_id <= 2**31 - 1:
            raise ValueError("capacity must be positive int32 and device_id nonnegative int32")
        if not ids or len(ids) > 2**31 - 1 or any(body < 0 for body in ids):
            raise ValueError("nonnegative Earth/asteroid IDs are required")
        elements = (Elements * len(ids))(*(body_elements(catalogue, body) for body in ids))
        path = os.environ.get("SPACEPDHCG_GTOC12_CUDA_LIBRARY")
        if not path:
            raise RuntimeError("CUDA ephemeris requires SPACEPDHCG_GTOC12_CUDA_LIBRARY")
        self.library = ct.CDLL(str(Path(path).resolve(strict=True)))
        try:
            create = self.library.spacepdhcg_gtoc12_ephemeris_create
            self._evaluate = self.library.spacepdhcg_gtoc12_ephemeris_host
            self._destroy = self.library.spacepdhcg_gtoc12_ephemeris_destroy
        except AttributeError as error:
            raise RuntimeError("CUDA route ephemerides require a rebuilt native library") from error
        create.argtypes = [
            ct.POINTER(Elements),
            ct.c_int32,
            ct.c_int32,
            ct.c_double,
            ct.c_int32,
            ct.POINTER(ct.c_void_p),
        ]
        self._evaluate.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int32, ct.c_void_p]
        self._destroy.argtypes = [ct.POINTER(ct.c_void_p)]
        for function in (create, self._evaluate, self._destroy):
            function.restype = ct.c_int
        self.handle = ct.c_void_p()
        self.owner = threading.get_ident()
        self.capacity = capacity
        self._body_index = {body: index for index, body in enumerate(ids)}
        self._requests = np.zeros(capacity, dtype=REQUEST)
        self._results = np.empty(capacity, dtype=RESULT)
        self.telemetry = {
            "backend": "cuda",
            "batches": 0,
            "state_requests": 0,
            "element_upload_bytes": ct.sizeof(elements),
            "request_upload_bytes": 0,
            "result_download_bytes": 0,
            "CPU_ephemeris_calls": 0,
        }
        self._check(
            create(elements, len(ids), capacity, MU_SUN_KM3_S2, device_id, ct.byref(self.handle))
        )

    @staticmethod
    def _check(status):
        if status:
            raise RuntimeError(f"CUDA ephemeris returned status {status}")

    def _owned(self):
        if not self.handle.value:
            raise RuntimeError("CUDA ephemeris workspace is closed")
        if threading.get_ident() != self.owner:
            raise RuntimeError("CUDA ephemeris workspace belongs to another thread")

    def states(self, body_ids, epochs):
        self._owned()
        ids = tuple(_integer(body, "body ID") for body in body_ids)
        times = np.asarray(epochs, dtype=np.float64)
        count = len(ids)
        if count > self.capacity or times.shape != (count,) or not np.all(np.isfinite(times)):
            raise ValueError("finite epoch vector and a batch within capacity are required")
        try:
            indices = [self._body_index[body] for body in ids]
        except KeyError as error:
            raise ValueError("body ID is absent from this immutable workspace") from error
        if not count:
            return np.empty((0, 3)), np.empty((0, 3))
        requests = self._requests[:count]
        requests["body"] = indices
        requests["epoch"] = times
        self._check(
            self._evaluate(self.handle, requests.ctypes.data, count, self._results.ctypes.data)
        )
        self.telemetry["batches"] += 1
        self.telemetry["state_requests"] += count
        self.telemetry["request_upload_bytes"] += count * REQUEST.itemsize
        self.telemetry["result_download_bytes"] += count * RESULT.itemsize
        results = self._results[:count]
        if np.any(results["status"] != 0) or not (
            np.all(np.isfinite(results["position"])) and np.all(np.isfinite(results["velocity"]))
        ):
            raise RuntimeError("CUDA ephemeris failed to produce finite converged orbital states")
        # Consumers keep boundaries across later batches; returned arrays do not
        # alias the workspace's overwritten transfer buffers.
        return results["position"].copy(), results["velocity"].copy()

    def close(self):
        if self.handle.value:
            self._owned()
            self._check(self._destroy(ct.byref(self.handle)))

    def __enter__(self):
        self._owned()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if getattr(self, "handle", None) and self.handle.value:
            try:
                self.close()
            except Exception:
                pass
