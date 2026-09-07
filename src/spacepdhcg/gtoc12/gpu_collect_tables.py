"""Immutable float32 device tables, with optional reads for host consumers."""

import ctypes as ct

import numpy as np

from . import constants as C
from .gpu_lambert import HopElements, body_elements


class GpuCollectTable:
    def __init__(self, gpu, table, source, target, tofs, end):
        self.gpu = gpu
        self.handle = ct.c_void_p()
        self.shape = (len(table.epochs), len(tofs))
        create = gpu.library.spacepdhcg_collect_table_create
        self.read_native = gpu.library.spacepdhcg_collect_table_read
        self.destroy = gpu.library.spacepdhcg_collect_table_destroy
        create.argtypes = [
            ct.c_void_p,
            ct.POINTER(HopElements),
            ct.c_void_p,
            ct.c_int32,
            ct.c_void_p,
            ct.c_int32,
            ct.c_double,
            ct.POINTER(ct.c_void_p),
        ]
        self.read_native.argtypes = [ct.c_void_p, ct.c_void_p]
        self.destroy.argtypes = [ct.c_void_p]
        for f in [create, self.read_native, self.destroy]:
            f.restype = ct.c_int
        elements = HopElements(
            body_elements(table.catalogue, source),
            body_elements(table.catalogue, target),
            C.MU_SUN_KM3_S2,
            0.0,
            C.MAX_VINF_EARTH_KM_S if target == 0 else 0.0,
        )
        epochs = np.ascontiguousarray(table.epochs, dtype=np.float64)
        tofs = np.ascontiguousarray(tofs, dtype=np.float64)
        gpu._prepare(256)
        gpu._check(
            create(
                gpu.handle,
                ct.byref(elements),
                epochs.ctypes.data,
                len(epochs),
                tofs.ctypes.data,
                len(tofs),
                end,
                ct.byref(self.handle),
            )
        )
        count = len(epochs) * len(tofs)
        gpu._record(2 * count)
        gpu.telemetry["completed_element_hops"] = (
            gpu.telemetry.get("completed_element_hops", 0) + count
        )
        gpu.telemetry["collect_resident_builds"] = (
            gpu.telemetry.get("collect_resident_builds", 0) + 1
        )
        table.lambert_evaluations += 2 * count
        gpu.collect_table_objects.add(self)

    def read(self):
        self.gpu._owned()
        result = np.empty(self.shape, dtype=np.float32)
        self.gpu._check(self.read_native(self.handle, result.ctypes.data))
        self.gpu.telemetry["collect_table_download_bytes"] = (
            self.gpu.telemetry.get("collect_table_download_bytes", 0) + result.nbytes
        )
        return result

    def close(self):
        if self.handle.value:
            self.gpu._check(self.destroy(self.handle))
            self.handle = ct.c_void_p()

    def __del__(self):
        if getattr(self, "handle", None) and self.handle.value:
            self.close()
