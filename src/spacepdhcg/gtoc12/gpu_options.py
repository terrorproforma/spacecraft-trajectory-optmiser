"""Owned immutable CUDA option rows, with explicit reads for host consumers."""

import ctypes as ct

import numpy as np


class GpuResidentOptions:
    def __init__(self, gpu):
        self.gpu = gpu
        self.handle = ct.c_void_p()
        self.count = 0
        self._host = None
        self._read = gpu.library.spacepdhcg_gtoc12_collection_options_read
        self._read.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int32]
        self._read.restype = ct.c_int
        self._destroy = gpu.library.spacepdhcg_gtoc12_collection_options_destroy
        self._destroy.argtypes = [ct.POINTER(ct.c_void_p)]
        self._destroy.restype = ct.c_int
        gpu.option_objects.add(self)

    def owned(self):
        self.gpu._owned()
        if not self.handle.value:
            raise RuntimeError("CUDA option table is closed")

    def __len__(self):
        return self.count

    def read(self):
        self.owned()
        if self._host is None:
            rows = np.empty((self.count, 3), dtype=np.float64)
            self.gpu._check(self._read(self.handle, rows.ctypes.data, self.count))
            rows.flags.writeable = False
            self._host = rows
            stats = self.gpu.telemetry
            stats["resident_option_read_bytes"] = (
                stats.get("resident_option_read_bytes", 0) + rows.nbytes
            )
        return self._host

    def __iter__(self):
        return iter(map(tuple, self.read().tolist()))

    def __getitem__(self, key):
        return self.read()[key]

    def close(self):
        if self.handle.value:
            self.gpu._owned()
            self.gpu._check(self._destroy(ct.byref(self.handle)))
        self._host = None

    def __del__(self):
        self.close()
