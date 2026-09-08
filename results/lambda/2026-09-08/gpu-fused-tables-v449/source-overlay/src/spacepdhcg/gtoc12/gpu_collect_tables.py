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

    def minimum_propellant(self, table, source, target, lo, hi, maximum_tof, mass):
        """Price a harvest window on the device and download one scalar."""
        from .gpu_collect_dp import Policy

        self.gpu._owned()
        try:
            minimum = self.gpu.library.spacepdhcg_collect_table_min_propellant
        except AttributeError as error:
            raise RuntimeError(
                "CUDA harvest-window pricing requires a rebuilt native library"
            ) from error
        minimum.argtypes = [
            ct.c_void_p,
            ct.POINTER(Policy),
            ct.c_int32,
            ct.c_int32,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.c_void_p,
            ct.POINTER(ct.c_double),
        ]
        minimum.restype = ct.c_int
        s = table.settings
        fit = s.inflation_fit
        policy = Policy()
        policy.hop_model = 2 if fit else int(s.hop_inflation_slope is not None)
        policy.thrust = C.THRUST_MAX_N
        policy.exhaust = C.ISP_S * C.G0_M_S2 * 1e-3
        policy.hop_ratio = s.hop_authority_ratio
        policy.hop_flat = s.hop_inflation
        policy.hop_floor = s.hop_inflation_floor
        policy.hop_slope = s.hop_inflation_slope or 0.0
        geometry_a, geometry_l = 0.0, None
        if fit:
            policy.fit_floor = fit.floor
            policy.fit_coefficients[:] = fit.coefficients
            geometry_a, longitude = table.pair_geometry(source, target, table.epochs[lo:hi])
            geometry_l = np.ascontiguousarray(longitude, dtype=np.float64)
        result = ct.c_double()
        self.gpu._check(
            minimum(
                self.handle,
                ct.byref(policy),
                lo,
                hi,
                maximum_tof,
                mass,
                geometry_a,
                None if geometry_l is None else geometry_l.ctypes.data,
                ct.byref(result),
            )
        )
        telemetry = self.gpu.telemetry
        telemetry["collect_window_queries"] = telemetry.get("collect_window_queries", 0) + 1
        telemetry["collect_window_download_bytes"] = (
            telemetry.get("collect_window_download_bytes", 0) + 8
        )
        telemetry["collect_window_geometry_upload_bytes"] = telemetry.get(
            "collect_window_geometry_upload_bytes", 0
        ) + (0 if geometry_l is None else geometry_l.nbytes)
        return result.value

    def close(self):
        if self.handle.value:
            self.gpu._check(self.destroy(self.handle))
            self.handle = ct.c_void_p()

    def __del__(self):
        if getattr(self, "handle", None) and self.handle.value:
            self.close()
