"""Retained CUDA catalogue snapshot for deterministic neighbour filtering/ranking."""

import ctypes as ct

import numpy as np

from . import constants as C

BODY = np.dtype(
    [
        ("id", "i8"),
        *[(key, "f8") for key in ["epoch", "a", "e", "inclination", "node", "perihelion", "mean"]],
    ],
    align=True,
)


class Query(ct.Structure):
    _fields_ = [
        ("source_index", ct.c_int32),
        ("neighbours", ct.c_int32),
        *[
            (key, ct.c_double)
            for key in ["epoch", "band_a", "band_e", "band_i", "band_phase", "filter_scale"]
        ],
    ]


class GpuNeighbours:
    """Catalogue and pool are immutable for this workspace's lifetime."""

    def __init__(self, library, catalogue, pool, tofs, device):
        self.library, self.catalogue = library, catalogue
        self.handle = ct.c_void_p()
        self.create = library.spacepdhcg_gtoc12_neighbours_create
        self.create.argtypes = [
            ct.c_void_p,
            ct.c_int32,
            ct.c_void_p,
            ct.c_int32,
            ct.c_void_p,
            ct.c_int32,
            ct.c_double,
            ct.c_double,
            ct.c_int32,
            ct.POINTER(ct.c_void_p),
        ]
        self.create.restype = ct.c_int
        self.evaluate = library.spacepdhcg_gtoc12_neighbours_host
        self.evaluate.argtypes = [
            ct.c_void_p,
            ct.POINTER(Query),
            ct.c_void_p,
            ct.c_int32,
            ct.POINTER(ct.c_int32),
        ]
        self.evaluate.restype = ct.c_int
        self.destroy = library.spacepdhcg_gtoc12_neighbours_destroy
        self.destroy.argtypes = [ct.POINTER(ct.c_void_p)]
        self.destroy.restype = ct.c_int
        bodies = np.empty(len(catalogue.ids), dtype=BODY)
        for field, source in [
            ("id", "ids"),
            ("epoch", "epoch_mjd"),
            ("a", "semi_major_axis_km"),
            ("e", "eccentricity"),
            ("inclination", "inclination_rad"),
            ("node", "ascending_node_rad"),
            ("perihelion", "argument_of_perihelion_rad"),
            ("mean", "mean_anomaly_rad"),
        ]:
            bodies[field] = getattr(catalogue, source)
        indices = np.ascontiguousarray(catalogue.index_of(pool), dtype=np.int32)
        tofs = np.ascontiguousarray(tofs, dtype=np.float64)
        if (
            not 0 < len(indices) <= np.iinfo(np.int32).max
            or not 0 < len(tofs) <= np.iinfo(np.int32).max
        ):
            raise ValueError("GPU neighbour pool and TOF grid must be nonempty int32-sized arrays")
        self.count = len(indices)
        self._check(
            self.create(
                bodies.ctypes.data,
                len(bodies),
                indices.ctypes.data,
                self.count,
                tofs.ctypes.data,
                len(tofs),
                C.MU_SUN_KM3_S2,
                C.AU_KM,
                device,
                ct.byref(self.handle),
            )
        )

    @staticmethod
    def _check(status):
        if status:
            raise RuntimeError(f"CUDA neighbour selection failed (native status {status})")

    def close(self):
        if self.handle.value:
            self._check(self.destroy(ct.byref(self.handle)))

    def candidates(self, source, epoch, settings):
        if not self.handle.value:
            raise RuntimeError("CUDA neighbour workspace is closed")
        if not isinstance(settings.neighbours, int) or not 1 <= settings.neighbours <= 2**31 - 1:
            raise ValueError("neighbours must be a positive int32")
        query = Query(
            int(self.catalogue.index_of(source)),
            settings.neighbours,
            epoch,
            settings.band_a_au,
            settings.band_e,
            settings.band_i_deg,
            settings.band_phase_deg,
            settings.filter_scale,
        )
        capacity = min(self.count, settings.neighbours + settings.neighbours // 2)
        result = np.empty(capacity, dtype=np.int64)
        count = ct.c_int32()
        self._check(
            self.evaluate(
                self.handle, ct.byref(query), result.ctypes.data, capacity, ct.byref(count)
            )
        )
        if not 0 <= count.value <= capacity:
            raise RuntimeError("CUDA neighbour query or ephemeris is invalid")
        return result[: count.value]
