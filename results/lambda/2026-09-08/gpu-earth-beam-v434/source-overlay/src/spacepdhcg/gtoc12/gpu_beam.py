"""Compact initial Earth beam from device ephemerides, screening and ranking."""

import ctypes as ct

import numpy as np

from . import constants as C
from .gpu_lambert import Elements, body_elements


class BeamConfig(ct.Structure):
    _fields_ = [("earth", Elements)] + [
        (name, ct.c_double)
        for name in (
            "mu",
            "allowance",
            "initial_mass",
            "inflation",
            "authority_ratio",
            "thrust",
            "exhaust_velocity",
            "horizon",
            "mining_rate",
            "year_days",
            "miner_mass",
            "propellant_weight",
            "cluster_bonus",
            "density_cap",
            "seed_bonus",
        )
    ]


TARGET = np.dtype(
    [
        ("elements", "f8", (7,)),
        ("asteroid", "i8"),
        ("weight", "f8"),
        ("density", "f8"),
        ("seeded", "f8"),
    ],
    align=True,
)
OPTION = np.dtype(
    [
        ("score", "f8"),
        ("asteroid", "i8"),
        ("departure", "f8"),
        ("tof", "f8"),
        ("delta_v", "f8"),
        ("propellant", "f8"),
    ],
    align=True,
)


def earth_beam(gpu, runner, pool, seeded):
    gpu._owned()
    settings = runner.settings
    epochs = np.ascontiguousarray(settings.launch_epochs, dtype=np.float64)
    tofs = np.ascontiguousarray(settings.earth_leg_tofs, dtype=np.float64)
    if epochs.ndim != 1 or tofs.ndim != 1:
        raise ValueError("Earth beam epochs and flight durations must be vectors")
    if not len(pool) or not len(epochs) or not len(tofs) or settings.first_level_limit == 0:
        return []
    if settings.first_level_limit < 0 or settings.earth_block < 1:
        raise ValueError("Earth beam limits must be nonnegative and block size positive")
    index = runner.catalogue.index_of(pool)
    targets = np.zeros(len(pool), dtype=TARGET)
    for i, name in enumerate(
        (
            "epoch_mjd",
            "semi_major_axis_km",
            "eccentricity",
            "inclination_rad",
            "ascending_node_rad",
            "argument_of_perihelion_rad",
            "mean_anomaly_rad",
        )
    ):
        targets["elements"][:, i] = getattr(runner.catalogue, name)[index]
    targets["asteroid"] = pool
    targets["weight"] = [runner.weights.get(int(a), 1.0) for a in pool]
    clustered = runner.clusters is not None and settings.cluster_bonus_kg > 0
    if clustered:
        targets["density"] = [runner.clusters.density_of(int(a)) for a in pool]
    if seeded is not None:
        targets["seeded"] = seeded
    inflation, ratio = runner.limits("earth_out")
    config = BeamConfig(
        body_elements(runner.catalogue, 0),
        C.MU_SUN_KM3_S2,
        C.MAX_VINF_EARTH_KM_S,
        settings.initial_mass,
        inflation,
        ratio,
        C.THRUST_MAX_N,
        C.ISP_S * C.G0_M_S2 * 1e-3,
        C.MISSION_END_MJD - 2 * C.YEAR_DAYS,
        C.MINING_RATE_KG_PER_YEAR,
        C.YEAR_DAYS,
        C.MINER_MASS_KG,
        settings.propellant_weight,
        settings.cluster_bonus_kg if clustered else 0.0,
        settings.cluster_density_cap if clustered else 1.0,
        settings.seed_bonus_kg if seeded is not None else 0.0,
    )
    native = gpu.library.spacepdhcg_orbitweaver_earth_beam_host
    native.argtypes = [
        ct.c_void_p,
        ct.POINTER(BeamConfig),
        ct.c_void_p,
        ct.c_size_t,
        ct.c_void_p,
        ct.c_size_t,
        ct.c_void_p,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_void_p,
        ct.c_size_t,
        ct.POINTER(ct.c_size_t),
    ]
    native.restype = ct.c_int
    gpu._prepare(256)
    out = np.empty(settings.first_level_limit, dtype=OPTION)
    selected = ct.c_size_t()
    gpu._check(
        native(
            gpu.handle,
            ct.byref(config),
            targets.ctypes.data,
            len(pool),
            epochs.ctypes.data,
            len(epochs),
            tofs.ctypes.data,
            len(tofs),
            settings.earth_block,
            len(out),
            out.ctypes.data,
            len(out),
            ct.byref(selected),
        )
    )
    count = len(pool) * len(epochs) * len(tofs)
    for first in range(0, len(pool), settings.earth_block):
        rows = min(settings.earth_block, len(pool) - first) * len(epochs) * len(tofs)
        for start in range(0, rows, gpu.capacity):
            gpu._record(2 * min(rows - start, gpu.capacity))
    gpu.telemetry["completed_element_hops"] = gpu.telemetry.get("completed_element_hops", 0) + count
    gpu.telemetry["completed_earth_beam_rows"] = (
        gpu.telemetry.get("completed_earth_beam_rows", 0) + selected.value
    )
    gpu.telemetry["earth_beam_download_bytes"] = (
        gpu.telemetry.get("earth_beam_download_bytes", 0) + selected.value * OPTION.itemsize + 8
    )
    return out[: selected.value].tolist()
