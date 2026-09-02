"""Lambert leg evaluation for GTOC12 screening.

Two implementations share one algorithm -- the universal-variable zero-revolution solver of
``cpp/include/spacepdhcg/orbitweaver/lambert.hpp`` (bracketing scan over ``z`` followed by
bisection):

* :func:`lambert_batch` is a vectorised NumPy port used for catalogue-scale screening;
* :class:`NativeLambert` compiles the repository C API on demand (``g++``) and calls
  ``spacepdhcg_lambert_zero_revolution`` / ``spacepdhcg_lambert_family_batch_cpu`` through
  ``ctypes``.  It is the CPU-parity truth path and the parity test asserts both agree.

The GPU Lambert batch (``spacepdhcg_orbitweaver_lambert_evaluate_async``) is deliberately not
used here: this track is CPU-first while the G4 campaign owns the device.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .constants import MU_SUN_KM3_S2

FloatArray = NDArray[np.float64]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_PI = np.pi


def _stumpff(z: FloatArray) -> tuple[FloatArray, FloatArray]:
    c = np.empty_like(z)
    s = np.empty_like(z)
    positive = z > 1e-8
    negative = z < -1e-8
    small = ~(positive | negative)
    root = np.sqrt(z[positive])
    c[positive] = (1.0 - np.cos(root)) / z[positive]
    s[positive] = (root - np.sin(root)) / root**3
    root = np.sqrt(-z[negative])
    c[negative] = (np.cosh(root) - 1.0) / (-z[negative])
    s[negative] = (np.sinh(root) - root) / root**3
    zs = z[small]
    c[small] = 0.5 - zs / 24.0 + zs**2 / 720.0 - zs**3 / 40320.0
    s[small] = 1.0 / 6.0 - zs / 120.0 + zs**2 / 5040.0 - zs**3 / 362880.0
    return c, s


def _residual(z, r1, r2, a_geom, tof, mu):
    """Universal-variable time residual; NaN where the evaluation is invalid (C<=0 or y<0)."""

    z, r1, r2, a_geom, tof = (
        np.ascontiguousarray(item, dtype=np.float64)
        for item in np.broadcast_arrays(z, r1, r2, a_geom, tof)
    )
    c, s = _stumpff(z)
    with np.errstate(invalid="ignore", divide="ignore"):
        valid = np.isfinite(c) & np.isfinite(s) & (c > 0.0)
        root_c = np.sqrt(np.where(valid, c, 1.0))
        y = r1 + r2 + a_geom * (z * s - 1.0) / root_c
        valid &= np.isfinite(y) & (y >= 0.0)
        x = np.sqrt(np.where(valid, y, 0.0) / np.where(valid, c, 1.0))
        time = (x**3 * s + a_geom * np.sqrt(np.where(valid, y, 0.0))) / np.sqrt(mu)
        residual = np.where(valid, time - tof, np.nan)
    return residual, np.where(valid, y, np.nan)


@dataclass(slots=True)
class LambertBatchResult:
    departure_velocity: FloatArray  # (n, 3) km/s
    arrival_velocity: FloatArray
    universal_parameter: FloatArray
    transfer_angle: FloatArray
    residual: FloatArray
    feasible: NDArray[np.bool_]


def lambert_batch(
    departure_position: FloatArray,
    arrival_position: FloatArray,
    time_of_flight_s: FloatArray,
    mu: float = MU_SUN_KM3_S2,
    *,
    long_way: bool | NDArray[np.bool_] = False,
    time_tolerance: float = 1e-8,
    scan_samples: int = 8192,
    maximum_iterations: int = 256,
) -> LambertBatchResult:
    """Vectorised zero-revolution Lambert solve (same bracketing scan + bisection as the C++)."""

    r1v = np.atleast_2d(np.asarray(departure_position, dtype=np.float64))
    r2v = np.atleast_2d(np.asarray(arrival_position, dtype=np.float64))
    tof = np.broadcast_to(np.asarray(time_of_flight_s, dtype=np.float64), (r1v.shape[0],)).astype(
        np.float64
    )
    long = np.broadcast_to(np.asarray(long_way, dtype=bool), (r1v.shape[0],))
    n = r1v.shape[0]
    r1 = np.linalg.norm(r1v, axis=1)
    r2 = np.linalg.norm(r2v, axis=1)
    cosine = np.clip(np.einsum("ij,ij->i", r1v, r2v) / (r1 * r2), -1.0, 1.0)
    sine = np.sqrt(np.maximum(0.0, 1.0 - cosine * cosine))
    sine = np.where(long, -sine, sine)
    denominator = 1.0 - cosine
    feasible = (
        (tof > 0.0) & (denominator > 1e-14) & (np.abs(sine) > 1e-14) & (r1 > 0.0) & (r2 > 0.0)
    )
    a_geom = sine * np.sqrt(r1 * r2 / np.where(denominator > 1e-14, denominator, 1.0))
    feasible &= np.abs(a_geom) > 1e-14

    lower_scan = -4.0 * _PI * _PI
    upper_scan = 4.0 * _PI * _PI - 1e-8
    grid = lower_scan + (upper_scan - lower_scan) * np.arange(scan_samples + 1) / scan_samples
    lower = np.zeros(n)
    upper = np.zeros(n)
    bracketed = np.zeros(n, dtype=bool)
    exact = np.full(n, np.nan)
    chunk = max(1, int(2_000_000 // (scan_samples + 1)))
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        idx = slice(start, stop)
        residual, _ = _residual(
            grid[None, :],
            r1[idx, None],
            r2[idx, None],
            a_geom[idx, None],
            tof[idx, None],
            mu,
        )
        # first grid point with |residual| <= tol counts as an exact root (C++ semantics)
        exact_mask = np.abs(residual) <= time_tolerance
        has_exact = exact_mask.any(axis=1)
        first_exact = np.argmax(exact_mask, axis=1)
        # sign change between consecutive *valid* samples: forward-fill the last valid sample
        valid = np.isfinite(residual)
        sign = np.sign(np.where(valid, residual, 0.0))
        # index of previous valid sample
        positions = np.where(valid, np.arange(scan_samples + 1)[None, :], -1)
        previous = np.maximum.accumulate(positions, axis=1)
        previous_shifted = np.concatenate(
            (np.full((stop - start, 1), -1), previous[:, :-1]), axis=1
        )
        has_previous = previous_shifted >= 0
        rows = np.arange(stop - start)[:, None]
        previous_sign = np.where(has_previous, sign[rows, np.maximum(previous_shifted, 0)], 0.0)
        change = valid & has_previous & (sign * previous_sign < 0.0)
        # the C++ stops at whichever comes first: exact root or sign change
        first_change = np.where(change.any(axis=1), np.argmax(change, axis=1), scan_samples + 5)
        first_exact_index = np.where(has_exact, first_exact, scan_samples + 5)
        use_exact = has_exact & (first_exact_index <= first_change)
        use_change = change.any(axis=1) & ~use_exact
        exact[idx] = np.where(use_exact, grid[np.minimum(first_exact, scan_samples)], np.nan)
        prev_index = previous_shifted[
            np.arange(stop - start), np.minimum(first_change, scan_samples)
        ]
        lower[idx] = np.where(use_change, grid[np.maximum(prev_index, 0)], 0.0)
        upper[idx] = np.where(use_change, grid[np.minimum(first_change, scan_samples)], 0.0)
        bracketed[idx] = use_change
    feasible &= bracketed | np.isfinite(exact)
    root = np.where(np.isfinite(exact), exact, 0.5 * (lower + upper))
    active = feasible & bracketed
    lower_residual, _ = _residual(lower, r1, r2, a_geom, tof, mu)
    lo = lower.copy()
    hi = upper.copy()
    for _ in range(maximum_iterations):
        if not active.any():
            break
        mid = 0.5 * (lo + hi)
        mid_residual, _ = _residual(mid, r1, r2, a_geom, tof, mu)
        invalid = active & ~np.isfinite(mid_residual)
        lo = np.where(invalid, mid, lo)
        converged = (
            active
            & np.isfinite(mid_residual)
            & ((np.abs(mid_residual) <= time_tolerance) | (np.abs(hi - lo) <= 1e-13))
        )
        root = np.where(active & np.isfinite(mid_residual), mid, root)
        opposite = active & np.isfinite(mid_residual) & (lower_residual * mid_residual < 0.0)
        hi = np.where(opposite & ~converged, mid, hi)
        same = active & np.isfinite(mid_residual) & ~opposite
        lo = np.where(same & ~converged, mid, lo)
        lower_residual = np.where(same & ~converged, mid_residual, lower_residual)
        active &= ~converged
    feasible &= ~active
    residual, y = _residual(root, r1, r2, a_geom, tof, mu)
    feasible &= np.isfinite(residual)
    y = np.where(np.isfinite(y), y, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = 1.0 - y / r1
        g = a_geom * np.sqrt(y / mu)
        g_dot = 1.0 - y / r2
        feasible &= np.isfinite(g) & (np.abs(g) > 1e-14)
        g_safe = np.where(np.abs(g) > 1e-14, g, 1.0)
        v1 = (r2v - f[:, None] * r1v) / g_safe[:, None]
        v2 = (g_dot[:, None] * r2v - r1v) / g_safe[:, None]
    angle = np.arccos(cosine)
    angle = np.where(long, 2.0 * _PI - angle, angle)
    v1 = np.where(feasible[:, None], v1, np.nan)
    v2 = np.where(feasible[:, None], v2, np.nan)
    return LambertBatchResult(v1, v2, root, angle, np.where(feasible, residual, np.nan), feasible)


# --- native parity path ---------------------------------------------------------------------


class _LambertResult(ctypes.Structure):
    _fields_ = [
        ("departure_velocity", ctypes.c_double * 3),
        ("arrival_velocity", ctypes.c_double * 3),
        ("universal_parameter", ctypes.c_double),
        ("transfer_angle_radians", ctypes.c_double),
        ("iterations", ctypes.c_uint64),
        ("time_of_flight_residual", ctypes.c_double),
    ]


class _FamilyRequest(ctypes.Structure):
    _fields_ = [
        ("deterministic_id", ctypes.c_uint64),
        ("departure_position", ctypes.c_double * 3),
        ("arrival_position", ctypes.c_double * 3),
        ("time_of_flight", ctypes.c_double),
        ("gravitational_parameter", ctypes.c_double),
        ("time_tolerance", ctypes.c_double),
        ("maximum_iterations", ctypes.c_uint64),
        ("maximum_revolutions", ctypes.c_uint64),
        ("scan_samples_per_band", ctypes.c_uint64),
        ("include_short_way", ctypes.c_int),
        ("include_long_way", ctypes.c_int),
    ]


class _FamilyResult(ctypes.Structure):
    _fields_ = [
        ("deterministic_id", ctypes.c_uint64),
        ("input_index", ctypes.c_uint64),
        ("family_index", ctypes.c_uint64),
        ("revolutions", ctypes.c_uint64),
        ("long_way", ctypes.c_int),
        ("parameter_branch", ctypes.c_int),
        ("status", ctypes.c_int),
        ("solution", _LambertResult),
    ]


def default_native_library_path() -> Path:
    return REPOSITORY_ROOT / "build" / "gtoc12" / "libspacepdhcg_c_api.so"


def compile_native_library(destination: Path | None = None) -> Path:
    """Compile ``cpp/src/c_api.cpp`` into a shared library (mirrors ``tests/test_cpp_c_api.py``)."""

    destination = destination or default_native_library_path()
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        raise RuntimeError("a C++20 compiler is required for the native Lambert parity path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-shared",
            "-fPIC",
            "-I",
            str(REPOSITORY_ROOT / "cpp" / "include"),
            str(REPOSITORY_ROOT / "cpp" / "src" / "c_api.cpp"),
            "-o",
            str(destination),
        ],
        check=True,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    return destination


class NativeLambert:
    """ctypes binding to the repository's CPU Lambert kernels (independent parity truth)."""

    def __init__(self, library_path: Path | None = None) -> None:
        path = library_path or Path(
            os.environ.get("SPACEPDHCG_GTOC12_C_API", default_native_library_path())
        )
        if not path.is_file():
            path = compile_native_library(path)
        self.library = ctypes.CDLL(str(path))
        lib = self.library
        lib.spacepdhcg_c_api_version.restype = ctypes.c_uint32
        lib.spacepdhcg_last_error.restype = ctypes.c_char_p
        lib.spacepdhcg_lambert_zero_revolution.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_uint64,
            ctypes.POINTER(_LambertResult),
        ]
        lib.spacepdhcg_lambert_zero_revolution.restype = ctypes.c_int
        lib.spacepdhcg_lambert_family_result_stride.argtypes = [ctypes.c_uint64]
        lib.spacepdhcg_lambert_family_result_stride.restype = ctypes.c_size_t
        lib.spacepdhcg_lambert_family_batch_cpu.argtypes = [
            ctypes.POINTER(_FamilyRequest),
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.POINTER(_FamilyResult),
            ctypes.c_size_t,
        ]
        lib.spacepdhcg_lambert_family_batch_cpu.restype = ctypes.c_int
        if lib.spacepdhcg_c_api_version() != 1:
            raise RuntimeError("unexpected SpacePDHCG C API version")

    def solve(self, r1, r2, tof_s: float, mu: float = MU_SUN_KM3_S2, *, long_way: bool = False):
        """Single zero-revolution solve; returns ``(v1, v2, z, angle, residual)`` or ``None``."""

        result = _LambertResult()
        r1c = (ctypes.c_double * 3)(*map(float, r1))
        r2c = (ctypes.c_double * 3)(*map(float, r2))
        status = self.library.spacepdhcg_lambert_zero_revolution(
            r1c, r2c, float(tof_s), float(mu), int(long_way), 1e-8, 256, ctypes.byref(result)
        )
        if status != 0:
            return None
        return (
            np.asarray(result.departure_velocity[:]),
            np.asarray(result.arrival_velocity[:]),
            result.universal_parameter,
            result.transfer_angle_radians,
            result.time_of_flight_residual,
        )

    def family_batch(
        self, r1s, r2s, tofs, mu: float = MU_SUN_KM3_S2, *, maximum_revolutions: int = 0
    ):
        """Fixed-layout CPU batch through ``spacepdhcg_lambert_family_batch_cpu``."""

        r1s = np.atleast_2d(np.asarray(r1s, dtype=np.float64))
        r2s = np.atleast_2d(np.asarray(r2s, dtype=np.float64))
        tofs = np.broadcast_to(np.asarray(tofs, dtype=np.float64), (r1s.shape[0],))
        n = r1s.shape[0]
        requests = (_FamilyRequest * n)()
        for k in range(n):
            requests[k].deterministic_id = k
            requests[k].departure_position = (ctypes.c_double * 3)(*r1s[k])
            requests[k].arrival_position = (ctypes.c_double * 3)(*r2s[k])
            requests[k].time_of_flight = float(tofs[k])
            requests[k].gravitational_parameter = float(mu)
            requests[k].time_tolerance = 1e-8
            requests[k].maximum_iterations = 256
            requests[k].maximum_revolutions = maximum_revolutions
            requests[k].scan_samples_per_band = 8192
            requests[k].include_short_way = 1
            requests[k].include_long_way = 1
        stride = int(self.library.spacepdhcg_lambert_family_result_stride(maximum_revolutions))
        results = (_FamilyResult * (n * stride))()
        status = self.library.spacepdhcg_lambert_family_batch_cpu(
            requests, n, maximum_revolutions, results, n * stride
        )
        if status != 0:
            raise RuntimeError(self.library.spacepdhcg_last_error().decode())
        rows = []
        for k in range(n * stride):
            item = results[k]
            rows.append(
                {
                    "input_index": int(item.input_index),
                    "family_index": int(item.family_index),
                    "revolutions": int(item.revolutions),
                    "long_way": bool(item.long_way),
                    "status": int(item.status),
                    "departure_velocity": np.asarray(item.solution.departure_velocity[:]),
                    "arrival_velocity": np.asarray(item.solution.arrival_velocity[:]),
                    "universal_parameter": float(item.solution.universal_parameter),
                }
            )
        return stride, rows


def native_available() -> bool:
    try:
        NativeLambert()
    except Exception:
        return False
    return True
