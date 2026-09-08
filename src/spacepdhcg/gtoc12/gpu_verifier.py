"""Batched, independent CUDA DOP853 propagation. CPU verifier remains the oracle.

This bridge packs inputs and downloads final records; the C API also accepts
device buffers directly. It never invokes a CPU numerical propagation fallback.
"""

from __future__ import annotations

import ctypes as ct
import os
import threading
from pathlib import Path

import numpy as np

LEG = np.dtype(
    [("initial", "f8", (7,)), ("duration_s", "f8"), ("arc_offset", "i4"), ("arc_count", "i4")],
    align=True,
)
ARC = np.dtype([("sample_offset", "i4"), ("sample_count", "i4")], align=True)
SAMPLE = np.dtype([("seconds", "f8"), ("thrust", "f8", (3,))], align=True)
RESULT = np.dtype(
    [
        ("final_state", "f8", (7,)),
        ("minimum_radius_km", "f8"),
        ("status", "i4"),
        ("accepted_steps", "i4"),
        ("rejected_steps", "i4"),
        ("evaluations", "i4"),
    ],
    align=True,
)


class GpuVerifier:
    """Retained buffers. Use as a context manager on one thread and CUDA device."""

    def __init__(self, legs: int, arcs: int, samples: int):
        for value, minimum in [(legs, 1), (arcs, 0), (samples, 0)]:
            if not isinstance(value, int) or not minimum <= value <= 2**31 - 1:
                raise ValueError("capacities must fit nonnegative int32; legs must be positive")
        path = os.environ.get("SPACEPDHCG_GTOC12_CUDA_LIBRARY")
        if not path:
            raise RuntimeError("CUDA verifier requires SPACEPDHCG_GTOC12_CUDA_LIBRARY")
        self.library = ct.CDLL(str(Path(path).resolve(strict=True)))
        self._create = self.library.spacepdhcg_gtoc12_verify_create
        self._create.argtypes = [ct.c_int] * 3 + [ct.POINTER(ct.c_void_p)]
        self._create.restype = ct.c_int
        self._destroy = self.library.spacepdhcg_gtoc12_verify_destroy
        self._destroy.argtypes = [ct.POINTER(ct.c_void_p)]
        self._destroy.restype = ct.c_int
        self._run = self.library.spacepdhcg_gtoc12_verify_host
        self._run.argtypes = [
            ct.c_void_p,
            ct.c_void_p,
            ct.c_int,
            ct.c_void_p,
            ct.c_int,
            ct.c_void_p,
            ct.c_int,
            ct.c_int,
            ct.c_void_p,
        ]
        self._run.restype = ct.c_int
        self._handle = ct.c_void_p()
        self._owner = threading.get_ident()
        self._check(self._create(legs, arcs, samples, ct.byref(self._handle)))

    @staticmethod
    def _check(status):
        if status:
            raise RuntimeError(f"CUDA verifier operation failed (CUDA status {status})")

    def close(self):
        if self._handle.value:
            if threading.get_ident() != self._owner:
                raise RuntimeError("CUDA verifier must be closed on its creating thread")
            self._check(self._destroy(ct.byref(self._handle)))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def propagate(self, legs, arcs, samples, *, max_steps=1_000_000):
        if not self._handle.value:
            raise RuntimeError("CUDA verifier is closed")
        if threading.get_ident() != self._owner:
            raise RuntimeError("CUDA verifier must run on its creating thread")
        if not isinstance(max_steps, int) or not 1 <= max_steps <= 2**31 - 1:
            raise ValueError("max_steps must be a positive int32")
        arrays = []
        for value, dtype in [(legs, LEG), (arcs, ARC), (samples, SAMPLE)]:
            array = np.ascontiguousarray(value, dtype=dtype)
            if array.ndim != 1 or len(array) > 2**31 - 1:
                raise ValueError("verifier inputs must be one-dimensional int32-sized arrays")
            arrays.append(array)
        legs, arcs, samples = arrays
        output = np.empty(len(legs), dtype=RESULT)
        self._check(
            self._run(
                self._handle,
                legs.ctypes.data,
                len(legs),
                arcs.ctypes.data,
                len(arcs),
                samples.ctypes.data,
                len(samples),
                max_steps,
                output.ctypes.data,
            )
        )
        return output


def pack_legs(solutions):
    """Pack emitted burn samples, preserving coast gaps and duplicate-epoch rules."""
    from . import constants as C

    legs = np.zeros(len(solutions), dtype=LEG)
    arcs, samples = [], []
    for i, solution in enumerate(solutions):
        boundary = solution.boundary
        legs[i]["initial"] = np.r_[
            boundary.departure_position,
            solution.departure_ship_velocity_km_s(),
            boundary.initial_mass,
        ]
        legs[i]["duration_s"] = (boundary.arrival_epoch - boundary.departure_epoch) * C.DAY_S
        legs[i]["arc_offset"] = len(arcs)
        for arc in solution.burn_arcs():
            epochs, thrust = arc.interior_arrays()
            if np.any(np.diff(epochs) < 0):
                raise ValueError("burn sample epochs must not decrease")
            keep = np.r_[True, np.diff(epochs) > 0]
            epochs, thrust = epochs[keep], thrust[keep]
            arcs.append((len(samples), len(epochs)))
            samples.extend(
                (float((epoch - boundary.departure_epoch) * C.DAY_S), vector)
                for epoch, vector in zip(epochs, thrust, strict=True)
            )
        legs[i]["arc_count"] = len(arcs) - legs[i]["arc_offset"]
    return legs, np.array(arcs, dtype=ARC), np.array(samples, dtype=SAMPLE)


class GpuVerifierSession:
    """Reuse propagation buffers across calls, growing only when capacity is insufficient."""

    def __init__(self):
        self._gpu = None
        self._capacity = (0, 0, 0)
        self._owner = threading.get_ident()
        self._closed = False

    def _owned(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError("CUDA verifier session must stay on its creating thread")
        if self._closed:
            raise RuntimeError("CUDA verifier session is closed")

    def propagate(self, legs, arcs, samples, *, max_steps=1_000_000):
        self._owned()
        required = (max(1, len(legs)), len(arcs), len(samples))
        if self._gpu is None or any(a > b for a, b in zip(required, self._capacity, strict=True)):
            capacity = tuple(max(a, b) for a, b in zip(required, self._capacity, strict=True))
            if self._gpu is not None:
                self._gpu.close()
                self._gpu = None
            self._gpu = GpuVerifier(*capacity)
            self._capacity = capacity
        return self._gpu.propagate(legs, arcs, samples, max_steps=max_steps)

    def close(self):
        if self._closed:
            return
        self._owned()
        if self._gpu is not None:
            self._gpu.close()
            self._gpu = None
        self._closed = True

    def __enter__(self):
        self._owned()
        return self

    def __exit__(self, *args):
        self.close()


def certify_legs_cuda(solutions, *, workspace=None):
    """Batched numerical certificates; final fleet certification remains separate."""
    from . import constants as C
    from .low_thrust import DU_KM, LegCertificate

    if not solutions:
        return []
    legs, arcs, samples = pack_legs(solutions)
    if workspace is None:
        with GpuVerifier(len(legs), len(arcs), len(samples)) as verifier:
            output = verifier.propagate(legs, arcs, samples)
    else:
        output = workspace.propagate(legs, arcs, samples)
    if np.any(output["status"]):
        failures = [(i, int(row["status"])) for i, row in enumerate(output) if row["status"]]
        raise RuntimeError(f"CUDA independent propagation failed: {failures}")
    return [
        LegCertificate(
            position_error_km=float(
                np.linalg.norm(row["final_state"][:3] - solution.boundary.arrival_position)
            ),
            velocity_error_km_s=float(
                np.linalg.norm(row["final_state"][3:6] - solution.arrival_ship_velocity_km_s())
            ),
            final_mass_kg=float(row["final_state"][6]),
            minimum_sun_distance_au=float(row["minimum_radius_km"] / C.AU_KM),
            maximum_thrust_n=float(np.max(np.linalg.norm(solution.thrust_n, axis=1))),
            rk4_vs_dop853_km=float(
                np.linalg.norm(row["final_state"][:3] - solution.states_scaled[-1, :3] * DU_KM)
            ),
        )
        for solution, row in zip(solutions, output, strict=True)
    ]


def pack_solution(solution):
    """Pack event-to-event legs; return descriptors and matching event metadata."""
    from . import constants as C
    from .solution import BurnArc, Event

    descriptors, arc_rows, sample_rows, metadata = [], [], [], []
    for ship in solution.ships:
        if not ship.items or not isinstance(ship.items[0], Event):
            continue  # The mission rule verifier reports malformed launch records.
        previous, burns = ship.items[0], []
        for item in ship.items[1:]:
            if isinstance(item, BurnArc):
                burns.append(item)
                continue
            initial = previous.after
            offset = len(arc_rows)
            for arc in sorted(burns, key=lambda item: item.start):
                epochs, thrust = arc.interior_arrays()
                if np.any(np.diff(epochs) < 0):
                    raise ValueError("burn sample epochs must not decrease")
                keep = np.r_[True, np.diff(epochs) > 0]
                epochs, thrust = epochs[keep], thrust[keep]
                arc_rows.append((len(sample_rows), len(epochs)))
                sample_rows.extend(
                    (float((epoch - initial.epoch) * C.DAY_S), vector)
                    for epoch, vector in zip(epochs, thrust, strict=True)
                )
            descriptors.append(
                (
                    np.r_[initial.position, initial.velocity, initial.mass],
                    (item.epoch - initial.epoch) * C.DAY_S,
                    offset,
                    len(arc_rows) - offset,
                )
            )
            metadata.append((ship.ship_id, previous, item, burns))
            previous, burns = item, []
    return (
        np.array(descriptors, dtype=LEG),
        np.array(arc_rows, dtype=ARC),
        np.array(sample_rows, dtype=SAMPLE),
        metadata,
    )


def propagate_solution_cuda(solution):
    """Batched propagation records for the existing independent mission checker."""
    from . import constants as C
    from .verifier import LegCheck

    legs, arcs, samples, metadata = pack_solution(solution)
    if not len(legs):
        return {}
    with GpuVerifier(len(legs), len(arcs), len(samples)) as gpu:
        output = gpu.propagate(legs, arcs, samples)
    if np.any(output["status"]):
        failures = [(i, int(row["status"])) for i, row in enumerate(output) if row["status"]]
        raise RuntimeError(f"CUDA independent propagation failed: {failures}")
    records = {}
    for row, (sid, previous, target, burns) in zip(output, metadata, strict=True):
        state = row["final_state"]
        records[(sid, id(previous), id(target))] = LegCheck(
            ship_id=sid,
            from_event=previous.event_id,
            to_event=target.event_id,
            from_line=previous.after.line_number,
            to_line=target.before.line_number,
            start_epoch=previous.after.epoch,
            end_epoch=target.epoch,
            burn_count=len(burns),
            position_error_km=float(np.linalg.norm(state[:3] - target.before.position)),
            velocity_error_km_s=float(np.linalg.norm(state[3:6] - target.before.velocity)),
            mass_error_kg=float(abs(state[6] - target.before.mass)),
            minimum_sun_distance_au=float(row["minimum_radius_km"] / C.AU_KM),
        )
    return records
