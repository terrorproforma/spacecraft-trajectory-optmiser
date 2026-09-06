"""Explicit CUDA interval dynamics for the GTOC12 refinement pipeline.

The native workspace and its device outputs persist across SCvx iterations.
This host bridge still downloads coefficients for the existing CPU sparse
assembly/Clarabel solve. It is not a claim of a complete GPU-native refiner.
Requesting CUDA never silently falls back to NumPy.
"""

from __future__ import annotations

import ctypes as ct
import os
import weakref
from pathlib import Path

import numpy as np

_DoublePointer = ct.POINTER(ct.c_double)


def _pointer(array):
    return array.ctypes.data_as(_DoublePointer)


class GpuDiscretisation:
    """One retained native workspace; calls on an instance must be serialized."""

    def __init__(self, model, node_times, substeps: int, hold: str = "zoh"):
        path = os.environ.get("SPACEPDHCG_GTOC12_CUDA_LIBRARY")
        if not path:
            raise RuntimeError("CUDA GTOC12 requires SPACEPDHCG_GTOC12_CUDA_LIBRARY")
        self.node_times = np.array(node_times, dtype=np.float64, order="C", copy=True)
        if self.node_times.ndim != 1 or len(self.node_times) < 2:
            raise ValueError("node_times must be a vector with at least two nodes")
        if hold not in {"zoh", "lagrange"}:
            raise ValueError("hold must be 'zoh' or 'lagrange'")
        if hold == "lagrange" and len(self.node_times) < 4:
            raise ValueError("Lagrange interpolation requires at least four nodes")
        if not isinstance(substeps, int) or not 1 <= substeps <= 2**31 - 1:
            raise ValueError("substeps must be a positive int32")
        self.substeps = substeps
        self.intervals = len(self.node_times) - 1
        if self.intervals >= 2**31 - 1:
            raise ValueError("too many nodes for CUDA discretisation")
        self.hold = hold
        self.stencils = (
            np.arange(self.intervals)[:, None]
            if hold == "zoh"
            else np.minimum(np.maximum(np.arange(self.intervals) - 1, 0), self.intervals - 3)[
                :, None
            ]
            + np.arange(4)[None, :]
        )
        self.node_times.setflags(write=False)
        self.stencils.setflags(write=False)
        self._library = ct.CDLL(str(Path(path).resolve(strict=True)))
        create = self._library.spacepdhcg_gtoc12_discretisation_create
        create.argtypes = [
            ct.c_int,
            ct.c_int,
            ct.c_double,
            ct.c_double,
            _DoublePointer,
            ct.POINTER(ct.c_void_p),
        ]
        create.restype = ct.c_int
        destroy = self._library.spacepdhcg_gtoc12_discretisation_destroy
        destroy.argtypes = [ct.c_void_p]
        destroy.restype = None
        self._evaluate = self._library.spacepdhcg_gtoc12_discretisation_evaluate_host
        self._evaluate.argtypes = [
            ct.c_void_p,
            _DoublePointer,
            _DoublePointer,
            ct.c_int,
            ct.c_int,
            _DoublePointer,
            _DoublePointer,
            _DoublePointer,
            _DoublePointer,
        ]
        self._evaluate.restype = ct.c_int
        self._workspace = ct.c_void_p()
        status = create(
            self.intervals,
            int(hold == "lagrange"),
            model.kappa,
            model.lam,
            _pointer(self.node_times),
            ct.byref(self._workspace),
        )
        if status:
            raise RuntimeError(f"CUDA GTOC12 workspace creation failed (status {status})")
        self._finalizer = weakref.finalize(self, destroy, self._workspace)

    def close(self):
        self._finalizer()

    def _run(self, states, controls, linearise):
        if not self._finalizer.alive:
            raise RuntimeError("CUDA GTOC12 workspace is closed")
        if not isinstance(self.substeps, int) or not 1 <= self.substeps <= 2**31 - 1:
            raise ValueError("substeps must be a positive int32")
        states = np.ascontiguousarray(states, dtype=np.float64)
        controls = np.ascontiguousarray(controls, dtype=np.float64)
        n = self.intervals
        if states.shape != (n + 1, 7) or controls.shape != (n + 1, 4):
            raise ValueError("CUDA GTOC12 requires states[nodes,7] and controls[nodes,4]")
        propagated = np.empty((n, 7))
        outputs = (
            (np.empty((n, 7, 7)), np.empty((n, self.stencils.shape[1], 7, 4)), np.empty((n, 7)))
            if linearise
            else ()
        )
        pointers = [_pointer(array) for array in outputs] if linearise else [None] * 3
        status = self._evaluate(
            self._workspace,
            _pointer(states),
            _pointer(controls),
            self.substeps,
            int(linearise),
            *pointers,
            _pointer(propagated),
        )
        if status:
            raise RuntimeError(f"CUDA GTOC12 evaluation failed (status {status})")
        return (*outputs, propagated) if linearise else propagated

    def linearise(self, states, controls):
        return self._run(states, controls, True)

    def propagate(self, states, controls):
        return self._run(states, controls, False)
