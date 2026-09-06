"""Native GPU GTOC12 subproblem solve with explicit conic residual qualification."""

from __future__ import annotations

import ctypes as ct
import math
import os
import weakref
from pathlib import Path

import numpy as np

from .gpu_conic import _Dimensions, _Parameters
from .gpu_discretisation import _DoublePointer, _pointer
from .low_thrust import _ConvexProblem


class _Report(ct.Structure):
    _fields_ = (
        [(name, ct.c_int) for name in ("qualified", "qoco_status", "iterations", "failure")]
        + [
            (name, ct.c_double)
            for name in (
                "requested_tolerance",
                "primal_residual",
                "dual_residual",
                "absolute_primal_residual",
                "absolute_dual_residual",
                "setup_seconds",
                "update_seconds",
                "solve_seconds",
                "residual_seconds",
            )
        ]
        + [
            (name, ct.c_uint64)
            for name in (
                "workspace_creations",
                "numeric_updates",
                "device_numeric_updates",
                "solves",
                "adapter_d2h_count",
                "adapter_d2h_bytes",
            )
        ]
    )
    _fields_ += [
        (name, ct.c_double)
        for name in (
            "primal_objective",
            "dual_objective",
            "absolute_gap",
            "relative_gap",
        )
    ]


class GpuQocoProblem(_ConvexProblem):
    """Retained device matrices and GPU solver with host setup and outer control."""

    def __init__(
        self,
        model,
        node_times,
        hold,
        free_dep,
        free_arr,
        boundary,
        fuel_weights,
        tolerance,
        ruiz_iterations=0,
    ):
        path = os.environ.get("SPACEPDHCG_GTOC12_CUDA_LIBRARY")
        qoco = os.environ.get("SPACEPDHCG_QOCO_LIBRARY")
        if not path or not qoco or not Path(qoco).is_file():
            raise RuntimeError(
                "GPU QOCO requires SPACEPDHCG_GTOC12_CUDA_LIBRARY and SPACEPDHCG_QOCO_LIBRARY"
            )
        times = np.ascontiguousarray(node_times, dtype=np.float64)
        fuel = np.ascontiguousarray(fuel_weights, dtype=np.float64)
        bnd = np.ascontiguousarray(
            np.concatenate([boundary[k] for k in ("r0", "v0", "rf", "vf")]), dtype=np.float64
        )
        if times.ndim != 1 or len(times) < 2 or fuel.shape != times.shape or bnd.shape != (12,):
            raise ValueError("invalid node times, fuel weights or boundary dimensions")
        if hold not in {"zoh", "lagrange"}:
            raise ValueError("hold must be 'zoh' or 'lagrange'")
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("conic tolerance must be finite and positive")
        if not isinstance(ruiz_iterations, int) or not 0 <= ruiz_iterations <= 100:
            raise ValueError("Ruiz iterations must be an integer in [0,100]")
        super().__init__(len(times), bool(free_dep), bool(free_arr))
        self._library = ct.CDLL(str(Path(path).resolve(strict=True)))
        create = self._library.spacepdhcg_gtoc12_qoco_create
        create.argtypes = (
            [ct.c_int] * 4
            + [ct.c_double] * 2
            + [_DoublePointer] * 3
            + [ct.c_double, ct.c_int, ct.POINTER(ct.c_void_p)]
        )
        create.restype = ct.c_int
        destroy = self._library.spacepdhcg_gtoc12_qoco_destroy
        destroy.argtypes, destroy.restype = [ct.c_void_p], None
        dimensions = self._library.spacepdhcg_gtoc12_qoco_get_dimensions
        dimensions.argtypes, dimensions.restype = [ct.c_void_p, ct.POINTER(_Dimensions)], ct.c_int
        self._solve = self._library.spacepdhcg_gtoc12_qoco_solve_host
        self._solve.argtypes = [
            ct.c_void_p,
            _DoublePointer,
            _DoublePointer,
            ct.POINTER(_Parameters),
            ct.c_int,
            _DoublePointer,
            ct.POINTER(_Report),
        ]
        self._solve.restype = ct.c_int
        self._workspace = ct.c_void_p()
        status = create(
            self.intervals,
            int(hold == "lagrange"),
            int(self.free_dep),
            int(self.free_arr),
            model.kappa,
            model.lam,
            _pointer(times),
            _pointer(bnd),
            _pointer(fuel),
            tolerance,
            ruiz_iterations,
            ct.byref(self._workspace),
        )
        if status:
            raise RuntimeError(f"GPU QOCO workspace creation failed (status {status})")
        self._finalizer = weakref.finalize(self, destroy, self._workspace)
        try:
            self.dimensions = _Dimensions()
            if (
                dimensions(self._workspace, ct.byref(self.dimensions))
                or self.dimensions.variables != self.n_variables
            ):
                raise RuntimeError("GPU QOCO variable layout mismatch")
        except BaseException:
            self.close()
            raise
        self.last_report = {}

    def close(self):
        self._finalizer()

    def solve_linearised(
        self,
        states,
        controls,
        substeps,
        trust_state,
        trust_control,
        virtual_weight,
        minimum_mass,
        radius_floor,
        vinf_max,
        smoothness_weight,
    ):
        if not self._finalizer.alive:
            raise RuntimeError("GPU QOCO workspace is closed")
        if not isinstance(substeps, int) or not 1 <= substeps <= 2**31 - 1:
            raise ValueError("substeps must be a positive int32")
        states = np.ascontiguousarray(states, dtype=np.float64)
        controls = np.ascontiguousarray(controls, dtype=np.float64)
        if states.shape != (self.nodes, 7) or controls.shape != (self.nodes, 4):
            raise ValueError("GPU QOCO requires states[nodes,7] and controls[nodes,4]")
        parameters = _Parameters(
            trust_state,
            trust_control,
            virtual_weight,
            minimum_mass,
            radius_floor,
            vinf_max,
            smoothness_weight,
        )
        report = _Report()
        primal = np.full(self.n_variables, np.nan)
        status = self._solve(
            self._workspace,
            _pointer(states),
            _pointer(controls),
            ct.byref(parameters),
            substeps,
            _pointer(primal),
            ct.byref(report),
        )
        self.last_report = {name: getattr(report, name) for name, _ in report._fields_}
        # Failed solves have no fresh audit. Use JSON null instead of Infinity.
        self.last_report = {
            name: None if isinstance(value, float) and not math.isfinite(value) else value
            for name, value in self.last_report.items()
        }
        self.last_report["api_status"] = status
        if status not in {0, 4}:
            raise RuntimeError(f"GPU QOCO evaluation failed (status {status}); no CPU fallback")
        ok = status == 0 and report.qualified == 1
        label = f"QOCO_{report.qoco_status}_{'qualified' if ok else 'unqualified'}"
        return ok, label, primal
