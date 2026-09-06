"""Native GPU linearisation/assembly with a temporary CPU conic-solver bridge."""

from __future__ import annotations

import ctypes as ct
import os
import weakref
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .gpu_discretisation import _DoublePointer, _pointer
from .low_thrust import _ConvexProblem

_IntPointer = ct.POINTER(ct.c_int)


class _Parameters(ct.Structure):
    _fields_ = [
        (name, ct.c_double)
        for name in (
            "trust_state",
            "trust_control",
            "virtual_weight",
            "minimum_mass",
            "radius_floor",
            "vinf_max",
            "smoothness_weight",
        )
    ]


class _Dimensions(ct.Structure):
    _fields_ = [
        (name, ct.c_int)
        for name in (
            "variables",
            "rows",
            "equalities",
            "inequalities",
            "soc_count",
            "a_nonzeros",
            "p_nonzeros",
        )
    ]


class GpuConvexProblem(_ConvexProblem):
    """Topology compiled once; numerical assembly and interval dynamics on CUDA.

    Calls must be serialized. Returned matrices own independent values, while
    their immutable topology is retained. CPU Clarabel remains outside this API.
    """

    def __init__(self, model, node_times, hold, free_dep, free_arr, boundary, fuel_weights):
        import clarabel

        path = os.environ.get("SPACEPDHCG_GTOC12_CUDA_LIBRARY")
        if not path:
            raise RuntimeError("CUDA GTOC12 requires SPACEPDHCG_GTOC12_CUDA_LIBRARY")
        times = np.ascontiguousarray(node_times, dtype=np.float64)
        fuel = np.ascontiguousarray(fuel_weights, dtype=np.float64)
        bnd = np.ascontiguousarray(
            np.concatenate([boundary[k] for k in ("r0", "v0", "rf", "vf")]), dtype=np.float64
        )
        if times.ndim != 1 or len(times) < 2 or fuel.shape != times.shape or bnd.shape != (12,):
            raise ValueError("invalid node times, fuel weights or boundary dimensions")
        if hold not in {"zoh", "lagrange"}:
            raise ValueError("hold must be 'zoh' or 'lagrange'")
        super().__init__(len(times), bool(free_dep), bool(free_arr))
        self._library = ct.CDLL(str(Path(path).resolve(strict=True)))
        create = self._library.spacepdhcg_gtoc12_conic_create
        create.argtypes = (
            [ct.c_int] * 4 + [ct.c_double] * 2 + [_DoublePointer] * 3 + [ct.POINTER(ct.c_void_p)]
        )
        create.restype = ct.c_int
        destroy = self._library.spacepdhcg_gtoc12_conic_destroy
        destroy.argtypes, destroy.restype = [ct.c_void_p], None
        dimensions = self._library.spacepdhcg_gtoc12_conic_get_dimensions
        dimensions.argtypes, dimensions.restype = [ct.c_void_p, ct.POINTER(_Dimensions)], ct.c_int
        topology = self._library.spacepdhcg_gtoc12_conic_copy_topology_host
        topology.argtypes, topology.restype = [ct.c_void_p] + [_IntPointer] * 4, ct.c_int
        self._evaluate = self._library.spacepdhcg_gtoc12_conic_evaluate_host
        self._evaluate.argtypes = [
            ct.c_void_p,
            _DoublePointer,
            _DoublePointer,
            ct.POINTER(_Parameters),
            ct.c_int,
            _DoublePointer,
        ]
        self._evaluate.restype = ct.c_int
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
            ct.byref(self._workspace),
        )
        if status:
            raise RuntimeError(f"CUDA GTOC12 conic creation failed (status {status})")
        self._finalizer = weakref.finalize(self, destroy, self._workspace)
        try:
            self.dimensions = d = _Dimensions()
            if dimensions(self._workspace, ct.byref(d)) or d.variables != self.n_variables:
                raise RuntimeError("CUDA GTOC12 conic layout mismatch")
            self._topology = (
                np.empty(d.variables + 1, dtype=np.int32),
                np.empty(d.a_nonzeros, dtype=np.int32),
                np.empty(d.variables + 1, dtype=np.int32),
                np.empty(d.p_nonzeros, dtype=np.int32),
            )
            if topology(self._workspace, *[a.ctypes.data_as(_IntPointer) for a in self._topology]):
                raise RuntimeError("CUDA GTOC12 topology copy failed")
            for array in self._topology:
                array.setflags(write=False)
            self._cones = [
                clarabel.ZeroConeT(d.equalities),
                clarabel.NonnegativeConeT(d.inequalities),
            ]
            self._cones.extend(clarabel.SecondOrderConeT(4) for _ in range(d.soc_count))
        except BaseException:
            self.close()
            raise

    def close(self):
        self._finalizer()

    def build_linearised(
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
            raise RuntimeError("CUDA GTOC12 conic workspace is closed")
        if not isinstance(substeps, int) or not 1 <= substeps <= 2**31 - 1:
            raise ValueError("substeps must be a positive int32")
        states = np.ascontiguousarray(states, dtype=np.float64)
        controls = np.ascontiguousarray(controls, dtype=np.float64)
        if states.shape != (self.nodes, 7) or controls.shape != (self.nodes, 4):
            raise ValueError("CUDA GTOC12 requires states[nodes,7] and controls[nodes,4]")
        parameters = _Parameters(
            trust_state,
            trust_control,
            virtual_weight,
            minimum_mass,
            radius_floor,
            vinf_max,
            smoothness_weight,
        )
        d = self.dimensions
        packed = np.empty(d.a_nonzeros + d.rows + d.variables + d.p_nonzeros)
        status = self._evaluate(
            self._workspace,
            _pointer(states),
            _pointer(controls),
            ct.byref(parameters),
            substeps,
            _pointer(packed),
        )
        if status:
            raise RuntimeError(f"CUDA GTOC12 conic evaluation failed (status {status})")
        b_start, q_start = d.a_nonzeros, d.a_nonzeros + d.rows
        p_start = q_start + d.variables
        ao, ai, po, pi = self._topology
        a = sp.csc_matrix((packed[:b_start], ai, ao), shape=(d.rows, d.variables), copy=False)
        p = sp.csc_matrix((packed[p_start:], pi, po), shape=(d.variables, d.variables), copy=False)
        return a, packed[b_start:q_start], packed[q_start:p_start], self._cones, p
