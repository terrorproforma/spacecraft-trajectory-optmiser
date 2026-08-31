#!/usr/bin/env python3
"""Exercise the persistent CUDA C ABI with real DLPack framework producers."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
from pathlib import Path
from typing import Any

from spacepdhcg.backends.dlpack_capsule import consume_producer

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

SUCCESS = 0
POINTER_CONTRACT = 4
READ_ONLY = 0
READ_WRITE = 1
DEVICE_CUDA = 2
ABI_VERSION = 1


class Device(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int32)]


class Stream(ctypes.Structure):
    _fields_ = [("device", Device), ("native_handle", ctypes.c_size_t)]


class Managed(ctypes.Structure):
    _fields_ = [
        ("managed_tensor", ctypes.c_void_p),
        ("kind", ctypes.c_int),
        ("access", ctypes.c_int),
    ]


class Topology(ctypes.Structure):
    _fields_ = [(name, Managed) for name in (
        "quadratic_offsets",
        "quadratic_indices",
        "scalar_offsets",
        "scalar_indices",
        "affine_offsets",
        "affine_indices",
    )]


class Numeric(ctypes.Structure):
    _fields_ = [(name, Managed) for name in (
        "quadratic",
        "scalar_constraint",
        "affine_cone",
        "linear_objective",
        "scalar_lower",
        "scalar_upper",
        "affine_offset",
        "variable_lower",
        "variable_upper",
    )]


class Iterates(ctypes.Structure):
    _fields_ = [("primal", Managed), ("dual", Managed)]


class Exchange(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("topology_fingerprint", ctypes.c_uint64),
        ("consumer_stream", Stream),
        ("topology", Topology),
        ("numeric", Numeric),
        ("iterates", Iterates),
    ]


class Structure(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("topology_fingerprint", ctypes.c_uint64),
        ("variables", ctypes.c_int32),
        ("scalar_rows", ctypes.c_int32),
        ("affine_rows", ctypes.c_int32),
        ("quadratic_nonzeros", ctypes.c_size_t),
        ("scalar_nonzeros", ctypes.c_size_t),
        ("affine_nonzeros", ctypes.c_size_t),
        ("affine_cones", ctypes.c_void_p),
        ("affine_cone_count", ctypes.c_size_t),
        ("variable_cones", ctypes.c_void_p),
        ("variable_cone_count", ctypes.c_size_t),
    ]


class CreateOptions(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("scaling_mode", ctypes.c_int),
        ("maximum_relative_matrix_change", ctypes.c_double),
        ("maximum_relative_vector_change", ctypes.c_double),
        ("maximum_scaling_reuse_updates", ctypes.c_uint64),
        ("debug_validate_aliases", ctypes.c_int32),
        ("external_lifetime_context", ctypes.c_void_p),
        ("retain_external", ctypes.c_void_p),
        ("release_external", ctypes.c_void_p),
    ]


class SolveOptions(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("optimality_tolerance", ctypes.c_double),
        ("feasibility_tolerance", ctypes.c_double),
        ("iteration_limit", ctypes.c_uint64),
        ("residual_check_frequency", ctypes.c_uint32),
    ]


class PointerSnapshot(ctypes.Structure):
    _fields_ = [(name, ctypes.c_size_t) for name in (
        "quadratic_offsets",
        "quadratic_indices",
        "scalar_offsets",
        "scalar_indices",
        "affine_offsets",
        "affine_indices",
        "quadratic_values",
        "scalar_values",
        "affine_values",
        "primal",
        "dual",
        "scaling",
    )]


class Provider:
    def __init__(self, name: str) -> None:
        import cupy as cp

        self.name = name
        self.cp = cp
        self.stream = cp.cuda.Stream(non_blocking=True)
        if name == "cupy":
            self.module = cp
        elif name == "torch":
            import torch

            self.module = torch
        elif name == "jax":
            import jax

            jax.config.update("jax_enable_x64", True)
            import jax.numpy as jnp

            self.module = jnp
        else:
            raise ValueError(f"unknown producer {name!r}")

    def array(self, values: list[float] | list[int], dtype: str) -> Any:
        if self.name == "cupy":
            return self.module.asarray(values, dtype=dtype)
        if self.name == "torch":
            dtype_value = {
                "int32": self.module.int32,
                "float64": self.module.float64,
                "float32": self.module.float32,
            }[dtype]
            return self.module.tensor(values, device="cuda", dtype=dtype_value)
        return self.module.asarray(values, dtype=dtype)

    def numpy(self, value: Any) -> Any:
        if self.name == "cupy":
            return self.cp.asnumpy(value)
        if self.name == "torch":
            return value.detach().cpu().numpy()
        import numpy as np

        return np.asarray(value)


def configure_library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    workspace_pointer = ctypes.POINTER(ctypes.c_void_p)
    library.spacepdhcg_cuda_workspace_create_from_dlpack.argtypes = [
        ctypes.POINTER(Structure),
        ctypes.POINTER(Exchange),
        ctypes.POINTER(CreateOptions),
        workspace_pointer,
    ]
    library.spacepdhcg_cuda_workspace_create_from_dlpack.restype = ctypes.c_int
    library.spacepdhcg_cuda_workspace_update_from_dlpack_async.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.POINTER(Numeric),
        Stream,
    ]
    library.spacepdhcg_cuda_workspace_update_from_dlpack_async.restype = ctypes.c_int
    library.spacepdhcg_cuda_workspace_solve_async.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(SolveOptions),
        Stream,
    ]
    library.spacepdhcg_cuda_workspace_solve_async.restype = ctypes.c_int
    library.spacepdhcg_cuda_workspace_wait.argtypes = [ctypes.c_void_p]
    library.spacepdhcg_cuda_workspace_wait.restype = ctypes.c_int
    library.spacepdhcg_cuda_workspace_pointer_snapshot.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PointerSnapshot),
    ]
    library.spacepdhcg_cuda_workspace_pointer_snapshot.restype = ctypes.c_int
    library.spacepdhcg_cuda_workspace_destroy.argtypes = [workspace_pointer]
    library.spacepdhcg_cuda_workspace_destroy.restype = ctypes.c_int
    return library


def managed(producer: Any, stream: int, access: int) -> Managed:
    consumed = consume_producer(producer, stream=stream)
    return Managed(consumed.managed_tensor, int(consumed.kind), access)


def arrays(provider: Provider, *, invalid_dtype: bool = False) -> dict[str, Any]:
    return {
        "q_offsets": provider.array([0, 1, 2], "int32"),
        "q_indices": provider.array([0, 1], "int32"),
        "a_offsets": provider.array([0, 1, 2], "int32"),
        "a_indices": provider.array([0, 0], "int32"),
        "f_offsets": provider.array([], "int32"),
        "f_indices": provider.array([], "int32"),
        "q": provider.array([1.0, 2.0], "float32" if invalid_dtype else "float64"),
        "a": provider.array([1.0, 1.0], "float64"),
        "f": provider.array([], "float64"),
        "c": provider.array([-1.0, -1.0], "float64"),
        "scalar_lower": provider.array([1.0], "float64"),
        "scalar_upper": provider.array([1.0], "float64"),
        "affine_offset": provider.array([], "float64"),
        "variable_lower": provider.array([0.0, 0.0], "float64"),
        "variable_upper": provider.array([1.0, 1.0], "float64"),
        "primal": provider.array([0.0, 0.0], "float64"),
        "dual": provider.array([0.0], "float64"),
    }


def exchange(values: dict[str, Any], stream: int) -> Exchange:
    def wrap(name: str, access: int) -> Managed:
        return managed(values[name], stream, access)

    return Exchange(
        ABI_VERSION,
        0x7CD07E0A4F9E0B61,
        Stream(Device(DEVICE_CUDA, 0), stream),
        Topology(
            wrap("q_offsets", READ_ONLY),
            wrap("q_indices", READ_ONLY),
            wrap("a_offsets", READ_ONLY),
            wrap("a_indices", READ_ONLY),
            wrap("f_offsets", READ_ONLY),
            wrap("f_indices", READ_ONLY),
        ),
        Numeric(
            wrap("q", READ_WRITE),
            wrap("a", READ_WRITE),
            wrap("f", READ_WRITE),
            wrap("c", READ_WRITE),
            wrap("scalar_lower", READ_WRITE),
            wrap("scalar_upper", READ_WRITE),
            wrap("affine_offset", READ_WRITE),
            wrap("variable_lower", READ_WRITE),
            wrap("variable_upper", READ_WRITE),
        ),
        Iterates(wrap("primal", READ_WRITE), wrap("dual", READ_WRITE)),
    )


def numeric(values: dict[str, Any], stream: int) -> Numeric:
    def wrap(name: str) -> Managed:
        return managed(values[name], stream, READ_WRITE)

    return Numeric(
        wrap("q"),
        wrap("a"),
        wrap("f"),
        wrap("c"),
        wrap("scalar_lower"),
        wrap("scalar_upper"),
        wrap("affine_offset"),
        wrap("variable_lower"),
        wrap("variable_upper"),
    )


def require(status: int, operation: str) -> None:
    if status != SUCCESS:
        raise RuntimeError(f"{operation} failed with status {status}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer", choices=("cupy", "torch", "jax"), required=True)
    parser.add_argument("--library", type=Path, required=True)
    arguments = parser.parse_args()

    provider = Provider(arguments.producer)
    library = configure_library(arguments.library)
    stream = int(provider.stream.ptr)
    structure = Structure(
        ABI_VERSION,
        0x7CD07E0A4F9E0B61,
        2,
        1,
        0,
        2,
        2,
        0,
        None,
        0,
        None,
        0,
    )
    options = CreateOptions(ABI_VERSION, 2, 0.25, 0.5, 4, 1, None, None, None)
    solve_options = SolveOptions(ABI_VERSION, 2.0e-6, 2.0e-6, 100_000, 25)

    create_values = arrays(provider)
    create_exchange = exchange(create_values, stream)
    workspace = ctypes.c_void_p()
    require(
        library.spacepdhcg_cuda_workspace_create_from_dlpack(
            ctypes.byref(structure),
            ctypes.byref(create_exchange),
            ctypes.byref(options),
            ctypes.byref(workspace),
        ),
        "create from DLPack",
    )
    before = PointerSnapshot()
    require(
        library.spacepdhcg_cuda_workspace_pointer_snapshot(workspace, ctypes.byref(before)),
        "pointer snapshot before",
    )

    update_values = arrays(provider)
    update_views = numeric(update_values, stream)
    require(
        library.spacepdhcg_cuda_workspace_update_from_dlpack_async(
            workspace,
            structure.topology_fingerprint,
            ctypes.byref(update_views),
            create_exchange.consumer_stream,
        ),
        "DLPack update",
    )
    del update_values
    gc.collect()
    require(library.spacepdhcg_cuda_workspace_wait(workspace), "DLPack update wait")
    require(
        library.spacepdhcg_cuda_workspace_solve_async(
            workspace,
            ctypes.byref(solve_options),
            create_exchange.consumer_stream,
        ),
        "solve",
    )
    require(library.spacepdhcg_cuda_workspace_wait(workspace), "solve wait")
    solution = provider.numpy(create_values["primal"])
    if max(abs(float(solution[0]) - 2.0 / 3.0), abs(float(solution[1]) - 1.0 / 3.0)) > 3e-5:
        raise AssertionError(f"unexpected solution {solution!r}")
    after = PointerSnapshot()
    require(
        library.spacepdhcg_cuda_workspace_pointer_snapshot(workspace, ctypes.byref(after)),
        "pointer snapshot after",
    )
    if (
        before.quadratic_offsets != after.quadratic_offsets
        or before.quadratic_values != after.quadratic_values
        or before.primal != after.primal
    ):
        raise AssertionError("persistent pointers changed after DLPack update")
    require(library.spacepdhcg_cuda_workspace_destroy(ctypes.byref(workspace)), "destroy")

    invalid_values = arrays(provider, invalid_dtype=True)
    invalid_exchange = exchange(invalid_values, stream)
    invalid_workspace = ctypes.c_void_p()
    invalid_status = library.spacepdhcg_cuda_workspace_create_from_dlpack(
        ctypes.byref(structure),
        ctypes.byref(invalid_exchange),
        ctypes.byref(options),
        ctypes.byref(invalid_workspace),
    )
    if invalid_status != POINTER_CONTRACT or invalid_workspace.value is not None:
        raise AssertionError(f"invalid dtype returned {invalid_status}")

    print(json.dumps({
        "case": "dlpack_producer_compat",
        "producer": arguments.producer,
        "non_default_stream": True,
        "premature_update_release": True,
        "invalid_dtype_rejected": True,
        "pointer_stable": True,
        "solution": [float(solution[0]), float(solution[1])],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
