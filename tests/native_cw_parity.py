"""Cross-check the dependency-free C ABI against the Python HCW oracle."""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import numpy as np

from spacepdhcg.models.cw import discretise_cw


def run(library_path: Path) -> None:
    library = ctypes.CDLL(str(library_path.resolve()))
    library.spacepdhcg_native_abi_version.argtypes = []
    library.spacepdhcg_native_abi_version.restype = ctypes.c_int
    library.spacepdhcg_cw_discretise.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
    ]
    library.spacepdhcg_cw_discretise.restype = ctypes.c_int

    if library.spacepdhcg_native_abi_version() != 1:
        raise RuntimeError("unexpected SpacePDHCG native ABI version")

    cases = (
        (1.13e-3, 1.0),
        (1.13e-3, 20.0),
        (7.5e-4, 120.0),
        (2.0e-3, 0.01),
        (2.0e-3, 500.0),
    )
    for mean_motion, step_seconds in cases:
        native_state = np.empty((6, 6), dtype=np.float64, order="C")
        native_control = np.empty((6, 3), dtype=np.float64, order="C")
        status = library.spacepdhcg_cw_discretise(
            mean_motion,
            step_seconds,
            native_state.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            native_state.size,
            native_control.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            native_control.size,
        )
        if status != 0:
            raise RuntimeError(
                f"native HCW discretisation failed for n={mean_motion}, dt={step_seconds}"
            )
        python_state, python_control = discretise_cw(mean_motion, step_seconds)
        np.testing.assert_allclose(native_state, python_state, rtol=2.0e-12, atol=2.0e-12)
        np.testing.assert_allclose(native_control, python_control, rtol=2.0e-12, atol=2.0e-10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("library", type=Path)
    arguments = parser.parse_args()
    run(arguments.library)


if __name__ == "__main__":
    main()
