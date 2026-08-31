from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from spacepdhcg.models import PoweredDescent3DOFModel

ROOT = Path(__file__).resolve().parents[1]
Double3 = ctypes.c_double * 3
Double4 = ctypes.c_double * 4
Double7 = ctypes.c_double * 7
Double28 = ctypes.c_double * 28
Double49 = ctypes.c_double * 49


class PoweredDescentConfig(ctypes.Structure):
    _fields_ = [
        ("gravity", Double3),
        ("mass_flow_coefficient", ctypes.c_double),
        ("minimum_mass", ctypes.c_double),
        ("maximum_thrust", ctypes.c_double),
        ("minimum_sigma", ctypes.c_double),
        ("maximum_tilt_radians", ctypes.c_double),
        ("glide_slope_radians", ctypes.c_double),
    ]


class LambertResult(ctypes.Structure):
    _fields_ = [
        ("departure_velocity", Double3),
        ("arrival_velocity", Double3),
        ("universal_parameter", ctypes.c_double),
        ("transfer_angle_radians", ctypes.c_double),
        ("iterations", ctypes.c_uint64),
        ("time_of_flight_residual", ctypes.c_double),
    ]


def _compile_library(tmp_path: Path) -> ctypes.CDLL:
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++20 compiler is required for the native C ABI gate")
    library_path = tmp_path / "libspacepdhcg_c_api.so"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-shared",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            "-I",
            str(ROOT / "cpp" / "include"),
            str(ROOT / "cpp" / "src" / "c_api.cpp"),
            "-o",
            str(library_path),
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    library = ctypes.CDLL(str(library_path))
    library.spacepdhcg_c_api_version.restype = ctypes.c_uint32
    library.spacepdhcg_native_version.restype = ctypes.c_char_p
    library.spacepdhcg_last_error.restype = ctypes.c_char_p
    library.spacepdhcg_default_powered_descent_3dof_config.argtypes = [
        ctypes.POINTER(PoweredDescentConfig)
    ]
    library.spacepdhcg_powered_descent_3dof_dynamics.argtypes = [
        ctypes.POINTER(PoweredDescentConfig),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    library.spacepdhcg_powered_descent_3dof_dynamics.restype = ctypes.c_int
    library.spacepdhcg_powered_descent_3dof_jacobians.argtypes = [
        ctypes.POINTER(PoweredDescentConfig),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    library.spacepdhcg_powered_descent_3dof_jacobians.restype = ctypes.c_int
    library.spacepdhcg_lambert_zero_revolution.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_uint64,
        ctypes.POINTER(LambertResult),
    ]
    library.spacepdhcg_lambert_zero_revolution.restype = ctypes.c_int
    return library


def test_c_api_matches_python_reference(tmp_path: Path) -> None:
    library = _compile_library(tmp_path)
    assert library.spacepdhcg_c_api_version() == 1
    assert library.spacepdhcg_native_version().decode() == "0.1.0.dev0"

    config = PoweredDescentConfig()
    library.spacepdhcg_default_powered_descent_3dof_config(ctypes.byref(config))
    state_values = np.asarray([20.0, -10.0, 120.0, 0.4, -0.2, -7.0, 2_000.0])
    thrust = np.asarray([1_200.0, -500.0, 8_000.0])
    control_values = np.concatenate((thrust, [np.linalg.norm(thrust)]))
    state = Double7(*state_values)
    control = Double4(*control_values)
    derivative = Double7()
    state_jacobian = Double49()
    control_jacobian = Double28()

    status = library.spacepdhcg_powered_descent_3dof_dynamics(
        ctypes.byref(config),
        state,
        control,
        derivative,
    )
    assert status == 0, library.spacepdhcg_last_error().decode()
    status = library.spacepdhcg_powered_descent_3dof_jacobians(
        ctypes.byref(config),
        state,
        control,
        state_jacobian,
        control_jacobian,
    )
    assert status == 0, library.spacepdhcg_last_error().decode()

    model = PoweredDescent3DOFModel()
    expected_state, expected_control = model.jacobians(state_values, control_values)
    np.testing.assert_allclose(derivative, model.dynamics(state_values, control_values), atol=1e-13)
    np.testing.assert_allclose(np.asarray(state_jacobian).reshape(7, 7), expected_state, atol=1e-13)
    np.testing.assert_allclose(
        np.asarray(control_jacobian).reshape(7, 4),
        expected_control,
        atol=1e-13,
    )

    result = LambertResult()
    status = library.spacepdhcg_lambert_zero_revolution(
        Double3(5_000.0, 10_000.0, 2_100.0),
        Double3(-14_600.0, 2_500.0, 7_000.0),
        3_600.0,
        398_600.0,
        0,
        1e-9,
        256,
        ctypes.byref(result),
    )
    assert status == 0, library.spacepdhcg_last_error().decode()
    np.testing.assert_allclose(result.departure_velocity, [-5.9925, 1.9254, 3.2456], atol=5e-3)
    np.testing.assert_allclose(result.arrival_velocity, [-3.3125, -4.1966, -0.3853], atol=5e-3)
    assert abs(result.time_of_flight_residual) < 1e-7
