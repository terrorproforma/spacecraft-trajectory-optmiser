"""Native free-final-time (``pd3_fft`` / ``pd6_fft``) transcriptions through the C API.

The C++ smoke tests (``cpp/tests/time_dilated_flow_smoke.cpp`` and
``cpp/tests/powered_descent_free_time_transcription_smoke.cpp``) cover the variational RK4 sigma
sensitivities, the quaternion tangent rule and the topology fingerprints.  These tests exercise
the same objects from Python: CQP layout, affine reconstruction through the emitted CSC rows, the
sigma column against a finite-difference oracle on the native replay, the new attitude-tilt cone,
and the outer loop that reproduces the CPU core's Szmuk 2018 time of flight.

They need the native library (``SPACEPDHCG_NATIVE_LIBRARY`` or the packaged wheel) and skip with
an explicit reason otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

from spacepdhcg.native import native_available

pytestmark = pytest.mark.skipif(
    not native_available(), reason="native spacepdhcg library is not available"
)


def _library_has_free_time() -> bool:
    from spacepdhcg.native._library import load_native_library

    return hasattr(load_native_library(), "spacepdhcg_pd6_fft_create")


requires_free_time = pytest.mark.skipif(
    native_available() and not _library_has_free_time(),
    reason="native library predates the free-final-time C API",
)

#: t_f of the Python FOH/K=50 core (``spacepdhcg.literature.pd6_szmuk_2018``), which matches the
#: paper's Figure 2.  The native transcription is ZOH with 4 RK4 substeps per interval; the
#: declared discretisation envelope between the two is 0.01 UT.
CPU_CORE_SZMUK_TF = 3.3901
SZMUK_TF_ENVELOPE_UT = 0.01


def _dense(csc, values: np.ndarray) -> np.ndarray:
    rows, columns = csc.shape
    dense = np.zeros((rows, columns))
    offsets = np.asarray(csc.indptr)
    indices = np.asarray(csc.indices)
    for column in range(columns):
        for position in range(offsets[column], offsets[column + 1]):
            dense[indices[position], column] = values[position]
    return dense


def _pd3_consistent_reference():
    from spacepdhcg.native.free_time import NativeFreeTimeTranscription, Pd3FreeTimeOptions

    options = Pd3FreeTimeOptions(intervals=6, substeps=2)
    transcription = NativeFreeTimeTranscription(options)
    lay = transcription.layout
    rng = np.random.default_rng(3)
    controls = np.zeros((lay.intervals, lay.control_dimension))
    for k in range(lay.intervals):
        thrust = np.array([200.0 * np.sin(k), -150.0 * np.cos(k), 7000.0 + 300.0 * k])
        controls[k] = transcription.project_control(np.concatenate((thrust, [0.0])))
    states = np.zeros((lay.intervals + 1, lay.state_dimension))
    states[0] = np.array([1500.0, -800.0, 2400.0, 40.0, 30.0, -75.0, 1905.0])
    states[0, :6] += rng.normal(scale=1.0, size=6)
    sigma = 48.0
    states[1:] = states[0]  # replay evaluates every interval: keep the mass positive
    for k in range(lay.intervals):
        states[k + 1] = transcription.replay(states, controls, sigma)[k]
    return transcription, states, controls, sigma


def _pd6_consistent_reference(**overrides):
    from spacepdhcg.native.free_time import NativeFreeTimeTranscription, Pd6FreeTimeOptions

    options = Pd6FreeTimeOptions(intervals=5, substeps=2, maximum_torque=50.0, **overrides)
    transcription = NativeFreeTimeTranscription(options)
    lay = transcription.layout
    controls = np.zeros((lay.intervals, lay.control_dimension))
    for k in range(lay.intervals):
        raw = np.array([120.0, -80.0, 9000.0 + 100.0 * k, 0.3, 0.1, -0.2, 0.0])
        controls[k] = transcription.project_control(raw)
    states = np.zeros((lay.intervals + 1, lay.state_dimension))
    states[0] = np.array(
        [
            400.0,
            -300.0,
            1200.0,
            20.0,
            -10.0,
            -40.0,
            0.98,
            0.1,
            -0.15,
            0.05,
            0.02,
            -0.01,
            0.03,
            1900.0,
        ]
    )
    states[0, 6:10] /= np.linalg.norm(states[0, 6:10])
    sigma = 25.0
    states[1:] = states[0]  # replay evaluates every interval: keep the mass positive
    for k in range(lay.intervals):
        states[k + 1] = transcription.replay(states, controls, sigma)[k]
    return transcription, states, controls, sigma


def _reference_vector(transcription, states, controls, sigma) -> np.ndarray:
    lay = transcription.layout
    z = np.zeros(lay.variables)
    z[: states.size] = states.reshape(-1)
    z[lay.control_offset : lay.control_offset + controls.size] = controls.reshape(-1)
    z[lay.sigma_index] = sigma
    return z


@requires_free_time
@pytest.mark.parametrize("family", ["pd3_fft", "pd6_fft"])
def test_layout_carries_a_single_sigma_column_after_the_controls(family: str) -> None:
    transcription, states, controls, _ = (
        _pd3_consistent_reference() if family == "pd3_fft" else _pd6_consistent_reference()
    )
    lay = transcription.layout
    assert transcription.family == family
    assert lay.control_offset == states.size
    assert lay.sigma_index == lay.control_offset + controls.size
    assert lay.virtual_offset == lay.sigma_index + 1
    virtual = lay.intervals * lay.state_dimension
    assert lay.epigraph_offset == lay.virtual_offset + virtual
    assert lay.variables == lay.epigraph_offset + virtual
    assert lay.topology_fingerprint != 0


@requires_free_time
@pytest.mark.parametrize("family", ["pd3_fft", "pd6_fft"])
def test_consistent_reference_satisfies_every_equality_row(family: str) -> None:
    """Affine reconstruction: with nu = 0 the reference itself solves the linearised rows."""

    transcription, states, controls, sigma = (
        _pd3_consistent_reference() if family == "pd3_fft" else _pd6_consistent_reference()
    )
    initial = states[0].copy()
    target = states[-1].copy() if family == "pd6_fft" else states[-1, :6].copy()
    values = transcription.values(states, controls, sigma, initial, target)
    matrix = _dense(transcription.structure.constraint, values.constraint)
    residual = matrix @ _reference_vector(transcription, states, controls, sigma)
    equality = np.isclose(values.lower, values.upper)
    assert equality.any()
    worst = float(np.max(np.abs(residual[equality] - values.lower[equality])))
    assert worst <= 1.0e-8 * max(1.0, float(np.max(np.abs(states))))
    # inequality rows are satisfied too
    assert np.all(residual >= values.lower - 1.0e-8)
    assert np.all(residual <= values.upper + 1.0e-8)


@requires_free_time
@pytest.mark.parametrize("family", ["pd3_fft", "pd6_fft"])
def test_sigma_column_matches_a_finite_difference_of_the_native_replay(family: str) -> None:
    transcription, states, controls, sigma = (
        _pd3_consistent_reference() if family == "pd3_fft" else _pd6_consistent_reference()
    )
    lay = transcription.layout
    initial = states[0].copy()
    target = states[-1].copy() if family == "pd6_fft" else states[-1, :6].copy()
    values = transcription.values(states, controls, sigma, initial, target)
    matrix = _dense(transcription.structure.constraint, values.constraint)
    sigma_column = matrix[:, lay.sigma_index]
    step = 1.0e-4 * sigma
    plus = transcription.replay(states, controls, sigma + step)
    minus = transcription.replay(states, controls, sigma - step)
    oracle = (plus - minus) / (2.0 * step)  # (K, n_x): d x_{k+1} / d sigma
    n_x = lay.state_dimension
    checked = 0
    worst = 0.0
    csc = transcription.structure.constraint
    structural_rows = np.asarray(csc.indices)[
        csc.indptr[lay.sigma_index] : csc.indptr[lay.sigma_index + 1]
    ]
    for row in structural_rows:
        # dynamics rows are the only rows touching sigma (structurally, even where the value is
        # zero, e.g. a zero throttle slack); identify (k, i) from the successor state
        # coefficient, which is the sole +-1 entry among the successor-state columns.
        row_values = matrix[row]
        successor_columns = np.flatnonzero(np.abs(row_values[: states.size]) > 0.0)
        # the row references x_k (many entries) and x_{k+1} (one entry): pick the one whose
        # node index is largest
        node_of = successor_columns // n_x
        k_plus_one = int(node_of.max())
        candidates = successor_columns[node_of == k_plus_one]
        assert candidates.size == 1
        i = int(candidates[0] % n_x)
        coefficient_next = row_values[candidates[0]]
        native_sensitivity = -sigma_column[row] / coefficient_next
        scale = max(1.0, abs(oracle[k_plus_one - 1, i]))
        worst = max(worst, abs(native_sensitivity - oracle[k_plus_one - 1, i]) / scale)
        checked += 1
    assert checked == lay.intervals * n_x
    assert worst <= 1.0e-6


@requires_free_time
def test_pd6_attitude_tilt_cone_is_present_and_bounds_the_quaternion_vector_part() -> None:
    """New pd6_fft path constraint: |[q_x, q_y]| <= sqrt((1 - cos theta_max)/2)."""

    theta = np.pi / 2.0
    transcription, states, controls, sigma = _pd6_consistent_reference(
        maximum_attitude_tilt_radians=theta
    )
    values = transcription.values(states, controls, sigma, states[0], states[-1])
    affine = _dense(transcription.structure.affine_cone, values.affine_cone)
    z = _reference_vector(transcription, states, controls, sigma)
    lhs = affine @ z + values.affine_offset
    bound = np.sqrt(0.5 * (1.0 - np.cos(theta)))
    tilt_cones = []
    for cone in transcription.structure.affine_cones:
        slots = cone.vector_dimension + 2
        block = slice(cone.start, cone.start + slots)
        if slots == 3 and np.isclose(values.affine_offset[cone.start + 2], bound):
            columns = [np.flatnonzero(affine[cone.start + j]) for j in range(2)]
            if all(c.size == 1 for c in columns):
                tilt_cones.append((cone.start, [int(c[0]) for c in columns], block))
    assert len(tilt_cones) == transcription.layout.intervals + 1
    n_x = transcription.layout.state_dimension
    for node, (_start, columns, block) in enumerate(sorted(tilt_cones)):
        assert columns == [node * n_x + 7, node * n_x + 8]
        vector = lhs[block][:2]
        assert np.allclose(vector, states[node, 7:9])
        assert np.isclose(lhs[block][2], bound)
    # A reference tilted past theta_max violates the cone (native cone violation reports it).
    tilted = states.copy()
    tilted[:, 6:10] = np.array([np.cos(np.deg2rad(60.0)), np.sin(np.deg2rad(60.0)), 0.0, 0.0])
    lhs_tilted = affine @ _reference_vector(transcription, tilted, controls, sigma)
    lhs_tilted += values.affine_offset
    _start, _, block = sorted(tilt_cones)[0]
    assert np.linalg.norm(lhs_tilted[block][:2]) > lhs_tilted[block][2] + 0.1


@requires_free_time
def test_pd6_fft_rejects_an_attitude_tilt_bound_outside_zero_pi() -> None:
    from spacepdhcg.native.free_time import (
        NativeFreeTimeTranscription,
        NativeLibraryError,
        Pd6FreeTimeOptions,
    )

    with pytest.raises(NativeLibraryError, match="tilt"):
        NativeFreeTimeTranscription(Pd6FreeTimeOptions(maximum_attitude_tilt_radians=4.0))


@requires_free_time
def test_native_pd6_fft_outer_loop_reproduces_the_cpu_core_szmuk_time_of_flight() -> None:
    from spacepdhcg.literature.pd6_szmuk_2018_native import reproduce_native

    result = reproduce_native()
    assert result.outcome.converged, result.outcome.termination
    assert result.outcome.replay_defect_inf <= 1.0e-6
    assert result.max_path_violation <= 1.0e-6, result.path_violations
    assert abs(result.time_of_flight - CPU_CORE_SZMUK_TF) <= SZMUK_TF_ENVELOPE_UT
    # theta_max = 90 deg is active on this problem (the optimum sits on the tilt cone).
    assert result.path_violations["tilt"] <= 1.0e-6
    tilt_active = np.max(
        1.0 - (1.0 - 2.0 * (result.outcome.states[:, 7] ** 2 + result.outcome.states[:, 8] ** 2))
    )
    assert tilt_active > 0.99


@requires_free_time
def test_without_the_tilt_bound_the_native_optimum_is_a_different_shorter_problem() -> None:
    """Evidence that the attitude cone is what separates 2.97 UT from the paper's 3.39 UT."""

    from spacepdhcg.literature import pd6_szmuk_2018_native as native
    from spacepdhcg.literature.pd6_szmuk_2018 import Szmuk2018Parameters

    p = Szmuk2018Parameters(tilt_max_deg=180.0)
    result = native.reproduce_native(p)
    assert result.outcome.converged
    assert result.time_of_flight < CPU_CORE_SZMUK_TF - 0.3
    violations = native.path_violations(
        Szmuk2018Parameters(), result.outcome.states, result.outcome.controls
    )
    assert violations["tilt"] > 0.3
