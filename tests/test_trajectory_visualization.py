from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "visualization"
    / "extract_verified_trajectories.py"
)
SPEC = importlib.util.spec_from_file_location("trajectory_visualization", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_decimation_preserves_endpoints_extrema_and_required_points() -> None:
    times = np.arange(101, dtype=np.float64)
    positions = np.column_stack(
        (
            np.sin(times / 10.0),
            np.cos(times / 7.0),
            (times - 50.0) ** 2,
        )
    )
    required = {17, 63}
    indices = MODULE.decimate_indices(times, positions, 24, required)

    expected = {0, 100, 17, 63}
    for axis in range(3):
        expected.add(int(np.argmin(positions[:, axis])))
        expected.add(int(np.argmax(positions[:, axis])))
    radii = np.linalg.norm(positions, axis=1)
    expected.update({int(np.argmin(radii)), int(np.argmax(radii))})

    assert expected.issubset(indices)
    assert indices == sorted(set(indices))
    assert len(indices) <= 24


def test_decimation_is_deterministic_and_never_interpolates() -> None:
    times = np.linspace(0.0, 10.0, 200)
    positions = np.column_stack((times, times**2, -times))

    first = MODULE.decimate_indices(times, positions, 20, {37})
    second = MODULE.decimate_indices(times, positions, 20, {37})

    assert first == second
    compact = MODULE.compact_points(times, positions, 20, {37})
    for index, point in zip(compact["selected_indices"], compact["points_txyz"], strict=True):
        assert point == [times[index], *positions[index]]


def test_state_history_classifier_rejects_scalar_trajectory_metrics() -> None:
    scalar = {"cpu_gpu_trajectory": 1.0e-16, "terminal": 1.0e-12}
    states = {"states": [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]}

    assert not MODULE.has_traceable_state_arrays(scalar)
    assert MODULE.has_traceable_state_arrays(states)
