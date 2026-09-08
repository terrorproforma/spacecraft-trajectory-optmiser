"""Immutable physical inputs for an explicitly supplied native ZOH rollout."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .low_thrust import LegBoundary, ScvxSettings


@dataclass(frozen=True, slots=True)
class ZohTrajectorySeed:
    """Archived initial state and exact ZOH thrust; GPU prepares the trajectory.

    State units are km, km/s and kg. Thrust is in N, with one vector per node
    and an inactive zero final vector. The source digest identifies the archive;
    it is not itself a physical certificate. Inputs are copied into immutable
    FP64 buffers; no thrust clipping, normalization or interpolation occurs here.
    """

    node_epochs_mjd: NDArray[np.float64]
    initial_state: NDArray[np.float64]
    thrust_n: NDArray[np.float64]
    source_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.source_sha256)
        ):
            raise ValueError("seed source_sha256 must be a lowercase SHA256 digest")
        arrays = {
            name: np.asarray(getattr(self, name), dtype=np.float64)
            for name in ("node_epochs_mjd", "initial_state", "thrust_n")
        }
        epochs, state, thrust = (arrays[name] for name in arrays)
        if epochs.ndim != 1 or len(epochs) < 2:
            raise ValueError("seed node_epochs_mjd must be a vector with at least two nodes")
        if state.shape != (7,) or thrust.shape != (len(epochs), 3):
            raise ValueError("seed requires initial_state shape (7,) and thrust_n shape (nodes, 3)")
        if not all(np.isfinite(array).all() for array in arrays.values()):
            raise ValueError("seed arrays must be finite")
        if not np.all(epochs[1:] > epochs[:-1]):
            raise ValueError("seed node epochs must be strictly increasing")
        if state[6] <= 0.0:
            raise ValueError("seed initial mass must be positive")
        if np.any(thrust[-1] != 0.0):
            raise ValueError("seed final inactive thrust must be zero")
        for name, array in arrays.items():
            # Immutable bytes also prevent callers from re-enabling write access.
            frozen = np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(array.shape)
            object.__setattr__(self, name, frozen)

    def validate_for(
        self,
        boundary: LegBoundary,
        settings: ScvxSettings,
        node_epochs_mjd: NDArray[np.float64],
    ) -> None:
        if settings.outer_loop_backend != "cuda" or settings.hold != "zoh":
            raise ValueError("a ZOH trajectory seed requires ZOH hold and CUDA outer loop")
        if settings.seed_backend == "numpy":
            raise ValueError("a ZOH trajectory seed cannot select the NumPy seed ablation")
        if not np.array_equal(self.node_epochs_mjd, node_epochs_mjd):
            raise ValueError("seed node epochs must exactly match the generated solver grid")
        if (
            not math.isfinite(boundary.initial_mass)
            or self.initial_state[6] != boundary.initial_mass
        ):
            raise ValueError("seed initial mass must exactly match the leg boundary")
        if not np.array_equal(self.initial_state[:3], boundary.departure_position):
            raise ValueError("seed initial position must exactly match the leg boundary")
        if not boundary.free_departure_vinf and not np.array_equal(
            self.initial_state[3:6], boundary.departure_velocity
        ):
            raise ValueError("seed fixed departure velocity must exactly match the body velocity")
        # A free Earth velocity is retained bit for bit, including tiny archival
        # cap excess. Native rollout and the independent verifier own physics.
