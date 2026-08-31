"""Solver-independent certificates for condensed scenario programmes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .condensed_cqp import CondensedScenarioCQPBundle

FloatArray = NDArray[np.float64]


def encode_condensed_primal(
    bundle: CondensedScenarioCQPBundle,
    local_primals: Sequence[FloatArray],
    *,
    consistency_tolerance: float = 1.0e-12,
) -> FloatArray:
    """Embed scenario-local vectors into one condensed global primal.

    Shared information-node controls appear once in the condensed vector. Every local
    vector that maps to the same global slot must therefore agree within
    ``consistency_tolerance``. The resulting vector is useful for formulation
    certificates and incumbent bounds that do not depend on a solver status string.
    """

    if len(local_primals) != bundle.scenario_count:
        raise ValueError("one local primal is required per scenario")
    if not np.isfinite(consistency_tolerance) or consistency_tolerance < 0.0:
        raise ValueError("consistency_tolerance must be finite and non-negative")

    global_primal = np.zeros(bundle.total_variables, dtype=np.float64)
    assigned = np.zeros(bundle.total_variables, dtype=bool)
    expected_size = bundle.local_structure.n_variables
    for scenario, local in enumerate(local_primals):
        vector = np.asarray(local, dtype=np.float64)
        if vector.shape != (expected_size,):
            raise ValueError(
                f"local primal {scenario} must have shape ({expected_size},)"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"local primal {scenario} must be finite")
        mapping = bundle.local_to_global(scenario)
        for local_index, global_index in enumerate(mapping):
            value = vector[local_index]
            if assigned[global_index] and not np.isclose(
                global_primal[global_index],
                value,
                atol=consistency_tolerance,
                rtol=0.0,
            ):
                raise ValueError(
                    "local primals disagree on a shared information-node control"
                )
            global_primal[global_index] = value
            assigned[global_index] = True

    if not np.all(assigned):
        raise AssertionError("condensed global primal contains unassigned variables")
    return global_primal
