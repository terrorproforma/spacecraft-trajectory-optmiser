"""Adaptive replay must resolve every change of thrust polynomial."""

import json
from itertools import pairwise
from pathlib import Path

import numpy as np

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.solution import make_burn_arc
from spacepdhcg.gtoc12.verifier import LagrangeThrust, PropagatedHistory, propagate_burn


def test_burn_mass_matches_independent_norm_quadrature():
    data = json.loads((Path(__file__).parent / "fixtures/gtoc12_lagrange_replay.json").read_text())
    times = np.asarray(data["times"])
    thrust = np.asarray(data["thrust"])
    initial = np.asarray(data["y0"])
    interpolant = LagrangeThrust(times, thrust)
    masses = []
    # Fuel loss depends only on the emitted thrust, independent of the orbit
    # integrator and optimiser. Two quadrature orders check reference accuracy.
    for points in (16, 32):
        nodes, weights = np.polynomial.legendre.leggauss(points)
        impulse = 0.0
        for lo, hi in pairwise(times):
            epochs = (lo + hi) / 2 + (hi - lo) / 2 * nodes
            impulse += (
                (hi - lo) / 2 * np.dot(weights, np.linalg.norm(interpolant.sample(epochs), axis=1))
            )
        masses.append(initial[6] - C.MASS_FLOW_PER_NEWTON_KG_S * impulse)
    assert abs(masses[0] - masses[1]) < 1e-10
    arc = make_burn_arc(times / C.DAY_S, thrust)
    history = PropagatedHistory()
    _, _, mass, _ = propagate_burn(
        0.0, initial[:3], initial[3:6], initial[6], arc, history=history, sample_days=3.7
    )
    assert abs(mass - masses[1]) < 1e-8
    assert np.all(np.diff(history.epochs_mjd) > 0)
    assert history.epochs_mjd[-1] == arc.end
    assert history.masses_kg[-1] == mass
