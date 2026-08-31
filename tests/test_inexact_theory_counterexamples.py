from __future__ import annotations

import math
from itertools import pairwise

import pytest


def test_fixed_inner_error_floor_prevents_stationarity() -> None:
    """C1: exact quadratic model plus a constant inner error stalls at epsilon."""

    epsilon = 1.0e-3
    x = 3.0
    for _ in range(50):
        exact_step = -x
        returned_step = exact_step + epsilon
        x += returned_step
    assert x == pytest.approx(epsilon)
    assert abs(x) > 0.0


def test_quartic_stationarity_residual_has_no_linear_error_bound() -> None:
    """C2: dist/residual = 1/x^2 is unbounded for f=x^4/4."""

    ratios = []
    for exponent in range(1, 9):
        x = 10.0 ** (-exponent)
        distance = abs(x)
        residual = abs(x**3)
        ratios.append(distance / residual)
    assert all(later > earlier for earlier, later in pairwise(ratios))
    assert ratios[-1] > 1.0e15


def test_inadequate_l1_penalty_rewards_constraint_violation() -> None:
    """C3: min -x, x=0 has merit -x+rho|x|, which decreases for rho<1."""

    rho = 0.5

    def merit(x: float) -> float:
        return -x + rho * abs(x)

    assert merit(1.0) < merit(0.0)
    assert merit(10.0) < merit(1.0)


def test_collapsing_scaling_can_fake_residual_convergence() -> None:
    """C4: scaled residual vanishes while the canonical residual remains one."""

    canonical = 1.0
    reported = [canonical / (iteration + 1) for iteration in range(1, 1001)]
    assert reported[-1] < 1.0e-3
    assert canonical == 1.0


def test_knot_only_path_checks_miss_mid_interval_violation() -> None:
    """C5: endpoints are feasible while the interval midpoint violates the path constraint."""

    def path_constraint(time: float) -> float:
        return 4.0 * time * (1.0 - time) - 0.5

    assert path_constraint(0.0) < 0.0
    assert path_constraint(1.0) < 0.0
    assert path_constraint(0.5) > 0.0


def test_resolve_before_shrink_distinguishes_solver_and_model_error() -> None:
    """C6: the zero step is poor because the inner KKT residual is large, not the model."""

    x = 0.0

    def objective(value: float) -> float:
        return 0.5 * (value - 1.0) ** 2

    under_solved_step = 0.0
    exact_step = 1.0
    inner_residual = abs((x + under_solved_step) - 1.0)
    assert inner_residual == pytest.approx(1.0)
    assert objective(x + exact_step) < objective(x + under_solved_step)


def test_omitting_nonanticipativity_solves_an_unattainable_policy() -> None:
    """C7: independent scenario controls beat every pre-observation shared control."""

    preferred = (-1.0, 1.0)
    independent_cost = sum((control - target) ** 2 for control, target in zip(preferred, preferred))

    def shared_cost(control: float) -> float:
        return 0.5 * sum((control - target) ** 2 for target in preferred)

    candidates = [index / 100.0 for index in range(-200, 201)]
    common_optimum = min(shared_cost(control) for control in candidates)
    assert independent_cost == 0.0
    assert common_optimum == pytest.approx(1.0)


def test_summable_schedule_is_finite_but_harmonic_schedule_is_not() -> None:
    """Property check for A11: p>1 is summable; p=1 keeps growing logarithmically."""

    summable = sum(1.0 / ((index + 1) ** 1.25) for index in range(200_000))
    harmonic_prefix = sum(1.0 / (index + 1) for index in range(200_000))
    assert math.isfinite(summable)
    assert summable < 5.0
    assert harmonic_prefix > 12.0
