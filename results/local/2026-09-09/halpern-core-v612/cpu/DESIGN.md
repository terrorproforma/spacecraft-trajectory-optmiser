# Bounded reflected Halpern experiment

This is an optional diagnostic replay algorithm. It does not select a new production GTOC12 backend, change the common accuracy gate, or claim a fleet-score or throughput gain.

The native API requires the common-KKT policy, exactly zero stored Hessian coefficients, free primal variables, an equality prefix followed by scalar upper rows, standard affine SOCs, and a supported positive cooperative grid. The replay also requires an unshifted snapshot and explicit positive `--execution-blocks`. Unsupported domains and nonzero coefficients, including subnormals, are rejected rather than approximated.

For the normal-dual sign convention `c + K^T y = 0`, each map first calculates

```
x_plus = x_work - (eta / omega) * scaling_x * (c + K^T y_work)
y_plus = prox_conjugate(y_work + eta * omega * scaling_y * K(2*x_plus - x_work))
R = 2*(x_plus, y_plus) - (x_work, y_work)
work_next = anchor/(inner+2) + (inner+1)/(inner+2) * R
```

The proximal point `(x_plus, y_plus) = T(work)` is the point exported and audited. The anchor, working point and reflection are private buffers; they may lie outside the dual cone. Every solve starts a new anchor and unit weight from the current public iterate. Disabling the option copies the exported primal into the ordinary solver's previous/extrapolated history.

`--halpern plain` uses this fixed-weight Halpern map. `--halpern adaptive` adds the pinned upstream restart structure: inspect every 200 iterations, force the first restart at 200, then use 0.2 sufficient reduction, 0.8 necessary reduction with worsening, or `inner >= .36*total`. The fixed-point metric is evaluated before the blend. Restart copies the proximal point into the anchor and working state, resets the inner counter, and measures the first new epoch reference at the next proximal step (201 after the first restart).

The metric for `delta = R(work)-work` is

```
omega * ||delta_x_scaled||^2 + ||delta_y_scaled||^2 / omega
    - 2*eta * delta_y_scaled^T K_scaled delta_x_scaled
```

The minus sign follows the native dual convention. If the metric is negative or arithmetic becomes nonfinite, the diagnostic stops with numerical failure; cancellation has priority.

Scaling and base steps retain the current 10 Ruiz / 20 power-iteration heuristic and 0.9 factor. The 20 power iterations do not prove an upper spectral bound, so this is an explicit experimental SPD precondition, not a universal convergence guarantee. Independent numerical norm estimates for the two selected captures give step products about 0.89805 and 0.90819. These estimates support the chosen local diagnostic and are not a proof for arbitrary input.

At restart, scaled displacement norms from anchor to proximal point drive the pinned 0.99/0.01 PI weight update with 0.3 integral smoothing and the original finite distance/ratio guards. This implementation deliberately uses the common normalized primal/dual residual ratio in that guard, which differs from the upstream native residual definition. Zero/nonfinite residual ratios fall back to the best previously recorded weight. Overflowing weight arithmetic is a numerical failure, never a silently clamped weight. Reciprocal primal/dual weight factors leave the base step product unchanged.

The private B/O scaling metadata is refreshed on enable. Checkpoint/restore are rejected while the option is active, preserving the old checkpoint ABI. Numeric updates remain subject to the common-policy restriction; disable/update/re-enable validates the domain anew.

Default versus plain changes both iteration ordering and reflected/anchored state. It is not a single-mechanism ablation. Plain versus adaptive isolates the restart/weight bundle. All arms must use the same captured equations, frozen binary, fixed grid, iteration/deadline caps, and original-equation external KKT gate.

The first build, v612a, compiled and passed CPU conversion/parser checks but changed compiler outlining for existing cooperative kernels. It has no GPU results. The isolated report-helper revision is v612b; resource inspection and independent review precede any GPU launch. Failed/superseded attempts are retained.
