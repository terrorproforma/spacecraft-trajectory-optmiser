# GPU SOC line-search correction

A captured GTOC12 QP produced a zero affine step even though the original
binary64 inputs permit a step of 0.5534743033528263. One cone incorrectly
restricted the step to zero; its independently calculated bound is
0.8768290586498959. Another cone correctly limits the whole vector to 0.55347.
The reference evaluates the quadratic boundary with 100-digit Decimal
arithmetic, converting each binary64 input exactly.

The SOC boundary is `c + b*alpha + a*alpha^2 = 0`, where the coefficients are
Lorentz inner products. Near the cone boundary, subtracting nearly equal
squared terms loses the positive determinant and can select the zero-step
branch. `qoco_soc_step.cuh` evaluates the coefficients and discriminant with
compensated double-double arithmetic on the GPU, then uses the existing stable
root formula, scalar cap and boundary cases. The original QP, physics gates,
solver tolerances and 0.99 fraction-to-boundary factor are unchanged.

`prepare_qoco_gpu.py` applies the correction to patched GPU builds and includes
the emitted header in its provenance manifest. `--unmodified` remains the
upstream control. The standalone preparer supports frozen prepared-source
diagnostics; it refuses an unexpected source function rather than silently
patching a different implementation.

## Verification and limits

- The old function fails the captured-cone regression. The final header passes
  14 analytic/live boundary cases, including dimensions 3, 4, 33 and 257, plus
  output canaries on RTX 5090 and H100. Four sanitizer modes pass on both GPUs.
- The final libraries pass 38 CLI/trajectory tests on each GPU.
- Original-equation QP qualification: final RTX build 31/32 versus baseline
  28/32; final H100 build 28/32 versus baseline 28/32. These observations do not
  establish a general reliability improvement; qualification failures remain.
- A preceding, numerically identical step-function prototype returned 548.255 kg
  in four complete H100 campaigns, all passing both mission checkers. Median
  runtime was 53.007 s corrected versus 54.105 s baseline (2.03% less time).
  Two observations per mode on one fixture are not a general speedup claim.
- The final packaged H100 library independently completes v360 in 54.010 s,
  returning 548.255 kg and passing both checkers. This confirmation is separate
  from the preceding matched timing comparison.

Fuller compensated Nesterov–Todd normalization was also tested. It reduces
scaling-matrix error on 8,424 actual captured cone pairs from a worst 0.898
relative error to below 1e-15, but does not improve complete-QP qualification.
It remains excluded. The observed trace itself qualified, and its synchronous
instrumentation changes scheduling; it is diagnostic evidence, not a benchmark
or proof that normalization alone causes every failure.

The fleet incumbent remains **12,805.194 weighted kg**. CPU orchestration and
broader solver qualification failures remain outstanding.

Subsequent local diagnostics excluded a combined rounded-endpoint/compensated-NT
candidate (28/32 qualified versus baseline 31/32) and per-factorization GPU KKT
equilibration (16/32 versus 28/32). A captured cone-division audit agrees with a
100-digit reference to 7.5e-16 on valid inputs; subsequent invalid inputs already
contain nonfinite values. [Complete reports and frozen experimental sources](../results/lambda/2026-09-08/gpu-qualification-v371/README.md)
preserve these negative results without enabling the candidates in production.

[Reproducible results, source hashes and visualiser instructions](../results/lambda/2026-09-08/gpu-soc-step-v359/README.md).
