# Exact singleton-bound replay diagnostic

The optional `--fold-singleton-bounds` representation is implemented and independently reviewed. All **seven** corrected tiny GPU checks passed both native stopping and the separate original-equation KKT audit. The source is frozen at `efaabd712df62c095ef3ee782ae561ee5afa8e05`, and every run loads the exact unchanged v603 core SHA256 `d4b0bea9672bb0cea612b7d4e8db5d448f1b79b9831be5710723276edd128633`.

The subsequent two-capture comparison was **negative**: neither generic nor folded representation qualified at 100,000 iterations. Generic mode reproduced the baseline metrics, while folding worsened the reported residuals. The option remains off by default. The parent publication README links the complete real comparison; these small tests establish conversion/lifecycle correctness, not a solver speed advantage or trajectory-backend integration. This evidence includes no subsequent tuning or further GPU runs.

## Exact conversion and dual reconstruction

A nonnegative row is a singleton only when exactly one stored coefficient is unequal to zero. Tiny and subnormal nonzero coefficients count; structural zeros elsewhere remain. An inequality `a*x[j] <= h` can become a native upper bound when `a>0`, or lower bound when `a<0`. Folding occurs only if `h/a` is provably exactly representable in binary64. The proof decomposes the inputs into odd integer mantissas and binary exponents, then checks integer divisibility and exponent range. Non-binary or out-of-range quotients remain as original scalar rows, with explicit counters. This prevents rounded quotients from collapsing a contradictory interval. All-zero rows remain, including constant infeasibility; exact conflicting native intervals are rejected before CUDA.

Bounds intersect at the tightest lower and upper values. When several rows attain the same tight bound, the deterministic owner has the largest coefficient magnitude, then the earliest original row index. Equality and retained scalar dual mappings survive row compression; SOC duals begin at the new scalar-row count. Folded multipliers are reconstructed from the original-coordinate stationarity normal: a positive normal can use a tight upper bound and a negative normal a tight lower bound. Fixed variables support either sign. Original `x/y/z/s` are exported and reaudited; no primal value is repaired.

Reconstruction requires exact contact between the local primal and native bound. An interior or one-ULP-off point receives no folded multiplier. The output reports off-contact normals, one-ULP misses and maximum bound distance, so conservative false unqualification remains observable. Nonrepresentable multipliers are explicitly unsupported and unqualified, rather than clamped. The unchanged original feasibility, dual-cone, stationarity, objective-gap and per-block complementarity gates remain decisive.

## Validation and failure history

The strict CPU test covers upper/lower/fixed/duplicate bounds, sign and shifted-coordinate reconstruction, structural zeros, tiny second coefficients, non-binary division, exact non-unit division, subnormals, ratio and multiplier overflow/underflow, conflicting bounds and deliberate one-ULP loss of active support. Numeric parsing uses `from_chars` so representable subnormal capture values are retained without an epsilon cutoff.

The original v604 attempt exposed a no-SOC C-ABI binding mismatch; its failed create call and three preceding passes remain in `initial-v604`. The corrected adapter supplies an empty affine-offset view only when there are no affine rows, while retaining normal CSC offsets for CPU calculations. The production C ABI and numerical core were not changed.

The v604b GPU batch used tolerance `1e-10`, 100,000 maximum iterations and ten-second solve deadlines. Default mixed and default pure-bound cases took 75 iterations; folded mixed and shifted cases took 75; folded duplicate bounds took 25; fixed-bound cold/retained cases each took one. All seven passed the independent common gate, all recovery counts were zero, and all folded cases had zero off-contact normals. No timing multiplier should be inferred from these tiny fixtures.

## Reproduction

The WSL build is `/home/angus/spacepdhcg-persistent-replay-v604b`. Its executable SHA256 is `d44ce03670072f2e192e47458248bd8afd1df23e206286b0a3f00041775974df`; manifest SHA256 is `34c82c97fad913af9664d48a9111996636ba3aed18d0a704805d762acc1eb4b4`. `source.tar.gz` freezes the complete C++ source, and `manifest.json` records actual compilation commands and every source hash. Linking against the existing v603 core isolates the representation change.

```bash
BASE=/home/angus/spacepdhcg-persistent-replay-v604b
"$BASE/build/cuda-tests/persistent_snapshot_replay" "$BASE/fixtures/bounds-duplicate.txt" \
  --fold-singleton-bounds --tolerance 1e-10 --iterations 100000 --deadline-seconds 10
```

Hold `/home/angus/.spacepdhcg-gpu.lock` during direct GPU invocation. The supplied `run_tiny.py NEW_OUTPUT_NAME` enforces a nonblocking shared lock, refuses existing outputs, verifies the core/executable identities, runs the finite batch and independently audits its original vectors. CPU-only inspection can use `--fold-singleton-bounds --validate-only` with `CUDA_VISIBLE_DEVICES` empty.
