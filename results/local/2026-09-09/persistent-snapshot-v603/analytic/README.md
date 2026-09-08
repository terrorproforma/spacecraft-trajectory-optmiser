# Persistent snapshot diagnostic adapter validation

This bundle validates importing existing `SPACEPDHCG_QOCO_QP_V1` conic captures into the production persistent C ABI. It does not integrate PDHCG into the GTOC12 SCvx route solver and does not qualify nonlinear trajectory physics or establish a performance advantage.

The isolated WSL build is `/home/angus/spacepdhcg-persistent-replay-v603`. `manifest.json` records all frozen C++ source hashes, build commands, source commit, executable and loaded core hashes. `source.tar.gz` contains that source. The source base is the validated v651 tree; the only overlays are the three new replay/test files and their CMake targets. No production numerical defaults changed.

## Mapping

The capture is `min 0.5*x'*P*x+c'*x`, `A*x=b`, `G*x+s=h`, with nonnegative rows followed by standard SOC blocks `(radius, vector...)`. Upper-triangular `P` is mirrored once, preserving explicit structural zeros. Equality rows become scalar bounds `[b,b]`; nonnegative rows become `G*x <= h`. Each SOC uses `F=-permutation(G)`, `offset=permutation(h)`, with the radius last. Persistent SOC `vector_dimension` is the captured cone size minus two.

The original equality and scalar upper duals retain their signs. The SOC dual is the negative inverse permutation of the persistent affine normal. For shifted captures the local primal is `u=x-origin`; translated coefficients and objective offset are checked before CUDA. Exported `x/y/z/s` use original coordinates. `s` is explicitly reconstructed as `h-G*x` because the C ABI exposes no independent slack vector.

Unsupported inputs fail before CUDA: SOC size two, unsorted or duplicate CSC topology, lower-triangular or full symmetric P storage, nonfinite/malformed/truncated captures, inconsistent translated coefficients, and problems beyond bounded diagnostic allocation limits. The solver currently selects CUDA device zero and fingerprints the loaded library using Linux `dladdr`. Convexity is proved only for zero or nonnegative symmetric diagonally dominant Hessians; other potentially PSD Hessians are accepted with an explicit assumption marker. A negative diagonal is rejected.

## Verification and evidence

The CPU test compiles with warnings as errors and passes exact mixed equality/scalar/SOC KKT fixtures, shifted objectives and dual signs, adjacent SOC sizes, non-diagonally-dominant PSD acceptance, malformed input cases, structural zeros, SHA-256 reference vectors and a block-complementarity cancellation regression. `lock-busy-attempt` records successful CPU-only validation with GPUs hidden and the first unsuccessful lock acquisition; no GPU solve ran in that attempt.

`tiny-success/report.json` records **six independently qualified GPU solves**, using the unchanged core tolerance `1e-10`, maximum 100,000 iterations and a ten-second solve deadline. All stopped optimal with no recovery iterations. Cold original/shifted solves took 75/100 iterations. The second reused-workspace solve took 75 iterations; full retained state took one iteration. These tiny fixtures test lifecycle behavior, so their timings are not throughput or trajectory-speed benchmarks.

Both the C++ audit and the separately implemented Python audit use original equations with FP64 parsed inputs and long-double products. The common gate is normalized primal residual, dual residual, objective gap and maximum scalar/SOC block complementarity at most `1e-9`, plus primal and dual cone violations at most `1e-8`. Complementarity normalizes by `max(1, abs(primal objective), abs(dual objective))`; the per-block gate prevents cancellation in the global sum. SOC violation means `max(0, norm(tail)-head)`, not Euclidean projection distance. Native stopping and the external gate are reported separately.

## Commands

From WSL, with the shared GPU lock held by the invoking harness:

```bash
BASE=/home/angus/spacepdhcg-persistent-replay-v603
"$BASE/build/cuda-tests/persistent_snapshot_replay" "$BASE/fixtures/mixed-shifted.txt" \
  --tolerance 1e-10 --iterations 100000 --deadline-seconds 10 \
  --mode full-retained --repeats 2
```

`--mode cold` recreates the workspace; `reuse` resets iterates while retaining scales; `full-retained` selects the exact production FULL_RETAINED warm-start enum. Independently unqualified predecessors are discarded and `fresh_workspace` makes every fallback explicit. A new batch can be run with `python3 run_tiny.py NEW_OUTPUT_NAME`; the driver itself enforces the shared nonblocking GPU lock and refuses existing output directories.

The stdout prefixes are `PERSISTENT_REPLAY_META`, `PERSISTENT_REPLAY` and `PERSISTENT_REPLAY_SUMMARY`, each followed by one JSON object. Completed unqualified solves still return process status zero; consumers must check `qualified_original`, the native termination enum and their independent audit. `repeat_seconds` covers setup through audit and cold/failed cleanup. It excludes shared parsing/library identification/CUDA initialization and JSON output. `process_seconds` covers argument parsing through final workspace cleanup and prior output. Native `peak_workspace_bytes` excludes borrowed input/output buffers. These scopes are also embedded in metadata.
