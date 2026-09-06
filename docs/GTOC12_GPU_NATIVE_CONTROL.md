# Device seed and SCvx control

The native GTOC12 path now generates its Lambert/Kepler seed on the GPU, retains
the trajectory there throughout SCvx, and computes merit, nonlinear defects,
virtual-control norms, acceptance, trust updates, polishing and final status in
CUDA kernels. Python calls the native solve once and formats its final result.
The independent CPU physics verifier remains an external acceptance test.

This is **not yet an end-to-end GPU-controlled optimiser**. Initial sparse
topology/conversion, parts of QOCO's setup, its predictor-corrector scalar control,
and native dispatch of the conic solves remain on the host. The CPU also enforces
wall time between subproblems. There is no mid-conic-solve cancellation. These
remaining operations must be removed or explicitly bounded before claiming the
whole numerical pipeline is GPU-native.

The optional [v108–v110 QOCO device-control extension](QOCO_DEVICE_CONTROL.md)
subsequently moves stopping/best-iterate decisions, regularization arithmetic,
combined-RHS scalars and NaN-direction recovery onto the GPU. Native IPM dispatch
and setup remain on the host, and the conditional-IR sanitizer issue remains
unresolved. This extension does not establish an end-to-end speedup.

## Selection

Use the core and QOCO libraries recorded in the checkpoint. The explicit CLI
selection is:

```text
--discretisation-backend cuda --assembly-backend cuda --convex-solver qoco --outer-loop-backend cuda
```

`--seed-backend auto` is the default: it selects GPU seed generation for the CUDA
outer loop and the existing NumPy seed otherwise. `--seed-backend numpy` is an
explicit comparison option with the CUDA outer loop. Requesting CUDA does not
fall back to a CPU numerical implementation. Default CPU refinement is unchanged.

The equivalent Python settings are:

```python
ScvxSettings(discretisation_backend="cuda", assembly_backend="cuda",
             convex_solver_backend="qoco", outer_loop_backend="cuda")
```

## Device ownership and arithmetic

The CUDA outer loop takes the qualified conic primal directly from the retained
QOCO workspace. Interval RK4 propagation uses the existing implementation, with
one independent interval per block. A two-stage reduction computes fuel, the
dead-zone nonlinear penalty, virtual-control L1 and infinity norms, maximum defect
and state/control step. A device controller applies the reference SCvx rules;
parallel kernels copy accepted states and controls into retained reference arrays.
Rejected candidates never overwrite the reference.

Only a 16-byte dispatch command crosses to the native host per outer attempt,
plus one initial command and any polishing refresh. Full trajectory arrays and
outer history are downloaded once at completion. QOCO's existing reports/audits
and internal traffic are additional; `outer_transfer_bytes` does **not** count
all CPU/GPU traffic. With the device seed, trajectory upload bytes are zero.
Fixed boundary/time/fuel metadata is still uploaded during setup.

The seed evaluates the existing 8193-point universal-variable scan in 66 GPU
blocks for the short and long branches. It retains the first exact root or sign
change across valid samples, bisects two roots, compares endpoint velocity costs,
and samples Kepler states in parallel across nodes. Degenerate Lambert geometry
uses the reference's straight interpolation seed on the GPU. Endpoint velocity
clipping and the Gamma seed are unchanged. Kepler convergence is per node instead
of a NumPy-wide stop reduction; parity is numerical, not bitwise.

Every new dynamics/SCvx translation unit disables FMA contraction, like the
existing native GTOC12 implementation. Parallel reductions can still change
rounding and subsequent iteration paths. No feasibility, conic residual, global
objective-gap or independent physics tolerance was relaxed.

## Validation and limitations

The v106 native controller passed 14 decision branches, 10 final-state cases and
eight reduction fixtures, including tails and nonfinite slack outside the state
and control slices. All four CUDA sanitizer tools passed these tests. Full native
SCvx with the host-controlled IR ablation passed memory/leak, initialization and
synchronization checks. Its full regression passed 304 tests.

The v107 GPU seed passed 20 comparisons against the independent NumPy
Lambert/Kepler implementation: normal, long-way, collinear, hyperbolic and dense
sampling, with all four free-endpoint combinations. Those seed tests passed all
four sanitizer tools. Full v107 native transfer and zero-time tests also passed
memory/leak, initialization and synchronization checks with the host IR ablation.
The ordinary v107 test disables the CPU seed and Python numerical-iteration
entrypoints and independently qualifies the resulting transfer against the same
2445.3111007852112 kg reference (1e-5 kg tolerance).
The final v107 regression passed all 324 tests without skips.

The separately selected v105 QOCO library uses GPU-controlled iterative
refinement. Its complete conditional-graph execution still has unresolved CUDA
sanitizer failures. Tests of the new outer loop with host IR isolate the new code;
they do **not** establish that v105's full conditional path is sanitizer-clean.
No unqualified result is promoted as a valid solution.

## Measurements

Each comparison has six rotated pairs, one warm-up and one measured trajectory
per fresh process. All 24 trajectories in each comparison passed independent
physics and the unchanged mass target. Both arms use the same QOCO105 library.

| Comparison | Python outer median | Native outer median | Mean, Python / native |
|---|---:|---:|---:|
| v106, CPU seed in both arms | 568.335 ms | 529.730 ms | 555.503 / 624.251 ms |
| v107, native arm uses GPU seed | 541.044 ms | 589.183 ms | 535.413 / 666.029 ms |

This is verified migration, **not an overall speedup claim**. Slow conic solves
dominate the distribution, and v107's sample median is slower. Preserve every
attempt when comparing convergence and time to the same verified accuracy.

The [checkpoint](../artifacts/performance/gtoc12-native-v107-checkpoint.json)
contains frozen runtime hashes, source snapshots, helpers, all timings and
validation logs. The next work is device QOCO predictor-corrector/stop state and
initial sparse conversion, followed by convergence and batched execution.
