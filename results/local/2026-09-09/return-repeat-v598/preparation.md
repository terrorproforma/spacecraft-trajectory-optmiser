# Isolate the H100 second-route return failure

Status: saved evidence audited and local replay fixture prepared. **No GPU replay
has been launched by this preparation.** Six CPU preflight passes stopped at the
native function boundary, recording zero CUDA calls and identical mathematical
input hashes within each of the two mass variants.

## Finding

The four H100 v636 runs use the same plan, epochs, settings, Python source hashes
and native library hashes. Their return-leg inputs are **not bit-identical**:
small upstream differences propagate into the initial mass on leg 17.

| Run | Return initial mass (kg) | Outer iterations | Return outcome |
| --- | ---: | ---: | --- |
| baseline0 | 1339.2959931031482 | 14 | Certified |
| candidate0 | 1339.2959931034932 | 14 | Certified |
| candidate1 | 1339.2959931028731 | 44 | `virtual control remains 1.380e-01` |
| baseline1 | 1339.2959931033315 | 14 | Certified |

The maximum difference is **6.200480129336938e-10 kg**, or
**4.629656297986603e-13** relative. The first leg starts at the same recorded mass
and epochs in all four runs; its independently propagated final masses already
span 7.571543392259628e-10 kg, and its first conic solve takes 33/36/34/34 iterations.
Every preceding leg certifies, with the same outer iteration counts across runs.
Raw native tensors and workspace state were not captured in the original campaign,
so the records do not establish that every original native input or initial
workspace byte was identical, even on the first leg.

The failure has a specific numerical branching point. Its return conic relative
gaps at outer iterations 8 and 9 are 1.1957453819789713e-9 and
1.0994670497726726e-9, just above the unchanged 1e-9 qualification gate, while
primal and dual residuals satisfy that gate. Iteration 10 remains unqualified.
The existing native controller retries an inaccurate result unchanged twice, then
shrinks the trust region. Successful baselines qualify at iteration 9 and reach
near-zero penalty by iteration 10. The failed run obtains qualified conic results
only at iterations 7 and 20, ultimately exhausting 40+4 attempts with nonzero
virtual control. The stored reports do not include its complete outer trust/
acceptance history; the replay now captures that history.

This points to conic qualification and nonlinear convergence sensitivity; it does
not establish whether the small input-mass perturbation or nondeterministic
linear algebra caused the branch. A device-selection effect is also not proved:
the emitted proxies are identical and another run with selection enabled succeeds.
The next step is to reproduce the instability, not relax the qualification gate.

The successfully certified second route carries 608.1587953456536 kg and retains
15.47678768 kg of propellant after unloading. Its full fleet scores approximately
12,808.703251580 fixed-bonus kg, below the retained 12,810.135953049. Recovering this
particular missed route would improve solver reliability; it would not promote a
better incumbent. The failed summary's zero collected mass is a failure sentinel,
not the payload to use when reconstructing its return boundary.

## Exact fixture and bounded replay

`audit.json` records all 18 legs' per-run masses/statuses, input hashes and complete
return conic reports. `fixture.json` reconstructs the return boundary with the
pinned catalogue and the original pipeline's mining calculation:

- Asteroid 13077 to Earth, MJD 69230 to 69805: **575 days, 289 nodes** at the
  existing 2-day grid, including the final one-day interval.
- Carried ore: 608.1587953456536 kg; minimum final mass: 1108.1587953456537 kg.
- Fixed position and velocity states, Earth arrival excess velocity allowed,
  no free departure excess velocity.
- Two exact initial masses, preserved as both round-trippable decimal and
  binary hexadecimal: failed `candidate1` and successful `candidate0`.
- Same six-call order: failed-mass / successful-mass, repeated three times in
  one process. Each call uses the existing cold-start `solve_leg`; there is no
  warm-start injection, mass rounding, tolerance change or retry-policy change.
- Same SCvx settings as the original: CUDA seed/discretization/assembly/outer loop,
  GPU QOCO, 40+4 outer iterations, 8/16 integration substeps, 1e-9 conic tolerance,
  5e-9 defect tolerance, QOCO Ruiz zero, graph execution. Original 900-second per-leg
  timeout is retained; a separate 600-second soft campaign limit stops between calls.

The script requires an explicit `--execute` to invoke CUDA and enforces both
validated local binary hashes. It acquires the shared GPU lock nonblocking and
makes **at most six return solves**, without replaying 17 preceding transfers.
It stores actual native numerical-input arrays, hashes, every outer-history row,
conic reports, output arrays and the ordinary pipeline clamp/independent rollout
for admissible solver statuses. Reported infeasible outputs remain rejected.
No route or fleet is promoted by this diagnostic.

Example for the root agent after selecting a frozen source snapshot and fresh
output directory, with the same cuDSS/CUDA loader paths as the v597 run:

```bash
export SPACEPDHCG_GTOC12_CUDA_LIBRARY=/home/angus/build-spacepdhcg-joint-v596/build/cuda/libspacepdhcg_cuda.so
export SPACEPDHCG_QOCO_LIBRARY=/home/angus/build-qoco-scaled-pool-v540/final/libqoco.so
export SPACEPDHCG_GTOC12_DATA=/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data
/home/angus/worktrees/spacepdhcg-literature-venv/bin/python \
  build/performance/return-repeat-v598/run.py \
  --repo /path/to/validated/frozen/source \
  --fixture build/performance/return-repeat-v598/fixture.json \
  --output /path/to/new-unique-output-directory \
  --max-solves 6 --wall-seconds 600 --execute
```

Omitting `--execute` performs only CPU construction and captures the inputs before
the native call. The completed preparation is in `prepared-native-inputs/`.
Native input hashes cover all mathematical arrays, physical scalars and settings;
they exclude the wall-clock start and the remaining timeout calculated immediately
before the C call. Preparation writes add a small amount of setup time, which is
counted in the unchanged per-leg timeout.

| Artifact | SHA-256 |
| --- | --- |
| `fixture.json` | `783caf8a0e14196111650d5e5fbef997b5324836017e6e9ada9b1409584c9259` |
| Failed-mass numerical input envelope | `2c54a5753bd5f2a1e13b7faf16f75bf9440027064fdc0367975ecb92dc5e401a` |
| Successful-mass numerical input envelope | `1ea3bc03e69a3678968ac3ce21f4db53e01103e5c2165d47e42374a1802079d9` |
| Required v596 core | `86d7fdc952e82f7e5557c1651a1722900d83cdd98d8a709a4b6274fd83af4671` |
| Required local QOCO540 | `0cc27a1d8bde1edd74f7ef00e04ec64c2d6c0f8fe526a1d94b04d509a74b3315` |

If identical mathematical input hashes yield different outcomes, the replay
demonstrates repeatability failure within that process. Consistent outcomes that
depend on the mass variant instead support numerical input sensitivity. If all
six certify, the local replay has not reproduced the H100 failure: hardware,
library builds, upstream process state and the small sample remain limitations.
Any later solver change must preserve the original qualification and independent
physics checks and be retested against both variants.
