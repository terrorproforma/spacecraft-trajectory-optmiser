# GPU-controlled cuDSS solve experiment

This is a feasibility experiment, not a production optimization or a speedup claim. The production iterative-refinement loop and its accuracy gates are unchanged. The diagnostic preserves the original solution, runs additional solves against the same factors, and then resumes the original algorithm. The full GPU-native goal remains incomplete.

The probe is `cpp/cuda/tests/qoco_cudss_conditional_probe.cuh`, injected only into isolated QOCO builds. The standalone `cuda_conditional_copy_probe.cu` isolates CUDA graph behavior without QOCO or cuDSS. Neither is selected by the normal build.

## Findings

Assigning a nonblocking stream after cuDSS data creation produced a misleading successful capture with only a copy and memset. No solve kernels were captured, and replay failed. Assigning the stream before data creation captured eight kernels, five copies, and five memsets on the tested systems. This establishes a stream-lifetime requirement for the next implementation; it does not establish that changing streams later works reliably.

Two captured eight-byte uploads used ordinary host memory. CUDA conditional bodies require their copy operands to be device-accessible. An isolated diagnostic copied those inputs into retained GPU allocations and changed the corresponding graph copies to device-to-device. Their observed bit patterns were `[0, Kn-1]` interpreted as two 32-bit integers, on all 15 systems inspected. This interpretation is inferred from the observations, not a supported cuDSS interface. The diagnostic rebuilds the graph for each factor state; it does not establish safe reuse across refactorizations or changed vendor allocations.

The conditional WHILE body then executed entirely on the GPU. Zero, one, and three executions, repeated graph launches, and the zero-execution output invariant all passed on five successive systems from each trajectory fixture:

| Fixture | KKT dimension | Control/replay checks | Strict numerical comparison |
| --- | ---: | ---: | --- |
| GTOC12 synthetic leg | 5,752 | 40 passed | 35 of 35 ordinary/conditional replays failed |
| PD6, 20 intervals | 3,026 | 40 passed | 35 of 35 passed |
| PD6, 500 intervals | 74,066 | 40 passed | 35 of 35 passed |

The comparison is `max(abs(replay-original)) <= 1e-12 * max(1, max(abs(original)))`. cuDSS uses nondeterministic atomics in its default mode. Ten additional ordinary GTOC12 solves also failed this comparison, while the corresponding 20 landing solves passed. True unrefined KKT residuals were recorded for both ordinary and captured solves. For example, GTOC12's first system had initial residual 0.00979656, ordinary repeats 0.01706576 and 0.00791337, and ordinary graph replay 0.00632483. This is evidence of existing pre-refinement variability, not proof that capture is numerically equivalent in every case.

All three full uninstrumented trajectory runs resumed the original algorithm and qualified. GTOC12 final mass was 2445.311100785131 kg against the unchanged 2445.3111007852112 kg reference and 1e-5 kg gate. The two landing runs independently certified. These successes validate those original-algorithm runs with diagnostics present; they do not qualify a GPU-controlled refinement implementation, which is not yet connected.

## Failed variants and validation limits

| Build | Purpose | Outcome |
| --- | --- | --- |
| 95 | Change stream after data creation | Capture omitted solve kernels; replay failed |
| 96 | Set stream before data creation | Captured solve; conditional instantiation rejected host copy operands |
| 97 | Inspect copies and require bitwise replay | Ordinary graph comparison failed |
| 98 | Quantify replay error, try GPU snapshots | Both variants failed the strict scaled comparison on GTOC12 |
| 99 | Record ordinary-solve controls, true residuals, and conditional behavior | Controls pass; numerical outcomes above remain explicit |
| 100 | Standalone nested child graph, with/without copy nodes | Normal runs pass; both memory checks fail with CUDA error 999; other six tool runs pass |
| 101 | Standalone directly populated conditional body | Normal runs pass; both memory checks and kernel-only racecheck fail; other five tool runs pass |
| 102 | Add synchronization before graph launch | Normal copy case passes; memory check still fails on replay; remaining runs interrupted by computer reset |
| 103 | Fresh post-reset build of the synchronization diagnostic | Both normal runs and six race/init/sync runs pass; both memory checks still fail |

QOCO99's full PD6 N20 memory check failed at the first nonzero conditional replay with CUDA error 999. The process exited before cleanup, producing 239 outstanding-allocation reports; this is not a successful leak check. Racecheck terminated with exit 11 despite reporting zero hazards. Initialization checking completed successfully. Synchronization checking failed with CUDA error 999. Planned N500 sanitizer checks were not reached. No blanket memory, race, or synchronization safety claim is justified.

The standalone failures reproduce without solver/vendor code, with both nested and directly populated bodies, and with copy nodes removed. That narrows investigation toward the local CUDA/driver/instrumentation interaction. It does not prove every failure in the solver probe has the same cause. An NVIDIA forum report describes related conditional graph sanitizer failures and an acknowledged investigation; its proposed pre-launch synchronization did not resolve this local probe.

The reset also removed the v102 executable from its WSL directory while its partial Windows-side results survived. Its binary identity cannot be recovered from those results; no hash is claimed. Build103 uses a fresh directory and records the retained source and executable.

## Reproduction and next implementation

`artifacts/performance/qoco-conditional-solve-v99-checkpoint.json` retains runtime hashes, prepared source overrides for builds 95–99, helper sources, raw outcomes, and reset interruption details. Frozen runtime directories are `/home/angus/build-qoco-gpu-conditional-probe-v95` through `v99`; all use the unchanged native core93 and cuDSS0.8. The original build helpers read the repository's current probe header, so reproducing an older variant requires restoring its recorded prepared overrides before compilation. Never overwrite the frozen builds.

The next production work is to retain a solver stream from creation, move true residual reduction and best-solution/stop decisions onto the device, and preserve the original refinement semantics. Graph-owned buffers and vendor data must have matching lifetimes. Reuse across numerical updates, cancellation/deadline behavior, difficult GTOC12 convergence, full independent qualification, and paired end-to-end timing still need validation. The host-scalar snapshot approach remains diagnostic until its lifetime and numerical assumptions are proved or replaced.

References: [cuDSS streams and graph support](https://docs.nvidia.com/cuda/cudss/general.html), [cuDSS stream and memory APIs](https://docs.nvidia.com/cuda/cudss/functions.html), [CUDA12.8 conditional graph requirements](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#conditional-graph-nodes), and [reported sanitizer/conditional-graph issue](https://forums.developer.nvidia.com/t/compute-sanitizer-and-cuda-graph-false-positives/373484).
