# GPU-controlled iterative refinement, v104–v105

The optional `scripts/gpu/prepare_qoco_device_ir.py` postprocessor adds a CUDA
conditional WHILE around actual cuDSS correction solves and the existing residual
arithmetic. The initial norm, acceptance/restoration of the best correction,
iteration budget and tolerance checks execute on device. One final refinement
state is downloaded per linear solve. Normal runtime builders are unchanged.

The cuDSS stream is assigned before data creation. Readiness/completion events
order default-stream operators with the dedicated solver stream. All participating
KKT, NT, vector and norm operations use the selected capture stream. A graph is
built per linear solve: reuse across numerical factorizations has not been proved.

The capture contains two eight-byte cuDSS host range inputs. The patch explicitly
checks their shape and contents before replacing them with retained device
constants. This is a guarded, cuDSS-0.8-specific assumption, not a general vendor
API guarantee. v105 initializes these constants on the solver stream at workspace
creation, removes v104's per-solve host copies and primes only pending sparse
views rather than redundantly computing the full residual before capture.

The scalar controller preserves equal/worse residual restoration, budgets and
the original NaN comparison semantics. Final conic and physics qualification
remains mandatory. Diagnostic controller tests replace only the conditional
setter with an observable flag; they do not test graph capture or vendor lifetime.

Both versions qualified the representative GTOC12 transfer and N20/N500 landing
fixtures against unchanged independent physics and objective gates. v105 passed
24 focused integration tests and seven convergence/accounting tests. Prepared
source reproduction matches all six changed v105 files exactly after LF
normalization.

The completed v105 longer paired benchmark contains 32 fresh processes and 64
qualified trajectory runs, including warm-ups. Sixteen measured samples per arm:
baseline v94 median 612.088 ms, mean 711.578 ms; device IR v105 median 545.038 ms,
mean 682.943 ms. This is about 11% lower median and 4% lower mean on this fixture,
not proof of a general speedup or complete GPU-native execution.

Full v104 conditional-graph sanitizer checks failed (CUDA999/unsuccessful process
termination); the host IR ablation passed memory and initialization checks, timed
out on racecheck and did not reach synccheck. Minimal vendor-free diagnostics in
the earlier conditional-solve checkpoint also fail under instrumentation. This
does not prove every integration failure shares the same cause. The full device
IR path is **not sanitizer-qualified**. All failures and timeout records are
retained alongside successful tests and timing tails in the
[native checkpoint](../artifacts/performance/gtoc12-native-v107-checkpoint.json).
