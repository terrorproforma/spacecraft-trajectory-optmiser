# Banded joint equality/L1 CPU reference: 0/2 qualify

One fixed worker completed the two prescribed cold inputs, changed stages of the same synthetic fixture. Both charged zero-start joint proximal calls passed. The early case failed its first outer attempt; the near-converged case committed 73 updates and failed attempt 74. No GPU or native solver ran.

There were 77 inner calls, 66 accepted directions, two successful full factors and two successful active factors. Both cases stopped at a failed merit-decrease check after Newton backtracking; neither reached an inner cap or timed out. Complete worker process time was 0.71946 seconds, including imports, setup, factors, initial calls, failed inner work, readouts and checks. This short failed run is not a speed improvement over completed 10,000-update runs.

All six saved original points fail the unchanged long-double and Decimal65 KKT gates. Virtual L1 cost and pair complementarity are zero on the saved candidates, while original stationarity, cone and gap errors remain. The failure candidates and last committed points have distinct roles. Native termination zero is an auditor-schema sentinel, not a claimed native solve result.

Saved-state arithmetic reproduces the failed inner metrics exactly. Early is blocked by normalized equality-dual debt 1.72282543e-10; near by normalized equality residual 4.06062792e-10, versus the fixed 1e-11 budget. The line search ignores a trial's passing residual flag before Armijo. Actual rejected trials/decrease values were not retained, so a discarded passing trial is not established. A separate future change can check trial residuals first and evaluate stable dual quadratic decrease; this package preserves the original failures and policy.

The ZIP contains the complete isolated v633 source, exact snapshots, unchanged v631 metric certificates, runtime identities, tiny and environment tests, worker/supervisor, all six readbacks, traces, unchanged audit sources/results, saved diagnosis and report. It also includes exactly the sealed eight-file v632 tiny evidence plus its index (56,515 indexed bytes); no compiled/runtime binaries, credentials or caches are included. A WSL startup failure before the diagnosis is retained; the successful diagnosis used Windows stdlib without new factors or directions.

The prior [v631 reference](../core-primal-reference-v631/README.md) explains the original equality/virtual-control obstruction. The repository [design and result summary](../../../../docs/PDHCG_JOINT_PROX_DESIGN.md) explains the joint split. All inputs required to verify these saved bytes are included; these links add context only.

Run the portable static verifier from this directory:

    python audit_package.py

It verifies every member/size/hash, the frozen source and runtime records, v632's sealed eight-file index, all work counts and saved original audit/vector bindings. It executes no archived numerical source, solver, factorization, propagation or GPU code. Runtime binaries are identified by hash only and are not recreated or independently reverified. The readout audit time (0.34964 seconds) is separate from the worker process time.
