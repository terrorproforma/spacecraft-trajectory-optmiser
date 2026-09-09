# What the result establishes

v627 demonstrates a concrete SCvx correction: including the original endpoint residuals in the merit function lets a useful seeded trajectory correction proceed, where the same fixed route previously stopped after 18 rejected outer steps. The changed leg now accepts four steps; the full route and waits are certified with sequential physical mass carry-forward. v628 then inserts that certified route into the current fleet and passes both original full-fleet checkers with a positive score gain.

The mission uses the native QOCO backend. This is progress in trajectory refinement and fleet mass, not a demonstration of PDHCG convergence or state-of-the-art PDHCG throughput. The isolated v626 GPU dual correction passes the original numerical KKT audit, but its frozen numerical-quality test rejects it and its inherited PDHCG termination remains `ITERATION_LIMIT`. Explaining the quality rejection does not retroactively make that run accepted.

Three claims still need separate evidence:

- **Native solver qualification:** an integrated PDHCG path must return accepted solutions under every unchanged original-coordinate gate, including global gap and cone complementarity, on representative changing SCvx subproblems. A saved-point correction on one favorable exact-zero face is insufficient; unsupported and failed cases must remain counted.
- **GPU pipeline scope:** the current path includes host orchestration, ephemeris work, transfers and full-fleet CPU verification. Report those scopes honestly. A claim that the numerical pipeline runs on GPU requires an implemented, measured path; a hybrid implementation may still be useful, but must be identified as such.
- **Matched score and runtime:** compare the same problems, initial conditions, required accuracy, hardware and complete mission acceptance against a strong qualified baseline. Count setup, initialization, transfers, inner solves, retries, correction, failed candidates, trajectory certification and final fleet checking. Compare time to the same verified quality or verified score at the same total budget. Neither a reduced kernel time nor a score increase on one route establishes SOTA.

No new algorithm, optimization, propagation or benchmark was run to produce this scope assessment. It describes the limits of the completed evidence.
