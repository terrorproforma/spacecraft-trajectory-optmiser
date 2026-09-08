# Independent known-point replay audit

All eight imported reference certificates pass the common original-equation KKT gate. Their encoded FP64 primal, equality dual, conic dual, and slack values match the raw imported reference records bit for bit. An independent reconstruction of the equality/scalar/SOC dual ordering also matches the exact native seed. Every pre-step record reports an unchanged internal primal/dual seed, zero seeded iterations, and native `UNSPECIFIED` termination after full reset.

The final result is **one of eight common-KKT passes and zero of eight native-qualified outputs**. All eight native solves report `ITERATION_LIMIT`, with successful API completion and no deadline or recovery event. A certificate's `passes_common_kkt_gate=true` does not mean that the persistent solver reported an accepted termination. The initial records have no accepted solver termination; their `qualified=false` is expected while `passes_common_kkt_gate=true` establishes the supplied certificate's accuracy.

The initial native absolute natural residual is approximately `3.81865e-8` for conditioning and `1.40971e-8` for difficult, in both representations. Both exceed the requested native `1e-9` threshold despite passing the common normalized KKT gate. These are different accuracy predicates. Pre-step metrics match exactly between the one- and 1,000-iteration runs within each problem/representation; bootstrap counters are not counted as seeded work.

| Capture | Representation | Seeded iterations | Final normalized gap | Common KKT gate |
|---|---|---:|---:|---|
| Conditioning | Generic | 1 | 6.04613e-7 | Fail |
| Conditioning | Folded bounds | 1 | 1.27414e-7 | Fail |
| Conditioning | Generic | 1,000 | 4.21818e-7 | Fail |
| Conditioning | Folded bounds | 1,000 | 1.84037e-6 | Fail |
| Difficult | Generic | 1 | 2.49764e-10 | Pass |
| Difficult | Folded bounds | 1 | 1.59897e-7 | Fail |
| Difficult | Generic | 1,000 | 1.23070e-7 | Fail |
| Difficult | Folded bounds | 1,000 | 1.81965e-8 | Fail |

The common normalized gap limit is `1e-9`. All final primal and cone checks pass. Conditioning with one folded step also exceeds the normalized dual limit; both conditioning cases at 1,000 steps exceed the maximum-block-complementarity limit. Native original-equation metrics and fresh Python long-double audits agree on every common-gate outcome. The difficult generic one-step result retains the common gate but has native natural residual `1.86882e-6`, so it still reports `ITERATION_LIMIT` and is not native-qualified.

The maximum change in a primal component is only about `5.68e-11` to `7.11e-9` across these cases, yet seven outputs lose gap accuracy. This demonstrates that a valid imported point is not necessarily preserved by the subsequent updates at this accuracy. It does not identify the numerical cause, prove global divergence, or establish that a larger iteration budget would recover the point. Strict reconstructed folded seed duals remain a separate diagnostic and are not substituted for the supplied original certificate during import.

The audit verified the saved snapshot, encoded point, original QOCO log, neutral point, adapter manifest, tiny report, runtime source/core, and raw-log hash bindings. `source-chain.json` additionally compares the encoded vectors with the selected QOCO repeat zero: conditioning has QOCO status 1 and difficult has status 2, both accepted by the recorded independent audit. Every encoded numerical value matches the recorded source value. Three equality-dual signed zeros per point were normalized during the earlier neutral extraction; the encoded files match those neutral files bit for bit, with no nonzero value changes. Both real captures are unshifted with numerically zero Hessians; the separate analytic suite covers shifted coordinates. These are conic subproblems, not complete nonlinear trajectories.

Recorded work comprises eight executable invocations, sixteen native solve API calls, **4,004 seeded iterations plus eight bootstrap iterations**, and no recovery iterations. Summed native solve-event time is `0.203400 s`; requested-solve wall time is `0.218888 s`; bootstrap wall time is `0.018453 s`; executable wall time is `3.638010 s`. Those scopes exclude queue wait and are not interchangeable. Each configuration was run once, so this is a mapping/stopping diagnostic, not a throughput or statistical performance benchmark. It produces no fleet-score change.

`findings.json` contains fresh audit values, exact identities, bitwise mapping checks, failure predicates, and work counts. `recheck.py` operates only on saved files and writes only this audit directory. The parent publication includes the full input/source/run bundle. No production files were edited and no GPU calls were launched by this audit.
