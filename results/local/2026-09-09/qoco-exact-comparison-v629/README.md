# Exact-QP cold QOCO comparison, v629

One cold QOCO solve of the unchanged v629 snapshot qualifies under the existing common original-equation gates. Native status remains **2, QOCO_SOLVED_INACCURATE**. The unchanged QOCO audit contract accepts native statuses 1 or 2 only when the independent numerical gates also pass. Decimal65 and Linux extended-precision audits agree; no tolerance was relaxed.

The process used one solve, two original numeric updates, 30 interior-point iterations and 177 refinement iterations. All thirteen snapshot settings, coefficients and quadratic structural zeros were preserved. No saved PDHCG point or reference solution was provided. Common normalized primal, stationarity and gap are 3.05783e-11, 1.33647e-14 and 6.97735e-11. Maximum normalized block complementarity is 8.61982e-18; primal/dual cone violations are 0 and 4.05944e-20.

Complete process time was **0.470431497 s**; synchronized host solve time was **0.105751260 s**. Context/reduction scope took 0.235141949 s, topology setup 0.046494122 s, the two updates 0.001341658 and 0.000395180 s, full download 0.000193056 s, serialization 0.014042104 s and cleanup 0.002320991 s. The complete phase table and library counters are in REPORT.md. These are host elapsed scopes, not kernel times; library counters overlap those scopes. The actual child and adopted-descendant inventories were empty, and the GPU lock was released after terminal cleanup.

This single sample establishes a qualified cold baseline on the same QP. It does not establish a matched speedup, median/p95 latency or SOTA result. The separate GPU correction consumed a pre-existing **10,000-step PDHCG point**; that preceding cost is not part of correction timing. Different initial states and setup/context costs must remain explicit in any end-to-end comparison. There is no mission or fleet-score result here.

The archive preserves the standalone driver, frozen source/header archives, all build/CPU-format/rejection logs, failed build-a validation, successful build-b, original snapshot, full raw x/y/z/s readbacks and binary vector hashes, supervisor/process helper, original process-ownership test source, both unchanged numerical-auditor dependencies, saved audit results, and actual compiler/library/runtime identity. Build a's only defect was JSON output of an unsigned-char verbose field as NUL; build b fixes that integer formatting. Neither build ran a solve. The helper's historical CPU test source is included; those subprocess tests were not repeated for this comparison. The build script defines and records the CPU format/rejection cases actually executed.

Compiled executables, solver libraries and cuDSS are omitted. Their exact paths, sizes and hashes are retained as external runtime dependencies in index.json and the manifests. The portable verifier does not reverify those unavailable binary bytes. It verifies every packaged source/header/readback hash and status binding instead. The local numerical audit ran after the timed solve; packaging performs no numerical audit, optimizer, factorization, propagation, native library load or GPU call.

Run the following with Python 3.10 or later, replacing INDEX_SHA256 with the published index digest:

```
python audit_package.py --package . --expected-index INDEX_SHA256 --roundtrip
```

The verifier uses only the standard library. It checks exact safe ZIP membership, byte counts and hashes; both nested source archives and the single format-only delta; source/settings/runtime/report identities; saved numerical-audit and native-status agreement; full-vector FP64 hashes; and terminal cleanup. `--roundtrip` additionally materializes repository-relative files into a fresh temporary directory, rereads every hash, and removes that temporary directory. It never executes archived source. Optional `--output PATH` refuses to overwrite an existing report. The package index binds this README, auditor and attributes, while its separately supplied digest is the trust anchor. Verification output remains outside the immutable package.

The sealed GPU correction/profile evidence is published separately at `../gpu-dual-isolation-v629/`; it is not duplicated here. Historical DESIGN/LAUNCH_PLAN and build-a records describe their original preparation states. The terminal REPORT, run-a readbacks and audit-findings.json are authoritative for this completed one-solve comparison.
