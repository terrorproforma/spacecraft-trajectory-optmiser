# Retrieved Lambda run status - v637

The already retrieved reports show partial trajectory progress and no certified complete route or fleet gain. Their `success` field describes process completion. Every attempted route has `certified=false`, and all four reports have zero fleet checks.

| Saved run | Process result | Reported trajectory outcome | Native solves | Cache hits | Campaign wall seconds |
|---|---|---|---:|---:|---:|
| v851 | Complete, success | Two initial legs certified; common third flight fails for all three requests | 3 | 6 | 3.233225 |
| v853 | Complete, failure | Cargo/mining validation rejects preparation before any solve | 0 | 0 | 0.847913 |
| v854 | Complete, success | Each retimed attempt certifies 16 of 17 legs; both Earth returns fail | 30 | 4 | 29.574029 |
| v855 | Complete, success | Three extended Earth returns fail; the same 16-flight prefix is reused | 3 | 48 | 41.266340 |

The retained failures include the v851 third-flight defect 4.341414e-4; v854 return defects 1.186157e-3 and 2.444980e-4; and v855 return defects 1.317715e-2, 2.015708e-3 and 7.722101e-1 at arrivals 69578, 69638 and 69698. Native failure labels are not mathematical infeasibility certificates. Qualified inner conic subproblems do not certify a whole trajectory.

There are 36 reported native solves, 58 cache hits and eight actual route attempts across these reports. The two planned v853 requests never became attempts. Cached legs must not be counted as fresh solves or fresh certifications. Per-run wall times include their own cache/preparation scopes and establish no matched performance comparison.

All reports retain incumbent Result SHA256 `1f420928bbef8da91e3a6f8b74d3bf00039c78d4257215c103a484bc55bb22da`. There is no certified replacement or fleet gain evidenced here.

The read-only observation at **2026-09-09 04:19:49.651137 UTC** found the H100 80GB at 0% utilization, 0 MiB reported memory use and an empty compute-process query. This is one observation, not an ongoing idle guarantee. Retrieval completed successfully in 11.364768 s with zero source uploads and zero launches; these are retrieval facts, not trajectory certification.

`evidence.zip` preserves the exact 245,516-byte `lambda-status-v637.json` (SHA256 `f00dd0dc96c70aaff10e633743fe30a02dcad7e241ed4498101410c714fdb493`), including all raw remote reports in `queries[2].stdout`, plus a compact derived summary and this packaging source. The summary records exact request IDs, failure states, timing/call scopes, core/settings identities and the prior preparation failure. Derived parsed-report hashes are labeled separately from original remote file-byte hashes; only the retrieved raw response is byte-pinned here. Source/library hashes are retained dependencies, not newly verified remote artifacts.

Run `python -B verify.py` from this directory, or pass the package path and `--index-sha256 EXPECTED_HASH`. The portable stdlib verifier checks all hashes and recomputes the summary from the saved raw response, including zero fleet checks and failed certification on every attempt. It performs no network, propagation, checker, solver or GPU work. Reported partial-leg certification has not been independently re-propagated in this package. `verification.json` records the saved-only check.
