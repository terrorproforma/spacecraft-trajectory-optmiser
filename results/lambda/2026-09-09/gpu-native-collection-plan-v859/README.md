# Native collection planning: downloaded RTX 5090 and H100 evidence

Moving mining rewards, subset masses and burn-pass selection onto CUDA improves
the paired whole-search medians by **5.3–7.5% on H100** and **2.6–5.9% locally**.
Both GPUs retain their previous candidate files exactly. This is a search-speed
change, not a new verified trajectory or fleet-score gain. The local measurements
include recorded desktop GPU activity; see the complete ranges in `saved-audit.json`.

Each archive contains the frozen source, native library and controller binary,
build/validation logs, original worker, eight benchmark runs and all candidate
files. The H100 archive also contains raw profiles. Both native builds pass 236
tests, 98 focused cases under each of three CUDA safety tools, controller checks,
and 31 full leak checks. All owned workers exited before packaging.

| Archive | Bytes | SHA-256 |
|---|---:|---|
| `local.tar.gz` | 29,857,163 | `8842184cf047d2ab5bb7e4c38e8409ed324063497a4e7a6c142fc78d7e1d9a64` |
| `h100.tar.gz` | 29,026,924 | `b58e4634f04c3d9950528021bbb2714a18bb2f36563bbb159e6a15e5c2f6cfc3` |

The common frozen source manifest is
`dd5686e55a266fc9fc6ee984fba5f10a48cc2e68b8e85e440e924b2b90a8ad4f`.
It records base `f2511e510938d78d61d2c492a78d2b0ba79793a6` and six source/test
overlays. Native library hashes are
`1e839308d2dc0e66089af9ad3d231787fd1d2dc807d69ec15f9279fb07d60084` (RTX) and
`8aebb21b45a3f427d155e72e8b27565799240b855bac5e1b6b31fd9efe9a9934` (H100).

Run the standard-library audit locally:

```powershell
Set-Location 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser'
python results/lambda/2026-09-09/gpu-native-collection-plan-v859/audit_saved.py
```

It streams both archives, verifies every manifest member and source file, binds
the binary identities to the runs, checks validation logs and unchanged candidate
hashes, recomputes medians/readback bytes, and verifies the raw H100 profile
summaries. It writes `saved-audit.json`. No archived code, GPU computation,
trajectory propagation or fleet verifier is executed. `reference.json` identifies
the preceding published candidate hashes. `source-binding.json` connects the
working-tree production changes to the frozen tested source.

Historical orchestration scripts in `reproduction/` retain their original host
paths. Rebuilding/rerunning requires the recorded CUDA toolchains, Python
environment, pinned upstream source and GTOC12 data. Those scripts are evidence of
the performed workflow, not self-contained portable launch commands. Raw archive
bytes are unchanged; presentation and reproduction text outside them uses LF
line endings. `index.json` hashes all other top-level/package files.

[Measurements, implementation and remaining CPU work](../../../../docs/GPU_NATIVE_COLLECTION_PLAN.md).
The verified score remains **13,526.961241 weighted kg**, and the visualiser keeps
the v633 fleet.
