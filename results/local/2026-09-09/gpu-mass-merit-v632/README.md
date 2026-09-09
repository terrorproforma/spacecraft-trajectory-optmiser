# GPU mass-merit evidence, v632

This package preserves the all-node minimum-mass merit change, its native validation, the matched return experiment, and the preceding saved-only diagnosis. Full rationale and measured limitations are in `docs/GPU_MASS_MERIT.md` inside `evidence.zip`.

Build passed: `True`. Native validation passed: `True`. Mission status: `candidate_return_failed`. Returned native calls: `2`; outer iterations: `28`. These labels retain failed outcomes; passing native unit tests alone does not certify a mission.

The fuel objective, cargo, schedule, conic constraints and independent physical/fleet gates are unchanged. Merit values are dimensionless normalized subproblem costs, not kg. The source archive is compared against the sealed v631 source: exactly three reviewed CUDA files differ. Runtime executable/library hashes are retained in manifests; compiled binaries and credentials are excluded.

Older fleet/prefix inputs and runtime dependencies listed in the mission's `external-evidence.json` remain hash-linked to the existing repository artifacts. They are not fetched or repackaged here. The prior v631 source archive/manifest and CPU logs are included only to verify the three-source delta and unchanged-host validation. This is a saved-byte/source/status verifier, not a fresh mission replay.

Run `python audit_package.py verify --roundtrip` for saved-byte verification only. An optional `--index-sha256 HASH` pins the independently supplied index. The verifier uses only the standard library, checks nested archives and source maps, and runs no archived code, optimizer, GPU computation, ephemerides, or propagation.
