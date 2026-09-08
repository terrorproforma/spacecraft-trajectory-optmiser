# Archived-control GPU initialization — v624 evidence

The previously failing Earth-to-30805 control now converges in one accepted
native SCvx step, with 20 QOCO iterations and a 0.416 s native call. Independent
GPU and CPU physics checks both pass: arrival error is approximately 57 m.
This regenerates a known feasible trajectory; fuel and fleet score are unchanged.
It is a single RTX 5090 result using GPU QOCO, not a PDHCG or SOTA speed claim.

[Implementation, measurements and limitations](../../../../docs/GPU_VERIFIED_TRAJECTORY_INITIALIZATION.md).

`evidence.zip` contains 373 files, 18,831,919 uncompressed bytes / 5,214,191 ZIP
bytes: exact runtime sources, archived input, raw and postprocessed outputs,
CUDA tests/sanitizer logs, the separate CPU reference, independent saved-data
audits, and the full-route parse preflight. Failed/intermediate builds remain
distinct from the final c build. Executable libraries and SSH credentials are
excluded; reported binary hashes identify the tested artifacts.

Archive SHA-256:
`4f43b84567bb9b2076e8e81d6bfe3fb26a5d454a08a8ee5ee3b7f6b368bb1316`

Index SHA-256:
`8a0029d2dc9163a7662154f761f5f9339884e8715d8b3a62c1312d3e0a0cfb38`

`verification.json` records the successful archive/source audit and two fresh
saved-data audits from an unpacking. Both reproduced the original findings
byte for byte, without invoking a GPU, optimizer or propagator.

From the repository root, these commands require only Python 3.12's standard
library and perform saved-evidence verification:

```text
python -m zipfile -e results/local/2026-09-09/native-trajectory-seed-v624/evidence.zip build/review-v624
python -B build/review-v624/build/performance/audit_native_seed_package_v624.py --package results/local/2026-09-09/native-trajectory-seed-v624 --expected-index-sha256 8a0029d2dc9163a7662154f761f5f9339884e8715d8b3a62c1312d3e0a0cfb38 --out build/review-v624/package-audit.json
python -B build/review-v624/build/performance/native-seed-result-review-v624/audit.py --root build/review-v624 --output build/review-v624/gpu-findings.json
python -B build/review-v624/build/performance/native-seed-cpu-certificate-v624/readback.py --root build/review-v624 --output build/review-v624/cpu-findings.json
```

The original one-run launchers are retained as experimental provenance. They
are not invoked by these audits. Full conic primal/dual vectors and the GPU
verifier's final physical vector were not captured; native conic qualification
is therefore reported from its retained audit. The separate CPU reference
retains its complete final physical state and all 160 segment endpoints.
