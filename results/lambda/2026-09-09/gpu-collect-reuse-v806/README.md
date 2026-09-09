# CUDA collection reuse and verified fleet v806

The downloaded H100 [Result.txt](h100-best/Result.txt) reaches
**12,999.452764189185 fixed-bonus weighted kg / 14,278.8501026699 raw kg**,
23 ships and 199 asteroids. Both original full-fleet physics checkers pass on
RTX 5090 and Lambda H100. The new ship-4 route adds 7.417102 weighted kg to v799.

[Implementation, paired timings, counts and limitations](../../../../docs/GPU_COLLECTION_WORKSPACE.md)
describe the collection-workspace change. The 55-test suite passes on both
GPUs, with 17 tests passing under each CUDA sanitizer. Paired searches create
51 workspaces instead of 2,184 with exact candidate bytes and 2.46–5.78% higher
whole-search throughput across the four measured GPU/route comparisons.

`local.tar.gz` and `h100.tar.gz` contain the complete frozen runtime source,
native core and QOCO libraries, source manifest, test and sanitizer logs, eight
paired benchmark runs, full route searches, all native refinement attempts,
certified pool inputs, CUDA selection and both final physics checks. Each
contains `FILES.json` covering 1,138 payload files. `*-receipt.json` binds
archive length and SHA-256. Extracted summaries and the H100 fleet are retained
for inspection. The frozen implementation predates the endpoint-merit patch.

Run the saved-byte audit with Python's standard library from any working
directory; it does not rerun physics or CUDA:

```powershell
python results/lambda/2026-09-09/gpu-collect-reuse-v806/audit_saved.py
```

The `reproduce` scripts preserve the executed preparation and launch steps.
They reference the recorded machine-specific environments and earlier frozen
inputs; they are provenance recipes, not a claim of a one-command fresh-machine
installation. No GPU job was rerun following the local disk-space interruption;
`retrieval-note.json` records the resumed local copy and exact recovered bytes.

The [subsequent composition](../gpu-collect-composition-v807/README.md) adds the
separately certified ship-8 correction and reaches 12,999.824843 weighted kg.
