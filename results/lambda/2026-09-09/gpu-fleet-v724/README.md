# CUDA fleet selector evidence

See `docs/GPU_FLEET_MASTER.md` in the repository for interpretation and limitations.
`summary.json` contains measured medians and work counts. The verified mission
score remains 12,810.135953 weighted kg; the CPU metadata candidate is not verified.

`local.tar.gz` and `h100.tar.gz` contain the original frozen builds, native
libraries, input pool and raw results. Each archive includes `FILES.json`, with
the exact member set, byte lengths and SHA-256 hashes. Adjacent receipts and the
retrieval record verify the archives. Extracted JSON/log files are convenient
copies, not replacements for the original records.

Original Python reports use `Infinity` for unavailable LP bounds. The derived
`v721/report.strict.json` files replace nonfinite values with JSON null for display.
The production reporting fix and regression tests are captured separately in
the v725/v726 follow-up archives. `followup-verification.json` records their
retrieval checks and final source comparisons (the final Python formatting change
preserves the tested AST). `sha256.json` covers every published file in this
directory except itself.

To reproduce the RTX 5090 packing benchmark from the repository root in Linux/WSL,
use the project's Python environment with NumPy/SciPy and CUDA runtime available:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
  results/lambda/2026-09-09/gpu-fleet-v724/reproduce/replay.py \
  results/lambda/2026-09-09/gpu-fleet-v724/local.tar.gz \
  --variant v720 --output fleet-replay.json --repeats 5
```

Use `h100.tar.gz` on H100. `--variant v713` runs the initial CUDA implementation.
The script verifies and unpacks the selected frozen source/library in a temporary
directory, acquires the shared GPU lock, and refuses to overwrite an output file.
Discard the first call when measuring warm medians. Output nodes are packing
decisions, not trajectory solves. No full-mission propagation is performed.

The visualiser dataset is `gtoc12-fleet-v724`. It reuses the unchanged verified
trajectory samples from `gtoc12-certificate-v712`; the source solution remains
`results/lambda/2026-09-09/gpu-leg-certificate-v712/h100-best/Result.txt`.
