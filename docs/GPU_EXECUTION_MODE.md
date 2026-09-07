# Explicit GPU loop selection

The development CLI selects the native SCvx/QOCO graph path automatically with
`--outer-loop-backend cuda`, or explicitly with `--gpu-execution graph`.
Use `--gpu-execution dispatch` for host-launch comparisons. Keep the existing
CUDA discretisation/assembly, QOCO solver and one-worker requirements.

This selects GPU numerical iteration after host setup and priming. It does not
eliminate Python search/fleet orchestration. The graph requires the pinned native
extension and SM90/SM120; unsupported configurations fail without CPU fallback.
Direct library users retain the existing API. Process-level flags are scoped and
restored at CLI command boundaries.

Both RTX 5090 and H100 pass 38 CLI/native-refinement checks. A balanced four-run
H100 comparison reduces complete-campaign median time from 60.828 to 58.847 s
(3.26%); all four missions return 548.255 kg and pass both checkers. This small
sample does not establish general reliability. Inner-QP qualification outliers
remain unresolved, and this candidate has not been merged into main.

[Evidence, numerical diagnosis and viewer loading](../results/lambda/2026-09-08/gpu-execution-v328/README.md).
