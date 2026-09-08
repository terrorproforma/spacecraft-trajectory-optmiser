# Resident CUDA joint geometry: RTX 5090 and H100

See [the implementation and measurements](../../../../docs/GPU_JOINT_GEOMETRY.md)
for execution flags, exact scope, remaining CPU work and visualiser loading.

`summary.json` contains four component cases per GPU and eight full-fleet recovery
processes. All fleet outputs pass independent and official physics checkers at
unchanged tolerances. The score reproduces 12,810.135953 weighted kg and
14,051.854894 physical kg with 23 ships, 195 collected asteroids and 196 deployed
miners. Component gains are 2.13–4.77x on RTX 5090 and 3.20–6.40x on H100, versus
the preceding staged batched CUDA wrapper. Full-process medians are effectively
unchanged; no overall speedup or score improvement is claimed.

Both GPUs pass 71 tests, with nine resident-geometry tests passing memcheck,
synccheck and racecheck. Each raw archive preserves the frozen source, fixtures,
core and QOCO binaries, commands, all test logs, timing samples and all four fleet
attempts. Archive records and member manifests verify every downloaded byte.
The local original 69-test run and later two additional input-gate tests are
retained separately. `published-source.json` records the subsequent documentation
comment change in the public header; runtime source is otherwise byte-identical.

`h100-best/Result.txt` is the complete downloaded H100 resident-geometry replay.
Its exported histories and campaign report accompany it. `viewer-dataset` is the
validated dataset displayed at:

http://127.0.0.1:4173/?dataset=gtoc12-geometry-v659&epoch=69807&preset=oblique&z=1

The test feature is opt-in. Python still creates moves, packs metadata and drives
the search, and independent CPU verification gates fleet acceptance. Computed
joint geometry costs currently recompute instead of populating the Python cache.
Sanitizer claims apply to resident geometry tests, not all QOCO/cuDSS code.
