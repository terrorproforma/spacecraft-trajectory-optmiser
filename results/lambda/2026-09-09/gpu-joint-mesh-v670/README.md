# CUDA mesh generation and H100 replay

[Implementation, results and loading instructions](../../../../docs/GPU_JOINT_MESH.md).

This opt-in path generates ordered timing moves on CUDA and feeds device epochs
directly into resident geometry and winner selection. Both GPUs pass 87 tests;
25 geometry/mesh tests pass memcheck, synccheck and racecheck on each GPU.
The component gains 7–15% locally and 34–42% on H100 against CPU-generated epochs
with the same resident geometry. These are not whole-mission speedups.

`summary.json` preserves all eight full-fleet process outcomes and timing samples.
Every retained best fleet passes independent and official checks at unchanged
tolerances: 12,810.135953 weighted kg, 14,051.854894 raw kg, 23 ships, 195 collected
asteroids. One local GPU-mesh run and one H100 CPU-mesh run fail to refine the
second, lower-scoring alternative. They retain the verified first result and
skip one full-fleet check, confounding process comparisons. No overall speedup
or score improvement is claimed. All failures remain in the raw archives.

`h100-best/Result.txt` is the downloaded H100 GPU-mesh fleet. Its viewer export,
campaign report and `viewer-dataset` are included. The displayed dataset is:
http://127.0.0.1:4173/?dataset=gtoc12-mesh-v670&epoch=69807&preset=oblique&z=1

The two raw archives contain the frozen source, core/QOCO binaries, fixtures,
build commands, complete test/sanitizer logs, all timing samples and all fleet
attempts. Member and archive hashes verify downloaded bytes. The local archive
also preserves the initial failed test fixture and its corrected final version;
runtime source did not change for that correction. `published-source.json`
verifies the final source against the frozen tested files.

Python still drives mesh progression, deadlines and metadata preparation.
Independent CPU checks gate fleet acceptance. Native conic convergence remains
unreliable on one recorded return: [nine repeated calls and rejected settings](../../../local/2026-09-09/return-replay-v672/).
