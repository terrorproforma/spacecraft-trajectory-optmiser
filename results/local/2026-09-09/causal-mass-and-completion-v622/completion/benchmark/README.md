# Completion packing benchmark v622 — prepared, not executed

This finite component comparison calls the frozen production `GpuCompletion.run`
through the same thread-owned compact-g library for both ordinary metadata and
compact native-model metadata. It does not measure mission search, trajectory
refinement or new solutions. No production default changes are part of this kit.

The inputs are 20 historical v595/v619 controls: five routes, two prefix-mass
choices and the unchanged flat/existing-fit models. These controls include passing
and final-mass-rejected requests. They are not the latest fleet. `plan.json`
retains the exact repeat indices and expected acceptance counts.

| Quantity | Fixed budget |
| --- | ---: |
| Models | 2: flat and existing five-feature fit; no refit |
| Batch sizes per model | 24, 48, 256, 1024 |
| Calls per arm and group | 1 first + 4 warm |
| Native evaluation calls | 80: 40 ordinary + 40 compact |
| Repeated proxy candidates per arm | 13,520 |
| Unique historical controls overall | 20 |
| Lambert solves / SCvx refinements / fleet promotions | 0 / 0 / 0 |
| Worker deadline | 180 seconds, then owned-group TERM/KILL cleanup |

The frozen native source launches one finishing kernel per ordinary call and
three kernels per nonempty compact call (candidate assembly, leg assembly,
finishing): 160 kernel launches are expected from the 80 evaluation calls. This
is a source-derived count, not a profiler measurement.

Both arms use the pinned official 60,000-asteroid catalogue. Saved
`features.json` is retained only as historical provenance; it is never injected
into a geometry cache. Ordinary metadata computes catalogue features through
the production table and retains its normal cache. Compact metadata computes
features on the GPU. Its source-array hashing and model-change validation run
on every call inside the measured interval; there is no identity-cache shortcut.

Each group owns separate fresh workspaces and request copies. First-call order
alternates between groups. Four subsequent measurements per arm use ABBA/BAAB
order. The timer encloses the complete production method, including packing,
model hashing/configuration, native execution/transfers and plan reconstruction.
Common workspace creation/close and catalogue/request construction are recorded
separately. First-call timings are first calls on fresh workspaces in one process,
not two independently cold processes. Compact model configuration remains timed.
Capture hooks, readback validation and evidence serialization are outside the
timer. Native kernel time and packing time remain available in telemetry.

Every call must reproduce the frozen CPU classification, exact cargo and pickup
order; floating values use the prespecified v621 tolerances (absolute 2e-10,
relative 2e-13). Qualification never uses those tolerances. Each arm's five
readbacks must also be bit-identical excluding timing statistics, and input
prescriptions must remain unchanged. No fallback, re-pricing, tolerance widening,
resampling or automatic retry is allowed after a failure. Raw output buffers are
saved immediately after the timed call, before telemetry or numeric assertions.
If the call raises during native work or reconstruction, available buffers are
still saved and explicitly marked unvalidated and potentially stale; no completed
native call count is inferred in that case. Per-call metadata and requested-work
journals distinguish attempts from returned calls. Outputs retain every failure.

`source/`, `source-full.tar.gz` and the source/report hash maps bind the exact
compact-g implementation. The original `inputs/compact-g-source.tar.gz` contains
the 12 owned native/host/test overlays, and is retained separately; the full
archive contains all 341 source and support files. Both archives are checked
against the copied tree. The ordinary reference and compact path use
the same `libspacepdhcg_completion.so` SHA
`9c3e6a250002893363d2f4dcf68babe3b7158508b5b97149f83c8261437f1fa7`.
The old `inputs/gpu-profile.json` supplies only its unchanged comparison policy;
its old runtime library paths are not used. `profile.json` is the actual profile.
No binary or external official catalogue is duplicated here; the loader checks
the catalogue's published byte size and SHA before use.

CPU preparation tests block native loading, search, finishing and geometry
evaluation. Eighteen tests passed, including failed-call/readback retention;
validation attempts and style-only failures are retained. These tests establish
construction, launch gates and evidence retention, not benchmark
timing or new GPU parity. Matching compact-g correctness passed separately in
`../completion-model-gpu-v622d/report.json` (7 tests, no skips; source/report
`56a65032e639a6eb9af08052e9f3df87f5f1c84cfe9cf2ece6f3f63eafb2a12a`).

After parent/reviewer approval, launch once from the repository root:

```powershell
wsl -d Ubuntu-22.04 -- /home/angus/worktrees/spacepdhcg-literature-venv/bin/python -B build/performance/completion-pack-benchmark-v622/launch.py --prerequisite build/performance/completion-model-gpu-v622d/report.json --prerequisite-sha256 643caf4e5550373ac3b06761197ae4ee1f7801569e614498786ba089b34a54ff
```

The foreground supervisor verifies the ready manifest, source/library/interpreter
identities and the exact passing prerequisite before acquiring the shared
`/home/angus/.spacepdhcg-gpu.lock` with `LOCK_NB`. It then checks RTX 5090 UUID and
active compute processes. A busy or failed preflight is retained, consumes this
single-use launch attempt and starts no child. The child inherits the lock;
timeout/interruption terminates only its owned process group and waits for exit.
Do not run `run.py` directly or reuse an existing output directory.

For CPU-only preparation checks, use `validate.py`; for a publication replay,
restore `source/` from `source-full.tar.gz` and preserve all frozen input paths.
`freeze.py` verifies every source/archive entry and records final inventory and
prerequisite identity without GPU access. `ready.json` excludes itself and any
future outputs, avoiding a self-referential hash.
