# GPU beam expansion: v835 evidence

CUDA now prices and sorts deploy-phase beam children in retained device buffers.
The measured change is enabled by default inside the CUDA screening backend.
Parent topology is retained and Python constructs only the ranked prefix needed
for the existing admission rules. FP64 dynamics and verification tolerances are unchanged.

| GPU | Route | Python expansion median | CUDA expansion median | Search throughput gain |
| --- | --- | ---: | ---: | ---: |
| RTX 5090 | Ship 10 | 20.851 s | 12.961 s | 1.61x |
| RTX 5090 | Ship 21 | 18.969 s | 10.818 s | 1.75x |
| H100 | Ship 10 | 18.113 s | 7.013 s | 2.58x |
| H100 | Ship 21 | 17.279 s | 6.343 s | 2.72x |

Each mode has one warm-up and three alternating measured samples per GPU, using
the same source and binary. Timings cover RouteSearch.run, excluding process
startup and catalogue loading. The environment flag
`SPACEPDHCG_TEST_GTOC12_GPU_EXPANSION=0/1` selects the comparison mode. Report
fields `fresh_seconds` and `reuse_seconds` mean Python and CUDA expansion here.

Both modes return 517 + 472 = **989 surrogate candidates**, with exactly matching
discrete fields, topology and ordering. Maximum numeric differences are
4.547e-13 on RTX 5090 and 9.095e-13 on H100. Candidate files are byte-identical
within each mode across repeats, but not across modes. The audit checks every
numeric field at absolute tolerance 2e-9 independently of physics certification.
Combined CUDA throughput is about **41.6 candidates/s on RTX 5090 and 74.1 on H100**.
These are search candidates, not certified solutions or PDHCG solves.

The two searches price 1,251,916 valid children out of 3,939,840 input slots.
Python materializes 10,123 + 95,632 = 105,755 children, **91.55% fewer** than
constructing every valid child. Each route uploads 141.95 MB of expansion inputs.
Host admission and the screening-to-expansion transfer remain optimization targets.

## Wider search and score

At beam width 128, both GPUs return **1,960 candidates**, screening **64,828,728
Lambert branches**. H100 search time is 26.950 s combined; RTX 5090 is 48.520 s.
There are 264 + 245 = **509 additional distinct asteroid orders** relative to the
narrower search. These candidates have not been trajectory-refined in this
checkpoint and cannot yet be counted as certified routes.

This checkpoint does not change its incumbent fleet's **13,023.704900978253
weighted kg / 14,291.0061601653 raw kg**, with 23 ships and 199 asteroids.
Separate fleet-selection work is outside this benchmark and may advance the
project independently. No new fleet dataset is published by v835.

## Validation and provenance

* Final default-on validation: **183 tests per GPU**.
* **45 tests per GPU under each of memcheck, racecheck and synccheck**, plus
  separate native endpoint-controller checks; no reported CUDA errors/hazards.
* Seven focused GPU tests pass full leak checking per GPU: zero bytes leaked,
  zero remaining allocations and zero errors.
* Frozen base: `3404e926355beb3383f391c3a4bb80ce3d618afa`, with the seven
  source/build/test overlays recorded in `source-binding.json`.
* v830 builds the core; v833 provides explicit-on measurements; v834 enables
  the same operator by default and adds its default test. All have identical
  C++ sources and native binaries. Later concurrent SCvx changes are excluded.
* The original v830 six-test fixture failure is retained. It tried assigning
  a read-only property before those tests exercised CUDA; subsequent fixed
  fixtures and expanded coverage pass.
* The H100 wider run is v837. A local WSL failure required a fresh local v838
  wider run; see `recovery-observation.json`. Prior completed evidence hashes
  were verified after recovery.

`local.tar.gz` and `h100.tar.gz` include exact runtime sources, CUDA binaries,
test logs, paired candidates and wider-search candidates. Each contains a
byte manifest. `saved-audit.json` records a fresh local audit of both archives.
Top-level extracted reports provide quick inspection; the archives are the
authoritative payload. Reproduction scripts preserve original host paths and
must be adapted for another machine. The portable saved-evidence audit uses
only Python's standard library and does not need a GPU.

## Load and audit locally

Full directory:
`C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-09\gpu-beam-expansion-v835`

Copy and paste into PowerShell:

```powershell
Set-Location 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser'
wsl -- python3 results/lambda/2026-09-09/gpu-beam-expansion-v835/audit_saved.py
Get-Content results/lambda/2026-09-09/gpu-beam-expansion-v835/saved-audit.json
```

The preceding verified fleet remains available in the existing web visualiser:
[Load v822 fleet](http://127.0.0.1:4173/?dataset=gtoc12-catalogue-v822&epoch=69807&preset=oblique&z=1).
The viewer server must already be running; this checkpoint does not change its
selected dataset. The wider surrogate candidates are evidence, not displayed
as independently verified trajectories.

[Implementation and remaining CPU work](../../../../docs/GPU_BEAM_EXPANSION.md)
