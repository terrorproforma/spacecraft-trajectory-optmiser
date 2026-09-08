# Retrieved Lambda joint evidence

Read-only SSH retrieval of the existing
`/home/ubuntu/spacepdhcg-joint-selection-v632` evidence completed at
`2026-09-08T14:54:24Z`. All 78 initially retrieved files, plus the two viewer files
retrieved at `2026-09-08T14:58:30Z`, passed their remote byte-count and SHA256
checks. This retrieval started no jobs, changed no remote files, uploaded
no source or tests, and did not rerun trajectory verification.

The 1,199,917-byte archive SHA256 is
`3774c7d36d9bcd0ca3c4f22599e264a7f48d136282bc72e22761f4f58142cd1b`.
`evidence/download-manifest.json` lists every original remote path and hash.

## Existing campaign results

All four v636 reports are complete and record both official and independent
verification passes for their best 23-ship fleet, mining 195 asteroids. All start
from 12,805.194102489 weighted kg / 14,047.802874744 raw kg and finish at
12,810.135953049 weighted kg / 14,051.854893909 raw kg. Candidate0 gains
4.941850560 weighted kg and 4.052019165 raw kg. The maximum weighted-score spread
across all four best results is 1.70985e-10 kg.

| Run | Device selection | Seconds | Native solves | Second refinement |
| --- | ---: | ---: | ---: | --- |
| baseline0 | 0 | 135.884 | 36 | Certified; incumbent retained |
| candidate0 | 1 | 137.600 | 36 | Certified; incumbent retained |
| candidate1 | 1 | 120.727 | 36 | Failed; incumbent retained |
| baseline1 | 0 | 136.146 | 36 | Certified; incumbent retained |

Recorded input hashes, Python source hashes and all four emitted proxy artifacts
are identical across runs. Each performs 23,142 joint evaluations in 132 batches.
Final joint-result downloads fall from 15,359,152 to 88,576 bytes; preflight
downloads remain 1,481,088 bytes.

These four durations do not establish an end-to-end device-selection speedup.
Candidate1's second refinement fails on return leg 17, asteroid 13077 to Earth,
with `virtual control remains 1.380e-01`. It consequently omits the successful
second candidate's downstream fleet verification work. Candidate0 has the same
certification outcomes as both baselines and takes slightly longer. The reason
for the differing refinement outcome is not established by these reports.

The downloaded representative best fleet is
`evidence/campaign-v636/candidate0/best/Result.txt`, SHA256
`e408ff46f547890aaaa8d65505b969829ff9497b0264bf7b558f1bf1d6b9fb17`.
The four best result files have different hashes despite their effectively equal
scores; `evidence/remote-provenance.json` records each hash.

The existing viewer export is also available at
`evidence/campaign-v636/candidate0/best/viewer/trajectories.json` and its sibling
`manifest.json`. The manifest exactly matches the campaign report, its source
hash matches the downloaded Result.txt, and the 10,557,251-byte trajectories
file has SHA256
`533226bd53b2e22c20d331f318de3f1581c86f4e4013a41df110c438ef76c59c`.
`viewer-download-verified.json` records these checks. The separate viewer archive
SHA256 is `ced0d495d988578aa6f7bb31e94f091d11b82c9e931ab44be32337fda229826b`.

## Configuration and source identity

All campaign runs use joint batching, CUDA screening, graph execution, and CUDA
seed/discretisation/assembly/outer-loop backends with QOCO. Only device selection
changes in A/B/B/A order. The runner requests ships 15 and 3, grids of 15 and 5
days, a 15-second joint budget, 1,800-second campaign budget, and at most four
certifications (two per ship). Each report records unchanged physics tolerances.
Exact settings and telemetry are in `audit-summary.json` and the original child
reports; the remote runner's relevant configuration lines are recorded in
`evidence/remote-provenance.json`.

| Component | SHA256 |
| --- | --- |
| H100 CUDA library | `29a6b10b6349a21fd49f3b2d3f425cdca0d4e3acc4d97fd14cd16c42de54b44e` |
| QOCO library | `b50d61903921659c1f431b7c2f3472e0f353e5e5a72159e4164d7d282d27c906` |
| Campaign/benchmark gpu_joint.py | `b8a42303852ba4241613a42d73c3ded1e147cb2bcdd145508ea2923cae4187d8` |
| jointopt.py | `80ae101b05681585bf9f6cb4cd57b71c88677e92318be75bf7dd0dd409741ecf` |
| Native joint source | `51bf1c995327da59763d84eaae3e89787c2ec606bb1837afebe4781a07d86014` |
| Joint C API header | `bb5eb7c6ca6600579664f59dbacad4318208bee05543ae209311b386c5c17979` |
| Later compatibility gpu_joint.py | `5f70cf173929b8324e9e40885200e71272732f3e8210b5372dc238a07ae25689` |

The native source, C API header and jointopt.py hashes match the local worktree at
retrieval. The campaign and timing wrapper predates the optional-ABI compatibility
guard. Existing `compatibility-validation-v640` reports separately validate the
final local wrapper hash on the same H100 core: 62 tests passed, zero skips,
33.62 seconds. Its compatibility test hash also matches the local file
(`80bec43e3c68d45e50a60ab0b2e3cc8532fb6ec61cb01021a87a8324fcc33cf4`).

## Documentation assessment

The H100 component measurements in `docs/GPU_JOINT_ITINERARY.md` are supported by
the retrieved `benchmark-v634` reports: scalar-versus-batch ratios are
12.5621x / 10.9173x for ship 1 empty/complete cache and 12.8055x / 11.0718x for
ship 2. All four candidate audits pass and winning indices agree. These are
surrogate/controller timings, not full campaign or physical-solver speedups.

The separate selection-only ratios are 0.99846x, 1.00264x, 0.99887x and 1.00506x,
consistent with the document's conclusion that reduced traffic has not
established substantial additional speedup. The documented final-download byte
counts and unchanged 64-byte-per-candidate preflight transfers agree with the
reports. This selection-only measurement forces an eligible winner with
`minimum=-inf`; the main ship-2 benchmark has no improving winner.

The original H100 validation records 50 tests passed without skips and both
native probes passing normal execution, memcheck, synccheck and racecheck. The
later 62-test compatibility result supplements that historical claim. These
checks do not certify arbitrary physical trajectories; the campaign's stored
official and independent reports provide the separate acceptance evidence above.
The RTX 5090 timing claims were outside this retrieval audit.

No documentation or production source was changed by this audit.
