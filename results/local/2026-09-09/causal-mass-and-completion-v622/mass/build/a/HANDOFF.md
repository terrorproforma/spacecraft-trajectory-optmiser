# v622a causal mass elimination: frozen diagnostic handoff

This is a default-off native persistent-solver experiment. It combines exact causal mass elimination with a direct original-coordinate diagonal metric inside the existing dual-first L1 path. The original common KKT gates, original objective, all virtual controls and the initial original-coordinate seed check remain authoritative. This is not a production trajectory backend or a measured cold-convergence result.

The C++/CUDA source is an uncommitted, immutable snapshot over base `fdf52ae31d259240ffebdcf79ed0965dce8b9298`, not that Git commit itself. The source tree SHA is `8d72fb3e441dfcdfcabe28d65d0e7d0a959e099622c05dfb8b5e93d1c83039a7`. The archive contains 384 files; only the 11 paths enumerated in the manifest differ from the pinned base. Concurrent compact-completion, search and fleet changes were excluded. No Git mutation was used to freeze the source.

Frozen root: `/home/angus/spacepdhcg-mass-v622a`.

| Artifact | SHA256 |
|---|---|
| Final build manifest | `ca970d7acd9893ba71d7e58f85393cb2a26c003ca931e2071e2718e4292ad026` |
| Source archive | `1b64328171dfe6cb1623fb75739006abfa2f31433c5ad2b12f92d766bd949511` |
| Full native library | `a89b8b30267399c033a88f3fd8ec7fa8ac1bceb1bd41e79b9841807af1060cb2` |
| Native tiny executable | `376c4119c45f53709330420174cc9538b357af370cb0a7cdc6c1fbf3fb6e408e` |
| Snapshot replay executable | `367f24e77bd43456c2fedd1c55e8492fef2b018eae9a3cc195a38ed567173457` |

All 16 build stages completed with their expected status, including CPU-only detector/conversion checks, six GPU-hidden replay validations, two explicit parser rejections and CTest enumeration. Compilation took 76.28 seconds. The resource check compares 12 existing default/common/Halpern/L1/initialization/residual/recovery kernel footprints against the previous full-core build: all match. New mass solve uses 198 registers, 40 stack bytes and 7,168 shared bytes; new initialization uses 48 registers, zero stack and 7,168 shared bytes. Occupancy is checked before launch; unsupported requested block counts are rejected.

## Representation and metric

Each descriptor proves one causal chain with `m0=1` and `m_i-m_(i-1)+g_i*Gamma_i-nu_i=a_i`. All `g_i` are positive and finite. The first experiment requires exactly zero Q, zero mass costs, no other mass coupling, exactly three singleton signed mass-bound rows per mass variable, disjoint active controls and an unshifted input. It retains all original allocation slots. The detector finds 211/234 mass nodes on the two fixed captures; with L1 the logical active dimensions become 3,580 variables/7,600 rows and 3,974 variables/8,432 rows.

The working forward product contains only the linear causal prefix. A precomputed affine prefix shifts the retained mass-bound upper values once. A reverse suffix computes the adjoint and recovers original mass equality duals. Public mass variables are reconstructed with the affine prefix after each update, then the original L1 pairs are completed. The original seed is checked before either completion, preserving legitimate weak interior multipliers and tiny nonzero virtual controls.

The direct diagonal steps use upward-rounded positive absolute coefficient sums, tied SOC row denominators, and downward division of the represented `theta=0.95`. Zero operator sums use a dummy denominator of one. Finite positive steps and thresholds, and an outward upper bound below one, are required. The metric is rebuilt every solve; old Ruiz diagonals and global B/O scales are not reused. Full original step arrays are exported by replay, with unit dummy entries for inactive slots. This is a simultaneous representation/metric comparison, not an isolated claim about either mechanism alone.

## Completed tiny evidence

Parent launched the reviewed tiny process once on RTX 5090. `../mass-tiny-v622/report.json` has SHA `471537ac89b61cf1623d03ec4d18a61f5f2a6fb83343a54387668cd25b234618`. It records seven solve APIs, summed caps of nine, and five actual updates. The raw log and executable-bound assertions cover a three-step independent mixed scalar/SOC/L1 oracle, mode-off versus fresh L1 from identical exported vectors, original weak seed acceptance at zero updates, initial cancellation, nonfinite seed, metric overflow and unsupported map rejection. The original weak seed's public primal/dual bits were preserved.

The tiny raw log stores outcome metrics. Full vector and actual-step comparisons run as assertions inside the pinned native executable against the independent CPU fixture; it should not be described as a separate raw-vector audit of seven logged vectors. The fixture/oracle and independent source/build reviews live in `../mass-native-oracle-v622` and `../mass-native-review-v622`.

The real runner is `../run_mass_real_v622.py`, SHA `f4bfb29c683eaa1ac394e5b8e52e1f2f59bea73a77be20daae4c8df896579f37`. Its CPU `--prepare-only` check passed in `../mass-real-v622-preflight`; that directory contains zero GPU calls. The finite plan is two original seed checks followed by four cold comparisons (two captures × unit L1/mass+unit L1), exactly six executions and at most eight solve APIs including two seed bootstraps. Every cold run is capped at 100,000 iterations and 30 seconds. All use the same frozen library and explicit 128 blocks, with no grid fallback, numerical tuning or retry. This handoff does not claim those real runs have completed.

Both runners require a matching device UUID, a nonblocking shared GPU lock inherited by the native child, fresh output directories and process timeouts. Raw failures are preserved. The real runner independently audits all final original x/y/z/s candidates, including unqualified limits/cancellations, and retains actual mass-step readbacks. Separate bootstrap calls are counted and timed; their full original vectors are not exported for independent audit.
