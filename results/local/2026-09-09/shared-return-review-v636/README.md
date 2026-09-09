# Independent saved-data review of shared return options

The published v846 performance and validation claims are supported by the saved
evidence. This review ran no GPU work, search, optimizer, propagation or checker.
The original [v846 package](../../../lambda/2026-09-09/gpu-shared-return-options-v846/README.md)
is unchanged and its archives are referenced rather than duplicated here.

The one issue is the original audit's Python-version-dependent profile sum.
On Python 3.12.14 it fails at line 89: four profile `total_seconds` values differ
from the saved values by 3–6 ULP, at most 1.07e-14 seconds. Explicit sequential
addition reproduces every saved total exactly. All profile call counts and
individual timing rows match exactly. This affects the profile audit, not the
three-sample benchmark medians or candidate data.

`compatibility_audit.py` binds the original audit SHA-256
`a13246091d565efea79199228b9e41220088988e54dc1594c75792c3942c3aa3`
and the original package manifest. It replaces only the profile-total expression
in memory with sequential addition. The original exact equality assertion and
every other audit assertion remain in place. No tolerance is introduced. Its
single recorded execution passed on Python 3.12.14 and reproduced the entire
published `saved-audit.json` exactly, validating 990 archived members per device.
`original-failure.json` preserves the earlier failure as an observation record.

An independent `git cat-file --batch` comparison found 876 runtime source files:
869 exactly match base
`9681f6ce41cfc11dac927b0965efa879605f4f9a`; the other seven exactly match the
declared overlays committed in `2771155e469adb0590a4726438c06d84aff7f650`.
Both runtime source manifests hash to `2cbf1221dc2eace0b0d1b10623a629be3daaa985bd1839769a55d306b6500014`.
The RTX core is `75a9640709216bdb5a381b9e7f895692489ca8d4e6cb8606103ee39c6a3dc15c`;
the H100 core is `a2ec27ce1b20ec472b0044a5c554751cb704b496f479aff487a852b7cda0527d`.

| GPU | Ship | Fresh median | Shared median | Throughput ratio |
| --- | --- | ---: | ---: | ---: |
| RTX 5090 | 10 | 13.208789 s | 10.123409 s | 1.304777x |
| RTX 5090 | 21 | 10.958894 s | 8.454162 s | 1.296272x |
| H100 | 10 | 6.954387 s | 6.019081 s | 1.155390x |
| H100 | 21 | 6.089910 s | 5.204798 s | 1.170057x |

Each mode has one warm-up and three measured samples. The combined rates are
989 divided by the sum of the two shared medians: **53.2362 proxy candidates/s
on RTX and 88.1157 on H100**. These are whole `RouteSearch.run` timings excluding
startup and catalogue loading. Each device performed eight two-route benchmark
runs: 7,912 candidate records from repeated workloads, not 7,912 new solutions.
All 517 ship-10 and 472 ship-21 candidate records are byte-identical across
modes/repeats on each device and to that device's preceding checkpoint. The
package does not establish cross-device byte identity; their candidate-file
hashes differ. The later profiles are separate executions outside these timers.

The saved two-route counts reconcile on both devices:
12,188 hits plus 158 misses equal 12,346 return queries. Shared rows avoid
12,992,408 branch requests; fresh branches fall from 35,358,344 to 22,365,936,
and resident option builds fall from 13,906 to 1,718. Native hit handling and
the Python `computed_hops` counter exclude reused rows from fresh work counts.

Raw logs and source/core bindings support 205 passing pytest cases per device,
67 under each of memcheck/racecheck/synccheck, separate passing controller
checks, and 29 focused cases under leak checking with zero reported leaks or
errors. These are retained test-case counts, not fresh GPU-call counts. No
tests were rerun for this review. The saved RTX profile also supports the
documented 6.4813 cumulative seconds in `_schedule_dp`, 1,152 tour preparations,
1,731 DP solves and 2.2793 seconds in workspace construction; nested cumulative
times overlap.

This performance change adds no mission certificate or score. The separately
verified incumbent remains 24 ships / 208 asteroids, 13,526.961241 weighted kg
and 14,915.044490 raw kg at this checkpoint.

From the repository root, the exact saved-data audit can be repeated with:

```powershell
python -B results/local/2026-09-09/shared-return-review-v636/compatibility_audit.py
```

This needs only the original v846 package and Python's standard library. It
does not load native libraries or use a GPU. The committed-source comparison
above was a separate read-only check; no Git operations are needed by this wrapper.
All evidence here is byte-preserved by the local `.gitattributes`; `index.json`
binds this review's files.
