# Fixed-cargo return-window rescue v607b

The bounded local RTX 5090 experiment **did not improve the fleet**. The retained
23-ship solution remains **14,051.854893908598 raw kg / 12,810.135953048577 fixed-bonus
weighted kg**. No candidate was promoted and no new visualizer result was added.

The test preserved the 16 certified prefix legs of the v606 ship-7 replacement,
including its cargo, deployment/collection events and emitted thrust samples.
It changed only waiting after final collection and the Earth-return window.
Actual work was **one original-prefix control return plus four candidate returns**,
with **zero whole-route reruns**. The successful control was freshly solved and
passed both fleet checkers. All four replacement returns remained uncertified.
The actual execution was local; **no H100 job or transfer occurred for v607b**.

| Recorded work | Count / time |
| --- | ---: |
| Legal screened windows | 7,022 coarse + 2,416 fine = 9,438 |
| Lambert direction requests | 18,876 |
| Fresh native return solves | 5 |
| SCvx iterations / accepted iterations | 142 / 53 |
| QOCO report rows / reported inner iterations | 142 / 12,117 |
| Total wall time | 86.221170 s |
| Native-wrapper wall time, including host overhead | 41.463645 s |
| Baseline and control full-fleet verification | 41.723452 s |

These work units describe stages of five return solves. They are not independent
whole-route solutions, and the wall times are not pure GPU kernel timings.

The replacement prefix left **91.543915 kg** of return propellant. The cheapest
screened return estimate was **213.617946 kg**, a **122.074031 kg** deficit.
All four selected windows received native refinement despite negative proxy
reserves. They reached the fixed optimizer fuel allowance with nonzero defects
of approximately 0.00218944, 0.00218944, 0.00744544 and 0.01000789. Every failure's
arrays, diagnostics and prescribed cargo were retained; no partial trajectory
was scored as a fleet.

The replacement prefix had already spent approximately **95.392567 kg more fuel**
than its original counterpart. The tested return-date changes did not recover
that loss. A distinct next hypothesis should reduce upstream prefix fuel use or
change the replacement geometry. Proxy estimates and failed SCvx solves do not
prove physical infeasibility, and this sampled epoch grid is not exhaustive.

The [independent artifact audit](audit/final-report.json) verified all five native
calls, the preserved control prefix and other 22 ships, both control fleet gates,
all raw screening arithmetic and deterministic diverse selection, and absence
of false promotion. Its [readable interpretation](audit/README.md) includes all
four failures. The frozen kit passed **44 CPU behavioral tests** and Ruff checks;
see [validation evidence](execution/validation/report.json). Those tests include
byte-identical original-control reconstruction, waiting-coast/perihelion checks,
immutable events/cargo, failed-call retention and exclusive launch guards.

`execution/` preserves the exact ready manifest, driver, foreground supervisor,
completed launch record, inherited profiles/settings and validation logs.
`source.tar.gz` contains all **190** frozen Python/benchmark source files, and
`inputs.tar.gz` contains all **78** frozen route/source-lineage inputs. Members
retain the original bytes. Restoring both archives alongside the execution
files reconstructs every one of the **286** ready-indexed files exactly.
`output/` contains **all 33 raw outputs**, including failed NPZs; `audit/` preserves
the unchanged independent audit. [Archive/copy audit](provenance/archive-audit.json)
checks full member roundtrips, all copied hashes and unchanged original files.
`sha256.json` indexes every publication file except itself; raw Git attributes
prevent line-ending conversion from changing evidence hashes. No compiled
binaries are included.

Python is frozen at `b08b1f5a`; native libraries are the demonstrated local v590
core and QOCO540, with exact hashes in [publication provenance](publication.json).
The [previously published native source](../joint-candidates-v596/original-v590/native-source.tar.gz)
has SHA256 `6f10438bd50bb52401f25d1fe3d44524fdf1028080c0e9e5644c0037e4dbdf33`.
Re-execution also requires the pinned catalogue, bonus data and native builds;
this publication neither includes binaries nor automatically launches a run.

`execution/preparation.json` and `execution/profiles.json` are inherited v606
lineage, including an unused H100 profile. They do not describe extra v607b work.
`execution/README.md` is the unchanged pre-run preparation record; its pending
execution statement is historical. The authoritative actual run is
[output/report.json](output/report.json) with the completed local
[launch record](execution/launch.json). Historical absolute paths identify the
original workspace and are preserved without rewriting raw evidence.
