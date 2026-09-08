# Fixed-cargo low-thrust truth set

**Both original routes qualify; both score-improving proposals fail their final
Earth-return refinement.** The verified incumbent remains **14,051.854893908598
raw kg / 12,810.135953048577 fixed-bonus weighted kg across 23 ships**. No candidate
was promoted and no new visualiser dataset is created.

The local RTX 5090 run completed all four prescribed route calls and 66 native
leg solves in **110.373791 seconds**, including fresh baseline and control
full-fleet verification. Sixty-four legs certify; the two remaining calls are
the probes' unchanged final Earth-return legs. This is one selected diagnostic
set, not a general throughput benchmark or an estimate of a surrogate's overall
false-negative rate.

| Case | Potential weighted gain | Certified legs | Return fuel available | Outcome |
| --- | ---: | ---: | ---: | --- |
| Original ship 2 | 0 | 16/16 | 192.423222 kg | Both fleet checkers pass |
| Ship 2: 31302 → 2181 | +6.236203 kg | 15/16 | 13.723155 kg | Return refinement unqualified |
| Original ship 7 | 0 | 17/17 | 186.936482 kg | Both fleet checkers pass |
| Ship 7: 15206 → 3150 | +9.908765 kg | 16/17 | 91.543915 kg | Return refinement unqualified |

The proposed changes retain the exact prescribed cargo and epochs. All four
changed incident transfers in each probe certify. The prefixes consume an extra
**178.700067 kg** and **95.392567 kg** of propellant relative to their controls.
The matching original return legs consume 189.744659 kg and 183.794239 kg, at
their respective control initial masses; those burns are not a proven minimum
for the lighter probes. The failed return solves retain virtual-control
residuals of about 0.002306424 and 0.002189442. An unqualified SCvx result does
not prove physical infeasibility, but these runs demonstrate no feasible proxy
false negative and no score gain. Full-route fuel use matters after changed
transfers have passed their individual checks. [Independent result audit](audit/report.json).

The wrapper keeps cargo fixed because the stock route refinement resets cargo
to maximum production and may shrink it after a deficit. It otherwise reuses
the validated scheduler, native CUDA SCvx driver, mass/event ordering, thrust
clamp, independent DOP853 leg certification and route master. A candidate needs
its corresponding actual solver control to pass, both full-fleet checkers,
the exact prescribed cargo, raw-mass ship-count admissibility and a higher
verified weighted score. No tolerance or production implementation changed.
The original 23-ship result is unchanged byte for byte.

The frozen kit passed **39 CPU tests**, Ruff checks, construction checks and
source compatibility checks. Python sources are pinned to `b08b1f5a`; the
successful v595 source archive was compared with raw hashes plus an explicit
CRLF-only normalization check. The local core is v590
`ffbae813f1f683d276213cb83ec8be17d1c2338c9c4a96f3ec5283e26a432b73`,
and QOCO is v540
`0cc27a1d8bde1edd74f7ef00e04ec64c2d6c0f8fe526a1d94b04d509a74b3315`.
The supervisor and child independently check their hashes and share the GPU
lock. Supervisor PID 298 and child PID 377 both finish; exit code is zero.

`source.tar.gz` contains the 257 reviewed preparation files under `execution/`.
The visible `execution/README.md` is the unchanged pre-run preparation record;
its statement that execution is pending is historical. `execution/launch.json`
records the completed local execution. `raw.tar.gz` contains complete per-leg
reports, all available solution arrays, native histories, original failure
details and the two control fleet results. `report.json` and
`native-solves.jsonl` are unchanged raw records. `audit/` retains the independent
CPU checks; `sha256.json` indexes the publication. No compiled binaries are
included.

An H100 profile was prepared and its existing library/source compatibility
checked. Automatic approval review rejected transfer of this exact source/test
payload pending specific user approval. **No payload was transferred and no H100
truth-set job ran.** The authorized local alternative consumed the same global
four-route budget; the experiment is not to be duplicated on H100 automatically.
There is no additional cost hidden as a second device campaign.
