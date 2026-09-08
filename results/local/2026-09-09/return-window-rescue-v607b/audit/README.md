# Independent artifact audit: v607b

The completed local experiment retained the incumbent: **14,051.854893908598 raw kg / 12,810.135953048577 fixed-bonus weighted kg**. There were no promotions. This audit performed no GPU calls, new solves or additional fleet-verifier runs; it checked the recorded gates, raw artifacts and frozen provenance, and independently replayed the screening arithmetic and selection policy on the CPU with native library loading prohibited.

The launch used supervisor 1048 and child 1053, after an empty compute-process observation. All **286 frozen kit hashes** still match ready manifest `9414730e31bc2d896285761dc6436277cefb7d92af8a297291d9e3116a2c28ec`. The runtime retained the reviewed Python b08, native v590, QOCO540 and SCvx settings. The audit verified five distinct native calls, one per case, and zero whole-route reruns.

The original-prefix control return passed the independent leg/route gates and both complete-fleet checkers. Its emitted prefix matches the archived control exactly, all 16 asteroid events preserve their epochs and mass changes, and the other 22 ships are unchanged. It used 183.794239138939 kg of propellant from its original, heavier starting mass. That is a demonstrated feasible control burn, not a lower bound for the lighter replacement.

Screening retained **7,022 coarse + 2,416 fine = 9,438 unique legal pairs**, representing **18,876 Lambert direction requests**. CPU recomputation matched all retained authority, inflation and propellant estimates exactly. All four refined windows match deterministic selection over those rows, with a minimum 20-day separation between selected basins. Their proxy reserves were all negative; the experiment still spent the four allowed refinements on them, so the proxy did not reject those four opportunities without a solve.

| Case | Departure / arrival MJD | SCvx iterations / accepted | Outcome |
| --- | --- | ---: | --- |
| Original-prefix control | 69218 / 69728 | 5 / 5 | Both fleet checkers pass |
| Candidate 1 | 69218 / 69726 | 40 / 8 | Defect 0.002189442300; uncertified |
| Candidate 2 | 69238 / 69730 | 25 / 15 | Defect 0.002189442299; stationary penalized point, explicitly no infeasibility certificate |
| Candidate 3 | 69218 / 69688 | 35 / 11 | Defect 0.007445437092; uncertified |
| Candidate 4 | 69239 / 69691 | 37 / 14 | Defect 0.010007889375; uncertified |

Every failed candidate has a retained prescription, outcome, solver diagnostics and hashed NPZ arrays. Each retained the exact **1,207.2044215842654 kg** return starting mass and **1,115.6605065023955 kg** minimum arrival mass: **91.54391508186995 kg** of propellant available. Their reported optimizer propellant reached that budget while substantial defects remained. Those reported burns belong to uncertified trajectories and must not be represented as actual physically valid consumption. No candidate received a full-fleet certificate or promotion.

The cheapest screened return estimate was **213.617945604302 kg**, exceeding the fixed available budget by **122.074030522432 kg**. The v606 replacement prefix had already spent approximately **95.392567 kg more propellant** than the original prefix, leaving approximately half its return allowance. Changing only return dates cannot undo that earlier expenditure. These observations explain why the tested date changes did not rescue the fixed prefix; the empirical proxy and failed SCvx solves do **not** prove physical infeasibility, and the discrete epoch grid is not exhaustive. A distinct next score hypothesis should reduce upstream prefix fuel expenditure or change the replacement geometry, rather than repeat this return-only grid.

Total recorded wall time was **86.221170 s**. The five native wrappers accounted for **41.463645 s**, including host overhead. Baseline and control full-fleet verification accounted for **41.723452 s**. The run recorded **142 SCvx iterations / 53 accepted iterations**, with 142 QOCO report rows and 12,117 reported inner iterations; 98 QOCO rows were qualified and 44 unqualified. These counts describe stages of five return solves, not 142 or 12,117 independent trajectory solutions, and the timings are not pure GPU kernel time.

`final-report.json` contains all case details, source/output hashes, screening selection and accounting. Its `output_files` index covers every raw output, including all failure evidence, in the unchanged sibling experiment. `audit.py` is the exact CPU audit source; `report.json` preserves the earlier successful audit before the source added its own hash and direct prefix fuel-total calculation. The audit does not mutate the frozen experiment or claim an improvement.
