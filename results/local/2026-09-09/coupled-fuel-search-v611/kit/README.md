# Coupled itinerary fuel search v611 — prepared, not executed

This finite local experiment searches all deployment and collection epochs together on four existing feasible routes. It tests whether a greater value for estimated remaining propellant leads the native itinerary search to a higher **verified fixed-bonus weighted cargo score**. It does not change asteroid order, miner count, ship count, acceptance tolerances, or the retained fleet. No GPU measurements have been made for v611 during preparation.

The retained 23-ship fleet has 14,051.854893908598 raw kg and 12,810.135953048577 fixed-bonus weighted kg. The 23-ship raw-mass floor is 14,043.495453372925 kg, leaving 8.359440535673 kg of fleet slack. Raw mass governs ship-count eligibility; weighted cargo governs improvement.

The four independent own-miner seeds are ships **20, 7, 10 and 11**, each with 18 visits and 17 flight legs. Among the five eligible 17-leg incumbents they have the largest measured single-leg propellant use, respectively 218.418435, 188.408299, 174.503525 and 164.976442 kg. All four original CPU reference evaluations are feasible and use measured costs for every leg. Their cargo matches the retained Result and their asteroid footprints do not intersect other ships.

Each seed is searched with margin prices **0.05, 0.25 and 1.0**. The search objective is estimated weighted cargo plus price times estimated spare propellant. The price term is only a search heuristic: shortlist and promotion use weighted cargo gain, and require the raw fleet mass floor. Each arm sees identical starting ships, mesh levels and caps; ordering alternates between seeds. Actual evaluations and wall time may differ between arms. This is not a speed comparison.

The initial preparation attempted four timing seeds for the v606 ship-7 substitution. `seed-audit-01.json` preserves their failures: the native conditional graph stops on an infeasible seed, so those starts would not test the intended coupled search. The executable experiment instead starts from the certified incumbents. `incumbent-inventory-01.json` and `incumbent-controls.json` document this change. `preparation.json` records the original 27-input stage; the final input index contains 33 files after adding incumbent summaries and archived control evidence.

## Finite budget and acceptance

- At most 12 native whole-itinerary searches, each with a 30-second soft deadline, mesh 30/10/3/1 days, and at most 12 moves per mesh level.
- At most 7,969 itinerary rows per search: 95,628 across all searches, or 3,251,352 paired-direction Lambert requests as an upper bound. Work counters retain actual values.
- At most one distinct eligible final candidate per margin price, hence at most three new full-route refinements and 51 new native leg solves. The absolute defensive caps remain four full refinements and 68 native leg solves.
- Exact original target-ship bytes match already both-checked local v707 control results under the same core/QOCO and equivalent critical Python source. These archived controls avoid redundant full-route reruns. The run still performs fresh independent and official checks on the entire retained fleet before any search.
- Selected candidate cargo and epochs are immutable through the copied fixed-cargo refiner. No automatic cargo shrinking, extra retiming or tolerance relaxation is permitted. Failed leg arrays and diagnostics are retained.
- Every candidate promotion needs both complete fleet checkers, exact expected raw/weighted totals and a weighted improvement. At most three qualified variants permit at most eight subset checks for compatibility; a combined winner is checked again as a complete fleet. A failed combination falls back to the best already checked individual.

The initial reviewed run budget is **600 seconds**. The supervisor waits at most wall budget plus 30 seconds, then terminates only its owned new-session child process group, allows 10 seconds for cleanup, and kills/reaps it if necessary before releasing the shared GPU lock. A timeout preserves the raw report and adds `supervisor-timeout.json`, explicitly marking saved counts as partial. It never retries. A native solve underway at the deadline can therefore be interrupted and remain uncertified. Individual SCvx settings remain the demonstrated v707 settings; the outer deadline controls the whole run.

## Frozen implementation and scope

`source/` contains 190 files exported from commit `f8b2ac7aeba9e29ea90f5814060bb8cfe5db22d7`; concurrent working-tree changes are excluded. The native search wrapper and all critical refinement dependencies match the validated v707 source. The extra frozen hop-inflation JSON was absent from that source manifest and is not reported as a conflicting implementation. `fixed_refine.py` is an exact copy of the demonstrated v606 fixed-cargo wrapper.

`profile.json` pins the actual local RTX 5090 libraries: v702 core SHA256 `65f1335e3af820d22451e0d4f587f2d26aeccf364a0ed3bb9597bfd009bac5aa` and v683 QOCO SHA256 `5b1b1a047f9d5041f821b64490562a3d598f51b47fedb08d89cdae211728940a`. Fresh file hashing and `nm` confirm the local `spacepdhcg_gtoc12_joint_search_host` export without loading CUDA. Library binaries are not copied into the kit.

The graph handles repeated geometry, itinerary evaluation, winner selection and mesh updates on the GPU. Python constructs jobs, retains artifacts, orchestrates full-route refinement and runs the independent fleet checkers. The propellant screen remains an estimate and cannot establish physical feasibility. Only the unchanged low-thrust certificates and complete fleet checkers establish acceptance.

Prior v707 evidence reports 111 tests on each GPU and local moving-loop racecheck/synccheck success. Its local active-loop memcheck returned unresolved CUDA error 999; this kit does not claim that path is memory-clean. The local whole-fleet control evidence also establishes actual local execution, despite its archive living under `results/lambda/.../gpu-device-search-v707`. No new H100 execution is part of this kit.

This experiment advances route quality by exploring coupled mission timing with explicit fuel valuation. It follows the score-search part of the roadmap and uses the existing native graph acceleration; it is distinct from the exhausted fixed substitution and return-window sweeps. A gain is not guaranteed, and a negative result cannot prove mission infeasibility.

## CPU validation and launch

`validate.py` runs the CPU behavior suite with native loading prohibited, Ruff checks, exact source/input/library hashing, control checks and local symbol inspection. It creates `validation/` and the final `ready-manifest.json` only after all checks pass. All indexed files are rehashed before launch. The same single-launch marker prevents another execution in place, including after a failed pre-GPU observation.

After root checks the actual GPU state, run from the repository in PowerShell:

```powershell
wsl -d Ubuntu-22.04 -- /home/angus/worktrees/spacepdhcg-literature-venv/bin/python -B /mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser/build/performance/coupled-fuel-search-v611/launch.py --execute --wall-seconds 600
```

Omit `--execute` to print the recipe without creating a marker, loading CUDA, or acquiring the GPU. The supervisor writes `launch.json`, `run.log` and a new `output/` directory. Keep its foreground tool session alive until the owned child exits. Expected resources are one local RTX 5090, serial search/refinement work, small 166-row device neighborhoods, and full-fleet verifier CPU time. Peak VRAM and total runtime remain unmeasured for this experiment.
