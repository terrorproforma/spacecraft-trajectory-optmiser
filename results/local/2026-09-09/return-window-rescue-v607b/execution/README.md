# Fixed-cargo return-window rescue v607b

This is the executable successor to the preserved v607 CPU plan. Preparation and tests use the CPU only; no new GPU measurement is claimed by this kit. Root owns the single subsequent launch.

The experiment retains ship 7's 16 certified replacement-prefix legs from v606, including every deployment, collection, cargo value and emitted burn sample. It changes only waiting after the final collection and the Earth-return window. The other 22 fleet ships remain unchanged. There are **zero whole-route reruns** and **at most five fresh native return solves: one original-prefix control, then at most four replacement-prefix candidates**. A failed control prevents all candidate screening and refinement.

The exact original control is first reconstructed and independently checked on the CPU. Its complete archived Result is reproduced byte for byte. Execution must still solve its MJD 69218→69728 return afresh and pass both fleet checkers; replaying that certificate is not the solver control. Imported legs require their saved successful certificate, verified array hash, consistent mass ledger, fresh independent propagation and the stock route-certification registry. An imported prefix alone is not a certified fleet.

The replacement collects at asteroid 59653 at MJD 69218. Its departure mass is **1,207.2044215842654 kg**, fixed cargo **615.6605065023956 kg**, minimum mass before Earth unloading **1,115.6605065023955 kg**, and available return propellant **91.54391508186995 kg**. Legal windows satisfy `69218 <= departure < arrival <= 69807`. Waiting adds no mining event or cargo. Independent coast propagation checks endpoint consistency and an analytic asteroid-perihelion lower bound of approximately **2.7467748 AU** protects the complete waiting path. Full-fleet checks remain mandatory.

GPU paired Lambert screening evaluates **7,022 coarse windows** and at most **2,592 additional fine windows**: at most **9,614 pairs / 19,228 direction requests**. The API includes the permitted Earth-arrival excess velocity. Ranking uses the frozen return-inflation/propellant model and thrust authority, followed by deterministic selection of four epoch basins separated by at least 20 days where possible. All raw rows, including invalid geometry and negative predicted reserves, are retained. The best estimates may be refined even when the proxy predicts no feasible window. These are ranking estimates, not low-thrust certificates; the sampled time domain is not exhaustive.

Each selected return uses the unchanged native v590 core, QOCO540, CUDA cold seed, graph execution and v606 SCvx settings. The fresh return must pass the existing independent leg and full-route gates. The emitter then checks every prefix event's epoch, position, velocity and mass exactly, checks every prefix burn sample, replaces only ship 7, and runs both independent and official fleet verifiers. Prescribed raw and weighted cargo totals must match; no scaling, earlier-prefix retiming or acceptance-tolerance changes are permitted. Failures and returned arrays/SCvx diagnostics are retained, and an exception consumes its native-call budget.

The retained fleet is **14,051.854893908598 raw kg / 12,810.135953048577 fixed-bonus weighted kg**. A qualified replacement would yield **14,053.086926762808 raw kg / 12,820.044717795105 weighted kg**, approximately **+1.232033 raw / +9.908765 weighted kg**. All candidate windows have that same fixed cargo objective. Among candidates that independently achieve it, larger final dry mass breaks a score tie, preserving a greater propellant margin. Failed or partial trajectories are never promoted.

`preparation.json`, `profiles.json`, `source-sha256.json` and `inputs-sha256.json` retain provenance. Frozen Python is `b08b1f5aa464659e558926713d29bcd364a3778f`; the native core is `ffbae813f1f683d276213cb83ec8be17d1c2338c9c4a96f3ec5283e26a432b73`, and QOCO is `0cc27a1d8bde1edd74f7ef00e04ec64c2d6c0f8fe526a1d94b04d509a74b3315`. The 190 source files and 78 copied inputs are rehashed before use. `fixed_refine.py` is copied byte for byte from v606. Concurrent production edits are excluded. The unused H100 profile is retained as original provenance; this supervisor permits only the local profile.

`validation/` records CPU tests, lint, source/input/native-library hashes and construction checks. Native library loading is blocked throughout the behavioral tests and prefix audit. These checks establish preparation correctness, not new return-solver convergence or a score gain.

The foreground supervisor checks the shared lock and actual compute-process inventory immediately before launch, passes its held lock to the child, records both PIDs and waits for completion. It creates `launch.json` exclusively, refuses existing output, retains busy/failed attempts and never retries or kills another job. The soft wall budget is 1,800 seconds, checked between operations; an in-flight call retains the unchanged 900-second SCvx setting. No claim is made that this is a hard process-termination deadline.

Review the command without launching:

```powershell
wsl -d Ubuntu-22.04 -- /home/angus/worktrees/spacepdhcg-literature-venv/bin/python -B /mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser/build/performance/return-window-rescue-v607b/launch.py
```

Root's one authorized execution, after reviewing the ready-manifest and shared GPU state:

```powershell
wsl -d Ubuntu-22.04 -- /home/angus/worktrees/spacepdhcg-literature-venv/bin/python -B /mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser/build/performance/return-window-rescue-v607b/launch.py --execute
```

Outputs are `output/report.json`, per-case prescriptions/arrays/outcomes/checker reports, `output/screening/` raw geometry and selection, and any both-checker promotion under `output/best/`. The existing visualizer and production code are not modified by preparation.
