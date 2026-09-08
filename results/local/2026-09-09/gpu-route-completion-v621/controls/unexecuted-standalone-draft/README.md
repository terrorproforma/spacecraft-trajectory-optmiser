# Historical completion controls and residual diagnosis (v621)

This CPU evidence uses the **23 historical v595/v616 routes**, not the newer retained fleet. It explains remaining cost-proxy rejection and prepares exact controls for the new batched GPU finishing kernel. No trajectory was solved, refitted, certified or promoted in this preparation.

## Residual diagnosis

`residuals.json` prices every flight at its own **archived measured burn mass**, using its saved Lambert DV and exact saved epochs. There are 404 flights: 23 Earth outbound, 173 deployment hops, 185 collection hops and 23 returns. `features.json` retains 358 deterministic catalogue feature calculations; no Lambert solve or search was called. Catalogue SHA256 is `99a42cc30d4498d99b8acf507790ab74f040ff2e202ef6c8e90bbb39b6c46675`.

Positive error means predicted fuel exceeds archived measured fuel. All values below are kg of propellant, not the weighted mission objective.

| Component | Flat-policy median error | Existing-fit median error | Flat-policy summed error | Existing-fit summed error |
|---|---:|---:|---:|---:|
| Deployment hops | +3.383 | +3.383 | −36.397 | −36.397 |
| Collection hops | +2.413 | +0.322 | +72.906 | +141.289 |
| Earth returns | +18.364 | +18.364 | +519.942 | +519.942 |

The existing five-feature fit applies to collection hops only; deployment keeps the frozen flat beam policy. Earth outbound cost is the measured seed cost by construction. The existing fit was **not refitted**, and these controls are not claimed to be statistically held out from its historical training data. A better median with a worse aggregate collection error is not a blanket improvement. Large negative outliers also make deployment sums and medians differ.

The generic return model overpredicts 20 of 23 archived returns at their actual burn masses. The largest positive residuals are ship 9 (+71.955 kg), 14 (+55.569), 12 (+54.917), 20 (+53.870), 7 (+50.417), and 10 (+37.248). The `ranked_positive_errors` field preserves **all** legs in descending error order, including negative entries; `legs` contains full role, epoch, mass, DV, feature and input provenance.

Every saved cargo quantity matches its archived plan and stays within the mining-duration bound. Every archived route has positive final dry margin (0.760–8.955 kg). These are historical certificates and inventory bindings, not fresh independent trajectory checks.

For historical ship 23, the measured-prefix/fit CPU completion still rejects at **−6.385482 kg** margin despite an archived **+2.861384 kg** dry margin with exactly the same cargo. At archived masses its fitted collection-hop overestimate totals +8.151714 kg and its return overestimate is +3.995258 kg. Their sum is a local component diagnostic: it is not the sequential forward-pass error, because predicted earlier burns alter later masses. Together with valid unchanged cargo, this supports a proxy false-negative diagnosis for this historical control. It does not prove arbitrary rejected new routes feasible.

`residual-audit.json` independently checks all 808 leg/model combinations with scalar arithmetic, matches all 23 return components to the v619 report, and finds a maximum fuel difference of 1.42e−13 kg. There are no new mission or score results here.

## Native-ready fixtures

`fixtures.json` contains the prespecified five v619 controls (ships 23, 1, 4, 7, 10), each with two prefix masses and two fixed models: **20 historical cases**, 172 deployment slots and 232 forward leg slots including camps. Eighteen fail the final dry-plus-cargo gate; two ship-4 flat-prefix cases pass. All original historical flight masses, inflation factors and propellant values match the saved v619 trace exactly.

Another **22 synthetic controls** cover missing collections, short stays, invalid mining input raising `ValueError`, authority before invalid inflation, certified-cell authority, camp/reposition behavior, repeated pickups, epoch tolerance, exact dry-margin boundaries, all supported cost models, nonfinite arithmetic and CPython 3.12 compensated cargo summation. These are numerical controls, not mission proposals. Both batches use the same policy, so the runner concatenates them into one 42-case C ABI call (196 deployment slots, 260 leg slots).

`gtoc12_completion_c_api.h` is frozen at SHA256 `e176bb0d5848981a140e264a71d10ac744ce1babc946d5edf5caf01dc3f9dc76`. Model and role lookup are resolved input provenance. `fixture_oracle.py` checks the exact frozen `_finish` gate flow with those fixed cost-model inputs; it does not rerun geometry or certified-cell lookup. Per-leg/early-failure numeric details follow the explicit native ABI, because Python returns only `None`, `RoutePlan`, or `ValueError`. Qualification and successful totals are checked separately against `_finish`.

Fixtures SHA256: `bd2581b1a7034961adce09f063a5bf57610c3636d4cd95591dcdbd36dab97713`.

The 52 dedicated CPU tests passed in 4.64 seconds; Ruff passed. Full logs and exact authoring-script hashes are in `validation-03`. `validation-01` and `validation-02` retain the earlier lint-only failures. The final frozen Python source tree contains 190 files, with search.py SHA256 `9c5d64e37ceea7e5fd8c680556cd41bfbbb1e5a282aacba5d3ede0cbe0797d18`.

To repeat the existing CPU fixture checks without model fitting, catalogue access or native loading, use the existing Python 3.12 environment:

```bash
cd /mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser/build/performance/completion-native-controls-v621
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /home/angus/worktrees/spacepdhcg-literature-venv/bin/python -B -m pytest -q -p no:cacheprovider test_fixture_evidence.py
```

## Prepared GPU check — not executed by this preparation

`gpu_runner.py` is a foreground supervisor with a single-use marker and the shared nonblocking GPU lock. It verifies the exact native build manifest, source files, binaries, fixtures, Python source and inputs; it also requires the pinned RTX 5090 UUID and an empty compute-process list. Busy or invalid preflight consumes the attempt and preserves the result, with no automatic retry.

The finite budget is one standalone native test execution (2 kernels, 262 case evaluations) followed, only on success, by one Python C ABI call (1 kernel, 42 case evaluations). Each child process has a 180-second deadline; the supervisor terminates and reaps only its own process group on timeout. All raw readbacks and mismatches are retained before classification. The standalone library is not a full-core integration build.

`gpu-profile.json` fixes FP64 comparison tolerances before execution: absolute 2e−10 and relative 2e−13 for ordinary arithmetic, exact classification/indices/stages, exact cargo arithmetic and exact boundary/summation case results. Numeric tolerance never changes a pass/fail decision. This is a parity check, not a throughput benchmark or mission-feasibility certificate.

Root must review `gpu-ready.json` and launch once; this preparation does not execute it:

```powershell
wsl -d Ubuntu-22.04 -- env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /home/angus/worktrees/spacepdhcg-literature-venv/bin/python -B /mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser/build/performance/completion-native-controls-v621/gpu_runner.py
```

The next score-directed hypothesis is a controlled correction or escalation for residual return/collection proxy overprediction, evaluated first on these historical positive controls and prespecified held-out evidence. Native completion must first reproduce the same admitted/rejected pool. Any new mission candidate still requires the unchanged full-route refinement and both full-fleet verifiers; this evidence does not authorize wider generation or lower cargo.
