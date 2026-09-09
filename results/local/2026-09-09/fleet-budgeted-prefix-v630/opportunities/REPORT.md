# Saved frontier opportunities, v630

The smallest useful next refinement set is **ship 22 rank 0 and ship 10 rank 0**, one fixed-cargo attempt each under the endpoint-merit-corrected core. Neither request has been attempted under that core. Their archived prefix failures therefore do not establish that this would repeat the same intervention. This report authorizes no execution and records no new trajectory qualification.

## Baseline and method

The checked frontier Result is `765cb7ef97d38926317dbbf4ae8700626dffec06c48af3d6f68dae47c154f3da`: 23 ships, 199 asteroids, 14,291.006160165 raw kg and 13,023.704900978004 fixed-bonus weighted kg. Against the original 23-ship minimum `(23 / .004) * ln(23 / 2)`, it has **247.5107067920743352162094538937037474998139622389621947017 raw kg** of fleet headroom. This is fleet headroom, not additional cargo or a trajectory feasibility guarantee.

The standalone standard-library `audit.py` verifies published archive/manifests and reads only saved plans, Result event cargo, bonus coefficients, source metadata and historical attempt records. It performs no ephemeris, Lambert, forward proxy, propagation, native call or refinement. Input/member hashes are retained in `findings.json`; exact request data and all aliases are in `positive-requests.json`.

Requests are deduplicated by the exact `FixedCargoRequest` serialization of ordered non-camp flights, deployment/collection epochs and cargo, independently reproduced without importing production code. Hardware aliases and numerical proxy metadata do not create distinct requests. The `1e-8 kg` reporting cutoff excludes Result serialization noise; it is not a revised production gate.

## Remaining positive requests

There are **20 distinct positive requests**: 14 for ship 19, five for ship 22 and one for ship 10. All fit the current fleet raw budget, retain their current ship's first Earth-leg body/epochs and have no asteroid conflict with the other current ships. Eligible choices from different ships also have no mutual asteroid conflicts. Thirteen requests have never been exactly attempted; seven have old-core attempts. All 20 are unattempted under the endpoint-merit-corrected core. The current ship 8 improvement has absorbed or dominated the positive ship 8 choices in these saved pools.

The maxima within the four distinct historically failed flight-prefix families are:

| Source request (local/H100 aliases deduplicate) | Weighted delta, kg | Raw delta, kg | Old failed leg (zero-based) | Exact historical attempt |
| --- | ---: | ---: | --- | --- |
| v802 / ship 19 / rank 0 | +15.368123680614 | +34.086242299859 | 3: 51417 to 22760, MJD 65498 to 65648 | Yes, old v803 core |
| v794 / ship 19 / rank 0 | +9.354866954317 | -4.106776180634 | 2: 37322 to 37066, MJD 65258 to 65438 | Yes, old v795 core |
| v794 or v802 / ship 22 / rank 0 | +7.698960992113 | -4.572210814499 | 1: 9443 to 32613, MJD 65093 to 65273 | No; lower-ranked requests shared that prefix |
| v794 or v802 / ship 10 / rank 0 | +5.528863601559 | +5.749486652825 | 17: 17126 to Earth, MJD 69263 to 69713 | Yes, old cores |

Exact request hashes, in table order:

```text
731c4cac5b231e83d0bc39a650265e7665b01b6382a7c57f49eea80731a26365
5a4a1de4cf26a8f7fc46febb87e5eca6cb515a06928f0793d974d4bbbfa86985
8cbeb4f32a0006010e5a567c0ecd4cd99f06ae5b95ef11eeb74c01590d8188ae
dcfc0573a6d96ca76120347a4e22ba91f33bbd53bfc1c59704c0481a8053a6af
```

If both recommended ship 22/10 requests certify unchanged, their prescribed cargo arithmetic would add about **13.227824593672 weighted kg and 1.177275838326 raw kg** to this pinned fleet. These are conditional values, not achieved gains. Each requires the original complete-route and fleet checks. Ship 19's two larger maxima remain separately identified alternatives; this report does not add them automatically to the proposed two-request set.

## Failure provenance and search diversity

All positive requests match a historically failed exact flight-epoch prefix, but those failures used the following old libraries:

```text
6019d43b540c11f2a326fba1927676d30e2e7326f5973ed16d28cdfa81cf644c
f8275f151608dca3de83fb0245e384117a223ae7884f7e1f25fdd075747bc66c
abac8a860c63ed1fcccc1f1fc80f69e7082f5fa2f31ef98a0c084934848d3700
d5b547cde1dbede90e60814b94fb992d47628123feb62141b0f66bc84c946b60
```

Their archived `gtoc12_scvx.cu` bytes all hash to `036de3500f31893d958e0f34b0c2f4c02d4579b80087b76f0b9691469a32516e` and lack the endpoint-merit penalty. The corrected core is `9597a56e2b3c033d419287bc1d6da6cca43b1f084475c42a2b7f4b64d301aeff`. Four completed v629 requests used that corrected core; none equals any of the 20 remaining positive requests. The audit retains all 92 attempt records and their exact source/core binding. Prefix identity alone does not prove identical starting mass, state or policy, and does not establish physical infeasibility.

For each hardware archive, v794 has 10,971 distinct requests and v802 has 11,007: 10,133 shared, 874 new in v802 and 838 present only in v794. Local/H100 request sets are identical within each phase. Saved search scripts keep the same incumbent first-level Earth seed per ship and zero first-level window, while v802 changes incumbent/exclusion context. Thus the newer pool is not a byte-identical rerun, but its positive frontier still occupies only the four prefix families above.

If the two corrected-core representatives fail, another automatic pass over sibling schedules with the same failing prefix would provide little new evidence. The next proposal should explicitly change Earth seed, hop-time/trajectory seed or another declared intervention and use a finite budget. The saved evidence does **not** prove that the broader route space is exhausted.

## Reproduction and retained failure

From the repository root, run `python -B build/performance/frontier-opportunities-v630/audit.py`. It reads the published v799/v806 source archives, current frontier Result/binding and the completed v629 batch evidence, all with retained hash checks. It is a saved-data audit and performs no numerical trajectory calculation. Outputs use finite JSON; historical nonfinite search-setting metadata is represented explicitly by `saved_float64` tags.

`attempt-a/` preserves the first writer failure: historical `Infinity` in a search-setting metadata field was rejected by strict JSON serialization after the main arithmetic outputs had been written. The final source adds explicit tagged metadata serialization and source-core differentiation. The successful run repeats only saved-data arithmetic. No original input was changed and no solver or GPU work was repeated.
