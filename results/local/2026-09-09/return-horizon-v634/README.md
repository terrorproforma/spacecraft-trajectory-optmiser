# Two return horizons: no certified gain

The single v634 run tested the two prescribed ship 10 Earth arrivals, MJD 69743 and 69773, using the unchanged archived thrust plus 30 or 60 days of zero-thrust coast and the existing GPU mass-scaled initializer. Both returned `infeasible` optimizer labels because substantial dynamics and virtual-control residuals remained. This is not a mathematical proof of physical infeasibility.

| Saved outcome | +30 days | +60 days |
| --- | ---: | ---: |
| Earth arrival MJD | 69743 | 69773 |
| Nodes | 241 | 256 |
| Native outer updates, including conditioning retries | 44 | 32 |
| Accepted outer updates | 7 | 4 |
| Qualified / unqualified inner conic readbacks | 19 / 25 | 15 / 17 |
| Conditioning retry calls, included above | 11 | 7 |
| Maximum reported normalized dynamics defect | 0.004298365325523523 | 0.024889981199908484 |
| Reported virtual-control infinity norm | 0.004298365325523491 | 0.024889981199908442 |
| Returned final node mass, **unverified kg** | 1177.207392197104 | 1177.2073921971253 |
| Native call time, seconds | 10.433911 | 10.364973 |

The required dynamics-defect limit remained **5e-9**. The original-window v632 return had a smaller, still failing defect of 0.0027817923491237906. These two later arrivals therefore did not resolve the dynamics failure with this seed policy. The returned node masses nearly equal the 1177.2073921971253 kg requirement, but a node mass from a dynamically inconsistent trajectory does not establish real delivered mass. The last accepted updates were 15 and 6; both runs subsequently ended with unqualified inner conic readbacks and small trust regions. The new residual axes/intervals were not replayed or localized, and the original virtual-control vectors were not saved.

Actual work was **two seeded native returns, 76 outer updates, one CUDA ephemeris batch with two Earth targets, zero return certificates and zero full-fleet checker calls**. The worker took 22.407053948 s. This is one unpaired mission diagnostic, not a throughput or speedup benchmark. Ephemeris traffic was 56 bytes of elements and 32 bytes of requests uploaded, and 112 bytes of states downloaded. The exact Earth request/readback arrays and initial/control seed payloads are retained. Root's foreground session exited zero; the supervisor reaped the worker and verified no owned descendants or remaining GPU compute processes.

The fixed 17-flight, one-wait prefix was imported from its saved certificates, with no prefix solve or new wait certificate. Departure MJD 69263, all deployment/collection events, 677.2073921971253 kg of cargo, certified starting mass 1438.8883038351994 kg, native source, QOCO settings and physical gates stayed fixed. The two new Earth targets came from the existing production CUDA ephemeris API. No candidate fleet was emitted because neither return qualified.

The verified incumbent remains the [24-ship v633 fleet](../fleet-addition-v633/README.md): **13,526.96124117307 weighted kg**, **14,915.0444900762 raw kg**, 24 ships and 208 asteroids, Result SHA-256 `1f420928bbef8da91e3a6f8b74d3bf00039c78d4257215c103a484bc55bb22da`. The prescribed candidate gains were never certified and are not added to that score.

The exact prelaunch ready manifest is `ac6379bdaf8bdc28f4296509ab9aa3bddbac2903c53c9edd15759529879d5edf`. All 49 prepared files, all 36 raw output files, prior preparation/binding reports, shared frozen Python source, prefix evidence and native source archives are preserved by original project path in `evidence.zip`. Its native library identity is `4fdeabc62fe80dfe957fc7ad1d44ced15a7c5f4ac2c6fd9e9f84634bc3837344`; the 744-member source archive is `f869b74efa3e2cce21045cdc74d7946ba53ad4409b2d419600ffb9a97417aac8`. No executables are packaged. Seven runtime/official-data assets are recorded by exact hash rather than copied; their identities were checked by the saved preflight. A portable archive audit does not rerun those programs or recertify their results.

`saved-audit.json` independently checks saved counters, exact seed/control retention, the zero extensions, copied GPU Earth targets, unchanged prefix/cargo, and the reported failure classifications. Run `python -B verify_package.py --index-sha256 <index SHA>` for a standard-library-only archive/member roundtrip and saved-output audit. No trajectory equation, optimizer, GPU call or verifier is invoked.

The preparation README and earlier reports inside the archive are historical protocol records. The terminal evidence and this README describe the actual completed run. See [NEXT.md](NEXT.md) for one distinct possible seed intervention; it has not been authorized or executed by this package.
