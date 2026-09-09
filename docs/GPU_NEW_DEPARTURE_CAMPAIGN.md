# New departure targets and native refinement, 9 September 2026

The broader H100 search found three new profitable **forecasts**, and changing an
early flight's timing produced a prefix with 16 certified flights. The Earth
return still fails. **No complete route or score improvement was certified.**
The incumbent remains **13,526.961241 weighted kg**, 14,915.044490 raw kg,
24 ships and 208 asteroids. Its original physics gates remain unchanged.

## Actual work

| Measurement | Observed result |
|---|---:|
| Different incumbent families searched | Ships 2, 5, 12 and 18 |
| Alternative Earth targets | Three per family, excluding its old departure target |
| Search policies | Unit mass and official bonus weights |
| Complete proxy plans | 3,502 |
| Distinct physical prescriptions across both policies | 2,076 |
| Standalone profitable, conflict-free prescriptions meeting the fleet raw budget | 3 |
| Recorded time for seven searches | 43.812279 seconds |
| Candidates in those seven searches | 3,019, or 68.91 candidates/second |
| Recorded Lambert branches in those seven searches | 104,076,562 |
| Unique native low-thrust solve calls across follow-ups | 38 |
| Native outer updates / accepted updates | 369 / 155 |
| Unique flight solutions meeting numerical qualification and receiving flight certificates | 30 |
| Flight certificate calls, including repeated prefix checks | 86 |
| New complete certified routes / full-fleet checker calls | 0 / 0 |

These are one campaign's counts, not a paired speedup measurement. The first
search's 483 plans were saved before reporting failed on an infinite settings
value. They were reused byte-for-byte. Its timer and branch count were not saved,
so neither is invented or included in the seven-search rate. The initial failure
and the reporting-only correction are archived.

The search used the previously validated shared-return runtime: CUDA Lambert
screening, collection operators, expansion, ranking and admission. Beam width was
64, maximum depth 10, with a 60-second cap per search. Candidate sets excluded
all other incumbent ships' asteroids before screening. There were no conflicts
among the saved candidates and the retained fleet. Search control and some
collection preparation remain Python; this is not yet a fully GPU-controlled
end-to-end planner.

## What the refinement established

Both objective policies found the same three ship-18 prescriptions. Their
forecast weighted gains were **8.076660, 6.433949 and 1.505818 kg**. The independent
saved-cargo audit recomputes these against the exact 24-ship Result and bonus
table, including its **5.604591 kg** fleet raw-mass headroom.

All three share their first flights. Their new Earth departure and first asteroid
hop passed CUDA flight certification, but the 42100-to-8846 flight at MJD
65078–65258 stalled at a normalized dynamics defect of **4.341414e-4**, against
the unchanged **5e-9** gate. Identical boundary/settings requests reused saved
solver readbacks; successful flights still received fresh independent checks.

Increasing the virtual-control penalty from 10,000 to 100,000 did not reduce that
defect. At 1,000,000 it worsened to **5.606928e-3**. The penalty remains unchanged
in production and in the subsequent timing experiments. These outcomes do not
establish mathematical infeasibility.

Giving that third flight **15 or 30 additional days**, shifting five subsequent
deployments and consuming part of the later wait at asteroid 1122, resolved the
early failure under the original solver policy. The prescribed cargo was reduced
by the actual lost mining time. Both variants passed **16 flight certificates**
before failing on the Earth return. These are flight certificates, not a complete
route certificate: waits and fleet composition would still need complete checks.

The first timing preparation stopped before any GPU solve because subtracting
lost mining mass differed by floating-point roundoff from the strict production
limit. The corrected preparation computes the allowed cargo directly from the
new mining duration. It changes no admission tolerance; both records are saved.

| Earth-return experiment | Maximum normalized dynamics defect | Qualified? |
|---|---:|---|
| +15-day deployment schedule, original return arrival | 0.001186157 | No |
| +30-day deployment schedule, original return arrival | 0.000244498 | No |
| Same +30-day schedule, Earth arrival +60 days | 0.013177147 | No |
| Same +30-day schedule, Earth arrival +120 days | 0.002015708 | No |
| Same +30-day schedule, Earth arrival +180 days | 0.772210057; reported virtual control infinite | No |

The final three tests reused the exact 16-flight prefix and held cargo fixed.
They used fresh cold CUDA initialization for each changed return, not a coast
extension of old controls. No failed return was emitted into a fleet.

Native initialization, propagation, derivatives, assembly, QOCO solves, outer
decisions, route ephemerides and flight certification ran on CUDA. Python still
handles orchestration, artifact writing and boundary/readback caching. Both
original full-fleet checkers remain mandatory before score promotion; none was
called because every proposed complete route had already failed a flight gate.

## Evidence and next decision

The search and all native follow-ups are downloaded locally, with exact controls,
state arrays, boundaries, prescriptions, certificates, solver histories, runtime
identities and failed preparations. Standard-library audits verify all **1,209
archive members**, independently recalculate cargo with Decimal arithmetic, and
compare every reused prefix's saved array payloads. These audits inspect saved
evidence; they do not repeat propagation or certify new trajectories.

- [Search evidence and portable audit](../results/lambda/2026-09-09/gpu-family-departures-v850/README.md)
- [Refinement, penalty and timing evidence](../results/lambda/2026-09-09/gpu-family-refinement-v855/README.md)
- [Previously validated native runtime and paired speed measurements](GPU_SHARED_RETURN_OPTIONS.md)

The useful next mission input is the saved **v854 rank-1 prefix**, ending at
asteroid 8846 before the Earth return. Further work should examine its return
defect and fuel requirement, or change collection/return geometry while charging
every cargo loss to the actual fleet objective. Repeating the tested penalty
increases or the three tested later arrivals has no measured support. Moving
collection-DP preparation and scheduling onto CUDA remains the separate next
performance target.

No solver defaults or production source changed in this campaign. All owned
Lambda workers exited; the final H100 inventory was idle. The visualiser retains
the verified [v633 fleet](http://127.0.0.1:4173/?dataset=gtoc12-fleet-v633&epoch=69807&preset=oblique&z=1).

To inspect the downloaded statistics and audit both packages in PowerShell:

```powershell
Set-Location 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser'
python results/lambda/2026-09-09/gpu-family-departures-v850/audit_saved.py
python results/lambda/2026-09-09/gpu-family-refinement-v855/audit_saved.py
```
