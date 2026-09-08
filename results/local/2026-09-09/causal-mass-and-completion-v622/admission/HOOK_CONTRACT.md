# Explicit integration contract

1. The normal native completion reconstruction loop checks the optional
   `search.completion_capture` callable before discarding a failed row. Without
   that callable, normal completion decisions and default selection are unchanged.
2. `CompletionRecorder(producer_sha256, consume, on_blocked=...)` receives each
   request/result slice. The launcher must establish the SHA of the actual loaded
   core. The recorder copies Result, LegResult and per-deployment cargo bytes,
   hashes source request/model metadata, and serializes the exact plan summary.
   Only outcomes 0/5 with all forward stages complete and internally consistent
   immutable cargo can reach the consumer. All other outcomes remain failures.
3. `consume(envelope)` constructs `AdmissionCandidate` from the envelope and an
   explicitly supplied fleet context: predicted weighted gain, proposed total raw
   fleet mass, and proposed ship count. It journals `queue.offer(...)`'s returned
   disposition and prunes its `request_sha256 -> plan_summary` mapping against
   `queue.retained()` after every offer. This prevents unbounded external storage.
4. `run_refinement_queue(queue, plans, catalogue=..., settings=...,
   verify_full_fleet=..., incumbent_weighted_kg=..., deadline=..., on_result=...)`
   accepts immutable summary JSON bytes, summary dictionaries, or copied
   RoutePlans. It consumes at most four claims, including at most two uncertain
   claims; failure never refunds a claim. It requires explicit CUDA outer-loop
   settings. The supplied finite deadline is checked between work items; the
   caller's foreground supervisor must enforce a hard process timeout and shared
   GPU lock for a stuck native call.
5. The fixed refiner uses the original requested cargo, including quantities
   below mining maximum. It makes one pass, with no resizing or retiming. It does
   not inspect `.feasible_proxy` to authorize the physical solve. Every leg and
   the complete route master must certify, and final dry mass must qualify.
6. `verify_full_fleet(refined_route, request_sha256)` must put that route into the
   supplied baseline fleet, emit exact Result bytes, and run the independent and
   official checkers. It returns:

   ```python
   {
       "request_sha256": request_sha256,
       "result_sha256": sha256_of_emitted_Result_bytes,
       "independent": {
           "ok": True,
           "result_sha256": sha256_of_emitted_Result_bytes,
           "ships": verified_ship_count,
           "total_mass_kg": verified_raw_fleet_mass,
           "weighted_score_fixed_bonus_kg": verified_weighted_fleet_score,
           # retain the full checker report as well
       },
       "official": {
           "ok": True,
           "result_sha256": sha256_of_emitted_Result_bytes,
           # retain the full checker report/stdout as well
       },
   }
   ```

7. `on_result(outcome)` persists the captured request, readback, claimed index,
   failed/successful refinement, final checker evidence and blockers before the
   next job. A `verified_gain` is eligible for the caller to persist; the executor
   itself never replaces the incumbent or inserts a proxy into the fleet master.

Control replays use equality only for inventory/ranking and explicitly refuse
claims. Any future fresh positive-control solve must have its own declared share
of the finite experiment budget; it cannot turn an archival replay flag into a
current certificate.

The CPU tests cover actual native reconstruction-hook placement using stubbed
readbacks, not another numerical native evaluation. Whole-route native execution
must still be measured from a coherent pinned source/library profile under the
shared GPU lock before making a performance or score claim.
