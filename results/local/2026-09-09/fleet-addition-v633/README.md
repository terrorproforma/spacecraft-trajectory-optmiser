Both original full-fleet checkers pass the final 24-ship, 208-asteroid Result.
The independent score is 13,526.96124117307 weighted kg and 14,915.0444900762 raw kg:
a gain of 503.256340195067 weighted kg and 624.0383299112 raw kg over the pinned
23-ship v629 frontier. The 24-ship raw-mass rule has about 5.604591348198 kg margin.
The official console rounds total mass to 14,915 kg; its full ScoreData is retained.

This gain comes from recovering an existing certified route from saved fleet data
and composing it with the current fleet. It is not a new trajectory optimization
or PDHCG performance result. The route was joint_itinerary_h100_v2 family_300012
ship_04, serialized as ship9 in fleet_master_v10. Its nine deployment/collection
pairs are self-contained and disjoint from the current 199 asteroids.

The final Result preserves the entire original 23-ship Result as an exact byte
prefix, adds one LF separator, then the recovered route with ship label9 changed
to24. Only the appended route's final CRLF was removed after the first official
checker rejected it as an empty EOF line. The 100,391 nonempty row payloads are
unchanged between attempts, and both complete independent-checker outputs are
byte-identical. The first EOF rejection and every raw report are preserved.

There were two independent plus two official full-fleet checker calls in total,
with 47.144631559 seconds summed worker wall time for the two pairs. Each ran
under a 120-second hard deadline and reaped its descendants. There were zero new
GPU, optimizer, search or extra leg-certificate calls. These timings describe
checking saved fleets; they establish no solver or pipeline speedup.

The archive includes 192 frozen v632 host source files, original wrappers,
supervisors, process guard, source maps, event/cargo proofs, exact added route,
complete independent leg results and official outputs. Executables and catalogue
copies are excluded with exact hashes. The original current Result and both
checked candidate variants reconstruct losslessly from top-level Result.txt.
The historical full fleet and full bonus table remain explicit external hash
dependencies; the extracted ship bytes, historical checker metadata and all208
exact bonus coefficient strings are included. They are sufficient for the
portable saved-data audit, which performs no propagation or archived-code execution.

The historical viewer trajectory array is absent locally. This package does not
invent a dense history: the added Result supplies20 saved event states; existing
23-ship replay arrays belong to the separately retained viewer dataset.

Run `python audit_package.py . --index-sha256 INDEX_SHA` from a copied package,
or supply its directory as the first argument. Only Python standard library is
needed. Re-running scientific checkers is separate from this static audit and
requires the pinned external official data and scientific runtime.
