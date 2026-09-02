# G4 execution contract v1

This contract corrects execution semantics without changing the frozen G4 regime
matrix. The authoritative logical ledger remains:

- 24,883,200 logical rows;
- 19,353,600 measured rows;
- 5,529,600 warm-up rows.

No timeout, OOM, applicability decision, scheduler optimization, or H5/H6 early
decision may rewrite those counts.

## Four record layers

1. **Logical ledger row**: one frozen matrix coordinate and repeat disposition.
2. **Execution group**: one physical instance and solver-policy coordinate,
   containing exactly two warm-ups followed by seven measurements.
3. **Raw attempt record**: one separately retained warm-up or measurement,
   including terminal disposition, reason, and timing. A completed measurement
   contains a strict `paper1_result` with `record_scope=measured_attempt`.
4. **Publication record**: a strict `paper1_result` aggregate built only from
   schema-valid measured records and scoped `publication_aggregate`; warm-ups
   never enter its statistic.

The full ledger therefore schedules 2,764,800 process/session/workspace groups,
each with nine raw attempts. Publication aggregation has 138,240 regime
coordinates before product-specific filtering: the full group count divided by
the 20 paired evaluation instances.

An execution group is a hard process boundary. The executor must create one
persistent session and workspace, run both warm-ups, and then run all seven
measurements in that same process. It must restore the policy-declared
independent reset boundary between attempts without destroying the persistent
workspace. A capability record that does not declare
`g4-persistent-group-v1` is rejected.

## Applicability

`benchmarks/g4_applicability.json` is the hash-pinned authoritative
family×policy×axis contract.

- All 18 family×policy pairs in the frozen matrix are executable.
- Family class axes are executable only for their owning family. Unrelated
  family axes are `not_applicable` with an authoritative reason; they are not
  silently defaulted into physical identity.
- Fixed-loose, adaptive, adaptive+polish, pure QOCO, and hybrid quality tiers
  remain executable because the selected tier is an explicit final
  matched-quality target even when the solver's internal tolerance is fixed.
- QOCO scaling is executable campaign-side coefficient equilibration. QOCO
  Ruiz iterations remain disabled; these are not the same mechanism.
- A pure-QOCO `primal_dual` request is executable as a primal warm start with
  `discarded_unsupported` dual disposition. This does not make the solver
  unsupported.
- Hybrid applies the requested warm state to PDHCG and records any QOCO dual
  discard at handoff.

Future pinned capabilities may classify a combination `unsupported` only with a
specific technical reason. `not_applicable` requires the authoritative
applicability reason. Neither disposition is winner-eligible.

## Physical and order identity

Physical instance IDs are versioned content addresses over:

`family + intervals N + all applicable family classes + evaluation seed`.

Repeat, policy, quality, scaling, and warm-start identities remain separate.
Solver-order rotation hashes the complete physical coordinate plus quality tier,
conditioning, scaling mode, and warm mode. Repeat does not change the group
rotation; all nine attempts retain one policy order.

## Terminal dispositions and censoring

`hybrid_handoff_ineligible` means PDHCG construction actually ran but failed the
frozen `1e-6` handoff gate. It is not max-iterations, generic unqualified, or
unsupported. `hybrid_handoff_ineligible`, `not_applicable`, and `unsupported`
all require a reason and explicit timing and are never winner-eligible.

Timeout and OOM can be recorded only for an attempt that was actually launched.
No failure at a smaller coordinate predicts or censors a larger coordinate.
Every larger logical row remains pending until it is launched or receives a
contract-authorized static applicability disposition.

## H5/H6 claim-resolution core

`benchmarks/g4_h5_h6_claim_core.json` preregisters 360 persistent groups and
3,240 solver invocations:

- P1-E low-thrust at `N={100,500,2000}`, trust class 1.0, combined transfer:
  fixed-tight, adaptive, pure-GPU-IPM, hybrid;
- P1-C 3-DoF powered descent at `N={20,50,100}`, dispersion class 0.05:
  fixed-tight and adaptive;
- 20 paired evaluation instances;
- two same-session warm-ups and seven measurements.

Thus `(3×4 + 3×2) × 20 × (2+7) = 3,240`.

The core may resolve only H5 and H6. Its definition explicitly permits no
F01–F12 or T01–T08 regime product, and the publication builder rejects any
claim-core run. It cannot replace, populate, complete, or freeze the full G4
regime map.

## Batched-executor integration

The batched executor must add `--g4-session <execution-group.json>` and emit nine
`case=g4_attempt` JSON records from one process. Each record must preserve the
group/instance/repeat identity, indicate whether it was launched, carry an exact
terminal disposition, reason, and elapsed time, and set
`statistics_eligible=false` for warm-ups. Every measured record must carry a
strict, semantically valid `paper1_result`.

Before campaign launch:

1. merge this branch's schemas, identity functions, applicability locks, and
   group scheduler into the batched executor branch;
2. implement the persistent session protocol in the native executable;
3. regenerate an executor capability record declaring
   `g4-persistent-group-v1`;
4. initialize a new grouped checkpoint. Do not reuse the row-oriented active
   campaign checkpoint because its lease and warm-up semantics are different;
5. retain the old campaign evidence as historical attempts rather than
   rewriting or deleting it.

## Native session protocol and lifecycle

The production CUDA executable accepts:

`--g4-session <execution-group.json> <policy-sha256> <matrix-sha256> <capability-sha256>`

It validates the process contract, hash syntax, leased group ID, coordinate, and exact
`warmup-0,warmup-1,measured-0..6` order before CUDA work. It emits a flushed
`g4_session_ready` record, one flushed `g4_attempt` record after every terminal attempt, and one
`g4_session_complete` record. Usage/manifest errors exit 64, hash mismatches exit 65, and unknown
native failures exit 70. Launched solver outcomes remain data dispositions rather than being
collapsed into process exit codes.

One family owner creates topology buffers, one PDHCG workspace, one SCvx driver, and at most one
compatible QOCO workspace. Every attempt restores the immutable physical reference, performs
values-only updates, solves, independently replays, and applies the trust/acceptance transaction.
`cold` clears all iterates; `primal` retains primal but clears dual and momentum;
`primal_dual` retains primal/dual while resetting momentum. Pinned QOCO retains only accepted
primal state and reports a requested dual as `discarded_unsupported`. A failed attempt forces the
next boundary to cold rather than leaking partial state.

Raw records carry PID/context/workspace generations, workspace address, topology and coefficient
fingerprints, allocation/copy counters, requested/actual warm mode, dual/recovery/QOCO
dispositions, full timing identity, and direct-NVML attempt energy. Workspace setup occurs once;
CUDA startup is separately reported and excluded. Group deadline expiry emits explicit unlaunched
records, while an abnormal process is archived and restarted no more than once from a fresh group
boundary.
