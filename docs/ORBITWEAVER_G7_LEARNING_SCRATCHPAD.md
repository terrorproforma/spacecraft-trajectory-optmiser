# OrbitWeaver G7 learning scratchpad

## Stable facts

- The base already provides CPU/native OrbitWeaver fidelity stages, warm-reference storage,
  robust risk semantics, route master/column generation, dynamic time discretisation and
  independent J2 certification.
- G3 exposes opaque persistent SCvx drivers. G7 should compose that public lifecycle rather
  than duplicate solver state.
- G5 is not stable on this branch. Ownership and backend interfaces are the correct merge seam.
- Deterministic fixed result regions make unsupported/no-root/error cases auditable and avoid
  nondeterministic GPU append ordering.
- A logical multi-rank test is interface evidence only.

## Guardrails retained

- Lower bounds never exceed returned costs.
- Failed arcs stay present.
- Scenario probabilities sum to one.
- Non-anticipative prefixes are checked before risk aggregation.
- Certification has an independent backend identity and five explicit checks.
- Optimizer status cannot set `certified=true`.
- Work queues and device storage are bounded by configuration.
- Frozen benchmark data are loaded only after SHA-256 verification.

## Remaining questions for stable branch integration

- Exact G5 adapter type names and rank-local cancellation propagation.
- Final G4 policy manifest fields to include in the G7 run manifest.
- Whether G5 exposes deterministic reductions as a runtime mode or build option.
- Physical multi-GPU machine topology and the approved Paper 2 campaign commit.

## Schema audit lessons

- Dataclass validation and hand-maintained JSON schemas drift unless one is generated from
  the other.
- Python equality treats `True == 1`; JSON Schema does not. Constant/enum validation needs
  JSON-type-aware equality.
- Python's JSON reader accepts `NaN` and infinity by default even though they are not
  standard JSON. Record readers must reject `parse_constant`.
- A schema-valid record can still be semantically invalid: gaps, bounds, telemetry totals,
  manifest hashes, seeds and repeat indices require cross-field checks.
- OOM, timeout and censored records can legitimately retain partial incumbents, but cannot
  be marked certified and must retain an explicit matching failure.
