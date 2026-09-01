# Paper 1 G6 freeze tooling

Status: tooling implementation only. This document does not assert that G4, G5, G6, or Paper 1
has passed or frozen.

Scope update: schema `1.1.0` configurations require `campaign_scope_id`. The active
`single-gpu-v1` scope may freeze after complete portable in-scope G4 evidence; H4 and
F07/F12/T06 are recorded as deferred rather than fabricated. Historical schema `1.0.0`
configurations remain readable as `full-multi-gpu-v1`, and that scope still refuses to freeze
without physical P1-F evidence at 2, 4, and 8 GPUs.

## Architecture

`spacepdhcg.paper1` is deliberately downstream of the G4/G5 run producers:

1. `evidence.py` loads each archived `run-manifest.json`, `paper1-result.json`, and
   `evidence-record.json`. It validates run-ID/commit/family/solver traceability, separate
   independent-residual and replay artifacts, immutable URIs, content hashes, internal-index
   hashes, and optional local payload bytes. Failure/censoring statuses remain first-class rows.
2. `aggregate.py` applies the frozen source-data contracts for F01-F12 and T01-T08. It writes
   canonical JSON for every product, PDF/PNG figures, and JSON/CSV/TeX tables. Numeric plot input
   comes only from validated records.
3. `decisions.py` emits H1-H6 records with the frozen practical thresholds, 10,000-sample paired
   percentile bootstraps, fixed seeds, 95% intervals, sustained-scale rules, censored run IDs, and
   explicit `supported`, `rejected`, `mixed`, or `unresolved` outcomes. A scoped physical-only
   hypothesis uses `deferred-not-in-scope`, which is not a scientific outcome.
4. `freeze.py` orchestrates products, decisions, evidence indexes, checksums, claim linkage,
   byte-reproducibility checks, and clean-clone checks. `freeze` rejects synthetic, dirty,
   incomplete, unpinned, local-only, hash-mismatched, timing-inconsistent, or untraceable input.
5. `synthetic.py` generates deterministic fixtures labelled as non-evidence. They exercise
   positive, negative, mixed, unresolved, OOM, timeout, no-crossover, failed-polish, numerical,
   unsupported-formulation, and infeasible-trajectory paths.

The versioned contracts are:

- `papers/paper1/PRODUCT_CONTRACT_RECONCILIATION.md`
- `experiments/schema/paper1_evidence.schema.json`
- `experiments/schema/paper1_decision.schema.json`
- `experiments/schema/paper1_campaign.schema.json`
- the existing run-manifest and compact-result schemas.

## Commands

From the isolated Python environment:

```bash
spacepdhcg-paper1 validate <campaign> --output evidence-index.json
spacepdhcg-paper1 build <campaign> <output>
spacepdhcg-paper1 verify-reproducible <campaign>
spacepdhcg-paper1 verify-clean-clone <repo-relative-campaign>
spacepdhcg-paper1 freeze <campaign> <campaign-config.json> <output> --repository .
```

For direct scoped build/reproducibility commands, add
`--campaign-scope-id single-gpu-v1`; `freeze` reads the mandatory ID from its schema-`1.1.0`
configuration.

The safe demonstration path is:

```bash
spacepdhcg-paper1 synthetic-demo \
  build/paper1-synthetic/campaign \
  build/paper1-synthetic/bundle
spacepdhcg-paper1 verify-reproducible \
  build/paper1-synthetic/campaign --synthetic
```

The generated README, source JSON, build manifest, and every record retain `synthetic=true` or a
prominent synthetic-only note. The freeze command always refuses that campaign.

## Real campaign directory

Each run directory must contain:

```text
<run-id>/
  run-manifest.json
  paper1-result.json
  evidence-record.json
  <optional locally mirrored payloads>
```

`evidence-record.json` links the compact record to its manifest and to three independently hashed
objects: canonical-residual evidence, nonlinear replay evidence, and the immutable archive. A
portable URI is mandatory even if a local mirror is supplied.

The campaign configuration lists every required in-scope coordinate and its minimum record, instance, and
measured-repeat counts. It also pins hardware manifests, toolchain manifests, and solver locks by
repository-relative path and SHA-256. Historical full-campaign H1-H6 each link to one or more
manuscript claim IDs. For schema `1.1.0`, active hypotheses require claim IDs while deferred
hypotheses require an empty array. Scoped product sources and build/freeze manifests repeat the
scope ID.

## Freeze refusal conditions

The command refuses to write `freeze-seal.json` if any of these checks fail:

- dirty repository or HEAD different from the configured 40-character commit;
- missing/hash-mismatched hardware, toolchain, or solver pin;
- missing required coordinate, instance, or measured repeat;
- mismatched run IDs, repository commit, family, instance, or solver;
- local-only/mutable evidence URI, payload hash mismatch, or reused residual/replay evidence;
- absent warm/cold classification, timing component, accepted-trajectory boundary, or timing sum;
- qualified comparison without matched quality and independent replay;
- archived status not mapped by at least one frozen product;
- missing F11 variational trials for any of 3-DoF, 6-DoF, or low-thrust;
- missing F12 expected, worst-case, or CVaR robust-iteration diagnostics;
- missing active-hypothesis decision or manuscript claim link;
- a manuscript claim for a deferred hypothesis, a 2/4/8-GPU or P1-F record in
  `single-gpu-v1`, or a full campaign without physical P1-F 2/4/8-GPU records.

`freeze-seal.json` is a completeness seal only. Its statement explicitly disclaims a scientific
PASS claim.

## Exact G4 inputs still required

Before a real G6 freeze, the G4 producer must supply all locked P1-C/P1-D/P1-E coordinates,
including failures, with:

- seven measured repeats after two warm-ups and at least 20 committed evaluation instances/seeds;
- exact repository/upstream/IPM commits, builds, policy SHA-256, requested and actual policy,
  quality tier, scaling mode, warm mode, solver order, and conditioning bin;
- converged-status and canonical primal/dual/cone/gap residuals from an independent implementation;
- independent higher-order replay, full family path inventory, continuous-time error, objective
  practical equivalence, virtual control, and explicit cached-residual non-use;
- complete coefficient/update/scaling/transfers/solve/recovery/residual/replay/acceptance and
  conversion/setup/polish timing identities with CUDA startup outside the measured boundary;
- adaptive forcing/re-solve fingerprints, final polish outcome, including failed polish,
  max-iteration, numerical, infeasible, timeout, OOM, unsupported, and unrun records;
- active/reserved memory, allocation/copy counts, energy trace validity, immutable artifact URIs,
  archive hashes, and internal evidence-index hashes.
- F11 trial-level analytic-versus-independent-finite-difference absolute/relative differences,
  declared tolerances, coefficient-fill timings, and 6-DoF quaternion radial sensitivities.
- paired measured-repeat timing arrays for F10 winner confidence; absent or incomplete pairs
  deterministically produce a tie rather than a unique winner.

Current known G4 failures must remain failures; this tooling will not promote them.

## Exact G5 inputs still required for `full-multi-gpu-v1`

The G5 producer must supply the complete P1-F matrix and all negative evidence with:

- qualified one-GPU distributed parity to monolithic/CPU truth before multi-GPU coordinates;
- fixed global CQP identity, scenario ordering/partition, rank-to-GPU mapping, topology and
  deterministic-mode metadata;
- strong and weak scaling coordinates for every available declared GPU count, including the
  same-machine one-GPU baseline;
- scenario-aware and generic partitions at matched global CQP/topology, with local compute,
  collective payload/count/exposed time, overlap, load imbalance, peak active/reserved bytes,
  throughput, and total accepted-trajectory time;
- independent nonlinear replay, canonical residuals, non-anticipativity and risk-epigraph
  residuals for expected, worst-case, and CVaR semantics;
- OOM, timeout, unsupported, infeasible, numerical, and absent-hardware coordinates represented
  explicitly rather than omitted;
- pinned MPI/NCCL/CUDA/driver/hardware manifests, sanitizer/race evidence, immutable archive URIs,
  content hashes, and internal evidence indexes.
- F12 per-outer-iteration expected/worst-case/CVaR dynamics, path, terminal, virtual-control,
  non-anticipativity, risk-epigraph, and canonical-KKT residuals with acceptance and trust radius.

If G5 remains unauthorised or unexecuted, `full-multi-gpu-v1` correctly refuses. In
`single-gpu-v1`, those coordinates and F07/F12/T06 are explicitly deferred and excluded by scope;
they are not empty generated products and cannot support a manuscript claim.

## Unified-roadmap validation

The synthetic campaign rebuilt byte-reproducibly (`52` files) and exercised all product and H1-H6
decision paths. The freeze command refused the labelled synthetic campaign as required. The
available unified evidence is also insufficient for a real campaign configuration: G4 has
incomplete repeat/instance coverage and unresolved H5/H6, while G5 has only one-rank
implementation evidence and no portable physical 2/4/8-GPU archive. Paper 1 is therefore not
frozen.
