# Paper 1 product-contract reconciliation

Reconciliation version: **1.0.0**
Status date: **2026-09-02**

This record resolves a scope mismatch without deleting or weakening any frozen requirement.

## Authority and determination

The reviewed Paper 1 sources have these roles:

1. `NOTATION.md` is normative for identifiers, symbols, units, residuals, and performance metrics.
2. `FIGURE_SCHEMA.md` explicitly freezes inclusion, aggregation, axes, failure handling, filenames,
   and rendering for F01-F12 and columns for T01-T08.
3. `CLAIMS_AND_DECISION_RULES.md` is normative for H1-H6 thresholds, winner selection,
   confidence intervals, sustained crossovers, censoring, and permissible claim language.
4. `OUTLINE.md` describes ten planned main-paper figures and appendices, but does not supersede
   the broader frozen figure/table schema.
5. `docs/BENCHMARK_PROTOCOL.md` and `docs/GPU_EXECUTION_ROADMAP.md` govern evidence quality,
   reproducibility, retention, and final-freeze completeness.
6. Versioned JSON schemas constrain the machine-readable inputs and generated products.

The earlier G6 request and registry named only F01-F08/T01-T06. That was a narrower implementation
scope, not an amendment to `FIGURE_SCHEMA.md`. The authoritative final inventory is therefore:

- mandatory primary/main-paper products: F01-F10 and T01-T08;
- contract-required generator with placement decided at manuscript assembly: F11, which the frozen
  schema permits in the main paper or appendix;
- contract-required diagnostic product: F12, which cannot substitute for robust aggregate scaling.

The final inventory is F01-F12 and T01-T08. All 12 figures and 8 tables are consequently generated,
indexed, schema validated, checksummed, and included in freeze completeness. F11 placement
optionality and F12 diagnostic status do not make their evidence contracts optional.

## Frozen generator mapping

| Product | Source contract | Selection/claim constraint |
|---|---|---|
| F01-F08 | `FIGURE_SCHEMA.md` sections 2-9 | Existing architecture, persistence, scaling, and timing rules |
| F09 | section 10 | Separate canonical/nonlinear panels; qualified nondominated frontier only |
| F10 | section 11 and winner rule | At least 10% advantage plus paired 95% interval; ties remain ties |
| F11 | section 12 | Independent finite-difference trials for 3-DoF, 6-DoF, and low-thrust |
| F12 | section 13 | Expected/worst-case/CVaR iteration anatomy; diagnostic only |
| T01-T06 | table sections | Existing manifest, dimensions, correctness, persistence, policy, scaling rules |
| T07 | table section T07 and H2/H3 | Rule-derived compute/memory crossover summary |
| T08 | table section T08 and H1-H6 | Every rejected, mixed, or unresolved result with portable support |

Every numeric coordinate originates in a validated archived result or its hashed manifest evidence.
Generator modules contain no experiment-result coordinates. Every source JSON sets
`manual_coordinates=false`, records its filter, units, aggregation, and contributing run IDs.

## Future contract changes

Changing this inventory requires:

1. a new reconciliation version;
2. an explicit edit to the authoritative source that changed;
3. matching registry/schema/test updates;
4. preservation of previously generated products and an explanation of compatibility.

Absence of real G4/G5 data does not remove a product. It makes the real campaign incomplete or the
corresponding decision unresolved, according to the frozen rules.
