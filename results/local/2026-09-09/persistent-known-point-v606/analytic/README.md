# Qualified initial-point diagnostic

This extends the captured-conic-problem diagnostic adapter with an explicitly imported primal/dual point. It does not integrate the persistent backend into GTOC12 or change the production solver, defaults, or tolerances. The loaded solver core is the immutable v603 library, SHA-256 `d4b0bea9672bb0cea612b7d4e8db5d448f1b79b9831be5710723276edd128633`.

The point file has seven whitespace-delimited records. Counts must exactly match the captured dimensions; all numbers are finite FP64 values. Its declared SHA-256 must match the exact captured snapshot bytes.

```text
SPACEPDHCG_QOCO_INITIAL_POINT_V1
snapshot_sha256 <64 lowercase hexadecimal digits>
coordinates original|translated
x <n> <n values>
y <p> <p values>
z <m> <m values>
s <m> <m values>
```

Only `x` changes coordinates. `y`, `z`, and `s` use the original QOCO equality and conic row order. Select it with `--initial-point PATH`; the default unseeded path is retained. `--validate-only` validates the input and emits CPU diagnostics without initializing CUDA. This bundle's `build_known_point_v606c.py` compiles the adapter, tests the conversion, and parses every JSON record from ten GPU-hidden validation cases, including the two real GTOC12 inputs with both representations. See `json-validation.json` and the raw `*-validate-no-gpu.log` files.

Before CUDA, the supplied original primal/dual/slack must qualify at normalized primal, dual, gap, and maximum-block-complementarity tolerances of `1e-9`, and absolute primal/dual cone tolerance `1e-8`. Any FP64 original-to-local-to-original coordinate roundtrip must retain that qualification using the supplied certificate. Equality and retained scalar duals are copied exactly; SOC duals are sign-reversed and permuted into native order. Source multipliers of exactly folded singleton rows remain audit-only reference values. The strict reconstruction of those omitted multipliers is reported separately; it is not an import condition because a qualified approximate point may lie slightly inside a bound. Post-solve original-coordinate reconstruction and qualification are unchanged, including exact local contact for reconstructed folded bound multipliers.

For each freshly imported point, the diagnostic first performs a bounded unseeded bootstrap of at most one iteration to establish the native report epoch. It then fully resets the workspace, imports the explicit `PRIMAL_DUAL` start, measures residuals without advancing the point, and verifies that the actual internal primal and dual buffers are bitwise unchanged. The pre-step record labels `seeded_iterations=0`, termination `UNSPECIFIED`, and inherited bootstrap iteration counters. Bootstrap, reset, seed, residual-only measurement, and actual requested solve timings are distinct. The native absolute natural residual and the original-equation normalized quality gate are different predicates.

The source archive and `manifest.json` pin the diagnostic sources, build commands, binary identity, and unchanged loaded core. The source archive includes the broader frozen build tree, but the implementation changes are limited to the snapshot diagnostic header, executable, and CPU conversion test.

`initial-v606b/` preserves the first partial tiny attempt and its original frozen source/build. Both completed final vectors qualify in the CPU sidecar `initial-attempt-vector-audit.json`; the ordinary runner stopped after its first seeded solve on malformed initial-point string metadata. The defect was C++ argument-dependent lookup selecting `std::quoted(string&)` instead of the local JSON-writing helper. v606c renames the helper to `json_string`. The original logs remain unmodified. The sidecar explicitly checks the intact snapshot identity fields and re-audits final vectors without repairing the malformed metadata. CPU checks now parse emitted JSON rather than relying on exit code alone.

`busy-v606c/` preserves the zero-call retry that could not acquire the shared GPU lock. `tiny-wait/` is the completed, fairly queued retry on the corrected binary: all six final outputs qualify in both native and independent original-equation audits and report native `OPTIMAL` termination.

| Case | Actual solve iterations | Bootstrap iterations | Original-equation result |
|---|---:|---:|---|
| Unseeded mixed control | 75 | 0 | Qualified |
| Original generic | 1 | 1 | Qualified |
| Translated folded | 1 | 1 | Qualified |
| Shifted original generic | 1 | 1 | Qualified |
| Shifted translated folded | 1 | 1 | Qualified |
| Weak interior bound, folded | 1 | 1 | Qualified |

Every seeded pre-step measurement reported `UNSPECIFIED` termination, zero seeded iterations, and bitwise unchanged actual internal primal/dual buffers. The weak interior-bound case was deliberately unqualified under strict folded-dual reconstruction before the solve; its supplied original certificate passed, and one actual solver step restored exact bound contact and qualified. This behavior preserves the input while leaving the post-solve gate unchanged.

Across both attempts, the work was eight completed executable invocations, fourteen native solve API calls, 156 iterations belonging to requested solves, and six separately recorded bootstrap iterations. The busy attempt launched no GPU work. `work-counts.json` records the counts without including queue wait or CPU audits as solver work. `sha256.json` indexes the complete evidence bundle; no compiled binaries are included.

The parent publication README links the complete real-capture comparison. These analytic diagnostics establish input mapping and state handling; they do not establish convergence superiority, whole-route physics validity, or a fleet score improvement.
