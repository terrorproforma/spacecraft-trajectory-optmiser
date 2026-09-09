# Catalogue cache v822 — measured searches and verified H100 replay

The downloaded [H100 Result.txt](h100-best/Result.txt) retains the frontier score:
**13,023.704900978253 fixed-bonus weighted kg / 14,291.0061601653 raw kg**,
23 ships and 199 asteroids. Both original full-fleet physics checkers pass on
both machines at unchanged tolerances. Tiny score differences from the input
fleet are roundoff, not improvements. H100 Result SHA-256:
`6d43d6f495d5766631d2a05ac131c236e8ebdb995b43dc01db01dba5fe837a41`.

[Implementation, paired measurements and limits](../../../../docs/GPU_CATALOGUE_OWNERSHIP.md)
explain the new immutable catalogue snapshots and bounded policy-prefix cache.
Across the two benchmark routes, catalogue hashes fall from 1,302 to four while
all candidate files match exactly within each GPU. Final H100 median search
throughput improves 7.33% and 7.48%; local changes are 0.91% and 3.74%.

The final v819 source passes 161 tests and native controller checks on each GPU.
Thirty-two tests plus the controller pass separately under memcheck, racecheck
and synccheck. The final cache policy is a host-only refinement of the preliminary
v808 source; all C++ source and native core bytes are unchanged. The 14-route
v815 replay uses that integrated core, with 147 native calls on each GPU and 138
converged calls. The remaining local statuses are six infeasible/three failed;
H100 reports three infeasible/six failed. Five complete routes certify on each
GPU; optimizer failure is not proof of physical impossibility.

The v817 CUDA selector uses the already published frontier as its warm incumbent
and exhausts the 49-column pool without a meaningful new score. That fleet then
passes both checkers on both hosts. The original frontier's ships 8 and 19 retain
their local GPU-refinement provenance; they were not reoptimized on H100 here.
The fleet exporter produces fresh dense verifier histories, including coasts.

`local.tar.gz` and `h100.tar.gz` each contain 1,194 payload files plus `FILES.json`:
the final frozen runtime, native core, controller executable, retained QOCO,
source manifests, both cache experiments, all native route attempts, historical
certified pool inputs and the final full-fleet checks. Original runner failures
are included: a wrong controller executable directory, an omitted `.gitignore`
fixture, and dependent jobs that stopped before starting any native route solve.
The final checks and measurements are v819 and v820. `*-receipt.json` binds each
archive's length and SHA-256; `saved-audit.json` independently recomputes timing
medians, work counts, emitted cargo and fixed-bonus score from those saved bytes.

Run the standard-library saved-byte audit from the repository root:

```powershell
python results/lambda/2026-09-09/gpu-catalogue-cache-v822/audit_saved.py
```

It does not rerun CUDA or physics. The reproduction scripts retain original
machine-specific paths and dependency bindings; they are execution provenance,
not a promise of a one-command installation on a fresh machine.

[Open the downloaded dense H100 replay](http://127.0.0.1:4173/?dataset=gtoc12-catalogue-v822&epoch=69807&preset=oblique&z=1).

```powershell
$viewer = 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
if (-not (Get-NetTCPConnection -LocalPort 4173 -State Listen -ErrorAction SilentlyContinue)) {
    Start-Process node -ArgumentList @('scripts/serve.mjs','--port=4173') -WorkingDirectory $viewer -WindowStyle Hidden
}
Start-Process 'http://127.0.0.1:4173/?dataset=gtoc12-catalogue-v822&epoch=69807&preset=oblique&z=1'
```

Full Result path:
`C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-09\gpu-catalogue-cache-v822\h100-best\Result.txt`.
