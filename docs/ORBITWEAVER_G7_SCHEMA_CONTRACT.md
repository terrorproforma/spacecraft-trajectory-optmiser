# OrbitWeaver G7 manifest and result contract

## Authority

`src/spacepdhcg/orbitweaver/contracts.py` is the authoritative contract. Run:

```bash
PYTHONPATH=src python scripts/generate_orbitweaver_g7_schemas.py
PYTHONPATH=src python scripts/generate_orbitweaver_g7_schemas.py --check
```

The generated files under `experiments/schema/` must not be edited independently.
Tests compare all four materialised schemas byte-semantically with their Python sources and
compare the dependency-free validator with Draft 2020-12 `jsonschema`.

## Pins and repeat identity

Every manifest records:

- exact 40-character repository commit;
- SHA-256 of the exact config bytes;
- frozen Paper 2 matrix SHA-256;
- deterministic seed and declared repeat count;
- backend, ownership and device IDs;
- available Python/compiler/CMake/CUDA versions;
- OS, CPU and GPU name/UUID/compute-capability/driver strings;
- evidence level.

Every checkpoint and result records the run ID, manifest SHA-256, matrix SHA-256, seed and
zero-based repeat index. Validation against a manifest rejects any mismatch or repeat index
outside the declared count.

## Result semantics

Allowed statuses are `converged`, `iteration_limit`, `infeasible`, `cancelled`, `failed`,
`censored`, `unsupported`, `oom`, and `timeout`.

- `converged` requires finite non-negative incumbent, lower bound and gap.
- `infeasible` and `unsupported` must not contain incumbent/bound/gap values.
- Every terminal failure/censoring status requires a matching retained failure record.
- Certified results require a converged or iteration-limited incumbent and an accepted
  independent certification with finite non-negative dynamics, path, terminal,
  uncertainty and integration checks.
- Bounds must satisfy `lower_bound <= incumbent`.
- The gap must equal `max(0, incumbent - lower_bound)`.
- Telemetry must satisfy `completed = feasible + failed + cancelled <= submitted`.
- Non-standard JSON `NaN`/`Infinity`, unknown fields and unknown nested fields are rejected.

OOM, timeout and censored records may retain finite incumbent/bound/gap values discovered
before termination. They remain unsuccessful, uncertified records and are not performance
or scaling evidence.

## Integration instructions

1. Import `RunManifest`, `Checkpoint` and `ResultRecord` from
   `spacepdhcg.orbitweaver`.
2. Create manifests with `RunManifest.capture(...)` or
   `spacepdhcg-orbitweaver-g7 create-manifest`.
3. Use the returned `manifest.sha256()` in every checkpoint and result.
4. Construct repeat records from the manifest seed and `repeat_index < repeat_count`.
5. Call `write(path, manifest)` so semantic and schema checks run before atomic output.
6. Validate archived records with `validate-manifest`, `validate-checkpoint --manifest`,
   and `validate-result --manifest`.
7. Regenerate schemas after intentional contract changes and require `--check` in CI.
8. Merge G4/G5 adapters without changing these record semantics; add fields only by
   versioning the schema and migration reader.

No field in these records constitutes throughput, scaling, energy, optimality or physical
multi-GPU evidence by itself.
