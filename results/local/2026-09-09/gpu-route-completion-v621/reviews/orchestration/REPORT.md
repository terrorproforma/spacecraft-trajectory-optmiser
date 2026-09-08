# Completion integration review — CPU only

The frozen-versus-candidate orchestration comparison passed 256 heuristic and 16 two-weight DP patterns. Selected plans, final `last_failure`, builder mode/penalty order and the first tour's burn supplied to the second DP agree. Successful plans and both builder and completion failures are included. Heuristic scenarios cover the first penalty round and a mixed second round; they do not exhaust every three-round combination.

The candidate `_finish_cpu` AST is identical to the `7eb8828f61bd35ec0abaf98c7651c5e82384ae27` `_finish` after renaming. The extracted `_score_chain_plan` mathematical body is identical to the original `_chain_score` suffix. `inputs.json` preserves the exact extracted source segments, complete source-file hashes and related adapter/test hashes.

Reproduce with the standard library only:

```text
python review_orchestration.py --inputs inputs.json --output reproduced-findings.json
```

The script executes the captured orchestration methods with deterministic route-builder and completion stubs. It imports no project modules and performs no native loading, GPU work, trajectory calculation or physical certification. The comparison does not establish numerical GPU parity or exercise custom overrides and exceptional side effects. The first evidence serialization attempt reached all passing assertions but could not encode an empty `banned_pairs` set; the serializer now records sets as sorted lists. No solver/source/test behavior changed for that correction.

Independent source review found the index-scatter and failure restoration in `_plans_from_tours` consistent with original ordering. The host adapter preserves prefix legs, reconstructs cargo dictionary insertion order from pickup records, and records the inflation actually costed by each native flight. It requires finite exact Python-float epochs; malformed metadata/lookup errors are explicitly whole-call input errors. This avoids silently substituting FP64 coercion for integer subtraction or CPython's generic sum on NumPy scalars. The native Neumaier accumulation matches CPython 3.12's exact-float fast path and finite-compensation guard ([primary source](https://raw.githubusercontent.com/python/cpython/v3.12.12/Python/bltinmodule.c)). Normal model parity remains a bounded FP64 comparison because NumPy matrix products and exponential evaluation need not be bitwise identical to CUDA.

Two requested fixes are present in the reviewed adapter: `_plan_from_tour` overrides are rejected so batching cannot bypass custom tour validation, and retained workspace capacities grow monotonically across all dimensions. The scalar CPU implementation remains unchanged.

Full-core build provenance was separately checked against manifest `43c3c2974af7a9d16ba060724855a56b5122acf9c53ff29d68243df612d82969`. All 377 source members and the source-tree/archive hashes match. Comparison to the pinned commit with `git -c core.autocrlf=false archive` found exactly the declared four completion overlay files; disabling Windows line-ending conversion matters for this byte comparison. The normal CMake build enumerates the new completion test. Its recorded completion kernel uses 90 registers and 160 stack bytes; existing cooperative default/common kernels retain 80/96 registers and zero stack. This is build/resource evidence, not measured execution speed or unchanged machine code.

Source-level review is clear for the separately pinned finite full-core GPU tests. No GPU test was run by this reviewer.
