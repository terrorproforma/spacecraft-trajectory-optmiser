# GPU mass-scaled ZOH initializer, v631

The new optional initializer scales an archived trajectory's initial mass and physical thrust by the same positive factor on CUDA, then uses the existing forward ZOH rollout. It preserves position/velocity dynamics mathematically and keeps the original optimizer and independent certification gates. It does not certify the seed, reduce cargo, change epochs or guarantee sufficient final mass.

The completed validation is **92 CPU tests passed, 27 GPU-dependent CPU-test skips, and four separately supervised native processes passed** on the local RTX 5090: original ZOH regression, scaled ZOH regression, SCvx controller regression, and scaled ZOH under compute-sanitizer memcheck. Memcheck reported zero errors. The scaled test runs once normally and once under memcheck; each invocation reports one reference inspection, three finite scaled inspections and eight invalid-input cases. These tests invoked **zero optimizer calls and produced no mission score, fuel gain or throughput claim**. The pending ship10 experiment and proposed ship22 experiment are excluded.

The scaled native test checks unit-scale bit identity, down/up scaling, nonuniform epochs, physical position/velocity tolerances, normalized mass, analytic fuel, Gamma, unchanged host inputs, and invalid factors/norm arithmetic. A finite positive factor that underflows a nonzero squared thrust norm is explicitly rejected. Scaling up is not clipped; later original physics gates retain responsibility for thrust feasibility. Native numerical assertions ran inside the preserved test source; the logs do not contain independent full trajectory readbacks.

The frozen build uses committed base `6c686cce48763a5ab34687a74d810f5ce693589b` plus exactly six reviewed source overlays. This base commit is not claimed to be the exact tested source commit. The exact frozen tree is `164f91c6b0b26228c654ee3120119f62667601b5eaeb1cada255ae1bf7d1bc65`, source archive `ff90bc2b1b9a66c8d37abc933990a55d9327bd61ff15eb610940d41b32372ab0`, and build manifest `b92df5cc06f0119e0958b85935334b82910967f580f0d02bab2bde8d39702e11`. CUDA 12.8 compiled fresh SM120 objects; build CPU checks hid the GPU. The measured library hash is `8047ad5e194ce29fcf4a4767f7838cfde2f32ef714fec99f23aa66f7cd882e48`.

`evidence.zip` preserves repository-relative paths for the frozen source archive, manifest, build/CPU logs, compile commands, all four native logs, exact supervisor and process guard, and both independent v631 reviews. The native-seed review includes its exact historical v630 saved inputs, original source archive, recovered source members and failed preliminary audit assumption. Those earlier trajectory failures are historical diagnosis, not results of this initializer's unit tests. The five inspected historical source files and six new reviewed files are bound separately.

Compiled libraries/executables, credentials, caches and Git data are excluded. Their necessary runtime identities remain declared in the build manifest and index; verification does not claim to check absent binary bytes. Historical launcher/build scripts retain their original machine paths as provenance. They are not executed by the portable verifier and are not advertised as portable launch tools.

From any directory, using Python 3.10 or later:

```
python /path/to/package/audit_package.py --package /path/to/package --expected-index INDEX_SHA256 --roundtrip --output NEW_REPORT.json
```

The verifier uses only the standard library. It checks exact ZIP names/member hashes, safe nested archives, both complete source trees against their manifests, six reviewed source hashes, sealed review indexes and historical input pins, all build/native log bindings, the original process guard, zero-optimizer scope and cleanup statuses. `--roundtrip` writes a fresh temporary tree and rereads every member hash. Verification requires no original absolute workspace paths, project imports, solver, propagation, native runtime or GPU. The output file must not already exist.

The historical seed-analysis scripts are retained without edits. Their broader optional numerical recheck expects the original repository layout and NumPy, including then-current source inspections; the portable static verifier instead binds the saved analysis and all of its recorded inputs directly. No numerical re-audit is needed for the package check.
