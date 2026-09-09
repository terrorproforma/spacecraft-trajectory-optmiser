# Nonzero-Q HCW coefficient capture - v637

The existing displaced HCW SOC N20 recipe was assembled once on the CPU and exported with its original coefficients, bounds, cones, reference arrays and settings. The snapshot has 186 variables, 132 equalities and 20 SOC4 blocks. All 186 diagonal P values are positive: 63 at 0.0002, 63 at 0.02 and 60 at 2.

This is coefficient capture, not a native trajectory solve or certification. There were zero optimizer and GPU calls. Final solver vectors are explicitly null. The declared cold zero candidate is separate from the physical reference rollout and is not claimed to be backend-generated initialization.

The native and QOCO views preserve Q/A/c bits. Native affine [vector, scalar] rows are permuted to [scalar, vector], with G = -Pi F and h = Pi offset. The canonical objective constant remains zero. The separate physical tracking reference constant, 9.259516874999995e-6, is metadata and must not enter the original solver gap or its normalization. All original numerical settings and subsequent qualification scopes are recorded in PROTOCOL.md.

The approved action used one compiler process (1.625722 s wall) and one exporter process (0.023756 s wall), with one host assembly and 20 HCW reference-step evaluations. These are capture timings, not solver or GPU throughput. The source, four headers, compiler/executable identities, raw logs and empty cleanup results are retained. A prior prose-writing encoding failure is preserved; no numerical attempt was repeated.

`evidence.zip` contains the entire 25-file sealed kit plus its original index, summary and this packaging source. The exporter executable and persistent/QOCO/cuDSS libraries are excluded; their hashes remain in the original manifest/report. Historical paths are provenance only and are not required for verification. External reference source/runtime dependencies remain hash-linked, not copied or executed here.

Run `python -B verify.py` from this directory, or `python -B verify.py /path/to/package --index-sha256 EXPECTED_HASH`. The standalone stdlib verifier checks every ZIP byte, the four headers, preparation/execution/log bindings, native/QOCO serialization bits, unchanged settings and the unsolved status. It performs no compilation, assembly, dynamics, factorization, solver or GPU work. `verification.json` is the saved portable audit result; trust the independently supplied index hash.

The diagonal Q fixture supports a matched future comparison; it does not by itself establish a quadratic-CG advantage or any integrated speedup. Backend execution requires a separate reviewed protocol.
