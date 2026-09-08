# Optional reflected Halpern: source and CPU evidence

The final build is **v612d**, frozen commit `6fbff324b152b75e6c27e3d91b4dc0835c65ab93`. It implements a diagnostic-only zero-Q primal-first Halpern path and an adaptive restart/weight variant, retaining the existing common original-equation accuracy gates. See [DESIGN.md](DESIGN.md) for the equations, scope and spectral precondition.

This package contains **no GPU solve results**. It records CPU conversion/oracle tests, six actual-input GPU-hidden JSON parser cases, four unsupported-input rejections, CUDA compilation, both CMake test registrations, source archives and compiler resource records. No binaries are included. The linked library reuses unchanged non-persistent object files identified in each manifest; this is not a rebuild or validation of every production backend.

The final core SHA256 is `49d0eef5702a7dee319ef3b7747ded844bec0f39313300405f9d6f30fc61730e`; replay `872ef9d6fe84e85aee2431e2d92caa1ad43bf2293ef9fc91bd9ee3ba938717f2`; tiny test `6da1ba381a566a23923da26d21cc9e72e9e6eb8159399c800350f6a79e73bed7`. [v612d/manifest.json](v612d/manifest.json) binds all compiled source and reused objects.

[resource-comparison.json](resource-comparison.json) verifies that existing listed solve, scaling, residual and recovery kernels retain their prior register/stack/shared-memory footprints. Cooperative default/common use80/96 registers with zero stack; single-block default/common use148/204 with40-byte stack. The new Halpern kernel uses94 registers and zero stack. Matching resources is not evidence of runtime parity.

The superseded a–c attempts are preserved. a completed compilation and CPU/parser checks but its configure launch could not locate CMake; b/c completed CPU/configure checks. All three exposed an unwanted158-register/40-byte-stack footprint in the old cooperative kernels and were superseded before GPU execution. The final narrow forced-inline report helper restores those resources. See [attempts.json](attempts.json).

The reviewed [tiny runner](scripts/run_halpern_tiny_v612.py) is prepared for two executions under the shared GPU lock: plain/adaptive, seven solve API calls each, at most416 requested iterations total and410 expected actual optimization iterations. It has no bootstrap, no automatic reruns and no GPU launch in this package. The parent run will preserve its own fresh logs and report separately. Tiny success would validate the bounded fixtures, not cold convergence, throughput, fleet score or SOTA.

The flat [sha256.json](sha256.json) index binds every packaged file except itself. Source archives are checked against their per-file manifest hashes; all nine final owned live files matched the frozen archive at packaging.
