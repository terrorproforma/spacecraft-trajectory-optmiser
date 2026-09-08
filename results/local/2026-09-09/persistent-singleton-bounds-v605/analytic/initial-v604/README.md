# Initial exact-singleton folding attempt

This preserves the first frozen optional bound-folding adapter, source commit `ffc2d88d7ad2fbba9148512dc9e73736243e957b`, linked to the unchanged v603 production core. CPU tests and compilation passed. The first three GPU cases (default mixed, folded mixed, folded shifted) passed the native and independent original-equation KKT gates.

The next pure-bound case failed at workspace creation with `SPACEPDHCG_CUDA_POINTER_CONTRACT`. The importer supplied `n+1` zero affine CSC offsets even though `affine_rows==0`; the production C ABI requires a zero-length/null affine-offset view in that case. No solver iteration ran for the failing fixture. `tiny-initial/folded-duplicates.log` preserves the error, and the incomplete batch report is intentionally retained.

The corrected source/build and seven successful checks are in `..`. This initial build remains immutable. `conditioning-validate-no-gpu.log` and `difficult-validate-no-gpu.log` confirm CPU-only conversion of the two real captures with GPUs hidden: 5,273 and 5,848 singleton rows are exactly foldable, with no nonrepresentable-ratio fallbacks.
