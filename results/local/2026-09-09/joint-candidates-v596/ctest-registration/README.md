# Native joint CTest registration

The test-registration-only CMake change was validated on 9 September 2026
(Australia/Sydney) in a fresh isolated copy of the frozen v596 source. The
existing v596 source and library were left unchanged.

- Release configuration: `BUILD_TESTING=ON`, `SPACEPDHCG_BUILD_CUDA=ON`, CUDA 12.8,
  architecture 120, and the pinned upstream PDHCG source.
- Only `gtoc12_joint_smoke` and `gtoc12_joint_selection_test`, plus their required
  library dependencies, were built. Both were run through CTest under
  `/home/angus/.spacepdhcg-gpu.lock`.
- CTest: **2 passed, 0 failed, 0 skipped**, 0.49 seconds.
- Both test compile commands contain `-UNDEBUG` after Release's `-DNDEBUG`.
  Both executables also retain an undefined `__assert_fail` symbol, confirming
  that their assertion checks were compiled in.
- The lock was released at `2026-09-08T14:51:03Z`.

The selector test already matched CMake's `*_test.cu` discovery pattern; the
smoke test did not. Both names are now explicitly appended and deduplicated.
The assertion flag applies only to these two targets.

The isolated build is at
`/home/angus/build-spacepdhcg-joint-ctest-registration-20260909/build`.
The recorded v596 library remains at
`/home/angus/build-spacepdhcg-joint-v596/build/cuda/libspacepdhcg_cuda.so` with
SHA256 `86d7fdc952e82f7e5557c1651a1722900d83cdd98d8a709a4b6274fd83af4671`.

The updated `cpp/cuda/CMakeLists.txt` SHA256 is
`1fca2deaa9ea496773f82e017d9849da47778717ca6daa7ce63a02c13e365b5a`.
The final `evidence/report.json` SHA256 is
`1d8e800d183edbaf7505dd32c432bffb082b8d315e98d69d567d6b5c48af1d3d`.

See `evidence/report.json` for the exact configure/build/CTest commands, source
and executable hashes, Release compile commands, and preservation checks;
`evidence/ctest.xml` and `evidence/ctest-two-targets.log` for test results; and
`evidence/cmake-test-registration.patch` for the CMake-only difference from
frozen v596. `runner.py` records the complete isolated validation procedure.
