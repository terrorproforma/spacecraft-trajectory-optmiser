Normal full-core completion build
=================================

This is a fresh normal project CMake/Ninja build of the full CUDA library, for
local SM120 validation. It uses committed base
`7eb8828f61bd35ec0abaf98c7651c5e82384ae27` plus only the four reviewed completion
files listed in `manifest.json`. All other frozen C++/CMake/third-party bytes
were checked against a read-only Git archive of that commit. No uncommitted
fleet or other foreign source was included; no compiled cache objects were reused.

The 377-file source fingerprint covers the build inputs in `source.tar.gz`:
`e12c5a94b940eb003d6b513b3418ba681fbf294996f78315b706dcd7655384ff`.
The assembled source is explicitly uncommitted; the base commit is not presented
as the exact tested source revision. Host Python runtime files are excluded and
must be frozen and hashed separately for production adapter tests. Pinned
upstream revision, cleanliness and patch checks remained enabled in CMake.

All eight recorded preparation stages passed. Configure took 16.5 seconds and
fresh compilation 69.2 seconds. Executed checks were GPU-hidden completion
fixture/ABI construction and CPU-only snapshot conversion/analytic KKT tests.
CTest enumeration confirms `gtoc12_completion_test` is registered. The full
library exports the new completion functions and existing Lambert workspace
functions. No GPU test, trajectory solve, throughput benchmark or fleet promotion
occurred in this build procedure.

Frozen root: `/home/angus/spacepdhcg-completion-fullcore-v621`.

- Full core: `build/cuda/libspacepdhcg_cuda.so`, 21,084,184 bytes,
  SHA256 `cb977ff09b206de807996a8a21d7ccec41bb6a2a4c35a37c10ebe4883b83d2fb`.
- Linked native test: `build/cuda-tests/gtoc12_completion_test`, 53,200 bytes,
  SHA256 `28cf9de6e7f3d8c07b2aacd0b12ebdd66437921514d5d96fb9689551694b5fb8`.
- Manifest SHA256:
  `43c3c2974af7a9d16ba060724855a56b5122acf9c53ff29d68243df612d82969`.

The completion kernel uses 90 registers and 160 stack bytes in this normal
relocatable CUDA build, versus 84/160 in the standalone SM120 build. Existing
persistent kernel resource records retain the prior footprints: cooperative
legacy/common 80/96 registers with zero stack; single-block legacy/common
148/204 with 40 stack bytes; Halpern 96/0; unit/weighted L1 94/0. Resource equality
is not a runtime regression test or evidence of throughput improvement.

The exact commands, source/compiled-object hashes, compiler identification,
compile database, exported symbols and resource records are retained here.
There were no failed attempts in this full-core preparation. The separate
standalone preparation remains intact at `../completion-native-core-v621`.
