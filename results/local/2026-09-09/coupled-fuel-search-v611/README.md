# Coupled itinerary fuel search v611 — no verified gain

The single local RTX 5090 run completed normally in **34.844880 seconds**. It evaluated **39,852 complete itinerary proxy candidates** across 12 device searches and retained 202 accepted timing moves. Actual screening work was 793,152 Lambert direction requests. The frozen weighted score remains **12,810.135953048577 kg**, with **14,051.854893908598 raw kg** across 23 ships. The retained baseline passed both full-fleet checkers during this run.

Four independent eight-miner routes (ships20,7,10,11) were searched with the same mesh/caps at margin prices0.05,0.25,1.0. All four price0.05 outcomes increased predicted weighted cargo. The frozen one-candidate-per-price policy selected only the largest gain: ship20, +6.033277350309 predicted weighted kg (+6.351813826147 raw kg). The other price settings increased estimated spare fuel at the expense of weighted cargo and failed the improvement gate.

The selected route used **17 native leg solves**. Its first16 legs obtained independent certificates; its Earth-return leg34525→Earth, MJD69298→69698, remained uncertified after25 SCvx iterations/four accepted steps. Dynamics defect was0.0136137792152. Its fixed598.220396988364kg cargo left292.503658693290kg return propellant, which the uncertified optimizer exhausted. Cargo and epochs stayed unchanged. No candidate full-fleet Result or promotion was emitted. This is a failed refinement, not proof of physical infeasibility.

The three unrefined price0.05 gains remain in the raw searches: ship7 +0.372079193363, ship10 +0.245577079815, ship11 +0.123515282719 weighted kg. They were outside the frozen one-per-price shortlist, not proven infeasible. The unused refinement allowance was not spent, and no retry or extra GPU job was launched.

Measured scopes: search wrappers0.450427451s (about88,476 proxy evaluations/s, including first-call overhead); native leg wrappers12.091912440s; fresh baseline independent+official checks20.329613125s. These are distinct stages. Proxy evaluations/s is not certified solutions/s, and this experiment is not an A/B speedup claim.

The one-shot supervisor ran PID290/child378 with a600s wall budget,630s outer timeout and10s termination grace. It exited0 before timeout. `execution/launch.json` preserves process, device, environment and deadline evidence; the CPU validation includes a real harmless child timeout test. The single-launch marker and all failures remain intact in the original kit.

## Reproducibility and evidence

- `output/` preserves all50 raw files, including17 leg NPZs and all12 search outcomes.
- `kit/ready-manifest.json` retains the exact247-file preparation index, SHA256 `1d9428be654cf30f49367d5ba98f4a2674d810fc4e2ee728906b4518a1cf8986`.
- `source.tar.gz` contains all190 frozen source files from `f8b2ac7aeba9e29ea90f5814060bb8cfe5db22d7`; `inputs.tar.gz` contains all33 exact inputs, including the incumbent and archived equivalent controls. Extract them into `kit/source/` and `kit/inputs/` respectively to reconstruct the prepared kit.
- `kit/validation/` retains31 passing CPU tests and Ruff check/format logs. No CUDA was loaded during that validation or the post-run audit.
- `kit/profile.json` pins the actual local v702 core (`65f1335e…`) and v683 QOCO (`5b1b1a04…`). Binaries are omitted. The original v707 implementation/runtime evidence is under `results/lambda/2026-09-09/gpu-device-search-v707`; this v611 run was local. Prior local active-loop memcheck CUDA999 remains unresolved and is not represented as a pass.
- `audit/report.json` independently checks indexed bytes, weighted totals, raw fleet eligibility, shortlist exclusions, actual work, all leg mass/event accounting, retained certificates and the absence of false promotion. It audits recorded physical certificates; it does not repeat numerical propagation.
- `archive-audit.json` verifies all archive members and reconstructs every prepared hash. `sha256.json` indexes every publication file except itself. `.gitattributes` preserves raw evidence bytes.

This fixed-order search does not establish that nine- or ten-miner route families are unavailable. The broader frozen search defaults permit ten deployments; several retained fleet ships already have nine miners. Further route-family work needs separate source/pool diagnosis. No new experiment is authorized by this evidence bundle.
