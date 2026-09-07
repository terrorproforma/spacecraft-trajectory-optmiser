# OrbitWeaver G7 implementation devlog

## 2026-09-01

- Created `feat/orbitweaver-gpu` at
  `a33e950e06b0a302815fb079dc95f356c13db5fd` in the isolated WSL worktree.
- Mapped existing Lambert, low-thrust, route-master, column-generation, dynamic-grid,
  robust-risk, certification and public G3 APIs.
- Chose fixed Lambert output slots rather than atomic compaction to preserve ordering and
  all failure classifications.
- Added CPU C ABI parity and CUDA persistent workspace paths.
- Added a solver-neutral persistent backend seam and rank/device ownership policy.
- Added bounded scheduler/backpressure telemetry, stable top-K, scenario risk,
  deterministic restart and independent certification.
- Added Python orchestration/records, CLI validation and frozen Paper 2 matrix hash.
- Configured unique native Debug/Release and CUDA Debug/Release directories.
- Built native Werror/sanitizer-capable and CUDA 12.8 `sm_120` targets.
- Passed bounded CPU, Python and actual one-GPU Lambert parity tests.
- Deliberately did not run timing, energy, full route, or physical multi-GPU experiments.

## Corrections made during the loop

- Corrected new-file destination paths to the isolated WSL worktree before compiling.
- Used the worktree-local Python environment to avoid the canonical editable-install hook.
- Replaced release-disabled test assertions where Werror exposed unused variables.
- Pinned the benchmark loader to the byte-exact worktree matrix hash.

## Unified-roadmap serialized validation

- Rebuilt the unified Release, Debug, and sanitizer-capable CUDA targets for `sm_120`.
- Passed the Release one-GPU OrbitWeaver test on the RTX 5090 after native-QOCO integration.
- Exercised the concrete persistent G3 backend callback and route-result propagation seam.
- Retained the boundary: this is one-GPU correctness only. No physical multi-GPU scaling,
  complete route campaign, energy claim, G7 acceptance, or Paper 2 claim is made.

## 2026-09-02 schema audit

- Confirmed Python manifests had drifted from their JSON schema by omitting the required
  Paper 2 matrix hash.
- Moved config, manifest, checkpoint and result schemas into one in-package authority and
  added deterministic schema generation/check mode.
- Added strict record read/write paths with atomic output, finite JSON enforcement,
  nested unknown-field rejection and manifest pin/repeat cross-checks.
- Expanded terminal status semantics for failed, censored, unsupported, OOM, timeout,
  infeasible and cancelled records while retaining partial bounds where valid.
- Added repository/config/matrix, toolchain, hardware, seed and repeat capture.
- Added round-trip, adversarial and seeded differential schema tests.
- Built and installed the wheel, ran its CLI, generated a pinned manifest and validated it
  independently with Draft 2020-12 `jsonschema`.
- Ran no GPU timing, energy, throughput or multi-GPU experiment.

## 2026-09-02 G3/G5 concrete adapters

- Created the isolated `feat/orbitweaver-g3-g5-adapter` worktree from unified commit
  `e95b902d718ceaf05523e469cbe21945013c2f41`.
- Cherry-picked only schema-parity commit
  `bf9d10af541c995f1bdcd10b031486cff6b4351e`; retained the integrated original G7 code.
- Added a bounded Python G3 adapter with persistent topology/rank/device workspaces,
  in-place numerical updates, opaque compatible warm states, separate canonical/replay/path/
  terminal diagnostics and explicit failure/censor classifications.
- Added the C++ `G3PersistentTrajectoryAdapter` over the public device-SCvx C API.
- Added deterministic G5 route/arc/scenario partitioning, rank-local ownership/backend
  adapters, checkpoint compatibility, status propagation and collective telemetry surfaces.
- Connected scenario risk results to route columns using real returned costs/lower bounds;
  only independently certified route combinations may become incumbents.
- Added deterministic fixture tests for the full coarse/refined/scenario/master/certification
  path and failure modes. Fixtures are labelled non-evidence.
- Added logical-rank ownership coverage and a CUDA compile/link contract test.
- Validation:
  - native Debug ASan/UBSan/Werror: 43/43 CPU tests passed;
  - native Release Werror: 43/43 CPU tests passed;
  - CUDA 12.8 + G5 Debug/Release compiled for `sm_120`;
  - G5 logical-rank contract passed in Debug and Release;
  - adapter/schema Python selection: 40 passed;
  - Ruff and generated-schema checks passed.
- Kept all GPU executables, energy collection and physical multi-GPU runs disabled while
  shared validation remained active.

## 2026-09-02 single-GPU scope

- Added schema-v2 G7 manifests with `campaign_scope_id`; schema-v1 historical records remain
  readable.
- Made `single-gpu-v1` require one-device ownership and reject physical-multi-GPU evidence labels.
- Defined `complete-in-scope` as the full coarse/refined/scenario/pricing-master/certification/
  visualisation flow with independently certified results.
- Kept physical route-by-scenario scaling, throughput, energy, memory crossover, and
  tractability-frontier claims in the preserved deferred backlog.


## 2026-09-07 — v137/v138 reference coordinates and Lambda failure snapshot (unmerged)

- [self] PROGRESS: validated reference-centred GPU primal coordinates x=delta+o, retained translated packed values c+Po/b-Ao/h-Go, physical reconstruction before original objective/KKT audit. Owned GPU prefix, no trajectory host copies, stable sparse gathers. Option STATE_ORIGIN=1 disabled by default. QOCO128 unchanged. v138 classifies first-invalid-origin numerical status3/zeroiterations and rebuilds even before first completed solve. Direct mixed-cone origin tests0/3Ruiz oracle/device pass, including changed prefix, nonzeroP/c, unique physical solution, pending overwrite/cold-only/recovery. Eight total nativecases+GTOCguard PASS;89 integration PASS7.43s. Sources frozen core138ninebinaries; nextcore139+,nextQOCO131+.
- [tool] v137 earlier89integration+332broadPASS.24 captured translation/auditcases plus mem/init/sync0errors. Initial raw135/136comparison confirms actual original dynamics residual7.34418854222032e-7 equalsGPUabsoluteaudit; objectiveCPUdiff4e-31.1351/48strictfail,1363/48strictfail(one normalizedreject).36centred137coast strictPASS versusoneeachuncentred136/137 failure. However v138 broad FAIL331passed1failed360.39s coast[False-False-zoh]originalequality1.3945688767003704e-9>1e-9. Option-disabledintegration FAIL88passed1failed8.13s samegate. CENTRING HAS NOT ELIMINATED FAILURE. No rerun-for-green/no tolerance relaxation. All failures retained. Do NOT merge candidate into main.
- [tool] v137threeway36fulltransfersALLunchangedphysicscertificate/mass2445.3111007852112+-1e-5kg;medians136273.928ms/original137327.900ms/origin137298.701ms,376qualificationpredicatematches280deferred. v138sixbalancedpairs24complete transfersALLqualified (12perruntime, warmupsincluded),mediansorigin137343.338ms/origin138291.283ms,320matchingqualificationreports272deferred. Widevariability/differentiterations, no reliable additional speedup claim. Allcommands8*(attempts+1), reportsseparate. No newfleet/score.
- [self] Checkpoint artifacts/performance/native-origin-v138-checkpoint.json3565464bytesSHA d6dd517570b76217e72bee95fe71d9fd2a222d1d65597b3ece3344dc94713374 embeds11sources,137source-snapshot,138runtimeidentities,helpers,allpositive/negativeevidence. release_qualified=false. README/docsGTOC12_STATE_ORIGIN.md updated; existingdeferred136failurecheckpoint preserved. Ruffinitial2longlinesfixedformat/checkPASS; no numericalsourcechangeafterfreeze.
- [tool] User asked Lambda status after long pause. 2026-09-07T06:10UTC livePID53138elapsed59h15m, H100100%,1603MiB,124W. Last completed groupordinal209at06:09:23; group210live. Read-onlySQLite+all210resultfiles:210/396completed53%,1running;1890rawattempts comprised1296timeouts and594numerical failures, ZERO successes. Last90alltimeouts. Policy pure-gpu-ipm594numerical/adaptive1188timeouts/hybrid108timeouts. Completed_group means record validation, NOT trajectory success. Old source1dbcae098fd8871d4e0ac087e2306534704eff59, not138/notfleetsearch. Recommended stopping legacy run to user but DID NOT interrupt (existing preservecampaignconstraint); no authorization tostopreceived. Nooffload/newinstance.
- [tool] Downloaded all210completed result.json/stdout/stderr/coordinate/execution/command/contamination files plusjournal/hardware and derivedsnapshot-summary under results/lambda/2026-09-07/g4-progress-v138. No runninggroupfiles; readonlyDB usedto selectcompleted. tar2425694bytesSHA a72cb735ae2584dcdf156805f945f2a7b98da5d8caa2cf0dae2ed358810f163d. Summarytimestamp06:14:56UTC. Helperpull_lambda_results_v138.py andsummaryembeddedcheckpoint. Rawarchive+summaryreadyforGit; extractedfileslocal. /tmpSSHkeyvanishedafterWSLresetbetween somecalls: restorefromrepokeyusingO_EXCL0600beforeeachstandalonehelper, neverprint. Failedauthstatusartifactretained.
- [self] ViewerappaddsdatedLambdacount/outcomerows; JSONshows138UNMERGED331passed1failed/24qualified/291.3ms/checksum andLambda210/396,1296timeout594numerical0success. npmcheck+38testsPASS. Publisherfirstinvokedwrongcwd failedwithoutwrite; correctedrootexecutionPASS. Serverhadstoppedafterreset. Start-Process hidden launch blockedbypolicy; safermanagedforegroundnodeexec succeeded session68906; HTTPbenchmarkJSON200. CUA bindingreset thenselectedbrowser1tab2; initialreloadconnectionrefused stored data:errorpage. Furtherreload/gotonowblockedbyBrowserURLpolicyoncurrentdataURL; NOTvisuallyverifiednewpanel, no bypass. Usercanmanuallyreload. Browserexistingprovider cd30cc5b-35b2-48fc-9ffd-89c3d4e86cf0. Noagents/SKILL used. AllGPUhandles terminal;server68906mayremainlive.
- [self] NEXT: save/pushcandidateonly, main0e31ecceunchanged. Investigate actualstopping/restoration mismatch (nativeconfiguredtol absolute+relative; QOCOinaccurate1e-5; originalstrictabs1e-9). Do not claim translationfix or loosenphysicsgates. Then shared QOCO graph-body emission into outer GPU WHILE and per-attempt device reports/recovery. Generated QOCO128 qoco_ipm_graph.cuh emitsguard/init/IPMwhile/stepIF/IRwhile; refactorbodysharedwithparentemitter, notcloneconditional. IMPORTANT everyIPM invocationwithinouterloopmustresetloopconditionalto1 anditeration/counts insidebody (defaultconditional resetonlyrootlaunch). Resourceenterneedsqueuedreductionscope. CurrentAPIrejectexternalcapture; nativefinishperattempt/setup/retries/fleetsearchremainCPU. FullconditionalIPMsanitizerissueunresolved. Goal ACTIVE/incomplete.
