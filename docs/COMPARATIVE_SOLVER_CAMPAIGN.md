# Comparative solver test campaign

Status date: **2026-09-01**.

This campaign determines whether SpacePDHCG is useful trajectory-optimisation software relative to
available alternatives. It supplements `docs/BENCHMARK_PROTOCOL.md`,
`benchmarks/paper1_matrix.json`, and `papers/paper1/CLAIMS_AND_DECISION_RULES.md`.

# Decision on published results and GTOPX

Published literature can supply:

- benchmark definitions and physical constants;
- reference trajectories and control histories when released;
- analytic optima and best-known objective values;
- published convergence, iteration, and function-evaluation results for context.

Published wall-clock times are not admissible as direct speed comparisons unless the implementation,
hardware, precision, stopping criteria, timing boundary, and achieved trajectory quality match the
SpacePDHCG run. In practice, the principal speed competitors must be run on the same hardware.

GTOPX is a valuable benchmark suite for derivative-free global interplanetary mission design, but it
does not by itself test the main SpacePDHCG contribution. GTOPX contains low-dimensional black-box
problems with 6 to 26 decision variables, while SpacePDHCG targets large, sparse, repeated conic
subproblems inside local successive convexification. GTOPX is therefore a secondary mission-level
track. It becomes a direct comparison only if SpacePDHCG is embedded in a complete outer global
search method that exposes the same GTOPX decision vector and objective.

# Primary evaluation question

At matched independently verified nonlinear trajectory quality, does SpacePDHCG provide at least one
material advantage over reproducible alternatives:

- lower end-to-end latency;
- higher repeated-solve or batch throughput;
- lower device memory;
- a larger solvable horizon or scenario set;
- better multi-GPU scaling;
- lower energy per accepted trajectory;
- equal or better convergence reliability;
- a better accuracy-time-memory Pareto point?

An inner-solver or kernel speedup is insufficient when setup, transcription, data transfer, nonlinear
replay, or failures erase the benefit.

# Comparison layers

## Layer A — identical conic subproblem

Purpose: isolate the contribution of the persistent factorisation-free CQP backend.

Every solver receives the same canonical `Q`, `c`, scalar rows, affine-cone rows, variable bounds,
cone order, requested tolerance, and warm-start data. No solver may receive a more favourable
transcription.

Required systems:

- Clarabel CPU;
- OSQP CPU for QP-compatible instances;
- upstream PDHCG one-shot;
- SpacePDHCG persistent;
- QOCO-GPU where the cone inventory is supported;
- CuClarabel where available;
- the declared structured PIPG implementation;
- SpacePDHCG followed by GPU interior-point polishing.

Primary families:

- P1-A known-optimum trajectory-banded QP/SOCP;
- P1-B HCW rendezvous QP/SOCP;
- convex subproblems captured from accepted iterations of P1-C through P1-F.

Layer A establishes solver accuracy, setup/update cost, repeated-solve performance, memory crossover,
and the effect of warm starts. It does not establish end-to-end trajectory-optimisation superiority.

## Layer B — complete trajectory-optimisation system

Purpose: determine whether the complete SpacePDHCG pipeline is competitive when every implementation
uses its native algorithm and transcription.

Each system receives the same continuous physical problem, boundary conditions, objective, path
constraints, initial-guess information, and tuning budget. Meshes and internal convexifications may
differ. All returned trajectories are judged by the same external high-accuracy nonlinear checker.

Required systems where technically applicable:

- SpacePDHCG CT-SCvx;
- OpenSCvx;
- SCPToolbox or its maintained successor;
- SCvxGEN for the released 6-DoF landing profile;
- CasADi with IPOPT using a declared direct multiple-shooting or collocation transcription;
- pykep with IPOPT for TOPS low-thrust instances.

Optional systems are SNOPT, GPOPS-II, or a published custom GPU SCP implementation when a legal,
reproducible implementation is available. A paper-only implementation is recorded as a literature
reference, not silently reimplemented and presented as the authors' software.

## Layer C — capability and scaling

Purpose: identify the regime in which SpacePDHCG is practically useful.

Sweep:

- trajectory intervals;
- state and control dimensions;
- cone inventory;
- conditioning;
- requested and achieved quality;
- cold, primal, and primal-dual starts;
- scenario count;
- common-control prefix;
- risk measure;
- GPU count and topology.

Continue each sweep until every implementation has either produced a qualified result, timed out, or
failed because of memory or unsupported features. OOM, timeout, unsupported cone type, and quality
failure remain visible outcomes.

## Layer D — GTOPX global mission-design track

Purpose: provide an externally recognizable global-search benchmark for future OrbitWeaver or a
future global outer loop.

Source:

- [GTOPX benchmark distribution](https://www.midaco-solver.com/index.php/about/benchmarks/gtopx)
- [GTOPX SoftwareX paper](https://doi.org/10.1016/j.softx.2021.100666)

Initial progression:

1. Cassini1 as the easiest continuous MGA instance.
2. Rosetta as a higher-dimensional MGA-1DSM instance.
3. Messenger reduced, then Messenger full, as difficult rugged instances.
4. GTOC1 for a constrained long fly-by sequence.
5. Cassini1-MINLP only after discrete outer decisions are supported.
6. Multi-objective variants only after a Pareto-front method and metric are preregistered.

GTOPX best-known objectives and vectors are validation targets. Published MIDACO runtime or
multi-year search history is contextual evidence only. A GTOPX speed ranking requires all global
optimisers to be run under a common evaluation budget and common hardware.

## Layer E — historical GTOC mission-challenge replay

Purpose: test the complete OrbitWeaver route-search and continuous-trajectory stack on recognized
mixed discrete-continuous mission-design challenges.

Archive sources:

- [official GTOC archive by edition](https://sophia.estec.esa.int/gtoc_portal/?page_id=94);
- [GTOC13 JPL site](https://gtoc.jpl.net/).

GTOC rankings and published solutions are objective-quality references. They are not software-speed
references: competition entries may combine weeks of computation, multiple machines, human
intervention, unpublished tools, and post-competition refinement.

Recommended progression:

1. GTOC1 through its GTOPX-compatible reduced black-box formulation.
2. GTOC5 “Penetrators” as a low-thrust multi-rendezvous and sequence-building challenge.
3. GTOC9 “Kessler Run” as the primary debris-removal and multi-spacecraft route benchmark.
4. GTOC12 “Asteroid Mining” as the primary large low-thrust multi-spacecraft benchmark.
5. GTOC11 “Dyson Sphere” only after scheduling and mixed impulsive/continuous arcs are supported.
6. GTOC13 “Altaira System” only after ballistic fly-by and solar-sail models are supported.

For each edition, first create a reduced deterministic subset with the same official evaluator and
objective semantics. Freeze subset selection before observing solver scores. Progress to the full
challenge only after the official example solution validates exactly.

# Pinned benchmark portfolio

## P1-A — known-optimum trajectory-banded CQP

Role: exact algebraic truth and synthetic scale control.

Reference source: repository-generated optimum and committed seeds.

Use for:

- objective and primal distance;
- canonical residual agreement;
- QP versus SOCP cone cost;
- conditioning and horizon scaling;
- allocation, update, and warm-start measurements.

## P1-B — HCW rendezvous

Role: spacecraft-specific exact-linear repeated-update baseline.

Literature/software compatibility profile:

- [OpenSCvx CW proximity operations](https://openscvx.github.io/OpenSCvx/latest/Examples/spacecraft/proxops_cw/)

The repository fixture remains the same-data Layer A profile. The OpenSCvx physical setup is a
separate Layer B compatibility profile because its free-final-time and approach-cone choices are not
identical to the current fixed-pattern fixture.

## P1-C — 3-DoF powered descent

Role: canonical nonlinear powered-descent accuracy and end-to-end SCvx lifecycle.

Primary sources:

- Açıkmeşe and Ploen, *Convex Programming Approach to Powered Descent Guidance for Mars Landing*,
  [DOI 10.2514/1.27553](https://doi.org/10.2514/1.27553);
- [UW CT-SCVX reference implementation](https://github.com/uw-acl/successive-convexification);
- [OpenSCvx 3-DoF PDG example](https://openscvx.github.io/OpenSCvx/latest/Examples/rocket/3DoF_pdg/).

Freeze one exact literature-compatible physical profile before timing. Preserve the existing
horizon and dispersion sweeps as scale profiles.

## P1-D — 14-state 6-DoF powered descent

Role: canonical nonlinear SCvx comparison and flight-relevant cone inventory.

Primary sources:

- Szmuk and Açıkmeşe, *Successive Convexification for 6-DoF Mars Rocket Powered Landing with
  Free-Final-Time*, [DOI 10.2514/6.2018-0617](https://doi.org/10.2514/6.2018-0617);
- [SCvxGEN rocket-landing example](https://scvxgen.com/examples/rocket_landing);
- [OpenSCvx 6-DoF PDG example](https://openscvx.github.io/OpenSCvx/latest/Examples/rocket/6DoF_pdg/).

The 2018 paper's 50-node nondimensional profile is the publication-compatibility case. SCvxGEN and
OpenSCvx profiles are separate reproducible software-comparison cases when their dynamics or
parameters differ.

## P1-D-MC — independent 6-DoF powered-descent batch

Role: batch throughput and initial-condition robustness, distinct from coupled robust optimisation.

Source:

- Chari et al., *Fast Monte Carlo Analysis for 6-DoF Powered-Descent Guidance via GPU-Accelerated
  Sequential Convex Programming*,
  [DOI 10.2514/6.2024-1762](https://doi.org/10.2514/6.2024-1762).

Reproduce the published initial-position distribution
`[U(6,9), U(3,6), U(1,2)]` and declared algorithm limits. Run committed samples at batch sizes that
include 1, 16, 64, 256, 1024, and 2048 where hardware permits.

This track measures independent trajectory throughput. It must not be described as scenario-coupled
robust optimisation because it has no shared controls or non-anticipativity constraints.

## P1-E — long-horizon low thrust

Role: expose horizon, conditioning, and sparse-factorisation memory crossovers.

Literature compatibility profiles:

- fixed-time minimum-fuel Earth-to-Mars rendezvous: 348.795 days, initial mass 1000 kg,
  `Isp = 2000 s`, maximum thrust 0.5 N, and published best final mass approximately 603.935 kg;
- fixed-time minimum-fuel Earth-to-Dionysus rendezvous: 3534 days, initial mass 4000 kg,
  `Isp = 3000 s`, maximum thrust 0.32 N, and published best final mass approximately 2718.33 kg.

Reference:

- Tafazzol and Taheri, *Comparison of Control Regularization Techniques for Minimum-Fuel Low-Thrust
  Trajectory Design Using Indirect Methods*,
  [arXiv:2409.01490](https://arxiv.org/abs/2409.01490).

The published final masses are objective checks, not sufficient feasibility certificates. The
SpacePDHCG trajectory must still pass independent endpoint, mass, throttle, and high-order dynamics
checks.

## P1-E-TOPS — low-thrust suite

Role: modern open benchmark coverage beyond two hand-selected transfers.

Sources:

- Izzo et al., *A Practical Guide to Implementing Zero-Order-Hold Interplanetary Trajectory Legs*,
  [arXiv:2605.11043](https://arxiv.org/abs/2605.11043);
- [pykep trajectory optimisation gym](https://esa.github.io/pykep/gym.html);
- [ESA zero-order-hold repository](https://gitlab.com/EuropeanSpaceAgency/zero-order-hold).

TOPS contains 28 problems across Cartesian two-body, modified equinoctial, CR3BP, and solar-sail
dynamics. Before execution, ingest the released JSON, pin the exact source revision and checksum,
then select:

- one easy two-body case;
- one multi-revolution two-body case;
- one inclination-change or high-eccentricity case;
- one CR3BP case.

The selection must be made from problem metadata before solver results are observed. Solar-sail
cases are deferred unless SpacePDHCG gains a matching sail-control model.

## P1-F — coupled robust powered descent

Role: primary scenario-aware multi-GPU contribution.

No established public benchmark currently matches the repository's shared-control,
non-anticipative, risk-aware formulation. The committed generated instances are therefore the
primary benchmark. The deterministic scenario and independently optimised scenario lower bounds
provide checks, while monolithic and partitioned operators must agree at tractable sizes.

Published independent Monte Carlo timing is not a substitute for this coupled problem.

## P2-F — historical GTOC challenge replay

Role: externally recognizable validation of the integrated route and trajectory oracle.

### GTOC5 — Penetrators

The mission must rendezvous with asteroids to deliver payloads and later revisit them by close
fly-by to deploy penetrators. It exercises low-thrust arc generation, target sequencing, repeated
visits, and score-aware search.

Sources:

- [official GTOC5 archive](https://sophia.estec.esa.int/gtoc_portal/?page_id=25);
- [reproducible beam P-ACO research implementation](https://github.com/lfsimoes/beam_paco__gtoc5).

Use as an intermediate benchmark after deterministic multi-destination routing works. Published
routes and scores are quality references; rerun available implementations for timing.

### GTOC9 — The Kessler Run

The mission designs multiple spacecraft routes that cumulatively remove 123 Sun-synchronous debris
objects while minimizing campaign cost. It closely matches Paper 2's servicing, debris-removal,
multi-spacecraft assignment, and time-dependent routing scope.

Sources:

- [official GTOC9 archive](https://sophia.estec.esa.int/gtoc_portal/?page_id=814);
- [ESA Kelvins problem statement](https://kelvins.esa.int/gtoc9-kessler-run/problem/);
- [JPL methods and results](https://ntrs.nasa.gov/citations/20210007756).

The first implementation uses a preregistered subset of debris and spacecraft while preserving the
official dynamics, waiting-time, cost, and validation rules. The full 123-object campaign is the
primary GTOC demonstration once reduced instances validate.

### GTOC12 — Sustainable Asteroid Mining

The mission sends multiple low-thrust mining ships from Earth to deploy and retrieve miners across
60,000 candidate asteroids during a 15-year window. It exercises multi-fidelity arc screening,
low-thrust refinement, beam search, fleet selection, and coupled campaign scoring.

Sources:

- [official GTOC12 problem and downloadable data](https://gtoc12.tsinghua.edu.cn/competition/theProblem);
- [official GTOC12 archive](https://sophia.estec.esa.int/gtoc_portal/?page_id=1261);
- [published SCP and beam-search approach](https://doi.org/10.1007/s42064-024-0219-3).

Archive the official problem data, fixed bonus coefficients, example solution, verification program,
and public reference solutions. Use the offline verifier for every score. Dynamic competition bonus
values must not be mixed with fixed post-competition scores.

### Deferred editions

- GTOC11 combines large scheduling, impulsive mothership tours, and low-thrust asteroid relocation;
  it is a later stress test rather than an initial validation case.
- GTOC13 uses ballistic gravity assists and optional solar sailing in a hypothetical exoplanetary
  system; it is deferred until those dynamics are implemented.

# Literature evidence policy

Every external datum receives one evidence label:

- `analytic`: mathematically known optimum or invariant;
- `published-reference`: extracted from a paper or official benchmark release but not rerun here;
- `reproduced-external`: obtained by running pinned external code on recorded hardware;
- `measured-local`: obtained from SpacePDHCG on recorded hardware.

Rules:

1. Store the source URL or DOI, version, page/table/equation, units, and extraction method.
2. Preserve all digits supplied by the source; do not infer precision from rounded displays.
3. Mark digitised plot values as approximate.
4. Do not mix objective conventions such as propellant used, final mass, total delta-v, or time.
5. Do not treat a paper's solver status as an independently verified feasibility result.
6. Published timing may appear in a contextual appendix, never in the common-hardware winner map.
7. Function evaluations may be compared only when the evaluation function and derivative accounting
   are identical.
8. If source code is available, pin its commit or archive checksum before relying on it.
9. If exact problem data cannot be recovered, label the profile `descriptive-only`.

# Execution phases

## Phase 0 — freeze inputs

- Pin SpacePDHCG and every external implementation.
- Archive benchmark definitions, parameter files, solution vectors, and checksums.
- Record licences and any restrictions on redistribution.
- Freeze hardware, compiler, CUDA, precision, power, and deterministic settings.
- Freeze solver options, tuning budgets, seeds, timeouts, and quality tiers.
- Create a provenance record for every literature-derived value.

No performance execution begins until this phase is complete.

## Phase 1 — reference reproduction

- Verify P1-A exact optima.
- Reproduce HCW and powered-descent reference trajectories on CPU.
- Reproduce the 2018 6-DoF 50-node profile.
- Reproduce Earth-to-Mars and Earth-to-Dionysus objective values to the declared discretisation
  envelope.
- Load TOPS instances and verify units, endpoint conventions, and gradients.
- Evaluate official GTOPX best-known vectors with the pinned GTOPX evaluator.

Failure to reproduce a published objective is investigated as a formulation, units, transcription,
or source-version discrepancy before timing proceeds.

## Phase 2 — same-CQP solver campaign

- Export canonical CQP snapshots from committed outer iterations.
- Run every compatible Layer A backend on each snapshot.
- Check canonical residuals with one independent implementation.
- Measure cold setup, first solve, repeated update, warm solve, memory, and energy.
- Produce accuracy-time and memory-time fronts.

## Phase 3 — end-to-end deterministic campaign

- Run HCW, 3-DoF, 6-DoF, and low-thrust profiles through each compatible Layer B system.
- Include all setup, transcription, differentiation, transfers, convex solves, nonlinear replay, and
  acceptance logic.
- Judge every result with the same external nonlinear checker.
- Report time to first qualified trajectory and steady repeated-solve time separately.

## Phase 4 — robustness and capability campaign

- Run committed initial-condition and parameter dispersions.
- Report convergence probability, objective distribution, and violation distribution.
- Sweep horizon until timeout or OOM.
- Sweep scenario count and common-prefix fraction.
- Run single- and multi-GPU strong/weak scaling.
- Compare generic and scenario-aware partitions.

## Phase 5 — secondary GTOPX campaign

Run only after a compatible global outer search exists:

- verify official best-known vectors first;
- select global optimisers and freeze a common objective-evaluation budget;
- run Cassini1 before harder problems;
- report best, median, quartiles, success probability, and evaluations-to-target over committed
  seeds;
- keep GTOPX conclusions separate from SCvx/CQP conclusions.

## Phase 6 — GTOC route-and-trajectory campaign

- Download official problem data, format specifications, validators, and published solutions.
- Pin archive checksums and confirm the official example solution reproduces its score.
- Define reduced cases by deterministic metadata rules, not by observed SpacePDHCG performance.
- Run OrbitWeaver with analytical, Lambert/Edelbaum, coarse-convex, and refined-SCvx arc levels.
- Validate every reduced and full solution with the official evaluator.
- Compare score versus compute budget, number of refined arcs, and final certified feasibility.
- Run available external implementations on common hardware when claiming a speed comparison.
- Report human intervention and offline database-generation cost separately.

# Required outcomes and metrics

## Quality

- objective and objective gap;
- independently propagated dynamics defect;
- terminal position, velocity, attitude, and mass error as applicable;
- maximum nodal and inter-node path violation;
- thrust, torque, pointing, angular-rate, quaternion, and dry-mass violations;
- virtual-control norm;
- canonical primal, dual, cone, and gap residuals;
- non-anticipativity and risk-epigraph residuals;
- convergence and accepted-trajectory status.

## Time

- dependency build or code-generation time, reported separately;
- one-time topology construction and workspace creation;
- coefficient generation and differentiation;
- host-to-device and device-to-host transfer;
- scaling/preconditioning;
- convex-solver time;
- nonlinear replay and path checks;
- acceptance and trust-region logic;
- total time to first qualified trajectory;
- total steady repeated-solve time;
- median, IQR, P95, P99, minimum, and maximum;
- accepted trajectories per second for a declared batch.

## Resources and scaling

- active and reserved host/device memory;
- allocation count after persistent creation;
- host/device bytes;
- inter-GPU collective bytes, count, and exposed time;
- load imbalance;
- strong- and weak-scaling efficiency;
- energy per accepted trajectory where counters are available;
- timeout, OOM, unsupported-feature, numerical-failure, and quality-failure counts.

## Software utility

Report separately from raw performance:

- supported problem and cone classes;
- supported CPU/GPU platforms and precision;
- installation/build reproducibility;
- code-generation or compilation latency;
- deterministic replay support;
- warm-start and repeated-update support;
- diagnostics and independent-check integration;
- amount of problem-specific tuning required.

These are factual capability observations, not a composite score unless a scoring rule is
preregistered.

# Fairness rules

1. Same-CQP comparisons use byte-identical problem data after declared coordinate conversion.
2. End-to-end comparisons use the same continuous physical problem and external checker.
3. A solver may use its native transcription, but achieved nonlinear quality must match.
4. Every system receives the same documented tuning budget; tuning time is reported.
5. Default settings and tuned settings are separate result series.
6. Proprietary solvers are optional and clearly labelled.
7. Warm and cold results are never mixed.
8. JIT, code generation, and allocator warm-up are reported and excluded only from the explicitly
   labelled steady-state series.
9. Runs on different hardware are never placed in the common-hardware speedup ratio.
10. Failures remain in robustness and performance-profile denominators.

# Decision products

The campaign produces:

- a reference-reproduction report;
- a same-CQP solver accuracy/time/memory comparison;
- an end-to-end system comparison;
- a robustness and failure report;
- a horizon-by-scenario crossover map;
- a multi-GPU scaling report;
- an accuracy-time-memory Pareto plot;
- a literature-context appendix;
- a separate GTOPX report when a compatible global-search layer exists.
- GTOC5, GTOC9, and GTOC12 reduced/full challenge reports as capability matures.

The winner and hypothesis language remains governed by
`papers/paper1/CLAIMS_AND_DECISION_RULES.md`.

# Minimum publishable campaign

If resources require a reduced campaign, the minimum defensible set is:

1. P1-A and P1-B same-CQP comparisons across all available conic backends.
2. One canonical 3-DoF and one canonical 6-DoF powered-descent end-to-end comparison.
3. Earth-to-Mars plus one preregistered multi-revolution TOPS low-thrust instance.
4. One deterministic and three increasing coupled-scenario coordinates.
5. SpacePDHCG, one CPU reference, one GPU interior-point solver, OpenSCvx, and one direct-NLP
   baseline where compatible.
6. Complete independent nonlinear checking, memory measurement, and failure accounting.

GTOPX does not replace any item in this minimum Paper 1 campaign.

# Implementation status (2026-09-03)

Phase 0 and Phase 1 are implemented as runnable targets on `feat/literature-targets`; see
`docs/LITERATURE_TARGETS.md` for the code map and `docs/REFERENCE_REPRODUCTION_REPORT.md` (with
its machine-readable twin `benchmarks/literature/reference_reproduction.json`) for the per-target
reproduced/gap/descriptive-only/unsupported/blocked outcome. The provenance store
`benchmarks/literature/provenance.json` carries the evidence label of every literature value used
above, and `benchmarks/literature/targets.json` is the registry consumed by the tests and by
`spacepdhcg literature run <id>`.

The GTOC12 replay track of P2-F is implemented on `feat/gtoc12-asteroid-mining`; see
`docs/GTOC12_TRACK.md` for the pinned official data, the exact independent verifier, the
preregistered reduced instance, and the officially verified routes driven by
`spacepdhcg gtoc12 ...`.
