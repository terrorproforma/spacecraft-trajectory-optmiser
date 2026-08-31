# Paper 1 claims and decision rules

This file preregisters the Paper 1 hypotheses before GPU results are collected. It prevents a
post-hoc change from “where does the method work?” to a cherry-picked universal speedup claim.

All comparisons are end-to-end and quality matched under `docs/BENCHMARK_PROTOCOL.md`. A run that
fails the declared canonical or nonlinear gate cannot support a performance hypothesis.

## Common statistical rule

Unless a hypothesis gives a stronger condition:

1. use committed instances and locked solver parameters;
2. discard warm-up runs only as declared in the manifest;
3. require at least five measured repeats per deterministic instance and twenty instances per
   randomised coordinate;
4. compare paired coordinates whenever possible;
5. use median time as the primary statistic and report IQR/min/max;
6. use a 95% paired bootstrap interval over instances/repeats for speedup or difference;
7. call a result `supported` only if the complete confidence interval clears the practical threshold;
8. call it `rejected` if the interval clears the threshold in the opposite direction;
9. otherwise call it `mixed/unresolved`;
10. retain timeout, OOM, and quality failure as censored outcomes.

A crossover is “sustained” only when the rule holds at three consecutive increasing scale
coordinates or at every remaining feasible coordinate when fewer than three remain.

## H1 — persistent device residency removes repeated setup

### Claim

After one workspace creation and warm-up, SpacePDHCG can update and solve same-topology CQP
sequences without rebuilding or retransferring topology, and repeated setup/transfer overhead is no
more than 5% of steady-state end-to-end CQP time in the regime where the GPU is otherwise utilised.

### Primary metric

\[
\omega_{m persist}
=
\frac{T_{\rm topo,repeated}+T_{\rm create,repeated}+T_{\rm topology\ h2d,repeated}}
     {T_{\rm CQP,steady}}.
\]

### Supported

- post-creation topology allocation/copy count is exactly zero;
- topology fingerprints remain identical;
- median \(\omega_{\rm persist}\le0.05\);
- upper 95% interval \(\le0.08\);
- persistent and one-shot solutions pass the same residual gate.

### Rejected

Any topology reallocation/copy occurs in steady state, or median overhead exceeds 10% in every
scale that saturates the tested GPU.

### Mixed

The 5% target holds only above a documented scale. Report that crossover rather than weakening the
claim.

## H2 — factorisation-free compute crossover

### Claim

For sufficiently large trajectory CQPs, persistent PDHCG has lower end-to-end time than the best
qualified available GPU interior-point method.

### Primary metric

\[
S_{\rm compute}
=
\frac{T_{\rm best\ qualified\ GPU\ IPM}}
     {T_{\rm spacepdhcg\ persistent}}.
\]

### Supported

A sustained scale region has median \(S_{\rm compute}\ge1.20\) and lower 95% interval above 1.0,
with matched nonlinear quality and all setup/update/replay time included.

### Rejected

Every common feasible scale has upper 95% interval below 1.0.

### Mixed

PDHCG wins only at loose tolerances, one cone inventory, or only in batched throughput but not
single-problem latency. State the bounded regime precisely.

## H3 — memory crossover

### Claim

The factorisation-free method extends the largest solvable trajectory/scenario scale beyond
factorisation-based GPU conic solvers.

### Supported by either condition

1. SpacePDHCG produces a qualified solution for at least one larger coordinate at which every
   declared GPU IPM fails with recorded OOM; or
2. at matched coordinates, SpacePDHCG peak active device memory is at most 60% of the best GPU IPM,
   with upper 95% interval below 75%.

Allocator reservation and active allocation are reported separately. A solver manually capped to a
smaller workspace is not an OOM comparison unless its documented maximum feasible setting was used.

### Rejected

SpacePDHCG OOMs no later than the best IPM and does not achieve the memory-ratio condition.

## H4 — scenario-aware partitioning improves multi-GPU execution

### Claim

Whole-scenario ownership plus shared-arrowhead collectives reduces exposed communication and total
time relative to generic nonzero-balanced partitioning for robust spacecraft CQPs.

### Supported

At matched global CQP and GPU topology, for a sustained robust scale region:

- measured collective bytes fall by at least 25%;
- exposed collective time falls by at least 20%;
- total end-to-end time falls by at least 10%;
- median load imbalance is no greater than 1.15;
- monolithic and distributed solutions pass the same canonical/nonlinear gate.

The lower 95% interval for total-time improvement must exceed zero.

### Rejected

Scenario-aware partitioning increases total time by at least 10% throughout the communication-bound
region, or fails quality due to incorrect collective semantics.

### Mixed

Communication decreases but local load imbalance erases total-time benefit. This remains a useful
architectural finding and must be reported as mixed.

## H5 — adaptive inexact forcing reduces end-to-end work

### Claim

Adaptive inner tolerances reduce total SCvx time versus a fixed tight tolerance without degrading
final nonlinear quality or convergence reliability.

### Baseline

The fixed-tight tolerance equals the final required canonical residual at every outer iteration.
The adaptive policy is the committed repair/progress/refinement/polish policy; no per-instance
retuning is permitted.

### Supported

Across at least two nonlinear families and a sustained scale region:

- median total SCvx time is at least 15% lower;
- lower 95% interval for time reduction is above zero;
- final objective difference is within the declared practical-equivalence margin;
- final canonical and nonlinear residuals pass the same gate;
- failure/rejection rate is not more than two percentage points worse;
- achieved residuals satisfy the forcing rule recorded in `INEXACT_SCVX_THEORY.md`.

### Rejected

Adaptive forcing is slower by at least 10% or materially increases quality/failure violations across
all tested nonlinear families.

### Mixed

Benefits occur only with warm starts, only before final polish, or only in some conditioning bins.
Report the interaction.

## H6 — hybrid PDHCG to IPM polishing expands the Pareto frontier

### Claim

Using PDHCG for construction and a GPU interior-point solver for terminal polishing can achieve
IPM-grade final residuals with lower end-to-end time than pure IPM, while improving final accuracy
relative to pure PDHCG.

### Supported

At a sustained scale region:

- hybrid final canonical/nonlinear residuals are within a factor of two of pure-IPM residuals and
  pass the same quality tier;
- hybrid total time is at least 10% below pure IPM with lower 95% interval above zero;
- hybrid residual is at least one decade lower than the corresponding unpolished PDHCG result, or
  the unpolished result fails the final quality tier;
- warm-start conversion and polish setup are included.

### Rejected

Polish overhead eliminates the time advantage at every common feasible scale, or warm-start transfer
causes unreliable convergence.

### Mixed

Hybrid dominates only at extreme tolerances or after a clear horizon threshold. This defines the
regime map.

# Secondary, non-hypothesis observations

The paper may report without promoting to a preregistered primary claim:

- analytic variational RK4 coefficient-fill speed versus finite differences;
- quaternion tangency error;
- scaling reuse frequency;
- CUDA graph-capture benefit;
- energy per accepted trajectory;
- single versus double precision;
- OrbitWeaver arc throughput as a downstream demonstration.

These observations still require complete manifests and quality gates.

# Winner rule for the regime map

At each family/scale/quality/hardware coordinate:

1. exclude unsupported and unqualified solvers;
2. retain OOM/timeout as censored failures;
3. identify the lowest median end-to-end time;
4. a unique winner requires at least 10% improvement over the runner-up and a paired 95% interval
   excluding zero;
5. otherwise label the cell `tie`;
6. if no solver qualifies, label `no qualified solver`;
7. if the apparent winner is selected from an unlocked tuning sweep, label `exploratory`, not a
   primary result.

# Claim language

Allowed:

- “supported in the tested regime”;
- “crossover observed at …”;
- “mixed because communication savings were offset by imbalance”;
- “rejected on the tested hardware”;
- “unresolved due to censored comparison.”

Forbidden:

- “GPU acceleration demonstrated” from a kernel-only timing;
- “multi-GPU scaling” without nonlinear replay and collective time;
- “globally optimal trajectory”;
- “flight qualified”;
- “universally faster”;
- replacing a preregistered threshold after observing results without labelling the new analysis
  exploratory.
