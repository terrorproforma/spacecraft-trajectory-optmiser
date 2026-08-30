# Scenario-aware block-arrow decomposition

## Status

This document freezes the first executable contract for contribution C. It is a
correctness and accounting layer; it does not yet claim NCCL or multi-GPU speedup.

## Information histories

A scenario carries one information label per control stage. Two scenarios share a
control at stage `k` exactly when their information-history prefixes through `k` are
identical. This produces deterministic equivalence classes and prevents accidental
coupling after uncertainty has been observed.

The first-paper baseline is a common open-loop prefix. With `P` shared stages,
controls satisfy

\[
u_{s,k}=\bar u_k,\qquad k<P,
\]

while stages `k >= P` may use scenario-local recourse. A fully open-loop policy uses
`P = N`.

## Global variable ordering

For `S` scenarios, the global vector is

\[
z = [z_1, z_2, \ldots, z_S, \bar u_{n_1}, \ldots, \bar u_{n_M}],
\]

where every `z_s` is contiguous and the shared information-node controls form the
narrow arrowhead. Within each local block, states precede controls and optional local
auxiliary variables.

For every shared information node `n` and member scenario `s`, the sparse operator
contains

\[
u_{s,k}-\bar u_n=0.
\]

The row and column order is immutable for a fixed tree, horizon and variable layout.
Only numerical coefficients and right-hand sides may change during SCvx.

## Communication model

The committed accounting model assumes a ring all-reduce over the shared vector.
For `G` devices and `B` payload bytes, one collective communicates approximately

\[
2\frac{G-1}{G}B
\]

bytes per device, or `2(G-1)B` aggregate bytes. The key hypothesis is now directly
testable: with a fixed common-control horizon, the collective payload is independent
of scenario count even though local variables and non-anticipativity rows grow with
`S`.

This is an accounting baseline, not a prediction of end-to-end runtime. Future NCCL
measurements must report actual bytes, collective latency, overlap and synchronisation.

## Scenario-axis partition

Scenario-local work is assigned with a deterministic longest-processing-time greedy
partition. Stable tie-breaking makes layouts reproducible across hosts and runs. The
logical accelerator grid is `G_s x G_t`:

- `G_s` owns scenario groups;
- `G_t` later partitions time blocks or variable columns inside each group.

The current implementation validates `G_s` assignment and rank order. Time-axis sparse
partitioning and NCCL execution remain subsequent native work.

## Acceptance gate

The C foundation is accepted when:

1. exact information-history groups are reproduced;
2. the sparse non-anticipativity operator has the expected shape and two nonzeros per
   scalar equality;
3. consistent local/shared controls produce zero residual;
4. perturbing one local control produces the exact expected residual;
5. collective payload remains fixed under scenario-count scaling for a fixed shared
   policy dimension; and
6. scenario assignment is deterministic and load-balanced within declared limits.
