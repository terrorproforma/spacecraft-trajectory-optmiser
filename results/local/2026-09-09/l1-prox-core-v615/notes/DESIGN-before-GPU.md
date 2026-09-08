# Exact L1 proximal diagnostic (default off)

Source base: `2288e103f585a46290d5a6a9635e8cb51575ec6e`. This is an opt-in
representation and preconditioning experiment within the existing dual-first
CUDA primal-dual solver. It is mutually exclusive with Halpern. It has not yet
been run on a GPU. No algorithm default or original accuracy gate changes.

Each selected positive-cost epigraph `t` must have exactly two actual nonzero
uses: `v-t<=0`, `-v-t<=0`, with zero RHS, and no equality/SOC/other coupling.
Every stored Q coefficient must be exactly zero. No coefficient tolerance is
used. Disjoint variable roles/targets and scalar rows are required. The CPU
importer proves the structure before CUDA; an additive native setter independently
checks the provided map on the device. Structural zeros stay in the topology.

Working state retains the original dimensions and allocations. A separate
DeviceProblem view masks the removed scalar rows and sets eliminated smooth
costs to zero; active masks exclude removed columns/rows from products, scaling,
power iterations and updates. Original coefficients and the original common-KKT
audit view remain untouched. This does not yet claim a compact memory layout.

Ten cone-preserving Ruiz passes run on the reduced operator. B uses retained
RHS terms. O is `1/(sqrt(sum_active(c/D)^2 + sum_pairs(lambda/D_v)^2)+1)`.
Both smooth gradient and L1 penalty use this same objective transformation.
The unchanged 20-step spectral heuristic determines eta; it is not a proved
upper bound. Original-coordinate prox step is `d=eta*scaling[v]` and
`v_next=soft(v_previous-d*g_retained,d*lambda)`.

Every completed step reconstructs original `t=abs(v)`. For nonzero v, pair
duals are `(lambda,0)` or `(0,lambda)` according to its strict sign. At exactly
zero v, `delta=clip(-g_retained,-lambda,lambda)` and the pair sums to lambda
with difference delta. The split avoids overflowing lambda+abs(delta).
Completed pair rows are always excluded from subsequent working products.

Before the first update or any reconstruction, the original supplied x/t/y/z
is audited untouched. A qualified approximate seed can have tiny nonzero v and
both pair multipliers positive; even replacing t with abs(v) prematurely loses
the difficult capture's certificate. No snapping or activity tolerance is used.
All completed outputs pass/fail the same original-equation relative 1e-9
primal/dual/gap/block-complementarity and absolute 1e-8 cone gates. Natural
residuals remain separate telemetry. Cancellation wins before initial acceptance
and at each iteration/check. Nonfinite prox inputs/thresholds or underflowed
positive thresholds yield numerical failure, never silent smoothing.

`--l1-prox --common-kkt-stop --execution-blocks N` requires N>0 and unshifted,
unfolded captures. API setup is synchronous and retained across solves. Enable
forces reduced scaling refresh. Disable restores original history and forces
full original scaling/spectral refresh before a later default solve. Old
checkpoint layouts are unchanged; checkpoint/restore are rejected while enabled.
Changing common policy or numeric values requires disabling L1 first.

CPU C++ tests already pass exact detection, structural-zero retention, tiny
nonzero coupling/RHS/coefficient rejection, positive/negative/zero soft prox,
original dual signs/complementarity and overflow-safe dual splitting. The two
captured maps independently match 1470/1631 pairs: logical dimensions are
3791 variables /7811 rows and 4208 /8666, respectively; retained dimensions
remain 5261 /10751 and 5839 /11928. GPU compile/resource and bounded tiny checks
are still pending. The first failed CPU compilation (missing project include
path) performed no solver call; the corrected strict-warning compilation passes.
