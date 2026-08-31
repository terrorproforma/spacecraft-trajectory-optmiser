# Paper 1 notation lock

This file is normative for the Paper 1 manuscript, equations, figures, tables, code comments, and
machine-readable summaries. New symbols require an explicit edit here. A symbol may not acquire a
second meaning in another section.

## 1. Indexing

| Symbol | Meaning |
|---|---|
| \(j\) | outer SCvx iteration, starting at zero |
| \(k\) | trajectory interval or node index, made explicit by context |
| \(s\) | uncertainty-scenario index; never an optimisation step |
| \(r\) | repeated benchmark run index |
| \(g\) | GPU/device index |
| \(p\) | generic model parameter; not probability |
| \(N\) | number of control intervals |
| \(S\) | number of uncertainty scenarios |
| \(G\) | number of GPUs |
| \(n_x,n_u\) | state and control dimensions |
| \(n,m\) | CQP primal and total dual-row dimensions |

Use zero-based indices in code and one-based mathematical subscripts only where doing so materially
improves exposition. Every table must state whether `nodes = N+1` or `intervals = N`.

## 2. Physical trajectory variables

| Symbol | Meaning |
|---|---|
| \(x_k\in\mathbb R^{n_x}\) | spacecraft state at node \(k\) |
| \(u_k\in\mathbb R^{n_u}\) | control held over interval \(k\) |
| \(m_k\) | spacecraft mass component |
| \(q_k\in\mathbb S^3\) | unit quaternion, scalar-first convention |
| \(\omega_k\in\mathbb R^3\) | body angular rate |
| \(T_k\in\mathbb R^3\) | thrust vector |
| \(\sigma_k\) | thrust-magnitude epigraph/control scalar |
| \(\tau_k\in\mathbb R^3\) | body torque |
| \(t_k\) | physical time at node \(k\) |
| \(\Delta t_k\) | interval duration |
| \(f(x,u,p)\) | continuous nonlinear dynamics |
| \(\varphi_{\Delta t}(x,u,p)\) | implemented discrete nonlinear flow map |

SI units are mandatory in stored data and manifests unless an explicit unit field says otherwise:
metres, seconds, kilograms, radians, newtons, newton-metres, metres per second, and metres per
second squared. Plotting may rescale axes to kilometres or milliseconds, but the scale must be in
the axis label and not silently applied to source data.

## 3. SCvx variables

| Symbol | Meaning |
|---|---|
| \(w_j\) | accepted nonlinear reference, including any virtual/slack state |
| \(z\) | stacked physical trajectory decision vector |
| \(v\) | virtual-control or feasibility-slack vector |
| \(s\) in Section 5 equations | trial SCvx step; write \(s_j\) when scenario index is nearby |
| \(\Delta_j\) | trust-region radius |
| \(D_j\) | trust-region scaling matrix |
| \(m_j(s)\) | convex merit model at outer iteration \(j\) |
| \(\Phi_\rho(w)\) | nonlinear exact-penalty merit function |
| \(\operatorname{pred}_j\) | predicted merit reduction |
| \(\operatorname{ared}_j\) | actual merit reduction |
| \(\rho_j\) | agreement ratio \(\operatorname{ared}_j/\operatorname{pred}_j\); penalty weights always carry subscripts such as \(\rho_c\) |
| \(\chi(w_j)\) | nonlinear first-order stationarity measure |
| \(\delta_j^{\mathrm{ct}}\) | continuous-time quadrature/certification error |

Never use \(\rho\) without a subscript for both model agreement and exact-penalty weight in the same
equation block.

## 4. Canonical CQP

The native convex subproblem is always written

\[
\begin{aligned}
\min_z\quad &\tfrac12 z^TQ_jz+c_j^Tz\\
\text{s.t.}\quad &\ell_j\le A_jz\le u_j,\\
&F_jz+f_j\in\mathcal K_j,\\
&\ell_j^x\le z\le u_j^x.
\end{aligned}
\tag{CQP}
\]

| Symbol | Meaning |
|---|---|
| \(Q_j\succeq0\) | native quadratic-objective matrix |
| \(c_j\) | linear objective vector |
| \(A_j\) | scalar equality/interval constraint matrix |
| \(\ell_j,u_j\) | scalar row bounds |
| \(F_j\) | affine-cone matrix |
| \(f_j\) | affine-cone offset |
| \(\mathcal K_j\) | ordered Cartesian product of declared cones |
| \(\ell_j^x,u_j^x\) | variable bounds |
| \(y_j\) | stacked scalar/conic dual vector |
| \(\mathcal S_j\) | exact CQP KKT solution set |

The symbol \(A\) is reserved for scalar constraints. The affine-cone matrix is always \(F\), not
another \(A\). CSC topology is denoted by `pattern(Q)`, `pattern(A)`, and `pattern(F)` in prose,
not by new mathematical symbols.

## 5. Residuals and accuracy

| Symbol | Meaning |
|---|---|
| \(\mathcal R_j(z,y)\) | unscaled natural conic KKT residual defined in `docs/INEXACT_SCVX_THEORY.md` |
| \(S_j\) | residual-reporting scaling matrix |
| \(\varepsilon_j=\|S_j\mathcal R_j\|_2\) | independently achieved canonical inner residual |
| \(\varepsilon_j^{\rm req}\) | requested inner residual tolerance |
| \(r_p,r_d,r_c,r_g\) | canonical primal, dual, cone/complementarity, and gap residual components |
| \(r_{\rm dyn}\) | independent nonlinear dynamics defect |
| \(r_{\rm path}\) | maximum independently checked path violation |
| \(r_{\rm term}\) | scaled terminal-state error |
| \(r_{\rm na}\) | non-anticipativity violation |
| \(r_{\rm risk}\) | risk-epigraph violation |
| \(r_{\rm vc}\) | virtual-control norm |

Use `requested tolerance` and `achieved residual`, never the ambiguous word `accuracy` by itself.
Solver-native residuals carry a `native_` prefix in data columns.

## 6. Variational integration

| Symbol | Meaning |
|---|---|
| \(\Phi_k=\partial\varphi/\partial x_k\) | one-step state-transition sensitivity |
| \(\Gamma_k=\partial\varphi/\partial u_k\) | one-step constant-control sensitivity |
| \(A_k,B_k,d_k\) | affine discrete model \(x_{k+1}=A_kx_k+B_ku_k+d_k\) |
| \(J_N(q)=(I-\hat q\hat q^T)/\|q\|\) | Jacobian of quaternion normalization |

`RK4 variational` means the same RK4 tableau integrates state, \(\Phi\), and \(\Gamma\), followed
by the exact Jacobian of any deterministic post-step projection. `RK4 finite-difference reference`
is the independent domain-aware numerical derivative and must not be labelled production.

## 7. Robust scenario notation

| Symbol | Meaning |
|---|---|
| \(x_k^{(s)},u_k^{(s)}\) | scenario-local state and control |
| \(\bar u_h\) | control associated with information-history node \(h\) |
| \(p_s\) | scenario probability |
| \(L_s\) | scenario loss |
| \(\mathbb E[L]\) | expected loss |
| \(\operatorname{VaR}_\alpha(L)\) | value at risk at confidence \(\alpha\) |
| \(\operatorname{CVaR}_\alpha(L)\) | conditional value at risk |
| \(G_s\times G_t\) | logical scenario-by-time/device grid; not a matrix |

Non-anticipativity is written \(u_k^{(s)}=\bar u_{h(s,k)}\). `Shared prefix` means scenarios have
identical information histories over that prefix, not merely numerically similar controls.

## 8. Performance notation

| Symbol | Meaning |
|---|---|
| \(T_{\rm topo}\) | topology construction time |
| \(T_{\rm coeff}\) | numerical coefficient-generation time |
| \(T_{\rm create}\) | workspace creation time |
| \(T_{\rm update}\) | numerical update time |
| \(T_{\rm h2d},T_{\rm d2h}\) | explicit host/device transfer times |
| \(T_{\rm solve}\) | inner solver iteration time |
| \(T_{\rm residual}\) | independent residual evaluation time |
| \(T_{\rm replay}\) | nonlinear replay/certification time |
| \(T_{\rm collective}\) | exposed collective communication time |
| \(T_{\rm CQP}\) | total convex-subproblem time |
| \(T_{\rm SCvx}\) | total outer solve time |
| \(M_{\rm peak}\) | peak allocated device memory |
| \(E_{\rm traj}\) | energy per accepted trajectory |
| \(P\) | accepted-trajectory throughput; never probability |
| \(\eta_G\) | multi-GPU parallel efficiency |

Speedup is always \(T_{\rm baseline}/T_{\rm candidate}\), so values greater than one favour the
candidate. Timing plots use medians unless explicitly labelled otherwise; uncertainty bands are
interquartile ranges.

## 9. Problem-family identifiers

The manuscript and data use exactly:

- `P1-A-banded`;
- `P1-B-hcw`;
- `P1-C-pd3`;
- `P1-D-pd6`;
- `P1-E-low-thrust`;
- `P1-F-robust-pd`.

Solver identifiers use exactly:

- `clarabel-cpu`;
- `osqp-cpu`;
- `pdhcg-upstream-one-shot`;
- `spacepdhcg-persistent`;
- `qoco-gpu`;
- `cuclarabel`;
- `structured-pipg`;
- `hybrid-pdhcg-ipm`.

A manifest may use another identifier only after adding it here.

## 10. Norms, statistics, and typography

- \(\|\cdot\|\) is the Euclidean norm; use \(\|\cdot\|_1\) and \(\|\cdot\|_\infty\) explicitly.
- `median [Q1,Q3]` is the primary timing summary.
- `N/A` means mathematically inapplicable; `not run` means absent experiment; `OOM`, `timeout`, and
  `failed quality gate` remain distinct outcomes.
- Italic capitals denote mathematical matrices; monospace names denote code/data identifiers.
- `GPU-native` means the declared measured region stays on the accelerator. It does not mean a
  host-controlled programme containing a GPU kernel.
- `Persistent` means the same concrete native workspace object owns topology, scaling state, and
  iterates across numerical updates. Reconstructing an upstream solver object is one-shot, even if
  an allocator cache makes it fast.
