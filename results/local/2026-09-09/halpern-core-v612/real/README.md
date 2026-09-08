# Completed real captured-problem comparison

Ten process executions completed once: four supplied-start checks plus six cold comparisons. They made 14 solve API calls because each supplied-start execution included one separately recorded one-iteration bootstrap; the actual seeded solves then accepted at zero iterations. There were four bootstrap iterations and 600,000 cold iterations, with no recovery or deadline cancellation.

Every arm used the same frozen v612d binary/core, original unshifted zero-Q captures, explicit 128 cooperative blocks on the RTX 5090, and the unchanged original-equation gate: relative primal, dual, gap and maximum block complementarity at most 1e-9, and primal/dual cone violation at most 1e-8. Cold calls had a 100,000-iteration cap and 30-second deadline. The supplied-start phases used a 20-second deadline. The logs retain separate setup, scaling, solve, download and audit times.

All four supplied starts remained qualified and preserved the original x/y/z FP64 bits. **None of the six cold calls qualified**; all reached exactly 100,000 iterations. The following are unitless conic gaps, not the GTOC12 kilogram score.

| Capture | Mode | Native solve seconds | Relative gap | Qualified |
|---|---|---:|---:|---|
| conditioning | off | 5.345601 | 0.9999243058 | No |
| conditioning | plain | 3.474233 | 1.370163221 | No |
| conditioning | adaptive | 3.401526 | 0.02554017084 | No |
| difficult | off | 5.520818 | 0.01122705047 | No |
| difficult | plain | 3.495427 | 1.027070844 | No |
| difficult | adaptive | 3.486708 | 0.01844017712 | No |

Adaptive restart reduced the conditioning gap substantially but still failed primal feasibility, cone membership and normalized maximum block complementarity (about 0.4059). Its worst absolute block product improved from 0.803863 to 0.405898; the objective normalization scale fell from 1212.65 to 1, so the normalized value increased despite that absolute improvement. On the difficult capture its gap was worse than the ordinary method. Both plain-Halpern gaps exceeded one. Lower recorded iteration-loop time is not a verified-solution speedup because every cold result failed the common gate. These single-run timings do not establish throughput or SOTA performance, and the algorithm remains opt-in.

The native solve timer includes iteration and residual-check work plus the empty recovery timer events; it excludes scaling, setup, output download and CPU audit. Default/off is the existing dual-first map. Plain/adaptive use a primal-first reflected map, so off-versus-plain changes multiple mechanisms. Plain-versus-adaptive isolates the restart/weight bundle.

[report.json](report.json) is the immutable completed runner report. [raw-logs.tar.gz](raw-logs.tar.gz) contains all ten full stdout/stderr logs, including original x/y/z/slack and native x_solver arrays; [raw-members.json](raw-members.json) binds each exact member. The four [inputs](inputs/conditioning.txt), [runner](run.py), [auditor](independent_auditor.py), and [manifest](manifest.json) are retained. [Fresh publication audits](../audit/real-findings.json) independently recomputed the saved-vector CPU gate and checked source/input/raw/report identities; all CPU, native diagnostic and GPU gate verdicts agree.

The separate [Decimal65 review](independent-review/REPORT.md) independently verifies all ten saved vector sets, four exact seed identities and the signed-gap decomposition. Its four original files are preserved verbatim. From repository root, reproduce those findings directly from this package with standard-library Python; the wrapper verifies and temporarily unpacks the raw archive and performs no CUDA calls:

```text
python -B results/local/2026-09-09/halpern-core-v612/real/independent-review/reproduce_from_package.py
```
