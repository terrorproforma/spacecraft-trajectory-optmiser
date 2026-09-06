#pragma once
#include "native_qoco_gpu.h"
#include "spacepdhcg/cuda/gtoc12_qoco_c_api.h"
#include <cmath>

namespace gtoc12_qoco {
// One scalar decision after the parallel objective and residual reductions.
// This is the same external gate as the synchronous bridge, including finite
// objective checks (a small relative gap alone is insufficient).
__global__ void qualify(const QocoReplayStatus* status, const QocoAuditResult* audit,
    const double* objective, double tolerance, spacepdhcg_gtoc12_qoco_report* report) {
    *report={};
    report->qoco_status=status->status; report->iterations=status->iterations;
    report->requested_tolerance=tolerance;
    report->primal_residual=audit->primal; report->dual_residual=audit->dual;
    report->absolute_primal_residual=audit->absolute_primal;
    report->absolute_dual_residual=audit->absolute_dual;
    report->primal_objective=objective[0]; report->dual_objective=objective[1];
    report->absolute_gap=objective[2]; report->relative_gap=objective[3];
    report->qualified=(status->status==1 || status->status==2)
        && isfinite(audit->primal) && isfinite(audit->dual)
        && audit->primal<=tolerance && audit->dual<=tolerance
        && isfinite(objective[0]) && isfinite(objective[1])
        && isfinite(objective[3]) && objective[3]<=tolerance;
}
}
