#include "spacepdhcg/scvx/policies.hpp"

#include <cmath>

int main() {
    using spacepdhcg::scvx::AdaptiveForcingRule;
    using spacepdhcg::scvx::InexactErrorLedger;
    using spacepdhcg::scvx::InexactSolveSample;
    using spacepdhcg::scvx::OuterResidual;
    using spacepdhcg::scvx::SolvePhase;
    using spacepdhcg::scvx::SolverStage;
    using spacepdhcg::scvx::TrustAction;
    using spacepdhcg::scvx::TrustRegionController;
    using spacepdhcg::scvx::hybrid_plan;

    const AdaptiveForcingRule forcing{};
    const auto repair = forcing.request(0U, OuterResidual{0.5, 0.0, 0.0, 1.0}, 0U);
    if (repair.phase != SolvePhase::repair || repair.tolerance > 1.0e-2) {
        return 1;
    }
    const auto progress = forcing.request(1U, OuterResidual{0.01, 0.01, 0.01, 0.4}, 1U, 0.8);
    if (progress.phase != SolvePhase::refinement || progress.tolerance > 1.0e-5) {
        return 2;
    }
    const auto polish = forcing.request(5U, OuterResidual{1.0e-5, 1.0e-5, 1.0e-5, 1.0e-4}, 4U, 0.9);
    if (polish.phase != SolvePhase::polish || polish.tolerance > 1.0e-8) {
        return 3;
    }
    if (!forcing.should_resolve(false, 3.0e-3, 2.0e-3, 1.0e-3)
        || std::abs(forcing.refined_tolerance(1.0e-3) - 1.0e-4) > 1.0e-15) {
        return 4;
    }

    TrustRegionController trust{};
    const auto expanded = trust.update(true, 0.9, 0.9);
    if (expanded.action != TrustAction::expand || expanded.radius_after <= expanded.radius_before) {
        return 5;
    }
    const auto shrunk = trust.update(false, -1.0, 0.2);
    if (shrunk.action != TrustAction::shrink || shrunk.radius_after >= shrunk.radius_before) {
        return 6;
    }

    InexactErrorLedger ledger{};
    ledger.append(InexactSolveSample{1.0e-3, 8.0e-4, 1.0e-1, 0.5});
    ledger.append(InexactSolveSample{1.0e-4, 7.0e-5, 1.0e-2, 0.1});
    if (!ledger.within_summable_budget(1.0e-3) || ledger.maximum_relative_forcing() <= 0.0) {
        return 7;
    }

    const auto plan = hybrid_plan(polish, true, true);
    if (plan.stage != SolverStage::interior_point_polish || !plan.transfer_primal
        || !plan.transfer_dual) {
        return 8;
    }
    return 0;
}
