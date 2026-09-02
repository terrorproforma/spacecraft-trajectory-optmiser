// Smoke test for the free-final-time transcriptions pd3_fft / pd6_fft.
//
// Checks (per family):
//   * the topology is NEW: its fingerprint differs from the frozen fixed-final-time one and
//     the sigma column is present in every dynamics row;
//   * a dynamically consistent reference (nonlinear time-dilated rollout) satisfies the
//     linearised dynamics rows to roundoff (affine reconstruction through the CQP fill);
//   * the sigma column equals the finite-difference derivative of the nonlinear map;
//   * pd6: Szmuk-style control model rows (Gamma = d̂·T, tau = r × T, terminal axial thrust)
//     and quaternion tangent rows hold at a projected reference; boundary masks turn rows into
//     vacuous 0 == 0 rows without changing the pattern.

#include "spacepdhcg/transcription/powered_descent_3dof.hpp"
#include "spacepdhcg/transcription/powered_descent_3dof_free_time.hpp"
#include "spacepdhcg/transcription/powered_descent_6dof.hpp"
#include "spacepdhcg/transcription/powered_descent_6dof_free_time.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(const bool condition, const std::string& label, const double value, const double bound) {
    std::printf("%-64s %10.3e (bound %.1e) %s\n", label.c_str(), value, bound, condition ? "ok" : "FAIL");
    if (!condition) {
        ++failures;
    }
}

template <typename Subproblem, typename Values>
double sigma_column_error(
    const Subproblem& subproblem,
    const Values& values,
    const std::vector<double>& decision,
    const double sigma,
    const std::size_t state_dimension
) {
    // Row r of dynamics: x_{k+1} - A x_k - B u_k - S sigma - nu = z.  Perturb sigma in the
    // decision and compare the change of the residual with the finite-difference of the map.
    const auto& structure = subproblem.structure();
    const auto& layout = subproblem.layout();
    const auto delta = 1.0e-4 * sigma;
    auto plus = decision;
    plus[layout.sigma_index()] = sigma + delta;
    auto minus = decision;
    minus[layout.sigma_index()] = sigma - delta;
    const auto residual_plus = spacepdhcg::transcription::free_time::matvec(
        structure.scalar_constraint, values.scalar_constraint, plus
    );
    const auto residual_minus = spacepdhcg::transcription::free_time::matvec(
        structure.scalar_constraint, values.scalar_constraint, minus
    );
    const auto decoded = subproblem.decode(decision);
    double worst = 0.0;
    for (std::size_t interval = 0; interval < layout.intervals; ++interval) {
        const auto step_plus = spacepdhcg::transcription::time_dilated_step(
            subproblem.model(), decoded.states[interval], decoded.controls[interval],
            sigma + delta, subproblem.config().d_tau(), subproblem.config().substeps
        );
        const auto step_minus = spacepdhcg::transcription::time_dilated_step(
            subproblem.model(), decoded.states[interval], decoded.controls[interval],
            sigma - delta, subproblem.config().d_tau(), subproblem.config().substeps
        );
        for (std::size_t component = 0; component < state_dimension; ++component) {
            const auto row = layout.dynamics_rows().start + state_dimension * interval + component;
            // residual = x_{k+1} - ... - S sigma  =>  d residual / d sigma = -S
            const auto analytic = -(residual_plus[row] - residual_minus[row]) / (2.0 * delta);
            const auto numeric = (step_plus[component] - step_minus[component]) / (2.0 * delta);
            worst = std::max(worst, std::abs(analytic - numeric));
        }
    }
    return worst;
}

void test_pd3() {
    using namespace spacepdhcg::transcription;
    constexpr std::size_t intervals = 12U;
    PoweredDescent3DofFreeTimeConfig config{};
    config.intervals = intervals;
    config.substeps = 2U;
    config.fuel_weight = 1.0e-3;
    config.time_weight = 1.0e-2;
    const PoweredDescent3DofFreeTimeSubproblem subproblem{spacepdhcg::dynamics::PoweredDescent3DofModel{}, config};

    PoweredDescentScvxConfig fixed_config{};
    fixed_config.intervals = intervals;
    const PoweredDescent3DofSubproblem fixed{spacepdhcg::dynamics::PoweredDescent3DofModel{}, fixed_config};
    check(
        subproblem.structure().fingerprint() != fixed.structure().fingerprint(),
        "pd3_fft topology differs from frozen pd3", 1.0, 0.0
    );
    check(
        subproblem.layout().variables() == fixed.layout().variables() + 1U,
        "pd3_fft adds exactly one variable (sigma)",
        static_cast<double>(subproblem.layout().variables()), 0.0
    );

    const double sigma = 48.0;
    const spacepdhcg::dynamics::PoweredDescentState initial{0.0, 0.0, 1'500.0, 0.0, 0.0, -40.0, 1'905.0};
    const spacepdhcg::dynamics::PoweredDescentControl control{0.0, 0.0, 9'000.0, 9'000.0};
    std::vector<spacepdhcg::dynamics::PoweredDescentState> states(intervals + 1U);
    std::vector<spacepdhcg::dynamics::PoweredDescentControl> controls(intervals, control);
    states.front() = initial;
    for (std::size_t interval = 0; interval < intervals; ++interval) {
        states[interval + 1U] = time_dilated_step<7U, 4U>(
            subproblem.model(), states[interval], controls[interval], sigma, config.d_tau(),
            config.substeps
        );
    }
    const std::array<double, 3U> target_position{states.back()[0], states.back()[1], states.back()[2]};
    const std::array<double, 3U> target_velocity{states.back()[3], states.back()[4], states.back()[5]};
    const auto values = subproblem.values(
        states, controls, sigma, initial, target_position, target_velocity
    );
    const auto decision = subproblem.reference_decision(states, controls, sigma);
    const auto diagnostics = subproblem.diagnostics(decision, values);
    check(
        diagnostics.linearised_dynamics_defect_inf < 1.0e-8,
        "pd3_fft affine reconstruction through CQP rows",
        diagnostics.linearised_dynamics_defect_inf, 1.0e-8
    );
    check(
        diagnostics.scalar_violation_inf < 1.0e-8, "pd3_fft consistent reference satisfies rows",
        diagnostics.scalar_violation_inf, 1.0e-8
    );
    check(
        diagnostics.cone_violation_inf < 1.0e-9, "pd3_fft consistent reference inside cones",
        diagnostics.cone_violation_inf, 1.0e-9
    );
    check(
        subproblem.replay_defect(states, controls, sigma) < 1.0e-9, "pd3_fft replay defect",
        subproblem.replay_defect(states, controls, sigma), 1.0e-9
    );
    const auto sigma_error = sigma_column_error(subproblem, values, decision, sigma, 7U);
    check(sigma_error < 1.0e-5, "pd3_fft sigma column vs finite differences", sigma_error, 1.0e-5);

    // Sigma trust box.
    check(
        values.variable_lower[subproblem.layout().sigma_index()] == sigma - config.sigma_trust_radius
            && values.variable_upper[subproblem.layout().sigma_index()]
                   == sigma + config.sigma_trust_radius,
        "pd3_fft sigma trust box", config.sigma_trust_radius, 0.0
    );
}

void test_pd6(const bool szmuk_mode) {
    using namespace spacepdhcg::transcription;
    constexpr std::size_t intervals = 10U;
    spacepdhcg::dynamics::PoweredDescent6DofConfig model_config{};
    model_config.gravity = {0.0, 0.0, -1.0};
    model_config.principal_inertia = {1.0e-2, 1.0e-2, 1.0e-2};
    model_config.mass_flow_coefficient = 1.0e-2;
    model_config.minimum_mass = 1.0;
    model_config.maximum_thrust = 5.0;
    model_config.minimum_sigma = 0.3;
    model_config.maximum_torque = 10.0;
    model_config.maximum_angular_rate = 1.0472;
    model_config.maximum_tilt_radians = 0.349;
    model_config.glide_slope_radians = 0.349;
    PoweredDescent6DofFreeTimeConfig config{};
    config.intervals = intervals;
    config.substeps = 2U;
    if (szmuk_mode) {
        config.thrust_norm_mode = FreeTimeThrustNormMode::linearised;
        config.torque_mode = FreeTimeTorqueMode::thrust_arm;
        config.thrust_arm = {0.0, 0.0, -1.0e-2};
        config.terminal_thrust_axial = true;
        for (std::size_t component = 6U; component < 10U; ++component) {
            config.initial_fixed[component] = false;
        }
    }
    const PoweredDescent6DofFreeTimeSubproblem subproblem{
        spacepdhcg::dynamics::PoweredDescent6DofModel{model_config}, config
    };
    const std::string tag = szmuk_mode ? "pd6_fft[szmuk]" : "pd6_fft[repo]";

    PoweredDescent6DofScvxConfig fixed_config{};
    fixed_config.intervals = intervals;
    const PoweredDescent6DofSubproblem fixed{spacepdhcg::dynamics::PoweredDescent6DofModel{model_config}, fixed_config};
    check(
        subproblem.structure().fingerprint() != fixed.structure().fingerprint(),
        tag + " topology differs from frozen pd6", 1.0, 0.0
    );

    const double sigma = 3.4;
    const double tilt = 0.15;
    spacepdhcg::dynamics::PoweredDescent6DofState initial{
        0.0, 0.0, 4.0, 0.0, 0.0, -0.5, std::cos(0.5 * tilt), std::sin(0.5 * tilt), 0.0, 0.0,
        0.0, 0.0, 0.0, 2.0,
    };
    // Small direct torque so the repo-mode reference is not stiff (inertia 1e-2): keeps the
    // central-difference truncation error well below the oracle bound.
    spacepdhcg::dynamics::PoweredDescent6DofControl raw{0.02, -0.01, 2.2, 0.003, 0.001, -0.002, 1.0};
    auto control = subproblem.project_control(raw);
    if (szmuk_mode) {
        // Last interval must be axial in Szmuk mode.
        raw[0] = 0.0;
        raw[1] = 0.0;
    }
    auto last_control = subproblem.project_control(raw);
    std::vector<spacepdhcg::dynamics::PoweredDescent6DofControl> controls(intervals, control);
    controls.back() = last_control;
    std::vector<spacepdhcg::dynamics::PoweredDescent6DofState> states(intervals + 1U);
    states.front() = initial;
    for (std::size_t interval = 0; interval < intervals; ++interval) {
        states[interval + 1U] = time_dilated_step<14U, 7U>(
            subproblem.model(), states[interval], controls[interval], sigma, config.d_tau(),
            config.substeps
        );
    }
    const auto values = subproblem.values(states, controls, sigma, initial, states.back());
    const auto decision = subproblem.reference_decision(states, controls, sigma);
    const auto diagnostics = subproblem.diagnostics(decision, values);
    check(
        diagnostics.linearised_dynamics_defect_inf < 1.0e-9,
        tag + " affine reconstruction through CQP rows",
        diagnostics.linearised_dynamics_defect_inf, 1.0e-9
    );
    check(
        diagnostics.scalar_violation_inf < 1.0e-9,
        tag + " consistent reference satisfies all scalar rows",
        diagnostics.scalar_violation_inf, 1.0e-9
    );
    const auto sigma_error = sigma_column_error(subproblem, values, decision, sigma, 14U);
    check(sigma_error < 1.0e-6, tag + " sigma column vs finite differences", sigma_error, 1.0e-6);

    // Quaternion tangent rule: the linearised flow's quaternion rows satisfy q̂ᵀ(A δx) = 0 for
    // any δx, i.e. the propagated quaternion row block is orthogonal to the propagated
    // quaternion.  Recover A from the CQP rows and test on the first interval.
    {
        const auto& layout = subproblem.layout();
        const auto& structure = subproblem.structure();
        double worst = 0.0;
        for (std::size_t column = 0; column < 14U; ++column) {
            std::vector<double> unit(layout.variables(), 0.0);
            unit[layout.state(0U).start + column] = 1.0;
            const auto response = free_time::matvec(
                structure.scalar_constraint, values.scalar_constraint, unit
            );
            double dot = 0.0;
            for (std::size_t component = 0; component < 4U; ++component) {
                const auto row = layout.dynamics_rows().start + 6U + component;
                dot += states[1U][6U + component] * response[row];
            }
            worst = std::max(worst, std::abs(dot));
        }
        check(worst < 1.0e-12, tag + " quaternion tangent rule on A columns", worst, 1.0e-12);
    }

    // Boundary masks: free rows are vacuous (-inf, +inf) without altering the pattern.
    {
        const auto& layout = subproblem.layout();
        const auto initial_quaternion_row = layout.initial_rows().start + 6U;
        const bool vacuous = !std::isfinite(values.scalar_lower[initial_quaternion_row]);
        check(
            vacuous == szmuk_mode, tag + " initial quaternion row is free iff unmasked",
            vacuous ? 1.0 : 0.0, 0.0
        );
        const auto mass_row = layout.terminal_rows().start + 13U;
        check(
            !std::isfinite(values.scalar_lower[mass_row]), tag + " terminal mass row is free",
            0.0, 0.0
        );
        const auto axial_row = layout.terminal_thrust_rows().start;
        const bool axial_active = values.scalar_lower[axial_row] == 0.0
            && values.scalar_upper[axial_row] == 0.0;
        check(
            axial_active == szmuk_mode, tag + " terminal axial thrust rows active iff enabled",
            axial_active ? 1.0 : 0.0, 0.0
        );
    }
}

}  // namespace

int main() {
    try {
        test_pd3();
        test_pd6(false);
        test_pd6(true);
    } catch (const std::exception& error) {
        std::printf("exception: %s\n", error.what());
        return 1;
    }
    std::printf("%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
