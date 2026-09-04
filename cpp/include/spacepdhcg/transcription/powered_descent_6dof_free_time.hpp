#pragma once

// Free-final-time 6-DoF powered descent (`pd6_fft`).
//
// Same time-dilation formulation as `pd3_fft` (see `powered_descent_3dof_free_time.hpp`), on
// the 14-state / 7-control rigid-body model of `dynamics/powered_descent_6dof.hpp`
// (state = r(3), v(3), q(4), omega(3), m; control = T_B(3), tau_B(3), Gamma).
//
// Additional structure specific to 6-DoF and to the Szmuk–Açıkmeşe 2018 profile:
//   * quaternion tangent rule: the linearised flow's quaternion rows are projected onto the
//     tangent space of the unit sphere at the propagated quaternion (done inside
//     `linearise_time_dilated_flow`), and each free quaternion node carries the linearised
//     unit-norm row  2 q̂·q = 1 + |q̂|²;
//   * configurable boundary masks (the paper leaves the initial attitude free and the final
//     mass free);
//   * optional Szmuk-style control model: torque tied to the thrust arm (tau = r_T × T) and a
//     linearised thrust-magnitude lower bound  T_min <= (T̂/|T̂|)·T  instead of the repository's
//     epigraph slack Gamma >= |T|;
//   * gimbal cone  |T| <= T_z / cos(delta_max)  (second-order cone, exact);
//   * optional terminal axial thrust (T_x = T_y = 0 on the last interval).
//
// All optional rows exist in the frozen CSC topology and are switched by bounds/values only,
// so one fingerprint serves every configuration.  The fixed-final-time P1-D topology is not
// touched.

#include "spacepdhcg/core/fixed_cqp.hpp"
#include "spacepdhcg/dynamics/powered_descent_6dof.hpp"
#include "spacepdhcg/transcription/free_time_common.hpp"
#include "spacepdhcg/transcription/time_dilated_flow_linearisation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace spacepdhcg::transcription {

enum class FreeTimeThrustNormMode : unsigned char {
    /// Gamma >= |T| (second-order cone) with Gamma in [minimum_sigma, maximum_thrust].
    epigraph = 0,
    /// Gamma = (T̂/|T̂|)·T (linearised norm), |T| <= maximum_thrust, Gamma >= minimum_sigma.
    linearised = 1,
};

enum class FreeTimeTorqueMode : unsigned char {
    /// tau_B is an independent control bounded by |tau| <= maximum_torque.
    direct = 0,
    /// tau_B = thrust_arm × T_B (Szmuk 2018 single-gimballed-engine model).
    thrust_arm = 1,
};

struct PoweredDescent6DofFreeTimeConfig {
    std::size_t intervals{20U};
    std::size_t substeps{1U};
    double sigma_minimum{1.0e-2};
    double sigma_maximum{1.0e3};
    double trust_radius{1.0};
    double sigma_trust_radius{1.0};
    double virtual_l1_weight{1.0e3};
    double virtual_quadratic_weight{1.0e-8};
    double virtual_epigraph_regularisation{1.0e-10};
    double fuel_weight{0.0};
    double time_weight{1.0};
    double sigma_tracking_weight{1.0e-6};
    /// Attitude tilt bound theta_max between the body thrust axis and the inertial vertical:
    /// cos(theta) = 1 - 2 (q_x^2 + q_y^2) >= cos(theta_max)  <=>  |[q_x, q_y]| <= sqrt((1 - cos theta_max)/2)
    /// (Szmuk 2018 eq. tilt constraint).  pi disables it (bound = 1 is implied by |q| = 1).
    double maximum_attitude_tilt_radians{3.141592653589793};
    FreeTimeThrustNormMode thrust_norm_mode{FreeTimeThrustNormMode::epigraph};
    FreeTimeTorqueMode torque_mode{FreeTimeTorqueMode::direct};
    std::array<double, 3U> thrust_arm{0.0, 0.0, -1.0e-2};
    bool terminal_thrust_axial{false};
    std::array<bool, 14U> initial_fixed{
        true, true, true, true, true, true, true, true, true, true, true, true, true, true,
    };
    std::array<bool, 14U> terminal_fixed{
        true, true, true, true, true, true, true, true, true, true, true, true, true, false,
    };
    std::array<double, 14U> state_tracking_weights{
        1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6,
        1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6,
    };
    std::array<double, 7U> control_tracking_weights{
        1.0e-8, 1.0e-8, 1.0e-8, 1.0e-8, 1.0e-8, 1.0e-8, 1.0e-8,
    };
    std::array<double, 14U> state_trust_scales{
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    };
    std::array<double, 7U> control_trust_scales{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    [[nodiscard]] double d_tau() const noexcept { return 1.0 / static_cast<double>(intervals); }

    /// sqrt((1 - cos theta_max) / 2): bound on |[q_x, q_y]| for the attitude tilt cone.
    [[nodiscard]] double attitude_tilt_bound() const noexcept {
        return std::sqrt(std::max(0.5 * (1.0 - std::cos(maximum_attitude_tilt_radians)), 0.0));
    }

    void validate() const {
        using free_time::require_nonnegative;
        using free_time::require_positive;
        if (intervals < 2U) {
            throw std::invalid_argument("pd6_fft needs at least two intervals");
        }
        if (substeps == 0U) {
            throw std::invalid_argument("pd6_fft needs at least one RK4 substep");
        }
        require_positive(sigma_minimum, "pd6_fft sigma minimum must be positive");
        if (!(sigma_maximum > sigma_minimum)) {
            throw std::invalid_argument("pd6_fft sigma maximum must exceed the minimum");
        }
        require_positive(trust_radius, "pd6_fft trust radius must be positive");
        require_positive(sigma_trust_radius, "pd6_fft sigma trust radius must be positive");
        require_positive(virtual_l1_weight, "pd6_fft virtual L1 weight must be positive");
        require_nonnegative(virtual_quadratic_weight, "pd6_fft virtual quadratic weight");
        require_positive(virtual_epigraph_regularisation, "pd6_fft epigraph regularisation");
        require_nonnegative(fuel_weight, "pd6_fft fuel weight");
        require_nonnegative(time_weight, "pd6_fft time weight");
        if (fuel_weight == 0.0 && time_weight == 0.0) {
            throw std::invalid_argument("pd6_fft needs a fuel or time objective");
        }
        require_nonnegative(sigma_tracking_weight, "pd6_fft sigma tracking weight");
        if (!std::isfinite(maximum_attitude_tilt_radians) || maximum_attitude_tilt_radians <= 0.0
            || maximum_attitude_tilt_radians > 3.141592653589793) {
            throw std::invalid_argument("pd6_fft attitude tilt bound must lie in (0, pi]");
        }
        for (const auto value : thrust_arm) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("pd6_fft thrust arm must be finite");
            }
        }
        for (const auto value : state_tracking_weights) {
            require_nonnegative(value, "pd6_fft state tracking weight");
        }
        for (const auto value : control_tracking_weights) {
            require_nonnegative(value, "pd6_fft control tracking weight");
        }
        for (const auto value : state_trust_scales) {
            require_positive(value, "pd6_fft state trust scale");
        }
        for (const auto value : control_trust_scales) {
            require_positive(value, "pd6_fft control trust scale");
        }
    }
};

struct PoweredDescent6DofFreeTimeLayout {
    explicit PoweredDescent6DofFreeTimeLayout(std::size_t interval_count)
        : intervals(interval_count) {}

    std::size_t intervals{0U};

    // Variables: states | controls | sigma | virtual | epigraph
    [[nodiscard]] std::size_t state_count() const noexcept { return (intervals + 1U) * 14U; }
    [[nodiscard]] std::size_t control_count() const noexcept { return intervals * 7U; }
    [[nodiscard]] std::size_t virtual_count() const noexcept { return intervals * 14U; }
    [[nodiscard]] std::size_t control_offset() const noexcept { return state_count(); }
    [[nodiscard]] std::size_t sigma_index() const noexcept {
        return control_offset() + control_count();
    }
    [[nodiscard]] std::size_t virtual_offset() const noexcept { return sigma_index() + 1U; }
    [[nodiscard]] std::size_t epigraph_offset() const noexcept {
        return virtual_offset() + virtual_count();
    }
    [[nodiscard]] std::size_t variables() const noexcept {
        return epigraph_offset() + virtual_count();
    }

    // Scalar rows: initial | dynamics | terminal | virtual epigraph | gamma linearisation |
    //              torque coupling | terminal axial thrust | quaternion norm
    [[nodiscard]] free_time::Range initial_rows() const noexcept { return {0U, 14U}; }
    [[nodiscard]] free_time::Range dynamics_rows() const noexcept {
        return {initial_rows().stop(), 14U * intervals};
    }
    [[nodiscard]] free_time::Range terminal_rows() const noexcept {
        return {dynamics_rows().stop(), 14U};
    }
    [[nodiscard]] free_time::Range virtual_epigraph_rows() const noexcept {
        return {terminal_rows().stop(), 28U * intervals};
    }
    [[nodiscard]] free_time::Range gamma_rows() const noexcept {
        return {virtual_epigraph_rows().stop(), intervals};
    }
    [[nodiscard]] free_time::Range torque_coupling_rows() const noexcept {
        return {gamma_rows().stop(), 3U * intervals};
    }
    [[nodiscard]] free_time::Range terminal_thrust_rows() const noexcept {
        return {torque_coupling_rows().stop(), 2U};
    }
    [[nodiscard]] free_time::Range quaternion_rows() const noexcept {
        return {terminal_thrust_rows().stop(), intervals + 1U};
    }
    [[nodiscard]] std::size_t scalar_rows() const noexcept { return quaternion_rows().stop(); }

    // Affine cone rows: thrust | gimbal | torque | glide | angular rate | attitude tilt |
    //                   stage trust | terminal trust
    [[nodiscard]] free_time::Range thrust_rows() const noexcept { return {0U, 4U * intervals}; }
    [[nodiscard]] free_time::Range gimbal_rows() const noexcept {
        return {thrust_rows().stop(), 4U * intervals};
    }
    [[nodiscard]] free_time::Range torque_rows() const noexcept {
        return {gimbal_rows().stop(), 4U * intervals};
    }
    [[nodiscard]] free_time::Range glide_rows() const noexcept {
        return {torque_rows().stop(), 3U * (intervals + 1U)};
    }
    [[nodiscard]] free_time::Range angular_rate_rows() const noexcept {
        return {glide_rows().stop(), 4U * (intervals + 1U)};
    }
    /// Per node: [q_x, q_y, bound] with the bound carried by the affine offset.
    [[nodiscard]] free_time::Range tilt_rows() const noexcept {
        return {angular_rate_rows().stop(), 3U * (intervals + 1U)};
    }
    [[nodiscard]] free_time::Range stage_trust_rows() const noexcept {
        return {tilt_rows().stop(), 22U * intervals};
    }
    [[nodiscard]] free_time::Range terminal_trust_rows() const noexcept {
        return {stage_trust_rows().stop(), 15U};
    }
    [[nodiscard]] std::size_t affine_rows() const noexcept { return terminal_trust_rows().stop(); }

    [[nodiscard]] free_time::Range state(std::size_t node) const {
        if (node > intervals) {
            throw std::out_of_range("pd6_fft state node is outside the trajectory");
        }
        return {14U * node, 14U};
    }
    [[nodiscard]] free_time::Range control(std::size_t interval) const {
        check(interval);
        return {control_offset() + 7U * interval, 7U};
    }
    [[nodiscard]] free_time::Range virtual_control(std::size_t interval) const {
        check(interval);
        return {virtual_offset() + 14U * interval, 14U};
    }
    [[nodiscard]] free_time::Range epigraph(std::size_t interval) const {
        check(interval);
        return {epigraph_offset() + 14U * interval, 14U};
    }

  private:
    void check(std::size_t interval) const {
        if (interval >= intervals) {
            throw std::out_of_range("pd6_fft interval is outside the trajectory");
        }
    }
};

struct PoweredDescent6DofFreeTimeDecision {
    std::vector<dynamics::PoweredDescent6DofState> states{};
    std::vector<dynamics::PoweredDescent6DofControl> controls{};
    double sigma{0.0};
    std::vector<dynamics::PoweredDescent6DofState> virtual_controls{};
};

class PoweredDescent6DofFreeTimeSubproblem {
  public:
    using State = dynamics::PoweredDescent6DofState;
    using Control = dynamics::PoweredDescent6DofControl;
    using Model = dynamics::PoweredDescent6DofModel;

    explicit PoweredDescent6DofFreeTimeSubproblem(
        Model model = Model{},
        PoweredDescent6DofFreeTimeConfig config = PoweredDescent6DofFreeTimeConfig{}
    )
        : model_(std::move(model)), config_(config), layout_(config_.intervals) {
        config_.validate();
        build_structure();
    }

    [[nodiscard]] const Model& model() const noexcept { return model_; }
    [[nodiscard]] const PoweredDescent6DofFreeTimeConfig& config() const noexcept {
        return config_;
    }
    [[nodiscard]] const PoweredDescent6DofFreeTimeLayout& layout() const noexcept {
        return layout_;
    }
    [[nodiscard]] const core::FixedStructure& structure() const noexcept { return structure_; }

    /// Make a control consistent with the configured control model: in `linearised` mode
    /// Gamma := |T|, in `thrust_arm` mode tau := r_T × T.  Used to project accepted iterates so
    /// the reference used for the next linearisation satisfies the nonlinear model exactly.
    [[nodiscard]] Control project_control(const Control& control) const noexcept {
        auto projected = control;
        if (config_.thrust_norm_mode == FreeTimeThrustNormMode::linearised) {
            projected[6U] = std::sqrt(
                control[0U] * control[0U] + control[1U] * control[1U] + control[2U] * control[2U]
            );
        }
        if (config_.torque_mode == FreeTimeTorqueMode::thrust_arm) {
            const auto& r = config_.thrust_arm;
            projected[3U] = r[1U] * control[2U] - r[2U] * control[1U];
            projected[4U] = r[2U] * control[0U] - r[0U] * control[2U];
            projected[5U] = r[0U] * control[1U] - r[1U] * control[0U];
        }
        return projected;
    }

    [[nodiscard]] core::NumericValues values(
        const std::vector<State>& reference_states,
        const std::vector<Control>& reference_controls,
        double reference_sigma,
        const State& initial_state,
        const State& target_state,
        double trust_radius = -1.0,
        double sigma_trust_radius = -1.0
    ) const {
        validate_reference(reference_states, reference_controls, reference_sigma);
        validate_state(initial_state, "pd6_fft initial state");
        validate_state(target_state, "pd6_fft target state");
        const auto radius = trust_radius > 0.0 ? trust_radius : config_.trust_radius;
        const auto sigma_radius =
            sigma_trust_radius > 0.0 ? sigma_trust_radius : config_.sigma_trust_radius;
        core::NumericValues result{};
        const auto infinity = std::numeric_limits<double>::infinity();
        result.quadratic = quadratic_values_;
        result.scalar_constraint.assign(structure_.scalar_constraint.nonzeros(), 0.0);
        result.affine_cone.assign(structure_.affine_cone->nonzeros(), 0.0);
        result.linear_objective.assign(layout_.variables(), 0.0);
        result.scalar_lower.assign(layout_.scalar_rows(), -infinity);
        result.scalar_upper.assign(layout_.scalar_rows(), infinity);
        result.affine_offset.assign(layout_.affine_rows(), 0.0);
        result.variable_lower.assign(layout_.variables(), -infinity);
        result.variable_upper.assign(layout_.variables(), infinity);

        fill_objective(result, reference_states, reference_controls, reference_sigma);
        fill_boundary(result, initial_state, target_state);
        fill_dynamics(result, reference_states, reference_controls, reference_sigma);
        fill_virtual_epigraph(result);
        fill_control_model(result, reference_controls);
        fill_quaternion(result, reference_states);
        fill_cones(result, reference_states, reference_controls, radius);
        fill_bounds(result, reference_sigma, sigma_radius);
        result.validate(structure_);
        return result;
    }

    [[nodiscard]] std::vector<double> reference_decision(
        const std::vector<State>& states,
        const std::vector<Control>& controls,
        double sigma
    ) const {
        validate_reference(states, controls, sigma);
        std::vector<double> decision(layout_.variables(), 0.0);
        for (std::size_t node = 0; node <= layout_.intervals; ++node) {
            std::copy(
                states[node].begin(), states[node].end(),
                decision.begin() + layout_.state(node).start
            );
        }
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            std::copy(
                controls[interval].begin(), controls[interval].end(),
                decision.begin() + layout_.control(interval).start
            );
        }
        decision[layout_.sigma_index()] = sigma;
        return decision;
    }

    [[nodiscard]] PoweredDescent6DofFreeTimeDecision decode(const std::vector<double>& decision)
        const {
        if (decision.size() != layout_.variables()) {
            throw std::invalid_argument("pd6_fft decision vector has the wrong size");
        }
        PoweredDescent6DofFreeTimeDecision decoded{};
        decoded.states.resize(layout_.intervals + 1U);
        decoded.controls.resize(layout_.intervals);
        decoded.virtual_controls.resize(layout_.intervals);
        for (std::size_t node = 0; node <= layout_.intervals; ++node) {
            std::copy_n(
                decision.begin() + static_cast<std::ptrdiff_t>(layout_.state(node).start), 14U,
                decoded.states[node].begin()
            );
        }
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            std::copy_n(
                decision.begin() + static_cast<std::ptrdiff_t>(layout_.control(interval).start),
                7U, decoded.controls[interval].begin()
            );
            std::copy_n(
                decision.begin()
                    + static_cast<std::ptrdiff_t>(layout_.virtual_control(interval).start),
                14U, decoded.virtual_controls[interval].begin()
            );
        }
        decoded.sigma = decision[layout_.sigma_index()];
        return decoded;
    }

    [[nodiscard]] double replay_defect(
        const std::vector<State>& states,
        const std::vector<Control>& controls,
        double sigma
    ) const {
        validate_reference(states, controls, sigma);
        double worst = 0.0;
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto next = time_dilated_step<14U, 7U>(
                model_, states[interval], controls[interval], sigma, config_.d_tau(),
                config_.substeps
            );
            for (std::size_t component = 0; component < 14U; ++component) {
                worst = std::max(
                    worst, std::abs(next[component] - states[interval + 1U][component])
                );
            }
        }
        return worst;
    }

    [[nodiscard]] free_time::FreeTimeDiagnostics diagnostics(
        const std::vector<double>& decision,
        const core::NumericValues& values
    ) const {
        if (decision.size() != layout_.variables()) {
            throw std::invalid_argument("pd6_fft decision vector has the wrong size");
        }
        values.validate(structure_);
        const auto scalar =
            free_time::matvec(structure_.scalar_constraint, values.scalar_constraint, decision);
        const auto affine =
            free_time::matvec(*structure_.affine_cone, values.affine_cone, decision);
        free_time::FreeTimeDiagnostics result{};
        for (std::size_t row = 0; row < scalar.size(); ++row) {
            result.scalar_violation_inf = std::max(
                result.scalar_violation_inf,
                std::max(values.scalar_lower[row] - scalar[row], scalar[row] - values.scalar_upper[row])
            );
        }
        for (std::size_t variable = 0; variable < decision.size(); ++variable) {
            result.variable_violation_inf = std::max(
                result.variable_violation_inf,
                std::max(
                    values.variable_lower[variable] - decision[variable],
                    decision[variable] - values.variable_upper[variable]
                )
            );
        }
        result.cone_violation_inf =
            free_time::cone_violation(structure_, affine, values.affine_offset);
        result.scalar_violation_inf = std::max(result.scalar_violation_inf, 0.0);
        result.variable_violation_inf = std::max(result.variable_violation_inf, 0.0);
        result.cone_violation_inf = std::max(result.cone_violation_inf, 0.0);
        const auto dynamics_rows = layout_.dynamics_rows();
        for (std::size_t local = 0; local < dynamics_rows.size; ++local) {
            const auto row = dynamics_rows.start + local;
            result.linearised_dynamics_defect_inf = std::max(
                result.linearised_dynamics_defect_inf,
                std::abs(scalar[row] - values.scalar_lower[row])
            );
        }
        for (std::size_t variable = layout_.virtual_offset();
             variable < layout_.epigraph_offset();
             ++variable) {
            result.virtual_control_inf =
                std::max(result.virtual_control_inf, std::abs(decision[variable]));
        }
        return result;
    }

  private:
    Model model_{};
    PoweredDescent6DofFreeTimeConfig config_{};
    PoweredDescent6DofFreeTimeLayout layout_;
    core::FixedStructure structure_{};
    free_time::ValueIndex quadratic_index_{};
    free_time::ValueIndex scalar_index_{};
    free_time::ValueIndex affine_index_{};
    std::vector<double> quadratic_values_{};

    void build_structure() {
        free_time::EntrySet q{};
        free_time::EntrySet a{};
        free_time::EntrySet f{};
        for (std::size_t variable = 0; variable < layout_.variables(); ++variable) {
            q.insert({variable, variable});
        }
        const auto sigma = layout_.sigma_index();
        const auto terminal_state = layout_.state(layout_.intervals);
        for (std::size_t component = 0; component < 14U; ++component) {
            a.insert({layout_.initial_rows().start + component, component});
            a.insert({layout_.terminal_rows().start + component, terminal_state.start + component});
        }
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto row_start = layout_.dynamics_rows().start + 14U * interval;
            const auto state = layout_.state(interval);
            const auto next_state = layout_.state(interval + 1U);
            const auto control = layout_.control(interval);
            const auto virtual_control = layout_.virtual_control(interval);
            for (std::size_t row_offset = 0; row_offset < 14U; ++row_offset) {
                const auto row = row_start + row_offset;
                for (std::size_t column = state.start; column < state.stop(); ++column) {
                    a.insert({row, column});
                }
                a.insert({row, next_state.start + row_offset});
                for (std::size_t column = control.start; column < control.stop(); ++column) {
                    a.insert({row, column});
                }
                a.insert({row, sigma});
                a.insert({row, virtual_control.start + row_offset});
            }
            // Gamma linearisation row: Gamma - d̂·T
            const auto gamma_row = layout_.gamma_rows().start + interval;
            for (std::size_t component = 0; component < 3U; ++component) {
                a.insert({gamma_row, control.start + component});
            }
            a.insert({gamma_row, control.start + 6U});
            // Torque coupling rows: tau_i - (r × T)_i
            for (std::size_t axis = 0; axis < 3U; ++axis) {
                const auto row = layout_.torque_coupling_rows().start + 3U * interval + axis;
                a.insert({row, control.start + 3U + axis});
                for (std::size_t component = 0; component < 3U; ++component) {
                    a.insert({row, control.start + component});
                }
            }
        }
        for (std::size_t flat = 0; flat < layout_.virtual_count(); ++flat) {
            const auto positive = layout_.virtual_epigraph_rows().start + 2U * flat;
            const auto virtual_variable = layout_.virtual_offset() + flat;
            const auto epigraph_variable = layout_.epigraph_offset() + flat;
            a.insert({positive, virtual_variable});
            a.insert({positive, epigraph_variable});
            a.insert({positive + 1U, virtual_variable});
            a.insert({positive + 1U, epigraph_variable});
        }
        {
            const auto last_control = layout_.control(layout_.intervals - 1U);
            a.insert({layout_.terminal_thrust_rows().start, last_control.start});
            a.insert({layout_.terminal_thrust_rows().start + 1U, last_control.start + 1U});
        }
        for (std::size_t node = 0; node <= layout_.intervals; ++node) {
            const auto state = layout_.state(node);
            const auto quaternion_row = layout_.quaternion_rows().start + node;
            for (std::size_t component = 0; component < 4U; ++component) {
                a.insert({quaternion_row, state.start + 6U + component});
            }
            const auto glide_start = layout_.glide_rows().start + 3U * node;
            for (std::size_t component = 0; component < 3U; ++component) {
                f.insert({glide_start + component, state.start + component});
            }
            const auto rate_start = layout_.angular_rate_rows().start + 4U * node;
            for (std::size_t component = 0; component < 3U; ++component) {
                f.insert({rate_start + component, state.start + 10U + component});
            }
            const auto tilt_start = layout_.tilt_rows().start + 3U * node;
            f.insert({tilt_start, state.start + 7U});
            f.insert({tilt_start + 1U, state.start + 8U});
        }
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto control = layout_.control(interval);
            const auto state = layout_.state(interval);
            const auto thrust_start = layout_.thrust_rows().start + 4U * interval;
            for (std::size_t component = 0; component < 3U; ++component) {
                f.insert({thrust_start + component, control.start + component});
            }
            f.insert({thrust_start + 3U, control.start + 6U});
            const auto gimbal_start = layout_.gimbal_rows().start + 4U * interval;
            for (std::size_t component = 0; component < 3U; ++component) {
                f.insert({gimbal_start + component, control.start + component});
            }
            f.insert({gimbal_start + 3U, control.start + 2U});
            const auto torque_start = layout_.torque_rows().start + 4U * interval;
            for (std::size_t component = 0; component < 3U; ++component) {
                f.insert({torque_start + component, control.start + 3U + component});
            }
            const auto trust_start = layout_.stage_trust_rows().start + 22U * interval;
            for (std::size_t component = 0; component < 14U; ++component) {
                f.insert({trust_start + component, state.start + component});
            }
            for (std::size_t component = 0; component < 7U; ++component) {
                f.insert({trust_start + 14U + component, control.start + component});
            }
        }
        for (std::size_t component = 0; component < 14U; ++component) {
            f.insert({layout_.terminal_trust_rows().start + component,
                      terminal_state.start + component});
        }
        std::vector<ConeBlockDescriptor> cones{};
        const auto k = layout_.intervals;
        free_time::append_uniform_cones(cones, layout_.thrust_rows().start, k, 4U, 2);
        free_time::append_uniform_cones(cones, layout_.gimbal_rows().start, k, 4U, 2);
        free_time::append_uniform_cones(cones, layout_.torque_rows().start, k, 4U, 2);
        free_time::append_uniform_cones(cones, layout_.glide_rows().start, k + 1U, 3U, 1);
        free_time::append_uniform_cones(cones, layout_.angular_rate_rows().start, k + 1U, 4U, 2);
        free_time::append_uniform_cones(cones, layout_.tilt_rows().start, k + 1U, 3U, 1);
        free_time::append_uniform_cones(cones, layout_.stage_trust_rows().start, k, 22U, 20);
        cones.push_back(ConeBlockDescriptor{
            ConeKind::second_order,
            static_cast<Index>(layout_.terminal_trust_rows().start),
            13,
            0.0,
        });
        structure_ = core::FixedStructure{
            free_time::make_pattern(layout_.variables(), layout_.variables(), q),
            free_time::make_pattern(layout_.scalar_rows(), layout_.variables(), a),
            free_time::make_pattern(layout_.affine_rows(), layout_.variables(), f),
            std::move(cones),
            {},
        };
        structure_.validate();
        quadratic_index_ = free_time::ValueIndex(structure_.quadratic);
        scalar_index_ = free_time::ValueIndex(structure_.scalar_constraint);
        affine_index_ = free_time::ValueIndex(*structure_.affine_cone);
        quadratic_values_.assign(structure_.quadratic.nonzeros(), 0.0);
        for (std::size_t node = 0; node <= layout_.intervals; ++node) {
            const auto state = layout_.state(node);
            for (std::size_t component = 0; component < 14U; ++component) {
                quadratic_index_.set(
                    quadratic_values_, state.start + component, state.start + component,
                    config_.state_tracking_weights[component]
                );
            }
        }
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto control = layout_.control(interval);
            for (std::size_t component = 0; component < 7U; ++component) {
                quadratic_index_.set(
                    quadratic_values_, control.start + component, control.start + component,
                    config_.control_tracking_weights[component]
                );
            }
            const auto virtual_control = layout_.virtual_control(interval);
            const auto epigraph = layout_.epigraph(interval);
            for (std::size_t component = 0; component < 14U; ++component) {
                quadratic_index_.set(
                    quadratic_values_, virtual_control.start + component,
                    virtual_control.start + component, config_.virtual_quadratic_weight
                );
                quadratic_index_.set(
                    quadratic_values_, epigraph.start + component, epigraph.start + component,
                    config_.virtual_epigraph_regularisation
                );
            }
        }
        quadratic_index_.set(quadratic_values_, sigma, sigma, config_.sigma_tracking_weight);
    }

    void fill_objective(
        core::NumericValues& values,
        const std::vector<State>& states,
        const std::vector<Control>& controls,
        double sigma
    ) const {
        for (std::size_t node = 0; node < states.size(); ++node) {
            const auto range = layout_.state(node);
            for (std::size_t component = 0; component < 14U; ++component) {
                values.linear_objective[range.start + component] =
                    -states[node][component] * config_.state_tracking_weights[component];
            }
        }
        double reference_throttle_sum = 0.0;
        for (std::size_t interval = 0; interval < controls.size(); ++interval) {
            const auto range = layout_.control(interval);
            for (std::size_t component = 0; component < 7U; ++component) {
                values.linear_objective[range.start + component] =
                    -controls[interval][component] * config_.control_tracking_weights[component];
            }
            values.linear_objective[range.start + 6U] +=
                config_.fuel_weight * config_.d_tau() * sigma;
            reference_throttle_sum += controls[interval][6U];
        }
        values.linear_objective[layout_.sigma_index()] =
            config_.time_weight + config_.fuel_weight * config_.d_tau() * reference_throttle_sum
            - sigma * config_.sigma_tracking_weight;
        for (std::size_t variable = layout_.epigraph_offset(); variable < layout_.variables();
             ++variable) {
            values.linear_objective[variable] = config_.virtual_l1_weight;
        }
    }

    void fill_boundary(core::NumericValues& values, const State& initial, const State& target)
        const {
        const auto final_state = layout_.state(layout_.intervals);
        for (std::size_t component = 0; component < 14U; ++component) {
            const auto initial_row = layout_.initial_rows().start + component;
            scalar_index_.set(values.scalar_constraint, initial_row, component, 1.0);
            if (config_.initial_fixed[component]) {
                values.scalar_lower[initial_row] = initial[component];
                values.scalar_upper[initial_row] = initial[component];
            }
            const auto terminal_row = layout_.terminal_rows().start + component;
            scalar_index_.set(
                values.scalar_constraint, terminal_row, final_state.start + component, 1.0
            );
            if (config_.terminal_fixed[component]) {
                values.scalar_lower[terminal_row] = target[component];
                values.scalar_upper[terminal_row] = target[component];
            }
        }
    }

    void fill_dynamics(
        core::NumericValues& values,
        const std::vector<State>& states,
        const std::vector<Control>& controls,
        double sigma
    ) const {
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto lin = linearise_time_dilated_flow<14U, 7U>(
                model_, states[interval], controls[interval], sigma, config_.d_tau(),
                config_.substeps
            );
            const auto row_start = layout_.dynamics_rows().start + 14U * interval;
            const auto state = layout_.state(interval);
            const auto next_state = layout_.state(interval + 1U);
            const auto control = layout_.control(interval);
            const auto virtual_control = layout_.virtual_control(interval);
            for (std::size_t row_offset = 0; row_offset < 14U; ++row_offset) {
                const auto row = row_start + row_offset;
                for (std::size_t column = 0; column < 14U; ++column) {
                    scalar_index_.set(
                        values.scalar_constraint, row, state.start + column,
                        -lin.state[row_offset * 14U + column]
                    );
                }
                scalar_index_.set(values.scalar_constraint, row, next_state.start + row_offset, 1.0);
                for (std::size_t column = 0; column < 7U; ++column) {
                    scalar_index_.set(
                        values.scalar_constraint, row, control.start + column,
                        -lin.control[row_offset * 7U + column]
                    );
                }
                scalar_index_.set(
                    values.scalar_constraint, row, layout_.sigma_index(), -lin.sigma[row_offset]
                );
                scalar_index_.set(
                    values.scalar_constraint, row, virtual_control.start + row_offset, -1.0
                );
                values.scalar_lower[row] = lin.offset[row_offset];
                values.scalar_upper[row] = lin.offset[row_offset];
            }
        }
    }

    void fill_virtual_epigraph(core::NumericValues& values) const {
        for (std::size_t flat = 0; flat < layout_.virtual_count(); ++flat) {
            const auto positive = layout_.virtual_epigraph_rows().start + 2U * flat;
            const auto negative = positive + 1U;
            const auto virtual_variable = layout_.virtual_offset() + flat;
            const auto epigraph_variable = layout_.epigraph_offset() + flat;
            scalar_index_.set(values.scalar_constraint, positive, virtual_variable, 1.0);
            scalar_index_.set(values.scalar_constraint, positive, epigraph_variable, -1.0);
            scalar_index_.set(values.scalar_constraint, negative, virtual_variable, -1.0);
            scalar_index_.set(values.scalar_constraint, negative, epigraph_variable, -1.0);
            values.scalar_upper[positive] = 0.0;
            values.scalar_upper[negative] = 0.0;
        }
    }

    void fill_control_model(core::NumericValues& values, const std::vector<Control>& controls)
        const {
        const bool linearised = config_.thrust_norm_mode == FreeTimeThrustNormMode::linearised;
        const bool thrust_arm = config_.torque_mode == FreeTimeTorqueMode::thrust_arm;
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto control = layout_.control(interval);
            const auto& reference = controls[interval];
            const auto gamma_row = layout_.gamma_rows().start + interval;
            std::array<double, 3U> direction{0.0, 0.0, 1.0};
            const auto norm = std::sqrt(
                reference[0U] * reference[0U] + reference[1U] * reference[1U]
                + reference[2U] * reference[2U]
            );
            if (norm > 0.0) {
                direction = {reference[0U] / norm, reference[1U] / norm, reference[2U] / norm};
            }
            for (std::size_t component = 0; component < 3U; ++component) {
                scalar_index_.set(
                    values.scalar_constraint, gamma_row, control.start + component,
                    linearised ? -direction[component] : 0.0
                );
            }
            scalar_index_.set(
                values.scalar_constraint, gamma_row, control.start + 6U, linearised ? 1.0 : 0.0
            );
            if (linearised) {
                values.scalar_lower[gamma_row] = 0.0;
                values.scalar_upper[gamma_row] = 0.0;
            }
            const auto& r = config_.thrust_arm;
            // (r × T)_x = r_y T_z - r_z T_y ; _y = r_z T_x - r_x T_z ; _z = r_x T_y - r_y T_x
            const std::array<std::array<double, 3U>, 3U> cross{{
                {0.0, r[2U], -r[1U]},
                {-r[2U], 0.0, r[0U]},
                {r[1U], -r[0U], 0.0},
            }};
            for (std::size_t axis = 0; axis < 3U; ++axis) {
                const auto row = layout_.torque_coupling_rows().start + 3U * interval + axis;
                scalar_index_.set(
                    values.scalar_constraint, row, control.start + 3U + axis, thrust_arm ? 1.0 : 0.0
                );
                for (std::size_t component = 0; component < 3U; ++component) {
                    scalar_index_.set(
                        values.scalar_constraint, row, control.start + component,
                        thrust_arm ? cross[axis][component] : 0.0
                    );
                }
                if (thrust_arm) {
                    values.scalar_lower[row] = 0.0;
                    values.scalar_upper[row] = 0.0;
                }
            }
        }
        const auto last_control = layout_.control(layout_.intervals - 1U);
        for (std::size_t lateral = 0; lateral < 2U; ++lateral) {
            const auto row = layout_.terminal_thrust_rows().start + lateral;
            scalar_index_.set(
                values.scalar_constraint, row, last_control.start + lateral,
                config_.terminal_thrust_axial ? 1.0 : 0.0
            );
            if (config_.terminal_thrust_axial) {
                values.scalar_lower[row] = 0.0;
                values.scalar_upper[row] = 0.0;
            }
        }
    }

    void fill_quaternion(core::NumericValues& values, const std::vector<State>& states) const {
        const bool terminal_quaternion_fixed = config_.terminal_fixed[6U]
            && config_.terminal_fixed[7U] && config_.terminal_fixed[8U] && config_.terminal_fixed[9U];
        const bool initial_quaternion_fixed = config_.initial_fixed[6U]
            && config_.initial_fixed[7U] && config_.initial_fixed[8U] && config_.initial_fixed[9U];
        for (std::size_t node = 0; node <= layout_.intervals; ++node) {
            const auto row = layout_.quaternion_rows().start + node;
            const auto state = layout_.state(node);
            const bool pinned = (node == layout_.intervals && terminal_quaternion_fixed)
                || (node == 0U && initial_quaternion_fixed);
            double norm_squared = 0.0;
            for (std::size_t component = 0; component < 4U; ++component) {
                const auto reference = states[node][6U + component];
                norm_squared += reference * reference;
                scalar_index_.set(
                    values.scalar_constraint, row, state.start + 6U + component,
                    pinned ? 0.0 : 2.0 * reference
                );
            }
            // Pinned nodes are already fixed by exact boundary equalities; a tangent plane about
            // a displaced reference would exclude that fixed unit target.  Keep 0 == 0.
            values.scalar_lower[row] = pinned ? 0.0 : 1.0 + norm_squared;
            values.scalar_upper[row] = pinned ? 0.0 : 1.0 + norm_squared;
        }
    }

    void fill_cones(
        core::NumericValues& values,
        const std::vector<State>& states,
        const std::vector<Control>& controls,
        double radius
    ) const {
        const bool linearised = config_.thrust_norm_mode == FreeTimeThrustNormMode::linearised;
        const auto inverse_cos_gimbal = 1.0 / model_.config().tilt_cosine();
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto control = layout_.control(interval);
            const auto thrust_start = layout_.thrust_rows().start + 4U * interval;
            for (std::size_t component = 0; component < 3U; ++component) {
                affine_index_.set(
                    values.affine_cone, thrust_start + component, control.start + component, 1.0
                );
            }
            // |T| <= Gamma (epigraph) or |T| <= maximum_thrust (linearised)
            affine_index_.set(
                values.affine_cone, thrust_start + 3U, control.start + 6U, linearised ? 0.0 : 1.0
            );
            values.affine_offset[thrust_start + 3U] =
                linearised ? model_.config().maximum_thrust : 0.0;
            // gimbal: |T| <= T_z / cos(delta_max)
            const auto gimbal_start = layout_.gimbal_rows().start + 4U * interval;
            for (std::size_t component = 0; component < 3U; ++component) {
                affine_index_.set(
                    values.affine_cone, gimbal_start + component, control.start + component, 1.0
                );
            }
            affine_index_.set(
                values.affine_cone, gimbal_start + 3U, control.start + 2U, inverse_cos_gimbal
            );
            const auto torque_start = layout_.torque_rows().start + 4U * interval;
            for (std::size_t component = 0; component < 3U; ++component) {
                affine_index_.set(
                    values.affine_cone, torque_start + component, control.start + 3U + component,
                    1.0
                );
            }
            values.affine_offset[torque_start + 3U] = model_.config().maximum_torque;
            const auto state = layout_.state(interval);
            const auto trust_start = layout_.stage_trust_rows().start + 22U * interval;
            for (std::size_t component = 0; component < 14U; ++component) {
                const auto scale = config_.state_trust_scales[component];
                affine_index_.set(
                    values.affine_cone, trust_start + component, state.start + component, scale
                );
                values.affine_offset[trust_start + component] =
                    -scale * states[interval][component];
            }
            for (std::size_t component = 0; component < 7U; ++component) {
                const auto row = trust_start + 14U + component;
                const auto scale = config_.control_trust_scales[component];
                affine_index_.set(values.affine_cone, row, control.start + component, scale);
                values.affine_offset[row] = -scale * controls[interval][component];
            }
            values.affine_offset[trust_start + 21U] = radius;
        }
        for (std::size_t node = 0; node <= layout_.intervals; ++node) {
            const auto state = layout_.state(node);
            const auto glide_start = layout_.glide_rows().start + 3U * node;
            affine_index_.set(values.affine_cone, glide_start, state.start, 1.0);
            affine_index_.set(values.affine_cone, glide_start + 1U, state.start + 1U, 1.0);
            affine_index_.set(
                values.affine_cone, glide_start + 2U, state.start + 2U,
                model_.config().glide_slope_tangent()
            );
            const auto rate_start = layout_.angular_rate_rows().start + 4U * node;
            for (std::size_t component = 0; component < 3U; ++component) {
                affine_index_.set(
                    values.affine_cone, rate_start + component, state.start + 10U + component, 1.0
                );
            }
            values.affine_offset[rate_start + 3U] = model_.config().maximum_angular_rate;
            // attitude tilt: |[q_x, q_y]| <= sqrt((1 - cos theta_max)/2)
            const auto tilt_start = layout_.tilt_rows().start + 3U * node;
            affine_index_.set(values.affine_cone, tilt_start, state.start + 7U, 1.0);
            affine_index_.set(values.affine_cone, tilt_start + 1U, state.start + 8U, 1.0);
            values.affine_offset[tilt_start + 2U] = config_.attitude_tilt_bound();
        }
        const auto final_state = layout_.state(layout_.intervals);
        const auto terminal_start = layout_.terminal_trust_rows().start;
        for (std::size_t component = 0; component < 14U; ++component) {
            const auto scale = config_.state_trust_scales[component];
            affine_index_.set(
                values.affine_cone, terminal_start + component, final_state.start + component,
                scale
            );
            values.affine_offset[terminal_start + component] =
                -scale * states.back()[component];
        }
        values.affine_offset[terminal_start + 14U] = radius;
    }

    void fill_bounds(core::NumericValues& values, double sigma, double sigma_radius) const {
        for (std::size_t node = 0; node <= layout_.intervals; ++node) {
            const auto state = layout_.state(node);
            values.variable_lower[state.start + 2U] = 0.0;
            values.variable_lower[state.start + 13U] = model_.config().minimum_mass;
        }
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto control = layout_.control(interval);
            values.variable_lower[control.start + 6U] = model_.config().minimum_sigma;
            values.variable_upper[control.start + 6U] = model_.config().maximum_thrust;
        }
        values.variable_lower[layout_.sigma_index()] =
            std::max(config_.sigma_minimum, sigma - sigma_radius);
        values.variable_upper[layout_.sigma_index()] =
            std::min(config_.sigma_maximum, sigma + sigma_radius);
        for (std::size_t variable = layout_.epigraph_offset(); variable < layout_.variables();
             ++variable) {
            values.variable_lower[variable] = 0.0;
        }
    }

    void validate_reference(
        const std::vector<State>& states,
        const std::vector<Control>& controls,
        double sigma
    ) const {
        if (states.size() != layout_.intervals + 1U || controls.size() != layout_.intervals) {
            throw std::invalid_argument("pd6_fft reference trajectory has the wrong horizon");
        }
        if (!std::isfinite(sigma) || sigma <= 0.0) {
            throw std::invalid_argument("pd6_fft reference sigma must be finite and positive");
        }
        for (const auto& state : states) {
            validate_state(state, "pd6_fft reference state");
        }
        for (const auto& control : controls) {
            for (const auto value : control) {
                if (!std::isfinite(value)) {
                    throw std::invalid_argument("pd6_fft reference control must be finite");
                }
            }
        }
    }

    static void validate_state(const State& state, const char* name) {
        for (const auto value : state) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument(std::string(name) + " must be finite");
            }
        }
        if (state[13U] <= 0.0) {
            throw std::invalid_argument(std::string(name) + " must have positive mass");
        }
    }
};

}  // namespace spacepdhcg::transcription
