#pragma once

// Free-final-time 3-DoF powered descent (`pd3_fft`).
//
// Time-dilation formulation: the trajectory lives on the normalised horizon tau in [0, 1]
// split into K equal intervals of d_tau = 1/K.  A single extra decision variable sigma (the
// time of flight, seconds) scales the physical time of every interval, so the discrete map is
//
//     x_{k+1} = F(x_k, u_k, sigma) = RK4 flow over sigma * d_tau seconds (ZOH control).
//
// Its linearisation about the reference (x̄, ū, σ̄) is
//
//     x_{k+1} = A_k x_k + B_k u_k + S_k sigma + z_k + nu_k
//
// with S_k = dF/dsigma computed by the augmented variational RK4 in
// `time_dilated_flow_linearisation.hpp`, and z_k the affine reconstruction offset.  The fuel
// integral sigma * d_tau * sum(Gamma_k) is bilinear; it is linearised about the reference
// (first-order in both factors) so the CQP stays convex.  The time weight makes the objective
// `time_weight * sigma + fuel_weight * ...`.
//
// This is a NEW topology.  The frozen fixed-final-time `PoweredDescent3DofSubproblem` is not
// modified and its P1-C fingerprint is unaffected.

#include "spacepdhcg/core/fixed_cqp.hpp"
#include "spacepdhcg/dynamics/powered_descent_3dof.hpp"
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

struct PoweredDescent3DofFreeTimeConfig {
    std::size_t intervals{20U};
    std::size_t substeps{1U};
    double sigma_minimum{1.0};
    double sigma_maximum{1.0e4};
    double trust_radius{1.0};
    double sigma_trust_radius{5.0};
    double virtual_l1_weight{1.0e3};
    double virtual_quadratic_weight{1.0e-8};
    double virtual_epigraph_regularisation{1.0e-10};
    double fuel_weight{1.0e-3};
    double time_weight{0.0};
    double sigma_tracking_weight{1.0e-6};
    std::array<double, 7U> state_tracking_weights{
        1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6,
    };
    std::array<double, 4U> control_tracking_weights{1.0e-8, 1.0e-8, 1.0e-8, 1.0e-8};
    std::array<double, 7U> state_trust_scales{
        1.0e-2, 1.0e-2, 1.0e-2, 1.0e-1, 1.0e-1, 1.0e-1, 1.0e-3,
    };
    std::array<double, 4U> control_trust_scales{1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4};

    [[nodiscard]] double d_tau() const noexcept { return 1.0 / static_cast<double>(intervals); }

    void validate() const {
        using free_time::require_nonnegative;
        using free_time::require_positive;
        if (intervals < 2U) {
            throw std::invalid_argument("pd3_fft needs at least two intervals");
        }
        if (substeps == 0U) {
            throw std::invalid_argument("pd3_fft needs at least one RK4 substep");
        }
        require_positive(sigma_minimum, "pd3_fft sigma minimum must be positive");
        if (!(sigma_maximum > sigma_minimum)) {
            throw std::invalid_argument("pd3_fft sigma maximum must exceed the minimum");
        }
        require_positive(trust_radius, "pd3_fft trust radius must be positive");
        require_positive(sigma_trust_radius, "pd3_fft sigma trust radius must be positive");
        require_positive(virtual_l1_weight, "pd3_fft virtual L1 weight must be positive");
        require_nonnegative(virtual_quadratic_weight, "pd3_fft virtual quadratic weight");
        require_positive(virtual_epigraph_regularisation, "pd3_fft epigraph regularisation");
        require_nonnegative(fuel_weight, "pd3_fft fuel weight");
        require_nonnegative(time_weight, "pd3_fft time weight");
        if (fuel_weight == 0.0 && time_weight == 0.0) {
            throw std::invalid_argument("pd3_fft needs a fuel or time objective");
        }
        require_nonnegative(sigma_tracking_weight, "pd3_fft sigma tracking weight");
        for (const auto value : state_tracking_weights) {
            require_nonnegative(value, "pd3_fft state tracking weight");
        }
        for (const auto value : control_tracking_weights) {
            require_nonnegative(value, "pd3_fft control tracking weight");
        }
        for (const auto value : state_trust_scales) {
            require_positive(value, "pd3_fft state trust scale");
        }
        for (const auto value : control_trust_scales) {
            require_positive(value, "pd3_fft control trust scale");
        }
    }
};

struct PoweredDescent3DofFreeTimeLayout {
    explicit PoweredDescent3DofFreeTimeLayout(std::size_t interval_count)
        : intervals(interval_count) {}

    std::size_t intervals{0U};

    // Variables: states | controls | sigma | virtual | epigraph
    [[nodiscard]] std::size_t state_count() const noexcept { return (intervals + 1U) * 7U; }
    [[nodiscard]] std::size_t control_count() const noexcept { return intervals * 4U; }
    [[nodiscard]] std::size_t virtual_count() const noexcept { return intervals * 7U; }
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

    // Scalar rows: initial | dynamics | terminal | virtual epigraph | tilt
    [[nodiscard]] free_time::Range initial_rows() const noexcept { return {0U, 7U}; }
    [[nodiscard]] free_time::Range dynamics_rows() const noexcept {
        return {initial_rows().stop(), 7U * intervals};
    }
    [[nodiscard]] free_time::Range terminal_rows() const noexcept {
        return {dynamics_rows().stop(), 6U};
    }
    [[nodiscard]] free_time::Range virtual_epigraph_rows() const noexcept {
        return {terminal_rows().stop(), 14U * intervals};
    }
    [[nodiscard]] free_time::Range tilt_rows() const noexcept {
        return {virtual_epigraph_rows().stop(), intervals};
    }
    [[nodiscard]] std::size_t scalar_rows() const noexcept { return tilt_rows().stop(); }

    // Affine cone rows: thrust | glide | stage trust | terminal trust
    [[nodiscard]] free_time::Range thrust_rows() const noexcept { return {0U, 4U * intervals}; }
    [[nodiscard]] free_time::Range glide_rows() const noexcept {
        return {thrust_rows().stop(), 3U * (intervals + 1U)};
    }
    [[nodiscard]] free_time::Range stage_trust_rows() const noexcept {
        return {glide_rows().stop(), 12U * intervals};
    }
    [[nodiscard]] free_time::Range terminal_trust_rows() const noexcept {
        return {stage_trust_rows().stop(), 8U};
    }
    [[nodiscard]] std::size_t affine_rows() const noexcept { return terminal_trust_rows().stop(); }

    [[nodiscard]] free_time::Range state(std::size_t node) const {
        if (node > intervals) {
            throw std::out_of_range("pd3_fft state node is outside the trajectory");
        }
        return {7U * node, 7U};
    }
    [[nodiscard]] free_time::Range control(std::size_t interval) const {
        check(interval);
        return {control_offset() + 4U * interval, 4U};
    }
    [[nodiscard]] free_time::Range virtual_control(std::size_t interval) const {
        check(interval);
        return {virtual_offset() + 7U * interval, 7U};
    }
    [[nodiscard]] free_time::Range epigraph(std::size_t interval) const {
        check(interval);
        return {epigraph_offset() + 7U * interval, 7U};
    }

  private:
    void check(std::size_t interval) const {
        if (interval >= intervals) {
            throw std::out_of_range("pd3_fft interval is outside the trajectory");
        }
    }
};

struct PoweredDescent3DofFreeTimeDecision {
    std::vector<dynamics::PoweredDescentState> states{};
    std::vector<dynamics::PoweredDescentControl> controls{};
    double sigma{0.0};
    std::vector<dynamics::PoweredDescentState> virtual_controls{};
};

class PoweredDescent3DofFreeTimeSubproblem {
  public:
    using State = dynamics::PoweredDescentState;
    using Control = dynamics::PoweredDescentControl;
    using Model = dynamics::PoweredDescent3DofModel;

    explicit PoweredDescent3DofFreeTimeSubproblem(
        Model model = Model{},
        PoweredDescent3DofFreeTimeConfig config = PoweredDescent3DofFreeTimeConfig{}
    )
        : model_(std::move(model)), config_(config), layout_(config_.intervals) {
        config_.validate();
        build_structure();
    }

    [[nodiscard]] const Model& model() const noexcept { return model_; }
    [[nodiscard]] const PoweredDescent3DofFreeTimeConfig& config() const noexcept {
        return config_;
    }
    [[nodiscard]] const PoweredDescent3DofFreeTimeLayout& layout() const noexcept {
        return layout_;
    }
    [[nodiscard]] const core::FixedStructure& structure() const noexcept { return structure_; }

    [[nodiscard]] core::NumericValues values(
        const std::vector<State>& reference_states,
        const std::vector<Control>& reference_controls,
        double reference_sigma,
        const State& initial_state,
        const std::array<double, 3U>& target_position,
        const std::array<double, 3U>& target_velocity,
        double trust_radius = -1.0,
        double sigma_trust_radius = -1.0
    ) const {
        validate_reference(reference_states, reference_controls, reference_sigma);
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
        fill_initial(result, initial_state);
        fill_dynamics(result, reference_states, reference_controls, reference_sigma);
        fill_terminal(result, target_position, target_velocity);
        fill_virtual_epigraph(result);
        fill_tilt(result);
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
            const auto range = layout_.state(node);
            std::copy(states[node].begin(), states[node].end(), decision.begin() + range.start);
        }
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto range = layout_.control(interval);
            std::copy(
                controls[interval].begin(),
                controls[interval].end(),
                decision.begin() + range.start
            );
        }
        decision[layout_.sigma_index()] = sigma;
        return decision;
    }

    [[nodiscard]] PoweredDescent3DofFreeTimeDecision decode(const std::vector<double>& decision)
        const {
        if (decision.size() != layout_.variables()) {
            throw std::invalid_argument("pd3_fft decision vector has the wrong size");
        }
        PoweredDescent3DofFreeTimeDecision decoded{};
        decoded.states.resize(layout_.intervals + 1U);
        decoded.controls.resize(layout_.intervals);
        decoded.virtual_controls.resize(layout_.intervals);
        for (std::size_t node = 0; node <= layout_.intervals; ++node) {
            const auto range = layout_.state(node);
            std::copy_n(
                decision.begin() + static_cast<std::ptrdiff_t>(range.start),
                7U,
                decoded.states[node].begin()
            );
        }
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            std::copy_n(
                decision.begin() + static_cast<std::ptrdiff_t>(layout_.control(interval).start),
                4U,
                decoded.controls[interval].begin()
            );
            std::copy_n(
                decision.begin()
                    + static_cast<std::ptrdiff_t>(layout_.virtual_control(interval).start),
                7U,
                decoded.virtual_controls[interval].begin()
            );
        }
        decoded.sigma = decision[layout_.sigma_index()];
        return decoded;
    }

    /// Nonlinear time-dilated replay defect `max_k |F(x_k,u_k,sigma) - x_{k+1}|_inf`.
    [[nodiscard]] double replay_defect(
        const std::vector<State>& states,
        const std::vector<Control>& controls,
        double sigma
    ) const {
        validate_reference(states, controls, sigma);
        double worst = 0.0;
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto next = time_dilated_step<7U, 4U>(
                model_, states[interval], controls[interval], sigma, config_.d_tau(),
                config_.substeps
            );
            for (std::size_t component = 0; component < 7U; ++component) {
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
            throw std::invalid_argument("pd3_fft decision vector has the wrong size");
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
    PoweredDescent3DofFreeTimeConfig config_{};
    PoweredDescent3DofFreeTimeLayout layout_;
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
        for (std::size_t component = 0; component < 7U; ++component) {
            a.insert({layout_.initial_rows().start + component, component});
        }
        const auto sigma = layout_.sigma_index();
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto row_start = layout_.dynamics_rows().start + 7U * interval;
            const auto state = layout_.state(interval);
            const auto next_state = layout_.state(interval + 1U);
            const auto control = layout_.control(interval);
            const auto virtual_control = layout_.virtual_control(interval);
            for (std::size_t row_offset = 0; row_offset < 7U; ++row_offset) {
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
        }
        const auto terminal_state = layout_.state(layout_.intervals);
        for (std::size_t component = 0; component < 6U; ++component) {
            a.insert({layout_.terminal_rows().start + component, terminal_state.start + component});
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
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto control = layout_.control(interval);
            const auto tilt_row = layout_.tilt_rows().start + interval;
            a.insert({tilt_row, control.start + 2U});
            a.insert({tilt_row, control.start + 3U});
            const auto thrust_start = layout_.thrust_rows().start + 4U * interval;
            for (std::size_t component = 0; component < 4U; ++component) {
                f.insert({thrust_start + component, control.start + component});
            }
            const auto state = layout_.state(interval);
            const auto trust_start = layout_.stage_trust_rows().start + 12U * interval;
            for (std::size_t component = 0; component < 7U; ++component) {
                f.insert({trust_start + component, state.start + component});
            }
            for (std::size_t component = 0; component < 4U; ++component) {
                f.insert({trust_start + 7U + component, control.start + component});
            }
        }
        for (std::size_t node = 0; node <= layout_.intervals; ++node) {
            const auto state = layout_.state(node);
            const auto glide_start = layout_.glide_rows().start + 3U * node;
            for (std::size_t component = 0; component < 3U; ++component) {
                f.insert({glide_start + component, state.start + component});
            }
        }
        for (std::size_t component = 0; component < 7U; ++component) {
            f.insert({layout_.terminal_trust_rows().start + component,
                      terminal_state.start + component});
        }
        std::vector<ConeBlockDescriptor> cones{};
        free_time::append_uniform_cones(cones, layout_.thrust_rows().start, layout_.intervals, 4U, 2);
        free_time::append_uniform_cones(
            cones, layout_.glide_rows().start, layout_.intervals + 1U, 3U, 1
        );
        free_time::append_uniform_cones(
            cones, layout_.stage_trust_rows().start, layout_.intervals, 12U, 10
        );
        cones.push_back(ConeBlockDescriptor{
            ConeKind::second_order,
            static_cast<Index>(layout_.terminal_trust_rows().start),
            6,
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
            for (std::size_t component = 0; component < 7U; ++component) {
                quadratic_index_.set(
                    quadratic_values_, state.start + component, state.start + component,
                    config_.state_tracking_weights[component]
                );
            }
        }
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto control = layout_.control(interval);
            for (std::size_t component = 0; component < 4U; ++component) {
                quadratic_index_.set(
                    quadratic_values_, control.start + component, control.start + component,
                    config_.control_tracking_weights[component]
                );
            }
            const auto virtual_control = layout_.virtual_control(interval);
            const auto epigraph = layout_.epigraph(interval);
            for (std::size_t component = 0; component < 7U; ++component) {
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
            for (std::size_t component = 0; component < 7U; ++component) {
                values.linear_objective[range.start + component] =
                    -states[node][component] * config_.state_tracking_weights[component];
            }
        }
        // Fuel: fuel_weight * sigma * d_tau * sum(Gamma_k), linearised about the reference:
        //   sigma_bar * sum(Gamma_k) + sum(Gamma_bar_k) * (sigma - sigma_bar)  (+ const).
        double reference_throttle_sum = 0.0;
        for (std::size_t interval = 0; interval < controls.size(); ++interval) {
            const auto range = layout_.control(interval);
            for (std::size_t component = 0; component < 4U; ++component) {
                values.linear_objective[range.start + component] =
                    -controls[interval][component] * config_.control_tracking_weights[component];
            }
            values.linear_objective[range.start + 3U] +=
                config_.fuel_weight * config_.d_tau() * sigma;
            reference_throttle_sum += controls[interval][3U];
        }
        values.linear_objective[layout_.sigma_index()] =
            config_.time_weight + config_.fuel_weight * config_.d_tau() * reference_throttle_sum
            - sigma * config_.sigma_tracking_weight;
        for (std::size_t variable = layout_.epigraph_offset(); variable < layout_.variables();
             ++variable) {
            values.linear_objective[variable] = config_.virtual_l1_weight;
        }
    }

    void fill_initial(core::NumericValues& values, const State& initial) const {
        for (std::size_t component = 0; component < 7U; ++component) {
            const auto row = layout_.initial_rows().start + component;
            scalar_index_.set(values.scalar_constraint, row, component, 1.0);
            values.scalar_lower[row] = initial[component];
            values.scalar_upper[row] = initial[component];
        }
    }

    void fill_dynamics(
        core::NumericValues& values,
        const std::vector<State>& states,
        const std::vector<Control>& controls,
        double sigma
    ) const {
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto lin = linearise_time_dilated_flow<7U, 4U>(
                model_, states[interval], controls[interval], sigma, config_.d_tau(),
                config_.substeps
            );
            const auto row_start = layout_.dynamics_rows().start + 7U * interval;
            const auto state = layout_.state(interval);
            const auto next_state = layout_.state(interval + 1U);
            const auto control = layout_.control(interval);
            const auto virtual_control = layout_.virtual_control(interval);
            for (std::size_t row_offset = 0; row_offset < 7U; ++row_offset) {
                const auto row = row_start + row_offset;
                for (std::size_t column = 0; column < 7U; ++column) {
                    scalar_index_.set(
                        values.scalar_constraint, row, state.start + column,
                        -lin.state[row_offset * 7U + column]
                    );
                }
                scalar_index_.set(values.scalar_constraint, row, next_state.start + row_offset, 1.0);
                for (std::size_t column = 0; column < 4U; ++column) {
                    scalar_index_.set(
                        values.scalar_constraint, row, control.start + column,
                        -lin.control[row_offset * 4U + column]
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

    void fill_terminal(
        core::NumericValues& values,
        const std::array<double, 3U>& position,
        const std::array<double, 3U>& velocity
    ) const {
        const auto final_state = layout_.state(layout_.intervals);
        for (std::size_t component = 0; component < 6U; ++component) {
            const auto row = layout_.terminal_rows().start + component;
            scalar_index_.set(values.scalar_constraint, row, final_state.start + component, 1.0);
            const auto target = component < 3U ? position[component] : velocity[component - 3U];
            values.scalar_lower[row] = target;
            values.scalar_upper[row] = target;
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

    void fill_tilt(core::NumericValues& values) const {
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto row = layout_.tilt_rows().start + interval;
            const auto control = layout_.control(interval);
            // cos(tilt) * Gamma - T_z <= 0
            scalar_index_.set(values.scalar_constraint, row, control.start + 2U, -1.0);
            scalar_index_.set(
                values.scalar_constraint, row, control.start + 3U, model_.config().tilt_cosine()
            );
            values.scalar_upper[row] = 0.0;
        }
    }

    void fill_cones(
        core::NumericValues& values,
        const std::vector<State>& states,
        const std::vector<Control>& controls,
        double radius
    ) const {
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto control = layout_.control(interval);
            const auto thrust_start = layout_.thrust_rows().start + 4U * interval;
            for (std::size_t component = 0; component < 4U; ++component) {
                affine_index_.set(
                    values.affine_cone, thrust_start + component, control.start + component, 1.0
                );
            }
            const auto state = layout_.state(interval);
            const auto trust_start = layout_.stage_trust_rows().start + 12U * interval;
            for (std::size_t component = 0; component < 7U; ++component) {
                const auto scale = config_.state_trust_scales[component];
                affine_index_.set(
                    values.affine_cone, trust_start + component, state.start + component, scale
                );
                values.affine_offset[trust_start + component] =
                    -scale * states[interval][component];
            }
            for (std::size_t component = 0; component < 4U; ++component) {
                const auto row = trust_start + 7U + component;
                const auto scale = config_.control_trust_scales[component];
                affine_index_.set(values.affine_cone, row, control.start + component, scale);
                values.affine_offset[row] = -scale * controls[interval][component];
            }
            values.affine_offset[trust_start + 11U] = radius;
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
        }
        const auto final_state = layout_.state(layout_.intervals);
        const auto terminal_start = layout_.terminal_trust_rows().start;
        for (std::size_t component = 0; component < 7U; ++component) {
            const auto scale = config_.state_trust_scales[component];
            affine_index_.set(
                values.affine_cone, terminal_start + component, final_state.start + component,
                scale
            );
            values.affine_offset[terminal_start + component] =
                -scale * states.back()[component];
        }
        values.affine_offset[terminal_start + 7U] = radius;
    }

    void fill_bounds(core::NumericValues& values, double sigma, double sigma_radius) const {
        for (std::size_t node = 0; node <= layout_.intervals; ++node) {
            const auto state = layout_.state(node);
            values.variable_lower[state.start + 2U] = 0.0;
            values.variable_lower[state.start + 6U] = model_.config().minimum_mass;
        }
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto control = layout_.control(interval);
            values.variable_lower[control.start + 3U] = model_.config().minimum_sigma;
            values.variable_upper[control.start + 3U] = model_.config().maximum_thrust;
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
            throw std::invalid_argument("pd3_fft reference trajectory has the wrong horizon");
        }
        if (!std::isfinite(sigma) || sigma <= 0.0) {
            throw std::invalid_argument("pd3_fft reference sigma must be finite and positive");
        }
        for (const auto& state : states) {
            for (const auto value : state) {
                if (!std::isfinite(value)) {
                    throw std::invalid_argument("pd3_fft reference state must be finite");
                }
            }
            if (state[6U] <= 0.0) {
                throw std::invalid_argument("pd3_fft reference mass must be positive");
            }
        }
        for (const auto& control : controls) {
            for (const auto value : control) {
                if (!std::isfinite(value)) {
                    throw std::invalid_argument("pd3_fft reference control must be finite");
                }
            }
        }
    }
};

}  // namespace spacepdhcg::transcription
