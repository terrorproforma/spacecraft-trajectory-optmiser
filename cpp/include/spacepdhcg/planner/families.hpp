#pragma once

// Family adapters: build the frozen transcriptions from user-supplied planner
// values, produce the device-driver layout metadata, generate dynamics-consistent
// initial references, replay controls independently (RK4 / exact HCW ZOH), and
// evaluate nonlinear quality with the same normalisation the device SCvx driver
// uses.  Everything here is host-only so the CUDA executable and the CPU C ABI
// share one implementation.

#include "spacepdhcg/core/fixed_cqp.hpp"
#include "spacepdhcg/dynamics/hcw.hpp"
#include "spacepdhcg/dynamics/low_thrust_two_body.hpp"
#include "spacepdhcg/dynamics/powered_descent_3dof.hpp"
#include "spacepdhcg/dynamics/powered_descent_6dof.hpp"
#include "spacepdhcg/planner/problem.hpp"
#include "spacepdhcg/scvx/powered_descent_3dof_driver.hpp"
#include "spacepdhcg/transcription/discretisation.hpp"
#include "spacepdhcg/transcription/hcw_rendezvous.hpp"
#include "spacepdhcg/transcription/low_thrust.hpp"
#include "spacepdhcg/transcription/powered_descent_3dof.hpp"
#include "spacepdhcg/transcription/powered_descent_6dof.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace spacepdhcg::planner {

/// Host mirror of `spacepdhcg_cuda_dynamics_config` (model tag excluded).
struct DynamicsParameters {
    double step_seconds{0.0};
    double mean_motion{0.0};
    std::array<double, 3U> gravity{0.0, 0.0, 0.0};
    double gravitational_parameter{0.0};
    double thrust_to_acceleration{0.0};
    double mass_flow_coefficient{0.0};
    std::array<double, 3U> principal_inertia{0.0, 0.0, 0.0};
};

/// Host mirror of the physical fields in `spacepdhcg_cuda_scvx_numeric_update`.
struct PhysicalLimits {
    double maximum_thrust{0.0};
    double maximum_torque{0.0};
    double maximum_angular_rate{0.0};
    double tilt_cosine{0.0};
    double glide_slope_tangent{0.0};
    double minimum_radius{0.0};
};

struct LayoutInfo {
    std::size_t state_dimension{0U};
    std::size_t control_dimension{0U};
    std::size_t intervals{0U};
    std::size_t terminal_dimension{0U};
    std::size_t variables{0U};
    std::size_t scalar_rows{0U};
    std::size_t affine_rows{0U};
    std::size_t dynamics_row_start{0U};
    std::size_t terminal_row_start{0U};
    std::size_t radial_row_start{0U};
    std::size_t quaternion_row_start{0U};
    std::size_t stage_trust_row_start{0U};
    std::size_t stage_trust_stride{0U};
    std::size_t terminal_trust_row_start{0U};
    std::size_t virtual_variable_offset{0U};
    std::size_t epigraph_variable_offset{0U};
    std::vector<int> state_variables{};
    std::vector<int> control_variables{};
    std::vector<int> virtual_variables{};
    std::vector<int> state_positions{};
    std::vector<int> control_positions{};
    std::vector<int> next_positions{};
    std::vector<int> virtual_positions{};
    std::vector<int> quadratic_diagonal_positions{};
    std::vector<int> radial_positions{};
    std::vector<int> quaternion_positions{};
    std::vector<double> state_trust_scales{};
    std::vector<double> control_trust_scales{};
    double fuel_weight{0.0};
    double virtual_l1_weight{0.0};
};

struct Trajectory {
    std::vector<double> states{};    // (N+1) * nx, row-major by node
    std::vector<double> controls{};  // N * nu, row-major by interval
};

struct PathComponent {
    std::string name{};
    double normalised{0.0};
    double physical{0.0};
};

struct Evaluation {
    double objective{0.0};
    std::string objective_definition{};
    double path_violation{0.0};
    std::vector<PathComponent> path{};
    double terminal_residual{0.0};
    std::vector<double> terminal_errors{};  // physical |x_N - target| per fixed component
    double terminal_position_error{0.0};
    double terminal_velocity_error{0.0};
    double propellant_used{0.0};
    double final_mass{0.0};
};

class FamilyAdapter {
  public:
    FamilyAdapter(const FamilyAdapter&) = delete;
    FamilyAdapter& operator=(const FamilyAdapter&) = delete;
    virtual ~FamilyAdapter() = default;

    [[nodiscard]] virtual Family family() const noexcept = 0;
    [[nodiscard]] virtual const core::FixedStructure& structure() const noexcept = 0;
    [[nodiscard]] virtual const LayoutInfo& layout() const noexcept = 0;
    [[nodiscard]] virtual DynamicsParameters dynamics() const = 0;
    [[nodiscard]] virtual PhysicalLimits limits() const = 0;
    [[nodiscard]] virtual core::NumericValues values(
        const Trajectory& reference,
        double trust_radius
    ) const = 0;
    [[nodiscard]] virtual Trajectory initial_reference() const = 0;
    /// Independent replay from `initial` under piecewise-constant `controls`.
    /// Returns (N*substeps + 1) * nx flat states.
    [[nodiscard]] virtual std::vector<double> rollout(
        const std::vector<double>& initial,
        const std::vector<double>& controls,
        std::size_t substeps
    ) const = 0;
    /// Nonlinear quality of node states and controls (N+1 states, N controls).
    [[nodiscard]] virtual Evaluation evaluate(
        const std::vector<double>& states,
        const std::vector<double>& controls
    ) const = 0;
    /// Path violation only, for arbitrary (M+1 states, M controls) sequences.
    [[nodiscard]] virtual std::vector<PathComponent> path_components(
        const std::vector<double>& states,
        const std::vector<double>& controls
    ) const = 0;

    [[nodiscard]] const PlannerProblem& problem() const noexcept { return problem_; }

    /// Trust-scaled infinity-norm defect between two node-state sequences.
    [[nodiscard]] double scaled_state_defect(
        const std::vector<double>& states,
        const std::vector<double>& replay
    ) const {
        if (states.size() != replay.size()) {
            throw std::invalid_argument("state defect requires equal-length sequences");
        }
        const auto& info = layout();
        double result = 0.0;
        for (std::size_t index = 0U; index < states.size(); ++index) {
            const std::size_t component = index % info.state_dimension;
            const double scale = info.state_trust_scales.empty()
                ? 1.0
                : info.state_trust_scales[component];
            result = std::max(result, std::abs(states[index] - replay[index]) * scale);
        }
        return result;
    }

    /// Unscaled infinity-norm distance between two equal-length sequences.
    [[nodiscard]] static double infinity_distance(
        const std::vector<double>& left,
        const std::vector<double>& right
    ) {
        if (left.size() != right.size()) {
            throw std::invalid_argument("infinity distance requires equal-length sequences");
        }
        double result = 0.0;
        for (std::size_t index = 0U; index < left.size(); ++index) {
            result = std::max(result, std::abs(left[index] - right[index]));
        }
        return result;
    }

    /// Extract node states/controls from a canonical primal vector.
    [[nodiscard]] Trajectory decode(const std::vector<double>& primal) const {
        const auto& info = layout();
        if (primal.size() != info.variables) {
            throw std::invalid_argument("primal vector has the wrong length for this transcription");
        }
        Trajectory result{};
        result.states.reserve(info.state_variables.size());
        result.controls.reserve(info.control_variables.size());
        for (const int variable : info.state_variables) {
            result.states.push_back(primal[static_cast<std::size_t>(variable)]);
        }
        for (const int variable : info.control_variables) {
            result.controls.push_back(primal[static_cast<std::size_t>(variable)]);
        }
        return result;
    }

    /// Maximum absolute virtual control (trust-scaled) inside a primal vector.
    [[nodiscard]] double scaled_virtual_control(const std::vector<double>& primal) const {
        const auto& info = layout();
        double result = 0.0;
        for (std::size_t index = 0U; index < info.virtual_variables.size(); ++index) {
            const double scale = info.state_trust_scales.empty()
                ? 1.0
                : info.state_trust_scales[index % info.state_dimension];
            result = std::max(
                result,
                std::abs(primal[static_cast<std::size_t>(info.virtual_variables[index])]) * scale
            );
        }
        return result;
    }

  protected:
    explicit FamilyAdapter(PlannerProblem problem) : problem_(std::move(problem)) {}

    PlannerProblem problem_;

    template <std::size_t Size>
    [[nodiscard]] static std::array<double, Size> to_array(
        const std::vector<double>& values,
        std::size_t offset = 0U
    ) {
        if (values.size() < offset + Size) {
            throw std::invalid_argument("vector is too short for the requested array view");
        }
        std::array<double, Size> result{};
        std::copy_n(values.begin() + static_cast<std::ptrdiff_t>(offset), Size, result.begin());
        return result;
    }

    template <typename State>
    [[nodiscard]] static std::vector<State> unflatten_states(
        const std::vector<double>& flat,
        std::size_t count
    ) {
        constexpr std::size_t dimension = std::tuple_size<State>::value;
        if (flat.size() != count * dimension) {
            throw std::invalid_argument("state sequence has the wrong length");
        }
        std::vector<State> result(count);
        for (std::size_t node = 0U; node < count; ++node) {
            std::copy_n(
                flat.begin() + static_cast<std::ptrdiff_t>(node * dimension),
                dimension,
                result[node].begin()
            );
        }
        return result;
    }

    template <typename Row>
    [[nodiscard]] static std::vector<double> flatten(const std::vector<Row>& rows) {
        std::vector<double> result;
        result.reserve(rows.size() * std::tuple_size<Row>::value);
        for (const auto& row : rows) {
            result.insert(result.end(), row.begin(), row.end());
        }
        return result;
    }

    [[nodiscard]] static std::map<std::pair<std::size_t, std::size_t>, int> positions(
        const core::CscPattern& pattern
    ) {
        std::map<std::pair<std::size_t, std::size_t>, int> result;
        for (Index column = 0; column < pattern.columns; ++column) {
            const auto begin = pattern.offsets[static_cast<std::size_t>(column)];
            const auto end = pattern.offsets[static_cast<std::size_t>(column) + 1U];
            for (Index slot = begin; slot < end; ++slot) {
                result[{
                    static_cast<std::size_t>(pattern.indices[static_cast<std::size_t>(slot)]),
                    static_cast<std::size_t>(column),
                }] = slot;
            }
        }
        return result;
    }

    template <typename StateRange, typename ControlRange, typename VirtualRange>
    static void fill_maps(
        LayoutInfo& info,
        const core::FixedStructure& structure,
        StateRange state_range,
        ControlRange control_range,
        VirtualRange virtual_range,
        bool has_virtual
    ) {
        const auto lookup = positions(structure.scalar_constraint);
        const auto q_lookup = positions(structure.quadratic);
        const std::size_t intervals = info.intervals;
        const std::size_t nx = info.state_dimension;
        const std::size_t nu = info.control_dimension;
        info.state_variables.reserve((intervals + 1U) * nx);
        info.control_variables.reserve(intervals * nu);
        for (std::size_t node = 0U; node <= intervals; ++node) {
            const std::size_t start = state_range(node);
            for (std::size_t index = 0U; index < nx; ++index) {
                info.state_variables.push_back(static_cast<int>(start + index));
            }
        }
        for (std::size_t interval = 0U; interval < intervals; ++interval) {
            const std::size_t start = control_range(interval);
            for (std::size_t index = 0U; index < nu; ++index) {
                info.control_variables.push_back(static_cast<int>(start + index));
            }
            if (has_virtual) {
                const std::size_t virtual_start = virtual_range(interval);
                for (std::size_t index = 0U; index < nx; ++index) {
                    info.virtual_variables.push_back(static_cast<int>(virtual_start + index));
                }
            }
        }
        for (std::size_t interval = 0U; interval < intervals; ++interval) {
            const std::size_t current = state_range(interval);
            const std::size_t next = state_range(interval + 1U);
            const std::size_t control = control_range(interval);
            const std::size_t virtual_start = has_virtual ? virtual_range(interval) : 0U;
            for (std::size_t row = 0U; row < nx; ++row) {
                const std::size_t matrix_row = info.dynamics_row_start + interval * nx + row;
                for (std::size_t column = 0U; column < nx; ++column) {
                    info.state_positions.push_back(lookup.at({matrix_row, current + column}));
                }
                for (std::size_t column = 0U; column < nu; ++column) {
                    info.control_positions.push_back(lookup.at({matrix_row, control + column}));
                }
                info.next_positions.push_back(lookup.at({matrix_row, next + row}));
                if (has_virtual) {
                    info.virtual_positions.push_back(
                        lookup.at({matrix_row, virtual_start + row})
                    );
                }
            }
        }
        for (const int variable : info.state_variables) {
            info.quadratic_diagonal_positions.push_back(q_lookup.at({
                static_cast<std::size_t>(variable),
                static_cast<std::size_t>(variable),
            }));
        }
        for (const int variable : info.control_variables) {
            info.quadratic_diagonal_positions.push_back(q_lookup.at({
                static_cast<std::size_t>(variable),
                static_cast<std::size_t>(variable),
            }));
        }
        info.variables = static_cast<std::size_t>(structure.variables());
        info.scalar_rows = static_cast<std::size_t>(structure.scalar_rows());
        info.affine_rows = static_cast<std::size_t>(structure.affine_rows());
        if (has_virtual && !info.virtual_variables.empty()) {
            info.virtual_variable_offset = static_cast<std::size_t>(info.virtual_variables.front());
            info.epigraph_variable_offset =
                info.virtual_variable_offset + info.virtual_variables.size();
        }
    }

    [[nodiscard]] static std::vector<double> repeat_controls(
        const std::vector<double>& controls,
        std::size_t control_dimension,
        std::size_t substeps
    ) {
        std::vector<double> result;
        result.reserve(controls.size() * substeps);
        const std::size_t intervals = controls.size() / control_dimension;
        for (std::size_t interval = 0U; interval < intervals; ++interval) {
            for (std::size_t repeat = 0U; repeat < substeps; ++repeat) {
                result.insert(
                    result.end(),
                    controls.begin() + static_cast<std::ptrdiff_t>(interval * control_dimension),
                    controls.begin()
                        + static_cast<std::ptrdiff_t>((interval + 1U) * control_dimension)
                );
            }
        }
        return result;
    }

    void require_trajectory_shape(
        const std::vector<double>& states,
        const std::vector<double>& controls
    ) const {
        const auto& info = layout();
        if (states.size() != (info.intervals + 1U) * info.state_dimension) {
            throw std::invalid_argument("node states must have (intervals + 1) * nx entries");
        }
        if (controls.size() != info.intervals * info.control_dimension) {
            throw std::invalid_argument("controls must have intervals * nu entries");
        }
    }

    [[nodiscard]] double clamp_thrust(double requested) const {
        return std::clamp(
            requested, problem_.vehicle.minimum_thrust, problem_.vehicle.maximum_thrust
        );
    }
};

// --------------------------------------------------------------------------
// HCW rendezvous
// --------------------------------------------------------------------------

class HcwAdapter final : public FamilyAdapter {
  public:
    explicit HcwAdapter(PlannerProblem problem)
        : FamilyAdapter(std::move(problem)), subproblem_(make_config(problem_)) {
        const auto& layout = subproblem_.layout();
        info_.state_dimension = 6U;
        info_.control_dimension = 3U;
        info_.intervals = problem_.intervals;
        info_.terminal_dimension = 6U;
        info_.dynamics_row_start = layout.dynamics_row();
        info_.terminal_row_start = layout.terminal_row();
        fill_maps(
            info_,
            subproblem_.structure(),
            [&layout](std::size_t node) { return layout.state_index(node, 0U); },
            [&layout](std::size_t interval) { return layout.control_index(interval, 0U); },
            [](std::size_t) { return std::size_t{0U}; },
            false
        );
    }

    [[nodiscard]] Family family() const noexcept override { return Family::hcw; }
    [[nodiscard]] const core::FixedStructure& structure() const noexcept override {
        return subproblem_.structure();
    }
    [[nodiscard]] const LayoutInfo& layout() const noexcept override { return info_; }

    [[nodiscard]] DynamicsParameters dynamics() const override {
        DynamicsParameters result{};
        result.step_seconds = problem_.step_seconds;
        result.mean_motion = problem_.vehicle.mean_motion;
        return result;
    }

    [[nodiscard]] PhysicalLimits limits() const override { return PhysicalLimits{}; }

    [[nodiscard]] core::NumericValues values(
        const Trajectory& reference,
        double trust_radius
    ) const override {
        static_cast<void>(trust_radius);
        require_trajectory_shape(reference.states, reference.controls);
        return subproblem_.values(
            to_array<6U>(problem_.initial_state), to_array<6U>(problem_.target_state)
        );
    }

    [[nodiscard]] Trajectory initial_reference() const override {
        std::vector<double> controls(problem_.intervals * 3U, 0.0);
        if (problem_.warm_start.has_value()) {
            controls = problem_.warm_start->controls;
        }
        return Trajectory{rollout(problem_.initial_state, controls, 1U), controls};
    }

    [[nodiscard]] std::vector<double> rollout(
        const std::vector<double>& initial,
        const std::vector<double>& controls,
        std::size_t substeps
    ) const override {
        if (substeps == 0U || initial.size() != 6U || controls.size() % 3U != 0U) {
            throw std::invalid_argument("HCW rollout arguments are malformed");
        }
        const std::size_t intervals = controls.size() / 3U;
        const auto discrete = substeps == 1U
            ? subproblem_.discrete_dynamics()
            : dynamics::discretise_hcw(
                  problem_.vehicle.mean_motion,
                  problem_.step_seconds / static_cast<double>(substeps)
              );
        std::vector<double> result;
        result.reserve((intervals * substeps + 1U) * 6U);
        dynamics::HcwState state = to_array<6U>(initial);
        result.insert(result.end(), state.begin(), state.end());
        for (std::size_t interval = 0U; interval < intervals; ++interval) {
            const auto control = to_array<3U>(controls, interval * 3U);
            for (std::size_t substep = 0U; substep < substeps; ++substep) {
                state = dynamics::hcw_step(discrete, state, control);
                result.insert(result.end(), state.begin(), state.end());
            }
        }
        return result;
    }

    [[nodiscard]] std::vector<PathComponent> path_components(
        const std::vector<double>& states,
        const std::vector<double>& controls
    ) const override {
        static_cast<void>(states);
        const double bound = problem_.vehicle.maximum_acceleration;
        double norm_violation = 0.0;
        double box_violation = 0.0;
        for (std::size_t interval = 0U; interval * 3U < controls.size(); ++interval) {
            const auto control = to_array<3U>(controls, interval * 3U);
            const double norm = std::sqrt(
                control[0U] * control[0U] + control[1U] * control[1U] + control[2U] * control[2U]
            );
            norm_violation = std::max(norm_violation, norm - bound);
            for (const double component : control) {
                box_violation = std::max(box_violation, std::abs(component) - bound);
            }
        }
        const double physical = problem_.vehicle.acceleration_norm_bound
            ? std::max(norm_violation, 0.0)
            : std::max(box_violation, 0.0);
        return {PathComponent{
            problem_.vehicle.acceleration_norm_bound ? "acceleration_norm" : "acceleration_box",
            physical / bound,
            physical,
        }};
    }

    [[nodiscard]] Evaluation evaluate(
        const std::vector<double>& states,
        const std::vector<double>& controls
    ) const override {
        require_trajectory_shape(states, controls);
        Evaluation result{};
        result.objective_definition = "0.5 * sum_k |a_k|^2 (m^2/s^4)";
        for (const double value : controls) {
            result.objective += 0.5 * value * value;
        }
        result.path = path_components(states, controls);
        result.path_violation = result.path.front().normalised;
        const std::size_t offset = problem_.intervals * 6U;
        result.terminal_errors.resize(6U);
        for (std::size_t component = 0U; component < 6U; ++component) {
            const double error = std::abs(states[offset + component] - problem_.target_state[component]);
            result.terminal_errors[component] = error;
            result.terminal_residual = std::max(result.terminal_residual, error);
            if (component < 3U) {
                result.terminal_position_error = std::max(result.terminal_position_error, error);
            } else {
                result.terminal_velocity_error = std::max(result.terminal_velocity_error, error);
            }
        }
        for (std::size_t interval = 0U; interval < problem_.intervals; ++interval) {
            const auto control = to_array<3U>(controls, interval * 3U);
            result.propellant_used += problem_.step_seconds * std::sqrt(
                control[0U] * control[0U] + control[1U] * control[1U] + control[2U] * control[2U]
            );
        }
        result.final_mass = 0.0;
        return result;
    }

  private:
    transcription::HcwRendezvousCqp subproblem_;
    LayoutInfo info_{};

    [[nodiscard]] static transcription::HcwRendezvousConfig make_config(
        const PlannerProblem& problem
    ) {
        transcription::HcwRendezvousConfig config{};
        config.intervals = problem.intervals;
        config.step_seconds = problem.step_seconds;
        config.mean_motion = problem.vehicle.mean_motion;
        config.maximum_acceleration = problem.vehicle.maximum_acceleration;
        config.control_set = problem.vehicle.acceleration_norm_bound
            ? transcription::HcwControlSet::second_order_cone
            : transcription::HcwControlSet::box;
        std::copy_n(problem.weights.state_weights.begin(), 6U, config.state_weights.begin());
        std::copy_n(problem.weights.control_weights.begin(), 3U, config.control_weights.begin());
        return config;
    }
};

// --------------------------------------------------------------------------
// 3-DoF powered descent
// --------------------------------------------------------------------------

class PoweredDescent3DofAdapter final : public FamilyAdapter {
  public:
    explicit PoweredDescent3DofAdapter(PlannerProblem problem)
        : FamilyAdapter(std::move(problem)),
          subproblem_(make_model(problem_), make_config(problem_)) {
        const auto& layout = subproblem_.layout();
        info_.state_dimension = 7U;
        info_.control_dimension = 4U;
        info_.intervals = problem_.intervals;
        info_.terminal_dimension = 6U;
        info_.dynamics_row_start = layout.dynamics_rows().start;
        info_.terminal_row_start = layout.terminal_rows().start;
        info_.stage_trust_row_start = layout.stage_trust_cone_rows().start;
        info_.stage_trust_stride = 12U;
        info_.terminal_trust_row_start = layout.terminal_trust_cone_rows().start;
        info_.state_trust_scales = problem_.weights.state_trust_scales;
        info_.control_trust_scales = problem_.weights.control_trust_scales;
        info_.fuel_weight = problem_.weights.fuel_weight;
        info_.virtual_l1_weight = problem_.weights.virtual_l1_weight;
        fill_maps(
            info_,
            subproblem_.structure(),
            [&layout](std::size_t node) { return layout.state(node).start; },
            [&layout](std::size_t interval) { return layout.control(interval).start; },
            [&layout](std::size_t interval) { return layout.virtual_control(interval).start; },
            true
        );
    }

    [[nodiscard]] Family family() const noexcept override {
        return Family::powered_descent_3dof;
    }
    [[nodiscard]] const core::FixedStructure& structure() const noexcept override {
        return subproblem_.structure();
    }
    [[nodiscard]] const LayoutInfo& layout() const noexcept override { return info_; }

    [[nodiscard]] DynamicsParameters dynamics() const override {
        DynamicsParameters result{};
        result.step_seconds = problem_.step_seconds;
        result.gravity = problem_.vehicle.gravity;
        result.mass_flow_coefficient = problem_.vehicle.mass_flow_coefficient;
        return result;
    }

    [[nodiscard]] PhysicalLimits limits() const override {
        PhysicalLimits result{};
        const auto& config = subproblem_.model().config();
        result.maximum_thrust = config.maximum_thrust;
        result.tilt_cosine = config.tilt_cosine();
        result.glide_slope_tangent = config.glide_slope_tangent();
        return result;
    }

    [[nodiscard]] core::NumericValues values(
        const Trajectory& reference,
        double trust_radius
    ) const override {
        require_trajectory_shape(reference.states, reference.controls);
        return subproblem_.values(
            unflatten_states<dynamics::PoweredDescentState>(reference.states, problem_.intervals + 1U),
            unflatten_states<dynamics::PoweredDescentControl>(reference.controls, problem_.intervals),
            to_array<7U>(problem_.initial_state),
            to_array<3U>(problem_.target_state, 0U),
            to_array<3U>(problem_.target_state, 3U),
            trust_radius
        );
    }

    [[nodiscard]] Trajectory initial_reference() const override {
        const auto initial = to_array<7U>(problem_.initial_state);
        std::vector<double> controls;
        if (problem_.warm_start.has_value()) {
            controls = problem_.warm_start->controls;
        } else {
            try {
                const auto reference = scvx::make_native_powered_descent_reference(
                    subproblem_.model(),
                    initial,
                    to_array<3U>(problem_.target_state, 0U),
                    to_array<3U>(problem_.target_state, 3U),
                    problem_.intervals,
                    problem_.step_seconds
                );
                controls = flatten(reference.second);
            } catch (const std::exception&) {
                controls.clear();
            }
            if (controls.empty()) {
                const double gravity = std::sqrt(
                    problem_.vehicle.gravity[0U] * problem_.vehicle.gravity[0U]
                    + problem_.vehicle.gravity[1U] * problem_.vehicle.gravity[1U]
                    + problem_.vehicle.gravity[2U] * problem_.vehicle.gravity[2U]
                );
                const double hover = clamp_thrust(initial[6U] * gravity);
                for (std::size_t interval = 0U; interval < problem_.intervals; ++interval) {
                    controls.insert(controls.end(), {0.0, 0.0, hover, hover});
                }
            }
        }
        return Trajectory{rollout(problem_.initial_state, controls, 1U), controls};
    }

    [[nodiscard]] std::vector<double> rollout(
        const std::vector<double>& initial,
        const std::vector<double>& controls,
        std::size_t substeps
    ) const override {
        if (substeps == 0U || initial.size() != 7U || controls.size() % 4U != 0U) {
            throw std::invalid_argument("3-DoF rollout arguments are malformed");
        }
        const auto& model = subproblem_.model();
        const double step = problem_.step_seconds / static_cast<double>(substeps);
        const std::size_t intervals = controls.size() / 4U;
        std::vector<double> result;
        result.reserve((intervals * substeps + 1U) * 7U);
        auto state = to_array<7U>(initial);
        result.insert(result.end(), state.begin(), state.end());
        for (std::size_t interval = 0U; interval < intervals; ++interval) {
            const auto control = to_array<4U>(controls, interval * 4U);
            for (std::size_t substep = 0U; substep < substeps; ++substep) {
                state = model.rk4_step(state, control, step);
                result.insert(result.end(), state.begin(), state.end());
            }
        }
        return result;
    }

    [[nodiscard]] std::vector<PathComponent> path_components(
        const std::vector<double>& states,
        const std::vector<double>& controls
    ) const override {
        const auto& model = subproblem_.model();
        const auto diagnostics = model.path_diagnostics(
            unflatten_states<dynamics::PoweredDescentState>(states, states.size() / 7U),
            unflatten_states<dynamics::PoweredDescentControl>(controls, controls.size() / 4U)
        );
        const double thrust = model.config().maximum_thrust;
        const double position_scale = std::max({
            info_.state_trust_scales[0U], info_.state_trust_scales[1U], info_.state_trust_scales[2U],
        });
        const double mass_scale = info_.state_trust_scales[6U];
        return {
            {"thrust_epigraph", diagnostics.thrust_epigraph / thrust, diagnostics.thrust_epigraph},
            {"throttle_lower", diagnostics.throttle_lower / thrust, diagnostics.throttle_lower},
            {"throttle_upper", diagnostics.throttle_upper / thrust, diagnostics.throttle_upper},
            {"tilt", diagnostics.tilt / thrust, diagnostics.tilt},
            {"minimum_mass", diagnostics.minimum_mass * mass_scale, diagnostics.minimum_mass},
            {"altitude", diagnostics.altitude * position_scale, diagnostics.altitude},
            {"glide_slope", diagnostics.glide_slope * position_scale, diagnostics.glide_slope},
        };
    }

    [[nodiscard]] Evaluation evaluate(
        const std::vector<double>& states,
        const std::vector<double>& controls
    ) const override {
        require_trajectory_shape(states, controls);
        Evaluation result{};
        result.objective_definition = "mean_k(sigma_k) / maximum_thrust (normalised fuel)";
        const double thrust = subproblem_.model().config().maximum_thrust;
        for (std::size_t interval = 0U; interval < problem_.intervals; ++interval) {
            result.objective += controls[interval * 4U + 3U]
                / (static_cast<double>(problem_.intervals) * thrust);
        }
        result.path = path_components(states, controls);
        for (const auto& component : result.path) {
            result.path_violation = std::max(result.path_violation, component.normalised);
        }
        terminal_errors(states, result, 6U, 7U, info_.state_trust_scales);
        result.final_mass = states[problem_.intervals * 7U + 6U];
        result.propellant_used = states[6U] - result.final_mass;
        return result;
    }

  private:
    transcription::PoweredDescent3DofSubproblem subproblem_;
    LayoutInfo info_{};

    [[nodiscard]] static dynamics::PoweredDescent3DofModel make_model(
        const PlannerProblem& problem
    ) {
        dynamics::PoweredDescent3DofConfig config{};
        config.gravity = problem.vehicle.gravity;
        config.mass_flow_coefficient = problem.vehicle.mass_flow_coefficient;
        config.minimum_mass = problem.vehicle.minimum_mass;
        config.maximum_thrust = problem.vehicle.maximum_thrust;
        config.minimum_sigma = problem.vehicle.minimum_thrust;
        config.maximum_tilt_radians = problem.vehicle.maximum_tilt_radians;
        config.glide_slope_radians = problem.vehicle.glide_slope_radians;
        return dynamics::PoweredDescent3DofModel{config};
    }

    [[nodiscard]] static transcription::PoweredDescentScvxConfig make_config(
        const PlannerProblem& problem
    ) {
        transcription::PoweredDescentScvxConfig config{};
        config.intervals = problem.intervals;
        config.step_seconds = problem.step_seconds;
        config.trust_radius = problem.solver.trust.initial_radius;
        config.virtual_l1_weight = problem.weights.virtual_l1_weight;
        config.virtual_quadratic_weight = problem.weights.virtual_quadratic_weight;
        config.virtual_epigraph_regularisation = problem.weights.virtual_epigraph_regularisation;
        config.fuel_weight = problem.weights.fuel_weight;
        config.discretisation = transcription::DiscretisationMethod::rk4_variational;
        std::copy_n(
            problem.weights.state_tracking_weights.begin(), 7U, config.state_tracking_weights.begin()
        );
        std::copy_n(
            problem.weights.control_tracking_weights.begin(),
            4U,
            config.control_tracking_weights.begin()
        );
        std::copy_n(problem.weights.state_trust_scales.begin(), 7U, config.state_trust_scales.begin());
        std::copy_n(
            problem.weights.control_trust_scales.begin(), 4U, config.control_trust_scales.begin()
        );
        return config;
    }

    void terminal_errors(
        const std::vector<double>& states,
        Evaluation& result,
        std::size_t fixed,
        std::size_t nx,
        const std::vector<double>& scales
    ) const {
        const std::size_t offset = problem_.intervals * nx;
        result.terminal_errors.resize(fixed);
        for (std::size_t component = 0U; component < fixed; ++component) {
            const double error =
                std::abs(states[offset + component] - problem_.target_state[component]);
            result.terminal_errors[component] = error;
            result.terminal_residual = std::max(result.terminal_residual, error * scales[component]);
            if (component < 3U) {
                result.terminal_position_error = std::max(result.terminal_position_error, error);
            } else if (component < 6U) {
                result.terminal_velocity_error = std::max(result.terminal_velocity_error, error);
            }
        }
    }
};

// --------------------------------------------------------------------------
// 6-DoF powered descent
// --------------------------------------------------------------------------

class PoweredDescent6DofAdapter final : public FamilyAdapter {
  public:
    explicit PoweredDescent6DofAdapter(PlannerProblem problem)
        : FamilyAdapter(std::move(problem)),
          subproblem_(make_model(problem_), make_config(problem_)) {
        const auto& layout = subproblem_.layout();
        info_.state_dimension = 14U;
        info_.control_dimension = 7U;
        info_.intervals = problem_.intervals;
        info_.terminal_dimension = 13U;
        info_.dynamics_row_start = layout.dynamics_rows().start;
        info_.terminal_row_start = layout.terminal_rows().start;
        info_.quaternion_row_start = layout.quaternion_rows().start;
        info_.stage_trust_row_start = layout.stage_trust_rows().start;
        info_.stage_trust_stride = 22U;
        info_.terminal_trust_row_start = layout.terminal_trust_rows().start;
        info_.state_trust_scales = problem_.weights.state_trust_scales;
        info_.control_trust_scales = problem_.weights.control_trust_scales;
        info_.fuel_weight = problem_.weights.fuel_weight;
        info_.virtual_l1_weight = problem_.weights.virtual_l1_weight;
        fill_maps(
            info_,
            subproblem_.structure(),
            [&layout](std::size_t node) { return layout.state(node).start; },
            [&layout](std::size_t interval) { return layout.control(interval).start; },
            [&layout](std::size_t interval) { return layout.virtual_control(interval).start; },
            true
        );
        const auto scalar_lookup = positions(subproblem_.structure().scalar_constraint);
        for (std::size_t node = 0U; node <= problem_.intervals; ++node) {
            for (std::size_t component = 0U; component < 4U; ++component) {
                info_.quaternion_positions.push_back(scalar_lookup.at({
                    info_.quaternion_row_start + node,
                    static_cast<std::size_t>(info_.state_variables[node * 14U + 6U + component]),
                }));
            }
        }
    }

    [[nodiscard]] Family family() const noexcept override {
        return Family::powered_descent_6dof;
    }
    [[nodiscard]] const core::FixedStructure& structure() const noexcept override {
        return subproblem_.structure();
    }
    [[nodiscard]] const LayoutInfo& layout() const noexcept override { return info_; }

    [[nodiscard]] DynamicsParameters dynamics() const override {
        DynamicsParameters result{};
        result.step_seconds = problem_.step_seconds;
        result.gravity = problem_.vehicle.gravity;
        result.mass_flow_coefficient = problem_.vehicle.mass_flow_coefficient;
        result.principal_inertia = problem_.vehicle.principal_inertia;
        return result;
    }

    [[nodiscard]] PhysicalLimits limits() const override {
        PhysicalLimits result{};
        const auto& config = subproblem_.model().config();
        result.maximum_thrust = config.maximum_thrust;
        result.maximum_torque = config.maximum_torque;
        result.maximum_angular_rate = config.maximum_angular_rate;
        result.tilt_cosine = config.tilt_cosine();
        result.glide_slope_tangent = config.glide_slope_tangent();
        return result;
    }

    [[nodiscard]] core::NumericValues values(
        const Trajectory& reference,
        double trust_radius
    ) const override {
        require_trajectory_shape(reference.states, reference.controls);
        return subproblem_.values(
            unflatten_states<dynamics::PoweredDescent6DofState>(
                reference.states, problem_.intervals + 1U
            ),
            unflatten_states<dynamics::PoweredDescent6DofControl>(
                reference.controls, problem_.intervals
            ),
            to_array<14U>(problem_.initial_state),
            to_array<14U>(problem_.target_state),
            trust_radius
        );
    }

    [[nodiscard]] Trajectory initial_reference() const override {
        std::vector<double> controls;
        if (problem_.warm_start.has_value()) {
            controls = problem_.warm_start->controls;
        } else {
            const double gravity = std::sqrt(
                problem_.vehicle.gravity[0U] * problem_.vehicle.gravity[0U]
                + problem_.vehicle.gravity[1U] * problem_.vehicle.gravity[1U]
                + problem_.vehicle.gravity[2U] * problem_.vehicle.gravity[2U]
            );
            const double hover = clamp_thrust(problem_.initial_state[13U] * gravity);
            for (std::size_t interval = 0U; interval < problem_.intervals; ++interval) {
                controls.insert(controls.end(), {0.0, 0.0, hover, 0.0, 0.0, 0.0, hover});
            }
        }
        return Trajectory{rollout(problem_.initial_state, controls, 1U), controls};
    }

    [[nodiscard]] std::vector<double> rollout(
        const std::vector<double>& initial,
        const std::vector<double>& controls,
        std::size_t substeps
    ) const override {
        if (substeps == 0U || initial.size() != 14U || controls.size() % 7U != 0U) {
            throw std::invalid_argument("6-DoF rollout arguments are malformed");
        }
        const auto& model = subproblem_.model();
        const double step = problem_.step_seconds / static_cast<double>(substeps);
        const std::size_t intervals = controls.size() / 7U;
        std::vector<double> result;
        result.reserve((intervals * substeps + 1U) * 14U);
        auto state = to_array<14U>(initial);
        result.insert(result.end(), state.begin(), state.end());
        for (std::size_t interval = 0U; interval < intervals; ++interval) {
            const auto control = to_array<7U>(controls, interval * 7U);
            for (std::size_t substep = 0U; substep < substeps; ++substep) {
                state = model.rk4_step(state, control, step);
                result.insert(result.end(), state.begin(), state.end());
            }
        }
        return result;
    }

    [[nodiscard]] std::vector<PathComponent> path_components(
        const std::vector<double>& states,
        const std::vector<double>& controls
    ) const override {
        const auto& model = subproblem_.model();
        const auto diagnostics = model.path_diagnostics(
            unflatten_states<dynamics::PoweredDescent6DofState>(states, states.size() / 14U),
            unflatten_states<dynamics::PoweredDescent6DofControl>(controls, controls.size() / 7U)
        );
        const auto& config = model.config();
        const double position_scale = std::max({
            info_.state_trust_scales[0U], info_.state_trust_scales[1U], info_.state_trust_scales[2U],
        });
        const double rate_scale = std::max({
            info_.state_trust_scales[10U], info_.state_trust_scales[11U],
            info_.state_trust_scales[12U],
        });
        const double mass_scale = info_.state_trust_scales[13U];
        const double thrust_physical = std::max({
            diagnostics.thrust_epigraph, diagnostics.throttle_lower, diagnostics.throttle_upper,
        });
        return {
            {"thrust", thrust_physical / config.maximum_thrust, thrust_physical},
            {"torque", diagnostics.torque / config.maximum_torque, diagnostics.torque},
            {"pointing", diagnostics.pointing / config.maximum_thrust, diagnostics.pointing},
            {"minimum_mass", diagnostics.minimum_mass * mass_scale, diagnostics.minimum_mass},
            {"altitude", diagnostics.altitude * position_scale, diagnostics.altitude},
            {"glide_slope", diagnostics.glide_slope * position_scale, diagnostics.glide_slope},
            {"angular_rate", diagnostics.angular_rate * rate_scale, diagnostics.angular_rate},
            {"quaternion_norm", diagnostics.quaternion_norm_error, diagnostics.quaternion_norm_error},
        };
    }

    [[nodiscard]] Evaluation evaluate(
        const std::vector<double>& states,
        const std::vector<double>& controls
    ) const override {
        require_trajectory_shape(states, controls);
        Evaluation result{};
        result.objective_definition = "mean_k(sigma_k) / maximum_thrust (normalised fuel)";
        const double thrust = subproblem_.model().config().maximum_thrust;
        for (std::size_t interval = 0U; interval < problem_.intervals; ++interval) {
            result.objective += controls[interval * 7U + 6U]
                / (static_cast<double>(problem_.intervals) * thrust);
        }
        result.path = path_components(states, controls);
        for (const auto& component : result.path) {
            result.path_violation = std::max(result.path_violation, component.normalised);
        }
        const std::size_t offset = problem_.intervals * 14U;
        result.terminal_errors.resize(13U);
        for (std::size_t component = 0U; component < 13U; ++component) {
            const double error =
                std::abs(states[offset + component] - problem_.target_state[component]);
            result.terminal_errors[component] = error;
            result.terminal_residual = std::max(
                result.terminal_residual, error * info_.state_trust_scales[component]
            );
            if (component < 3U) {
                result.terminal_position_error = std::max(result.terminal_position_error, error);
            } else if (component < 6U) {
                result.terminal_velocity_error = std::max(result.terminal_velocity_error, error);
            }
        }
        result.final_mass = states[offset + 13U];
        result.propellant_used = states[13U] - result.final_mass;
        return result;
    }

  private:
    transcription::PoweredDescent6DofSubproblem subproblem_;
    LayoutInfo info_{};

    [[nodiscard]] static dynamics::PoweredDescent6DofModel make_model(
        const PlannerProblem& problem
    ) {
        dynamics::PoweredDescent6DofConfig config{};
        config.gravity = problem.vehicle.gravity;
        config.principal_inertia = problem.vehicle.principal_inertia;
        config.mass_flow_coefficient = problem.vehicle.mass_flow_coefficient;
        config.minimum_mass = problem.vehicle.minimum_mass;
        config.maximum_thrust = problem.vehicle.maximum_thrust;
        config.minimum_sigma = problem.vehicle.minimum_thrust;
        config.maximum_torque = problem.vehicle.maximum_torque;
        config.maximum_angular_rate = problem.vehicle.maximum_angular_rate;
        config.maximum_tilt_radians = problem.vehicle.maximum_tilt_radians;
        config.glide_slope_radians = problem.vehicle.glide_slope_radians;
        return dynamics::PoweredDescent6DofModel{config};
    }

    [[nodiscard]] static transcription::PoweredDescent6DofScvxConfig make_config(
        const PlannerProblem& problem
    ) {
        transcription::PoweredDescent6DofScvxConfig config{};
        config.intervals = problem.intervals;
        config.step_seconds = problem.step_seconds;
        config.trust_radius = problem.solver.trust.initial_radius;
        config.virtual_l1_weight = problem.weights.virtual_l1_weight;
        config.virtual_quadratic_weight = problem.weights.virtual_quadratic_weight;
        config.virtual_epigraph_regularisation = problem.weights.virtual_epigraph_regularisation;
        config.fuel_weight = problem.weights.fuel_weight;
        config.discretisation = transcription::DiscretisationMethod::rk4_variational;
        std::copy_n(
            problem.weights.state_tracking_weights.begin(), 14U, config.state_tracking_weights.begin()
        );
        std::copy_n(
            problem.weights.control_tracking_weights.begin(),
            7U,
            config.control_tracking_weights.begin()
        );
        std::copy_n(problem.weights.state_trust_scales.begin(), 14U, config.state_trust_scales.begin());
        std::copy_n(
            problem.weights.control_trust_scales.begin(), 7U, config.control_trust_scales.begin()
        );
        return config;
    }
};

// --------------------------------------------------------------------------
// Low-thrust two-body transfer
// --------------------------------------------------------------------------

class LowThrustAdapter final : public FamilyAdapter {
  public:
    explicit LowThrustAdapter(PlannerProblem problem)
        : FamilyAdapter(std::move(problem)),
          subproblem_(make_model(problem_), make_config(problem_)) {
        const auto& layout = subproblem_.layout();
        info_.state_dimension = 7U;
        info_.control_dimension = 4U;
        info_.intervals = problem_.intervals;
        info_.terminal_dimension = 6U;
        info_.dynamics_row_start = layout.dynamics_rows().start;
        info_.terminal_row_start = layout.terminal_rows().start;
        info_.radial_row_start = layout.radial_rows().start;
        info_.stage_trust_row_start = layout.stage_trust_rows().start;
        info_.stage_trust_stride = 12U;
        info_.terminal_trust_row_start = layout.terminal_trust_rows().start;
        info_.state_trust_scales = problem_.weights.state_trust_scales;
        info_.control_trust_scales = problem_.weights.control_trust_scales;
        info_.fuel_weight = problem_.weights.fuel_weight;
        info_.virtual_l1_weight = problem_.weights.virtual_l1_weight;
        fill_maps(
            info_,
            subproblem_.structure(),
            [&layout](std::size_t node) { return layout.state(node).start; },
            [&layout](std::size_t interval) { return layout.control(interval).start; },
            [&layout](std::size_t interval) { return layout.virtual_control(interval).start; },
            true
        );
        const auto scalar_lookup = positions(subproblem_.structure().scalar_constraint);
        for (std::size_t node = 0U; node <= problem_.intervals; ++node) {
            for (std::size_t component = 0U; component < 3U; ++component) {
                info_.radial_positions.push_back(scalar_lookup.at({
                    info_.radial_row_start + node,
                    static_cast<std::size_t>(info_.state_variables[node * 7U + component]),
                }));
            }
        }
    }

    [[nodiscard]] Family family() const noexcept override { return Family::low_thrust; }
    [[nodiscard]] const core::FixedStructure& structure() const noexcept override {
        return subproblem_.structure();
    }
    [[nodiscard]] const LayoutInfo& layout() const noexcept override { return info_; }

    [[nodiscard]] DynamicsParameters dynamics() const override {
        DynamicsParameters result{};
        result.step_seconds = problem_.step_seconds;
        result.gravitational_parameter = problem_.vehicle.gravitational_parameter;
        result.thrust_to_acceleration = problem_.vehicle.thrust_to_acceleration;
        result.mass_flow_coefficient = problem_.vehicle.mass_flow_coefficient;
        return result;
    }

    [[nodiscard]] PhysicalLimits limits() const override {
        PhysicalLimits result{};
        const auto& config = subproblem_.model().config();
        result.maximum_thrust = config.maximum_thrust;
        result.minimum_radius = config.minimum_radius;
        return result;
    }

    [[nodiscard]] core::NumericValues values(
        const Trajectory& reference,
        double trust_radius
    ) const override {
        require_trajectory_shape(reference.states, reference.controls);
        return subproblem_.values(
            unflatten_states<dynamics::LowThrustState>(reference.states, problem_.intervals + 1U),
            unflatten_states<dynamics::LowThrustControl>(reference.controls, problem_.intervals),
            to_array<7U>(problem_.initial_state),
            to_array<7U>(problem_.target_state),
            trust_radius
        );
    }

    [[nodiscard]] Trajectory initial_reference() const override {
        std::vector<double> controls(problem_.intervals * 4U, 0.0);
        if (problem_.warm_start.has_value()) {
            controls = problem_.warm_start->controls;
        }
        return Trajectory{rollout(problem_.initial_state, controls, 1U), controls};
    }

    [[nodiscard]] std::vector<double> rollout(
        const std::vector<double>& initial,
        const std::vector<double>& controls,
        std::size_t substeps
    ) const override {
        if (substeps == 0U || initial.size() != 7U || controls.size() % 4U != 0U) {
            throw std::invalid_argument("low-thrust rollout arguments are malformed");
        }
        const auto& model = subproblem_.model();
        const double step = problem_.step_seconds / static_cast<double>(substeps);
        const std::size_t intervals = controls.size() / 4U;
        std::vector<double> result;
        result.reserve((intervals * substeps + 1U) * 7U);
        auto state = to_array<7U>(initial);
        result.insert(result.end(), state.begin(), state.end());
        for (std::size_t interval = 0U; interval < intervals; ++interval) {
            const auto control = to_array<4U>(controls, interval * 4U);
            for (std::size_t substep = 0U; substep < substeps; ++substep) {
                state = model.rk4_step(state, control, step);
                result.insert(result.end(), state.begin(), state.end());
            }
        }
        return result;
    }

    [[nodiscard]] std::vector<PathComponent> path_components(
        const std::vector<double>& states,
        const std::vector<double>& controls
    ) const override {
        const auto& model = subproblem_.model();
        const auto diagnostics = model.path_diagnostics(
            unflatten_states<dynamics::LowThrustState>(states, states.size() / 7U),
            unflatten_states<dynamics::LowThrustControl>(controls, controls.size() / 4U)
        );
        const double thrust = model.config().maximum_thrust;
        const double position_scale = std::max({
            info_.state_trust_scales[0U], info_.state_trust_scales[1U], info_.state_trust_scales[2U],
        });
        const double mass_scale = info_.state_trust_scales[6U];
        return {
            {"thrust_epigraph", diagnostics.thrust_epigraph / thrust, diagnostics.thrust_epigraph},
            {"throttle_upper", diagnostics.throttle_upper / thrust, diagnostics.throttle_upper},
            {"minimum_mass", diagnostics.minimum_mass * mass_scale, diagnostics.minimum_mass},
            {"minimum_radius", diagnostics.minimum_radius * position_scale, diagnostics.minimum_radius},
        };
    }

    [[nodiscard]] Evaluation evaluate(
        const std::vector<double>& states,
        const std::vector<double>& controls
    ) const override {
        require_trajectory_shape(states, controls);
        Evaluation result{};
        result.objective_definition = "mean_k(sigma_k) / maximum_thrust (normalised fuel)";
        const double thrust = subproblem_.model().config().maximum_thrust;
        for (std::size_t interval = 0U; interval < problem_.intervals; ++interval) {
            result.objective += controls[interval * 4U + 3U]
                / (static_cast<double>(problem_.intervals) * thrust);
        }
        result.path = path_components(states, controls);
        for (const auto& component : result.path) {
            result.path_violation = std::max(result.path_violation, component.normalised);
        }
        const std::size_t offset = problem_.intervals * 7U;
        result.terminal_errors.resize(6U);
        for (std::size_t component = 0U; component < 6U; ++component) {
            const double error =
                std::abs(states[offset + component] - problem_.target_state[component]);
            result.terminal_errors[component] = error;
            result.terminal_residual = std::max(
                result.terminal_residual, error * info_.state_trust_scales[component]
            );
            if (component < 3U) {
                result.terminal_position_error = std::max(result.terminal_position_error, error);
            } else {
                result.terminal_velocity_error = std::max(result.terminal_velocity_error, error);
            }
        }
        result.final_mass = states[offset + 6U];
        result.propellant_used = states[6U] - result.final_mass;
        return result;
    }

  private:
    transcription::LowThrustSubproblem subproblem_;
    LayoutInfo info_{};

    [[nodiscard]] static dynamics::LowThrustTwoBodyModel make_model(const PlannerProblem& problem) {
        dynamics::LowThrustTwoBodyConfig config{};
        config.gravitational_parameter = problem.vehicle.gravitational_parameter;
        config.thrust_to_acceleration = problem.vehicle.thrust_to_acceleration;
        config.mass_flow_coefficient = problem.vehicle.mass_flow_coefficient;
        config.minimum_mass = problem.vehicle.minimum_mass;
        config.maximum_thrust = problem.vehicle.maximum_thrust;
        config.minimum_radius = problem.vehicle.minimum_radius;
        return dynamics::LowThrustTwoBodyModel{config};
    }

    [[nodiscard]] static transcription::LowThrustScvxConfig make_config(
        const PlannerProblem& problem
    ) {
        transcription::LowThrustScvxConfig config{};
        config.intervals = problem.intervals;
        config.step_seconds = problem.step_seconds;
        config.trust_radius = problem.solver.trust.initial_radius;
        config.virtual_l1_weight = problem.weights.virtual_l1_weight;
        config.virtual_quadratic_weight = problem.weights.virtual_quadratic_weight;
        config.virtual_epigraph_regularisation = problem.weights.virtual_epigraph_regularisation;
        config.fuel_weight = problem.weights.fuel_weight;
        config.discretisation = transcription::DiscretisationMethod::rk4_variational;
        std::copy_n(
            problem.weights.state_tracking_weights.begin(), 7U, config.state_tracking_weights.begin()
        );
        std::copy_n(
            problem.weights.control_tracking_weights.begin(),
            4U,
            config.control_tracking_weights.begin()
        );
        std::copy_n(problem.weights.state_trust_scales.begin(), 7U, config.state_trust_scales.begin());
        std::copy_n(
            problem.weights.control_trust_scales.begin(), 4U, config.control_trust_scales.begin()
        );
        return config;
    }
};

[[nodiscard]] inline std::unique_ptr<FamilyAdapter> make_adapter(PlannerProblem problem) {
    switch (problem.family) {
        case Family::hcw:
            return std::make_unique<HcwAdapter>(std::move(problem));
        case Family::powered_descent_3dof:
            return std::make_unique<PoweredDescent3DofAdapter>(std::move(problem));
        case Family::powered_descent_6dof:
            return std::make_unique<PoweredDescent6DofAdapter>(std::move(problem));
        case Family::low_thrust:
            return std::make_unique<LowThrustAdapter>(std::move(problem));
    }
    throw ProblemError("unsupported planner family");
}

}  // namespace spacepdhcg::planner
