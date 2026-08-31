#include "spacepdhcg/native/cw_problem.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace spacepdhcg::native {
namespace {

template <std::size_t Columns, typename Matrix>
const double& entry(const Matrix& matrix, std::size_t row, std::size_t column) {
    return matrix[row * Columns + column];
}

void require_finite(std::span<const double> values, const char* name) {
    for (double value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(std::string(name) + " must be finite");
        }
    }
}

[[nodiscard]] bool same_pattern(const CscMatrix& left, const CscMatrix& right) {
    return left.rows == right.rows && left.columns == right.columns &&
           left.offsets == right.offsets && left.indices == right.indices;
}

}  // namespace

void CwRendezvousConfig::validate() const {
    if (intervals < 2) {
        throw std::invalid_argument("CW rendezvous requires at least two intervals");
    }
    if (!std::isfinite(step_seconds) || step_seconds <= 0.0) {
        throw std::invalid_argument("CW step duration must be finite and positive");
    }
    if (!std::isfinite(mean_motion) || mean_motion <= 0.0) {
        throw std::invalid_argument("CW mean motion must be finite and positive");
    }
    if (!std::isfinite(maximum_acceleration) || maximum_acceleration <= 0.0) {
        throw std::invalid_argument("maximum acceleration must be finite and positive");
    }
    require_finite(state_weights, "state weights");
    require_finite(control_weights, "control weights");
    if (std::ranges::any_of(state_weights, [](double value) { return value <= 0.0; }) ||
        std::ranges::any_of(control_weights, [](double value) { return value <= 0.0; })) {
        throw std::invalid_argument("CW objective weights must be positive");
    }
}

Index CwRendezvousLayout::state_variables() const noexcept {
    return (intervals + 1) * static_cast<Index>(cw_state_dimension);
}

Index CwRendezvousLayout::control_variables() const noexcept {
    return intervals * static_cast<Index>(cw_control_dimension);
}

Index CwRendezvousLayout::variables() const noexcept {
    return state_variables() + control_variables();
}

Index CwRendezvousLayout::scalar_constraints() const noexcept {
    const Index core = static_cast<Index>(cw_state_dimension) * (intervals + 2);
    return core + (thrust_constraint == CwThrustConstraint::box ? control_variables() : 0);
}

Index CwRendezvousLayout::affine_rows() const noexcept {
    return thrust_constraint == CwThrustConstraint::second_order_cone ? 4 * intervals : 0;
}

Index CwRendezvousLayout::dynamics_row() const noexcept {
    return static_cast<Index>(cw_state_dimension);
}

Index CwRendezvousLayout::terminal_row() const noexcept {
    return dynamics_row() + intervals * static_cast<Index>(cw_state_dimension);
}

Index CwRendezvousLayout::control_row() const noexcept {
    return terminal_row() + static_cast<Index>(cw_state_dimension);
}

Index CwRendezvousLayout::state_offset(Index node) const {
    if (node < 0 || node > intervals) {
        throw std::out_of_range("CW state node lies outside the horizon");
    }
    return node * static_cast<Index>(cw_state_dimension);
}

Index CwRendezvousLayout::control_offset(Index interval) const {
    if (interval < 0 || interval >= intervals) {
        throw std::out_of_range("CW control interval lies outside the horizon");
    }
    return state_variables() + interval * static_cast<Index>(cw_control_dimension);
}

double CwRendezvousDiagnostics::maximum_violation() const noexcept {
    return std::max({initial_error, terminal_error, dynamics_defect, control_violation});
}

CwRendezvousProblem::CwRendezvousProblem(CwRendezvousConfig config)
    : config_(std::move(config)),
      layout_{config_.intervals, config_.thrust_constraint},
      dynamics_(discretise_cw(config_.mean_motion, config_.step_seconds)) {
    config_.validate();
    prototype_ = build_prototype();
    prototype_.validate();
}

OwnedCqp CwRendezvousProblem::build_prototype() const {
    const Index variables = layout_.variables();
    CscBuilder quadratic_builder(variables, variables);
    for (Index node = 0; node <= config_.intervals; ++node) {
        const Index offset = layout_.state_offset(node);
        for (std::size_t component = 0; component < cw_state_dimension; ++component) {
            quadratic_builder.add(
                offset + static_cast<Index>(component),
                offset + static_cast<Index>(component),
                2.0 * config_.state_weights[component]
            );
        }
    }
    for (Index interval = 0; interval < config_.intervals; ++interval) {
        const Index offset = layout_.control_offset(interval);
        for (std::size_t component = 0; component < cw_control_dimension; ++component) {
            quadratic_builder.add(
                offset + static_cast<Index>(component),
                offset + static_cast<Index>(component),
                2.0 * config_.control_weights[component]
            );
        }
    }

    CscBuilder scalar_builder(layout_.scalar_constraints(), variables);
    for (std::size_t component = 0; component < cw_state_dimension; ++component) {
        scalar_builder.add(
            layout_.initial_row() + static_cast<Index>(component),
            layout_.state_offset(0) + static_cast<Index>(component),
            1.0
        );
    }
    for (Index interval = 0; interval < config_.intervals; ++interval) {
        const Index row =
            layout_.dynamics_row() + interval * static_cast<Index>(cw_state_dimension);
        const Index state = layout_.state_offset(interval);
        const Index next_state = layout_.state_offset(interval + 1);
        const Index control = layout_.control_offset(interval);
        for (std::size_t output = 0; output < cw_state_dimension; ++output) {
            for (std::size_t input = 0; input < cw_state_dimension; ++input) {
                scalar_builder.add(
                    row + static_cast<Index>(output),
                    state + static_cast<Index>(input),
                    -entry<cw_state_dimension>(dynamics_.state, output, input)
                );
            }
            scalar_builder.add(
                row + static_cast<Index>(output),
                next_state + static_cast<Index>(output),
                1.0
            );
            for (std::size_t input = 0; input < cw_control_dimension; ++input) {
                scalar_builder.add(
                    row + static_cast<Index>(output),
                    control + static_cast<Index>(input),
                    -entry<cw_control_dimension>(dynamics_.control, output, input)
                );
            }
        }
    }
    for (std::size_t component = 0; component < cw_state_dimension; ++component) {
        scalar_builder.add(
            layout_.terminal_row() + static_cast<Index>(component),
            layout_.state_offset(config_.intervals) + static_cast<Index>(component),
            1.0
        );
    }
    if (config_.thrust_constraint == CwThrustConstraint::box) {
        for (Index interval = 0; interval < config_.intervals; ++interval) {
            for (std::size_t component = 0; component < cw_control_dimension; ++component) {
                const Index flattened =
                    interval * static_cast<Index>(cw_control_dimension) +
                    static_cast<Index>(component);
                scalar_builder.add(
                    layout_.control_row() + flattened,
                    layout_.control_offset(interval) + static_cast<Index>(component),
                    1.0
                );
            }
        }
    }

    CscBuilder affine_builder(layout_.affine_rows(), variables);
    std::vector<ConeBlockDescriptor> affine_cones;
    if (config_.thrust_constraint == CwThrustConstraint::second_order_cone) {
        affine_cones.reserve(static_cast<std::size_t>(config_.intervals));
        for (Index interval = 0; interval < config_.intervals; ++interval) {
            const Index row = 4 * interval;
            for (std::size_t component = 0; component < cw_control_dimension; ++component) {
                affine_builder.add(
                    row + static_cast<Index>(component),
                    layout_.control_offset(interval) + static_cast<Index>(component),
                    1.0
                );
            }
            affine_cones.push_back(ConeBlockDescriptor{
                ConeKind::second_order,
                row,
                2,
                0.0,
            });
        }
    }

    OwnedCqp problem{
        quadratic_builder.build(),
        scalar_builder.build(),
        affine_builder.build(),
        std::vector<double>(static_cast<std::size_t>(variables), 0.0),
        std::vector<double>(static_cast<std::size_t>(layout_.scalar_constraints()), 0.0),
        std::vector<double>(static_cast<std::size_t>(layout_.scalar_constraints()), 0.0),
        std::vector<double>(static_cast<std::size_t>(layout_.affine_rows()), 0.0),
        std::vector<double>(
            static_cast<std::size_t>(variables),
            -std::numeric_limits<double>::infinity()
        ),
        std::vector<double>(
            static_cast<std::size_t>(variables),
            std::numeric_limits<double>::infinity()
        ),
        std::move(affine_cones),
        {},
    };

    if (config_.thrust_constraint == CwThrustConstraint::box) {
        for (Index row = layout_.control_row(); row < layout_.scalar_constraints(); ++row) {
            problem.scalar_lower[static_cast<std::size_t>(row)] = -config_.maximum_acceleration;
            problem.scalar_upper[static_cast<std::size_t>(row)] = config_.maximum_acceleration;
        }
    } else {
        for (Index interval = 0; interval < config_.intervals; ++interval) {
            problem.affine_offset[static_cast<std::size_t>(4 * interval + 3)] =
                config_.maximum_acceleration;
        }
    }
    return problem;
}

OwnedCqp CwRendezvousProblem::make_cqp(
    std::span<const double, cw_state_dimension> initial_state,
    std::span<const double, cw_state_dimension> target_state
) const {
    OwnedCqp result = prototype_;
    update_numerical_values(result, initial_state, target_state);
    return result;
}

void CwRendezvousProblem::update_numerical_values(
    OwnedCqp& problem,
    std::span<const double, cw_state_dimension> initial_state,
    std::span<const double, cw_state_dimension> target_state
) const {
    require_finite(initial_state, "initial state");
    require_finite(target_state, "target state");
    assert_compatible(problem);

    std::fill(problem.linear.begin(), problem.linear.end(), 0.0);
    for (Index node = 0; node <= config_.intervals; ++node) {
        const double fraction =
            static_cast<double>(node) / static_cast<double>(config_.intervals);
        const Index offset = layout_.state_offset(node);
        for (std::size_t component = 0; component < cw_state_dimension; ++component) {
            const double reference =
                (1.0 - fraction) * initial_state[component] +
                fraction * target_state[component];
            problem.linear[
                static_cast<std::size_t>(offset) + component
            ] = -2.0 * config_.state_weights[component] * reference;
        }
    }

    for (std::size_t component = 0; component < cw_state_dimension; ++component) {
        const auto initial_row =
            static_cast<std::size_t>(layout_.initial_row()) + component;
        const auto terminal_row =
            static_cast<std::size_t>(layout_.terminal_row()) + component;
        problem.scalar_lower[initial_row] = initial_state[component];
        problem.scalar_upper[initial_row] = initial_state[component];
        problem.scalar_lower[terminal_row] = target_state[component];
        problem.scalar_upper[terminal_row] = target_state[component];
    }
    problem.validate();
}

void CwRendezvousProblem::assert_compatible(const OwnedCqp& problem) const {
    if (!same_pattern(problem.quadratic, prototype_.quadratic) ||
        !same_pattern(problem.scalar_constraint, prototype_.scalar_constraint) ||
        !same_pattern(problem.affine_cone, prototype_.affine_cone) ||
        problem.affine_cones.size() != prototype_.affine_cones.size() ||
        problem.variable_cones.size() != prototype_.variable_cones.size()) {
        throw std::invalid_argument("CW numerical update received a different CQP topology");
    }
}

CwRendezvousDiagnostics CwRendezvousProblem::diagnostics(
    std::span<const double> decision,
    std::span<const double, cw_state_dimension> initial_state,
    std::span<const double, cw_state_dimension> target_state
) const {
    if (decision.size() != static_cast<std::size_t>(layout_.variables())) {
        throw std::invalid_argument("CW decision vector has the wrong dimension");
    }
    require_finite(decision, "decision vector");

    CwRendezvousDiagnostics result{};
    const auto initial_offset = static_cast<std::size_t>(layout_.state_offset(0));
    const auto terminal_offset =
        static_cast<std::size_t>(layout_.state_offset(config_.intervals));
    for (std::size_t component = 0; component < cw_state_dimension; ++component) {
        result.initial_error = std::max(
            result.initial_error,
            std::abs(decision[initial_offset + component] - initial_state[component])
        );
        result.terminal_error = std::max(
            result.terminal_error,
            std::abs(decision[terminal_offset + component] - target_state[component])
        );
    }

    for (Index interval = 0; interval < config_.intervals; ++interval) {
        const auto state_offset = static_cast<std::size_t>(layout_.state_offset(interval));
        const auto next_offset =
            static_cast<std::size_t>(layout_.state_offset(interval + 1));
        const auto control_offset =
            static_cast<std::size_t>(layout_.control_offset(interval));
        const std::span<const double, cw_state_dimension> state{
            decision.data() + state_offset,
            cw_state_dimension,
        };
        const std::span<const double, cw_control_dimension> control{
            decision.data() + control_offset,
            cw_control_dimension,
        };
        const auto prediction = propagate_cw(dynamics_, state, control);
        double norm_squared = 0.0;
        for (std::size_t component = 0; component < cw_state_dimension; ++component) {
            result.dynamics_defect = std::max(
                result.dynamics_defect,
                std::abs(decision[next_offset + component] - prediction[component])
            );
        }
        for (double value : control) {
            result.maximum_component_acceleration = std::max(
                result.maximum_component_acceleration,
                std::abs(value)
            );
            norm_squared += value * value;
        }
        const double norm = std::sqrt(norm_squared);
        result.maximum_acceleration_norm = std::max(result.maximum_acceleration_norm, norm);
        const double activity =
            config_.thrust_constraint == CwThrustConstraint::box
                ? result.maximum_component_acceleration
                : norm;
        result.control_violation = std::max(
            result.control_violation,
            activity - config_.maximum_acceleration
        );
    }
    result.control_violation = std::max(result.control_violation, 0.0);
    return result;
}

}  // namespace spacepdhcg::native
