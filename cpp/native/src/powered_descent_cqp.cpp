#include "spacepdhcg/native/powered_descent_cqp.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace spacepdhcg::native {
namespace {

void require_finite(std::span<const double> values, const char* name) {
    for (double value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(std::string(name) + " must be finite");
        }
    }
}

template <std::size_t Columns, typename Matrix>
const double& entry(const Matrix& matrix, std::size_t row, std::size_t column) {
    return matrix[row * Columns + column];
}

void set_coefficient(CscMatrix& matrix, Index row, Index column, double value) {
    if (column < 0 || column >= matrix.columns) {
        throw std::out_of_range("sparse coefficient column lies outside the matrix");
    }
    const auto begin_index = matrix.offsets[static_cast<std::size_t>(column)];
    const auto end_index = matrix.offsets[static_cast<std::size_t>(column) + 1U];
    const auto begin = matrix.indices.begin() + begin_index;
    const auto end = matrix.indices.begin() + end_index;
    const auto found = std::lower_bound(begin, end, row);
    if (found == end || *found != row) {
        throw std::logic_error("requested coefficient is absent from the fixed sparse pattern");
    }
    const auto position = static_cast<std::size_t>(std::distance(matrix.indices.begin(), found));
    matrix.values[position] = value;
}

[[nodiscard]] bool same_pattern(const CscMatrix& left, const CscMatrix& right) {
    return left.rows == right.rows && left.columns == right.columns &&
           left.offsets == right.offsets && left.indices == right.indices;
}

[[nodiscard]] bool same_cones(
    const std::vector<ConeBlockDescriptor>& left,
    const std::vector<ConeBlockDescriptor>& right
) {
    if (left.size() != right.size()) {
        return false;
    }
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (left[index].kind != right[index].kind || left[index].start != right[index].start ||
            left[index].vector_dimension != right[index].vector_dimension ||
            left[index].power_alpha != right[index].power_alpha) {
            return false;
        }
    }
    return true;
}

}  // namespace

void PoweredDescentCqpConfig::validate() const {
    if (intervals < 2) {
        throw std::invalid_argument("powered-descent CQP requires at least two intervals");
    }
    for (const auto& item : std::array{
             std::pair{"step_seconds", step_seconds},
             std::pair{"trust_radius", trust_radius},
             std::pair{"virtual_l1_weight", virtual_l1_weight},
             std::pair{"virtual_quadratic_weight", virtual_quadratic_weight},
             std::pair{"virtual_epigraph_regularisation", virtual_epigraph_regularisation},
             std::pair{"fuel_weight", fuel_weight},
         }) {
        if (!std::isfinite(item.second) || item.second <= 0.0) {
            throw std::invalid_argument(std::string(item.first) + " must be finite and positive");
        }
    }
    require_finite(state_tracking_weights, "state tracking weights");
    require_finite(control_tracking_weights, "control tracking weights");
    require_finite(state_trust_scales, "state trust scales");
    require_finite(control_trust_scales, "control trust scales");
    if (std::ranges::any_of(state_tracking_weights, [](double value) { return value <= 0.0; }) ||
        std::ranges::any_of(control_tracking_weights, [](double value) { return value <= 0.0; }) ||
        std::ranges::any_of(state_trust_scales, [](double value) { return value <= 0.0; }) ||
        std::ranges::any_of(control_trust_scales, [](double value) { return value <= 0.0; })) {
        throw std::invalid_argument("powered-descent weights and scales must be positive");
    }
}

Index PoweredDescentCqpLayout::state_count() const noexcept {
    return (intervals + 1) * static_cast<Index>(powered_descent_state_dimension);
}

Index PoweredDescentCqpLayout::control_count() const noexcept {
    return intervals * static_cast<Index>(powered_descent_control_dimension);
}

Index PoweredDescentCqpLayout::virtual_count() const noexcept {
    return intervals * static_cast<Index>(powered_descent_state_dimension);
}

Index PoweredDescentCqpLayout::virtual_epigraph_count() const noexcept {
    return virtual_count();
}

Index PoweredDescentCqpLayout::control_offset() const noexcept {
    return state_count();
}

Index PoweredDescentCqpLayout::virtual_offset() const noexcept {
    return control_offset() + control_count();
}

Index PoweredDescentCqpLayout::virtual_epigraph_offset() const noexcept {
    return virtual_offset() + virtual_count();
}

Index PoweredDescentCqpLayout::variables() const noexcept {
    return virtual_epigraph_offset() + virtual_epigraph_count();
}

Index PoweredDescentCqpLayout::dynamics_row() const noexcept {
    return static_cast<Index>(powered_descent_state_dimension);
}

Index PoweredDescentCqpLayout::terminal_row() const noexcept {
    return dynamics_row() + intervals * static_cast<Index>(powered_descent_state_dimension);
}

Index PoweredDescentCqpLayout::virtual_epigraph_row() const noexcept {
    return terminal_row() + 6;
}

Index PoweredDescentCqpLayout::tilt_row() const noexcept {
    return virtual_epigraph_row() + 2 * virtual_count();
}

Index PoweredDescentCqpLayout::scalar_rows() const noexcept {
    return tilt_row() + intervals;
}

Index PoweredDescentCqpLayout::glide_cone_row() const noexcept {
    return 4 * intervals;
}

Index PoweredDescentCqpLayout::stage_trust_cone_row() const noexcept {
    return glide_cone_row() + 3 * (intervals + 1);
}

Index PoweredDescentCqpLayout::terminal_trust_cone_row() const noexcept {
    return stage_trust_cone_row() + 12 * intervals;
}

Index PoweredDescentCqpLayout::affine_rows() const noexcept {
    return terminal_trust_cone_row() + 8;
}

Index PoweredDescentCqpLayout::state_offset(Index node) const {
    if (node < 0 || node > intervals) {
        throw std::out_of_range("powered-descent state node lies outside the horizon");
    }
    return node * static_cast<Index>(powered_descent_state_dimension);
}

Index PoweredDescentCqpLayout::control_offset(Index interval) const {
    if (interval < 0 || interval >= intervals) {
        throw std::out_of_range("powered-descent control interval lies outside the horizon");
    }
    return control_offset() + interval * static_cast<Index>(powered_descent_control_dimension);
}

Index PoweredDescentCqpLayout::virtual_offset(Index interval) const {
    if (interval < 0 || interval >= intervals) {
        throw std::out_of_range("powered-descent virtual interval lies outside the horizon");
    }
    return virtual_offset() + interval * static_cast<Index>(powered_descent_state_dimension);
}

Index PoweredDescentCqpLayout::virtual_epigraph_offset(Index interval) const {
    if (interval < 0 || interval >= intervals) {
        throw std::out_of_range("powered-descent epigraph interval lies outside the horizon");
    }
    return virtual_epigraph_offset() +
           interval * static_cast<Index>(powered_descent_state_dimension);
}

PoweredDescentCqp::PoweredDescentCqp(
    PoweredDescent3DofModel model,
    PoweredDescentCqpConfig config
)
    : model_(std::move(model)), config_(std::move(config)), layout_{config_.intervals} {
    config_.validate();
    prototype_ = build_prototype();
    prototype_.validate();
}

OwnedCqp PoweredDescentCqp::build_prototype() const {
    const Index variables = layout_.variables();
    CscBuilder quadratic_builder(variables, variables);
    for (Index node = 0; node <= config_.intervals; ++node) {
        const Index offset = layout_.state_offset(node);
        for (std::size_t component = 0; component < powered_descent_state_dimension; ++component) {
            quadratic_builder.add(
                offset + static_cast<Index>(component),
                offset + static_cast<Index>(component),
                config_.state_tracking_weights[component]
            );
        }
    }
    for (Index interval = 0; interval < config_.intervals; ++interval) {
        const Index control = layout_.control_offset(interval);
        const Index virtual_control = layout_.virtual_offset(interval);
        const Index epigraph = layout_.virtual_epigraph_offset(interval);
        for (std::size_t component = 0; component < powered_descent_control_dimension; ++component) {
            quadratic_builder.add(
                control + static_cast<Index>(component),
                control + static_cast<Index>(component),
                config_.control_tracking_weights[component]
            );
        }
        for (std::size_t component = 0; component < powered_descent_state_dimension; ++component) {
            quadratic_builder.add(
                virtual_control + static_cast<Index>(component),
                virtual_control + static_cast<Index>(component),
                config_.virtual_quadratic_weight
            );
            quadratic_builder.add(
                epigraph + static_cast<Index>(component),
                epigraph + static_cast<Index>(component),
                config_.virtual_epigraph_regularisation
            );
        }
    }

    CscBuilder scalar_builder(layout_.scalar_rows(), variables);
    for (std::size_t component = 0; component < powered_descent_state_dimension; ++component) {
        scalar_builder.add(
            layout_.initial_row() + static_cast<Index>(component),
            layout_.state_offset(0) + static_cast<Index>(component),
            1.0
        );
    }
    for (Index interval = 0; interval < config_.intervals; ++interval) {
        const Index row = layout_.dynamics_row() +
                          interval * static_cast<Index>(powered_descent_state_dimension);
        const Index state = layout_.state_offset(interval);
        const Index next_state = layout_.state_offset(interval + 1);
        const Index control = layout_.control_offset(interval);
        const Index virtual_control = layout_.virtual_offset(interval);
        for (std::size_t output = 0; output < powered_descent_state_dimension; ++output) {
            for (std::size_t input = 0; input < powered_descent_state_dimension; ++input) {
                scalar_builder.add(
                    row + static_cast<Index>(output),
                    state + static_cast<Index>(input),
                    1.0
                );
            }
            scalar_builder.add(
                row + static_cast<Index>(output),
                next_state + static_cast<Index>(output),
                1.0
            );
            for (std::size_t input = 0; input < powered_descent_control_dimension; ++input) {
                scalar_builder.add(
                    row + static_cast<Index>(output),
                    control + static_cast<Index>(input),
                    1.0
                );
            }
            scalar_builder.add(
                row + static_cast<Index>(output),
                virtual_control + static_cast<Index>(output),
                1.0
            );
        }
    }
    const Index terminal_state = layout_.state_offset(config_.intervals);
    for (Index component = 0; component < 6; ++component) {
        scalar_builder.add(
            layout_.terminal_row() + component,
            terminal_state + component,
            1.0
        );
    }
    for (Index flat = 0; flat < layout_.virtual_count(); ++flat) {
        const Index positive_row = layout_.virtual_epigraph_row() + 2 * flat;
        const Index negative_row = positive_row + 1;
        const Index virtual_column = layout_.virtual_offset() + flat;
        const Index epigraph_column = layout_.virtual_epigraph_offset() + flat;
        scalar_builder.add(positive_row, virtual_column, 1.0);
        scalar_builder.add(positive_row, epigraph_column, 1.0);
        scalar_builder.add(negative_row, virtual_column, 1.0);
        scalar_builder.add(negative_row, epigraph_column, 1.0);
    }
    for (Index interval = 0; interval < config_.intervals; ++interval) {
        const Index row = layout_.tilt_row() + interval;
        const Index control = layout_.control_offset(interval);
        scalar_builder.add(row, control + 2, 1.0);
        scalar_builder.add(row, control + 3, 1.0);
    }

    CscBuilder affine_builder(layout_.affine_rows(), variables);
    for (Index interval = 0; interval < config_.intervals; ++interval) {
        const Index row = layout_.thrust_cone_row() + 4 * interval;
        const Index control = layout_.control_offset(interval);
        for (Index component = 0; component < 4; ++component) {
            affine_builder.add(row + component, control + component, 1.0);
        }
    }
    for (Index node = 0; node <= config_.intervals; ++node) {
        const Index row = layout_.glide_cone_row() + 3 * node;
        const Index state = layout_.state_offset(node);
        for (Index component = 0; component < 3; ++component) {
            affine_builder.add(row + component, state + component, 1.0);
        }
    }
    for (Index interval = 0; interval < config_.intervals; ++interval) {
        const Index row = layout_.stage_trust_cone_row() + 12 * interval;
        const Index state = layout_.state_offset(interval);
        const Index control = layout_.control_offset(interval);
        for (Index component = 0; component < 7; ++component) {
            affine_builder.add(row + component, state + component, 1.0);
        }
        for (Index component = 0; component < 4; ++component) {
            affine_builder.add(row + 7 + component, control + component, 1.0);
        }
    }
    const Index terminal_trust = layout_.terminal_trust_cone_row();
    for (Index component = 0; component < 7; ++component) {
        affine_builder.add(terminal_trust + component, terminal_state + component, 1.0);
    }

    std::vector<ConeBlockDescriptor> cones;
    cones.reserve(static_cast<std::size_t>(2 * config_.intervals + config_.intervals + 2));
    for (Index interval = 0; interval < config_.intervals; ++interval) {
        cones.push_back({ConeKind::second_order, 4 * interval, 2, 0.0});
    }
    for (Index node = 0; node <= config_.intervals; ++node) {
        cones.push_back({
            ConeKind::second_order,
            layout_.glide_cone_row() + 3 * node,
            1,
            0.0,
        });
    }
    for (Index interval = 0; interval < config_.intervals; ++interval) {
        cones.push_back({
            ConeKind::second_order,
            layout_.stage_trust_cone_row() + 12 * interval,
            10,
            0.0,
        });
    }
    cones.push_back({
        ConeKind::second_order,
        layout_.terminal_trust_cone_row(),
        6,
        0.0,
    });

    return OwnedCqp{
        quadratic_builder.build(),
        scalar_builder.build(),
        affine_builder.build(),
        std::vector<double>(static_cast<std::size_t>(variables), 0.0),
        std::vector<double>(static_cast<std::size_t>(layout_.scalar_rows()), 0.0),
        std::vector<double>(static_cast<std::size_t>(layout_.scalar_rows()), 0.0),
        std::vector<double>(static_cast<std::size_t>(layout_.affine_rows()), 0.0),
        std::vector<double>(
            static_cast<std::size_t>(variables),
            -std::numeric_limits<double>::infinity()
        ),
        std::vector<double>(
            static_cast<std::size_t>(variables),
            std::numeric_limits<double>::infinity()
        ),
        std::move(cones),
        {},
    };
}

OwnedCqp PoweredDescentCqp::make_cqp(
    std::span<const PoweredDescentState> reference_states,
    std::span<const PoweredDescentControl> reference_controls,
    std::span<const double, powered_descent_state_dimension> initial_state,
    std::span<const double, 3> target_position,
    std::span<const double, 3> target_velocity,
    double trust_radius
) const {
    OwnedCqp problem = prototype_;
    update_numerical_values(
        problem,
        reference_states,
        reference_controls,
        initial_state,
        target_position,
        target_velocity,
        trust_radius
    );
    return problem;
}

void PoweredDescentCqp::update_numerical_values(
    OwnedCqp& problem,
    std::span<const PoweredDescentState> reference_states,
    std::span<const PoweredDescentControl> reference_controls,
    std::span<const double, powered_descent_state_dimension> initial_state,
    std::span<const double, 3> target_position,
    std::span<const double, 3> target_velocity,
    double trust_radius
) const {
    if (reference_states.size() != static_cast<std::size_t>(config_.intervals + 1) ||
        reference_controls.size() != static_cast<std::size_t>(config_.intervals)) {
        throw std::invalid_argument("powered-descent reference has the wrong horizon");
    }
    if (!std::isfinite(trust_radius) || trust_radius <= 0.0) {
        throw std::invalid_argument("trust radius must be finite and positive");
    }
    require_finite(initial_state, "initial state");
    require_finite(target_position, "target position");
    require_finite(target_velocity, "target velocity");
    for (const auto& state : reference_states) {
        require_finite(state, "reference state");
        if (state[6] <= 0.0) {
            throw std::invalid_argument("reference mass must be positive");
        }
    }
    for (const auto& control : reference_controls) {
        require_finite(control, "reference control");
    }
    assert_compatible(problem);

    problem.quadratic.values = prototype_.quadratic.values;
    std::fill(problem.scalar_constraint.values.begin(), problem.scalar_constraint.values.end(), 0.0);
    std::fill(problem.affine_cone.values.begin(), problem.affine_cone.values.end(), 0.0);
    std::fill(problem.linear.begin(), problem.linear.end(), 0.0);
    std::fill(
        problem.scalar_lower.begin(),
        problem.scalar_lower.end(),
        -std::numeric_limits<double>::infinity()
    );
    std::fill(
        problem.scalar_upper.begin(),
        problem.scalar_upper.end(),
        std::numeric_limits<double>::infinity()
    );
    std::fill(problem.affine_offset.begin(), problem.affine_offset.end(), 0.0);
    std::fill(
        problem.variable_lower.begin(),
        problem.variable_lower.end(),
        -std::numeric_limits<double>::infinity()
    );
    std::fill(
        problem.variable_upper.begin(),
        problem.variable_upper.end(),
        std::numeric_limits<double>::infinity()
    );

    for (Index node = 0; node <= config_.intervals; ++node) {
        const Index state = layout_.state_offset(node);
        for (std::size_t component = 0; component < powered_descent_state_dimension; ++component) {
            problem.linear[static_cast<std::size_t>(state) + component] =
                -reference_states[static_cast<std::size_t>(node)][component] *
                config_.state_tracking_weights[component];
        }
    }
    for (Index interval = 0; interval < config_.intervals; ++interval) {
        const Index control = layout_.control_offset(interval);
        for (std::size_t component = 0; component < powered_descent_control_dimension; ++component) {
            problem.linear[static_cast<std::size_t>(control) + component] =
                -reference_controls[static_cast<std::size_t>(interval)][component] *
                config_.control_tracking_weights[component];
        }
        problem.linear[static_cast<std::size_t>(control + 3)] +=
            config_.fuel_weight * config_.step_seconds;
    }
    for (Index index = layout_.virtual_epigraph_offset(); index < layout_.variables(); ++index) {
        problem.linear[static_cast<std::size_t>(index)] = config_.virtual_l1_weight;
    }

    for (Index component = 0; component < 7; ++component) {
        const Index row = layout_.initial_row() + component;
        set_coefficient(problem.scalar_constraint, row, component, 1.0);
        problem.scalar_lower[static_cast<std::size_t>(row)] =
            initial_state[static_cast<std::size_t>(component)];
        problem.scalar_upper[static_cast<std::size_t>(row)] =
            initial_state[static_cast<std::size_t>(component)];
    }

    for (Index interval = 0; interval < config_.intervals; ++interval) {
        const auto discrete = model_.linearised_euler(
            reference_states[static_cast<std::size_t>(interval)],
            reference_controls[static_cast<std::size_t>(interval)],
            config_.step_seconds
        );
        const Index row = layout_.dynamics_row() + 7 * interval;
        const Index state = layout_.state_offset(interval);
        const Index next_state = layout_.state_offset(interval + 1);
        const Index control = layout_.control_offset(interval);
        const Index virtual_control = layout_.virtual_offset(interval);
        for (std::size_t output = 0; output < powered_descent_state_dimension; ++output) {
            for (std::size_t input = 0; input < powered_descent_state_dimension; ++input) {
                set_coefficient(
                    problem.scalar_constraint,
                    row + static_cast<Index>(output),
                    state + static_cast<Index>(input),
                    -entry<powered_descent_state_dimension>(
                        discrete.state_matrix,
                        output,
                        input
                    )
                );
            }
            set_coefficient(
                problem.scalar_constraint,
                row + static_cast<Index>(output),
                next_state + static_cast<Index>(output),
                1.0
            );
            for (std::size_t input = 0; input < powered_descent_control_dimension; ++input) {
                set_coefficient(
                    problem.scalar_constraint,
                    row + static_cast<Index>(output),
                    control + static_cast<Index>(input),
                    -entry<powered_descent_control_dimension>(
                        discrete.control_matrix,
                        output,
                        input
                    )
                );
            }
            set_coefficient(
                problem.scalar_constraint,
                row + static_cast<Index>(output),
                virtual_control + static_cast<Index>(output),
                -1.0
            );
            const auto row_index = static_cast<std::size_t>(row) + output;
            problem.scalar_lower[row_index] = discrete.offset[output];
            problem.scalar_upper[row_index] = discrete.offset[output];
        }
    }

    const Index terminal_state = layout_.state_offset(config_.intervals);
    for (Index component = 0; component < 6; ++component) {
        const Index row = layout_.terminal_row() + component;
        set_coefficient(problem.scalar_constraint, row, terminal_state + component, 1.0);
        const double target = component < 3
                                  ? target_position[static_cast<std::size_t>(component)]
                                  : target_velocity[static_cast<std::size_t>(component - 3)];
        problem.scalar_lower[static_cast<std::size_t>(row)] = target;
        problem.scalar_upper[static_cast<std::size_t>(row)] = target;
    }

    for (Index flat = 0; flat < layout_.virtual_count(); ++flat) {
        const Index positive_row = layout_.virtual_epigraph_row() + 2 * flat;
        const Index negative_row = positive_row + 1;
        const Index virtual_column = layout_.virtual_offset() + flat;
        const Index epigraph_column = layout_.virtual_epigraph_offset() + flat;
        set_coefficient(problem.scalar_constraint, positive_row, virtual_column, 1.0);
        set_coefficient(problem.scalar_constraint, positive_row, epigraph_column, -1.0);
        set_coefficient(problem.scalar_constraint, negative_row, virtual_column, -1.0);
        set_coefficient(problem.scalar_constraint, negative_row, epigraph_column, -1.0);
        problem.scalar_upper[static_cast<std::size_t>(positive_row)] = 0.0;
        problem.scalar_upper[static_cast<std::size_t>(negative_row)] = 0.0;
    }

    for (Index interval = 0; interval < config_.intervals; ++interval) {
        const Index row = layout_.tilt_row() + interval;
        const Index control = layout_.control_offset(interval);
        set_coefficient(problem.scalar_constraint, row, control + 2, -1.0);
        set_coefficient(
            problem.scalar_constraint,
            row,
            control + 3,
            model_.config().tilt_cosine()
        );
        problem.scalar_upper[static_cast<std::size_t>(row)] = 0.0;
    }

    for (Index interval = 0; interval < config_.intervals; ++interval) {
        const Index row = layout_.thrust_cone_row() + 4 * interval;
        const Index control = layout_.control_offset(interval);
        for (Index component = 0; component < 4; ++component) {
            set_coefficient(problem.affine_cone, row + component, control + component, 1.0);
        }
    }
    for (Index node = 0; node <= config_.intervals; ++node) {
        const Index row = layout_.glide_cone_row() + 3 * node;
        const Index state = layout_.state_offset(node);
        set_coefficient(problem.affine_cone, row, state, 1.0);
        set_coefficient(problem.affine_cone, row + 1, state + 1, 1.0);
        set_coefficient(
            problem.affine_cone,
            row + 2,
            state + 2,
            model_.config().glide_slope_tangent()
        );
    }
    for (Index interval = 0; interval < config_.intervals; ++interval) {
        const Index row = layout_.stage_trust_cone_row() + 12 * interval;
        const Index state = layout_.state_offset(interval);
        const Index control = layout_.control_offset(interval);
        for (std::size_t component = 0; component < powered_descent_state_dimension; ++component) {
            const double scale = config_.state_trust_scales[component];
            set_coefficient(
                problem.affine_cone,
                row + static_cast<Index>(component),
                state + static_cast<Index>(component),
                scale
            );
            problem.affine_offset[static_cast<std::size_t>(row) + component] =
                -scale * reference_states[static_cast<std::size_t>(interval)][component];
        }
        for (std::size_t component = 0; component < powered_descent_control_dimension; ++component) {
            const Index cone_row = row + 7 + static_cast<Index>(component);
            const double scale = config_.control_trust_scales[component];
            set_coefficient(
                problem.affine_cone,
                cone_row,
                control + static_cast<Index>(component),
                scale
            );
            problem.affine_offset[static_cast<std::size_t>(cone_row)] =
                -scale * reference_controls[static_cast<std::size_t>(interval)][component];
        }
        problem.affine_offset[static_cast<std::size_t>(row + 11)] = trust_radius;
    }
    const Index terminal_trust = layout_.terminal_trust_cone_row();
    for (std::size_t component = 0; component < powered_descent_state_dimension; ++component) {
        const double scale = config_.state_trust_scales[component];
        set_coefficient(
            problem.affine_cone,
            terminal_trust + static_cast<Index>(component),
            terminal_state + static_cast<Index>(component),
            scale
        );
        problem.affine_offset[static_cast<std::size_t>(terminal_trust) + component] =
            -scale * reference_states.back()[component];
    }
    problem.affine_offset[static_cast<std::size_t>(terminal_trust + 7)] = trust_radius;

    for (Index node = 0; node <= config_.intervals; ++node) {
        const Index state = layout_.state_offset(node);
        problem.variable_lower[static_cast<std::size_t>(state + 2)] = 0.0;
        problem.variable_lower[static_cast<std::size_t>(state + 6)] =
            model_.config().minimum_mass;
    }
    for (Index interval = 0; interval < config_.intervals; ++interval) {
        const Index sigma = layout_.control_offset(interval) + 3;
        problem.variable_lower[static_cast<std::size_t>(sigma)] =
            model_.config().minimum_sigma;
        problem.variable_upper[static_cast<std::size_t>(sigma)] =
            model_.config().maximum_thrust;
    }
    for (Index index = layout_.virtual_epigraph_offset(); index < layout_.variables(); ++index) {
        problem.variable_lower[static_cast<std::size_t>(index)] = 0.0;
    }
    problem.validate();
}

void PoweredDescentCqp::assert_compatible(const OwnedCqp& problem) const {
    if (!same_pattern(problem.quadratic, prototype_.quadratic) ||
        !same_pattern(problem.scalar_constraint, prototype_.scalar_constraint) ||
        !same_pattern(problem.affine_cone, prototype_.affine_cone) ||
        !same_cones(problem.affine_cones, prototype_.affine_cones) ||
        !same_cones(problem.variable_cones, prototype_.variable_cones)) {
        throw std::invalid_argument("powered-descent update received a different CQP topology");
    }
}

PoweredDescentDecision PoweredDescentCqp::decode(std::span<const double> decision) const {
    if (decision.size() != static_cast<std::size_t>(layout_.variables())) {
        throw std::invalid_argument("powered-descent decision has the wrong dimension");
    }
    require_finite(decision, "powered-descent decision");
    PoweredDescentDecision result{};
    result.states.resize(static_cast<std::size_t>(config_.intervals + 1));
    result.controls.resize(static_cast<std::size_t>(config_.intervals));
    result.virtual_controls.resize(static_cast<std::size_t>(config_.intervals));
    result.virtual_epigraphs.resize(static_cast<std::size_t>(config_.intervals));
    for (Index node = 0; node <= config_.intervals; ++node) {
        const auto offset = static_cast<std::size_t>(layout_.state_offset(node));
        std::copy_n(
            decision.begin() + static_cast<std::ptrdiff_t>(offset),
            powered_descent_state_dimension,
            result.states[static_cast<std::size_t>(node)].begin()
        );
    }
    for (Index interval = 0; interval < config_.intervals; ++interval) {
        const auto control = static_cast<std::size_t>(layout_.control_offset(interval));
        const auto virtual_control = static_cast<std::size_t>(layout_.virtual_offset(interval));
        const auto epigraph =
            static_cast<std::size_t>(layout_.virtual_epigraph_offset(interval));
        std::copy_n(
            decision.begin() + static_cast<std::ptrdiff_t>(control),
            powered_descent_control_dimension,
            result.controls[static_cast<std::size_t>(interval)].begin()
        );
        std::copy_n(
            decision.begin() + static_cast<std::ptrdiff_t>(virtual_control),
            powered_descent_state_dimension,
            result.virtual_controls[static_cast<std::size_t>(interval)].begin()
        );
        std::copy_n(
            decision.begin() + static_cast<std::ptrdiff_t>(epigraph),
            powered_descent_state_dimension,
            result.virtual_epigraphs[static_cast<std::size_t>(interval)].begin()
        );
    }
    return result;
}

PoweredDescentCqpDiagnostics PoweredDescentCqp::diagnostics(
    std::span<const double> decision,
    const OwnedCqp& problem,
    std::span<const double, 3> target_position,
    std::span<const double, 3> target_velocity
) const {
    assert_compatible(problem);
    const auto decoded = decode(decision);
    const auto scalar_activity = problem.scalar_constraint.multiply(decision);

    PoweredDescentCqpDiagnostics result{};
    result.convex = problem.diagnostics(decision);
    for (Index interval = 0; interval < config_.intervals; ++interval) {
        const auto prediction = model_.euler_step(
            decoded.states[static_cast<std::size_t>(interval)],
            decoded.controls[static_cast<std::size_t>(interval)],
            config_.step_seconds
        );
        const Index row = layout_.dynamics_row() + 7 * interval;
        for (std::size_t component = 0; component < powered_descent_state_dimension; ++component) {
            result.nonlinear_dynamics_defect = std::max(
                result.nonlinear_dynamics_defect,
                std::abs(
                    decoded.states[static_cast<std::size_t>(interval + 1)][component] -
                    prediction[component]
                )
            );
            const auto row_index = static_cast<std::size_t>(row) + component;
            result.linearised_dynamics_defect = std::max(
                result.linearised_dynamics_defect,
                std::abs(scalar_activity[row_index] - problem.scalar_lower[row_index])
            );
            result.virtual_control = std::max(
                result.virtual_control,
                std::abs(
                    decoded.virtual_controls[static_cast<std::size_t>(interval)][component]
                )
            );
        }
    }
    const auto& terminal = decoded.states.back();
    for (std::size_t component = 0; component < 3; ++component) {
        result.terminal_error = std::max(
            result.terminal_error,
            std::abs(terminal[component] - target_position[component])
        );
        result.terminal_error = std::max(
            result.terminal_error,
            std::abs(terminal[3 + component] - target_velocity[component])
        );
    }
    return result;
}

}  // namespace spacepdhcg::native
