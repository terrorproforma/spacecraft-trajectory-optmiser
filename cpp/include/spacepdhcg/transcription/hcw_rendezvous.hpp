#pragma once

#include "spacepdhcg/core/fixed_cqp.hpp"
#include "spacepdhcg/dynamics/hcw.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace spacepdhcg::transcription {

enum class HcwControlSet : std::uint8_t {
    box,
    second_order_cone,
};

struct HcwRendezvousConfig {
    std::size_t intervals{40U};
    double step_seconds{20.0};
    double mean_motion{1.13e-3};
    double maximum_acceleration{5.0e-2};
    HcwControlSet control_set{HcwControlSet::second_order_cone};
    std::array<double, 6U> state_weights{
        1.0e-4,
        1.0e-4,
        1.0e-4,
        1.0e-2,
        1.0e-2,
        1.0e-2,
    };
    std::array<double, 3U> control_weights{1.0, 1.0, 1.0};

    void validate() const {
        if (intervals < 2U) {
            throw std::invalid_argument("HCW rendezvous requires at least two intervals");
        }
        for (const auto value : {step_seconds, mean_motion, maximum_acceleration}) {
            if (!std::isfinite(value) || value <= 0.0) {
                throw std::invalid_argument(
                    "HCW rendezvous scales must be finite and positive"
                );
            }
        }
        for (const auto value : state_weights) {
            if (!std::isfinite(value) || value <= 0.0) {
                throw std::invalid_argument("HCW state weights must be finite and positive");
            }
        }
        for (const auto value : control_weights) {
            if (!std::isfinite(value) || value <= 0.0) {
                throw std::invalid_argument("HCW control weights must be finite and positive");
            }
        }
    }
};

struct HcwRendezvousLayout {
    std::size_t intervals{0U};
    HcwControlSet control_set{HcwControlSet::second_order_cone};

    [[nodiscard]] std::size_t state_variables() const noexcept {
        return (intervals + 1U) * dynamics::hcw_state_dimension;
    }
    [[nodiscard]] std::size_t control_variables() const noexcept {
        return intervals * dynamics::hcw_control_dimension;
    }
    [[nodiscard]] std::size_t variables() const noexcept {
        return state_variables() + control_variables();
    }
    [[nodiscard]] std::size_t scalar_rows() const noexcept {
        const auto core = dynamics::hcw_state_dimension
                          + intervals * dynamics::hcw_state_dimension
                          + dynamics::hcw_state_dimension;
        return control_set == HcwControlSet::box ? core + control_variables() : core;
    }
    [[nodiscard]] std::size_t affine_rows() const noexcept {
        return control_set == HcwControlSet::second_order_cone ? 4U * intervals : 0U;
    }
    [[nodiscard]] std::size_t initial_row() const noexcept { return 0U; }
    [[nodiscard]] std::size_t dynamics_row() const noexcept {
        return dynamics::hcw_state_dimension;
    }
    [[nodiscard]] std::size_t terminal_row() const noexcept {
        return dynamics_row() + intervals * dynamics::hcw_state_dimension;
    }
    [[nodiscard]] std::size_t control_row() const noexcept {
        return terminal_row() + dynamics::hcw_state_dimension;
    }
    [[nodiscard]] std::size_t state_index(std::size_t node, std::size_t component) const {
        if (node > intervals || component >= dynamics::hcw_state_dimension) {
            throw std::out_of_range("HCW state index is outside the trajectory");
        }
        return node * dynamics::hcw_state_dimension + component;
    }
    [[nodiscard]] std::size_t control_index(
        std::size_t interval,
        std::size_t component
    ) const {
        if (interval >= intervals || component >= dynamics::hcw_control_dimension) {
            throw std::out_of_range("HCW control index is outside the trajectory");
        }
        return state_variables() + interval * dynamics::hcw_control_dimension + component;
    }
};

struct HcwRendezvousDiagnostics {
    double initial_error{0.0};
    double terminal_error{0.0};
    double dynamics_defect{0.0};
    double control_violation{0.0};
    double maximum_component_acceleration{0.0};
    double maximum_acceleration_norm{0.0};
};

namespace hcw_detail {

using Entries = std::set<std::pair<std::size_t, std::size_t>>;  // column, row

inline core::CscPattern make_pattern(
    std::size_t rows,
    std::size_t columns,
    const Entries& entries
) {
    if (rows > static_cast<std::size_t>(std::numeric_limits<Index>::max())
        || columns > static_cast<std::size_t>(std::numeric_limits<Index>::max())) {
        throw std::overflow_error("HCW CQP exceeds the native index range");
    }
    core::CscPattern result{};
    result.rows = static_cast<Index>(rows);
    result.columns = static_cast<Index>(columns);
    result.offsets.resize(columns + 1U, 0);
    for (std::size_t column = 0; column < columns; ++column) {
        result.offsets[column] = static_cast<Index>(result.indices.size());
        auto iterator = entries.lower_bound({column, 0U});
        while (iterator != entries.end() && iterator->first == column) {
            result.indices.push_back(static_cast<Index>(iterator->second));
            ++iterator;
        }
    }
    result.offsets[columns] = static_cast<Index>(result.indices.size());
    result.validate();
    return result;
}

inline std::map<std::pair<std::size_t, std::size_t>, std::size_t> lookup(
    const core::CscPattern& pattern
) {
    std::map<std::pair<std::size_t, std::size_t>, std::size_t> result{};
    for (Index column = 0; column < pattern.columns; ++column) {
        const auto begin = static_cast<std::size_t>(pattern.offsets[static_cast<std::size_t>(column)]);
        const auto end = static_cast<std::size_t>(
            pattern.offsets[static_cast<std::size_t>(column) + 1U]
        );
        for (std::size_t position = begin; position < end; ++position) {
            result.emplace(
                std::pair{
                    static_cast<std::size_t>(pattern.indices[position]),
                    static_cast<std::size_t>(column),
                },
                position
            );
        }
    }
    return result;
}

inline double maximum_absolute(const dynamics::HcwState& values) noexcept {
    double maximum{0.0};
    for (const auto value : values) {
        maximum = std::max(maximum, std::abs(value));
    }
    return maximum;
}

}  // namespace hcw_detail

class HcwRendezvousCqp {
  public:
    explicit HcwRendezvousCqp(
        HcwRendezvousConfig config = HcwRendezvousConfig{}
    )
        : config_(config),
          layout_{config.intervals, config.control_set},
          dynamics_(dynamics::discretise_hcw(config.mean_motion, config.step_seconds)) {
        config_.validate();
        structure_ = build_structure();
        structure_.validate();
        build_position_maps();
        fixed_values_ = build_fixed_values();
    }

    [[nodiscard]] const HcwRendezvousConfig& config() const noexcept { return config_; }
    [[nodiscard]] const HcwRendezvousLayout& layout() const noexcept { return layout_; }
    [[nodiscard]] const dynamics::HcwDiscreteDynamics& discrete_dynamics() const noexcept {
        return dynamics_;
    }
    [[nodiscard]] const core::FixedStructure& structure() const noexcept { return structure_; }

    [[nodiscard]] core::NumericValues values(
        const dynamics::HcwState& initial,
        const dynamics::HcwState& target
    ) const {
        validate_state(initial);
        validate_state(target);
        auto result = fixed_values_;
        result.linear_objective.assign(layout_.variables(), 0.0);
        result.scalar_lower.assign(layout_.scalar_rows(), 0.0);
        result.scalar_upper.assign(layout_.scalar_rows(), 0.0);

        for (std::size_t node = 0; node <= layout_.intervals; ++node) {
            const auto fraction = static_cast<double>(node)
                                  / static_cast<double>(layout_.intervals);
            for (std::size_t component = 0;
                 component < dynamics::hcw_state_dimension;
                 ++component) {
                const auto reference = (1.0 - fraction) * initial[component]
                                       + fraction * target[component];
                result.linear_objective[layout_.state_index(node, component)] =
                    -2.0 * config_.state_weights[component] * reference;
            }
        }

        for (std::size_t component = 0;
             component < dynamics::hcw_state_dimension;
             ++component) {
            result.scalar_lower[layout_.initial_row() + component] = initial[component];
            result.scalar_upper[layout_.initial_row() + component] = initial[component];
            result.scalar_lower[layout_.terminal_row() + component] = target[component];
            result.scalar_upper[layout_.terminal_row() + component] = target[component];
        }
        if (layout_.control_set == HcwControlSet::box) {
            for (std::size_t index = 0; index < layout_.control_variables(); ++index) {
                result.scalar_lower[layout_.control_row() + index] =
                    -config_.maximum_acceleration;
                result.scalar_upper[layout_.control_row() + index] =
                    config_.maximum_acceleration;
            }
        }
        result.validate(structure_);
        return result;
    }

    [[nodiscard]] core::FixedCQP problem(
        const dynamics::HcwState& initial,
        const dynamics::HcwState& target
    ) const {
        return core::FixedCQP(structure_, values(initial, target));
    }

    [[nodiscard]] std::pair<std::vector<dynamics::HcwState>, std::vector<dynamics::HcwControl>>
    decode(const std::vector<double>& primal) const {
        if (primal.size() != layout_.variables()) {
            throw std::invalid_argument("HCW primal has the wrong size");
        }
        if (!std::all_of(primal.begin(), primal.end(), [](double value) {
                return std::isfinite(value);
            })) {
            throw std::invalid_argument("HCW primal must be finite");
        }
        std::vector<dynamics::HcwState> states(layout_.intervals + 1U);
        std::vector<dynamics::HcwControl> controls(layout_.intervals);
        for (std::size_t node = 0; node <= layout_.intervals; ++node) {
            for (std::size_t component = 0;
                 component < dynamics::hcw_state_dimension;
                 ++component) {
                states[node][component] = primal[layout_.state_index(node, component)];
            }
        }
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            for (std::size_t component = 0;
                 component < dynamics::hcw_control_dimension;
                 ++component) {
                controls[interval][component] = primal[
                    layout_.control_index(interval, component)
                ];
            }
        }
        return {std::move(states), std::move(controls)};
    }

    [[nodiscard]] HcwRendezvousDiagnostics diagnostics(
        const std::vector<double>& primal,
        const dynamics::HcwState& initial,
        const dynamics::HcwState& target
    ) const {
        const auto [states, controls] = decode(primal);
        HcwRendezvousDiagnostics result{};
        dynamics::HcwState initial_error{};
        dynamics::HcwState terminal_error{};
        for (std::size_t component = 0;
             component < dynamics::hcw_state_dimension;
             ++component) {
            initial_error[component] = states.front()[component] - initial[component];
            terminal_error[component] = states.back()[component] - target[component];
        }
        result.initial_error = hcw_detail::maximum_absolute(initial_error);
        result.terminal_error = hcw_detail::maximum_absolute(terminal_error);
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto predicted = dynamics::hcw_step(
                dynamics_,
                states[interval],
                controls[interval]
            );
            for (std::size_t component = 0;
                 component < dynamics::hcw_state_dimension;
                 ++component) {
                result.dynamics_defect = std::max(
                    result.dynamics_defect,
                    std::abs(states[interval + 1U][component] - predicted[component])
                );
            }
            double norm_squared{0.0};
            for (const auto value : controls[interval]) {
                result.maximum_component_acceleration = std::max(
                    result.maximum_component_acceleration,
                    std::abs(value)
                );
                norm_squared += value * value;
            }
            const auto norm = std::sqrt(norm_squared);
            result.maximum_acceleration_norm = std::max(
                result.maximum_acceleration_norm,
                norm
            );
            const auto measure = layout_.control_set == HcwControlSet::box
                                     ? result.maximum_component_acceleration
                                     : norm;
            result.control_violation = std::max(
                result.control_violation,
                measure - config_.maximum_acceleration
            );
        }
        result.control_violation = std::max(0.0, result.control_violation);
        return result;
    }

  private:
    HcwRendezvousConfig config_{};
    HcwRendezvousLayout layout_{};
    dynamics::HcwDiscreteDynamics dynamics_{};
    core::FixedStructure structure_{};
    core::NumericValues fixed_values_{};
    std::vector<std::size_t> quadratic_positions_{};
    std::vector<std::size_t> scalar_positions_{};
    std::vector<std::size_t> affine_positions_{};

    [[nodiscard]] core::FixedStructure build_structure() const {
        hcw_detail::Entries quadratic_entries{};
        hcw_detail::Entries scalar_entries{};
        hcw_detail::Entries affine_entries{};
        for (std::size_t variable = 0; variable < layout_.variables(); ++variable) {
            quadratic_entries.emplace(variable, variable);
        }
        for (std::size_t component = 0;
             component < dynamics::hcw_state_dimension;
             ++component) {
            scalar_entries.emplace(
                layout_.state_index(0U, component),
                layout_.initial_row() + component
            );
            scalar_entries.emplace(
                layout_.state_index(layout_.intervals, component),
                layout_.terminal_row() + component
            );
        }
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto row_base = layout_.dynamics_row()
                                  + interval * dynamics::hcw_state_dimension;
            for (std::size_t row = 0; row < dynamics::hcw_state_dimension; ++row) {
                for (std::size_t column = 0;
                     column < dynamics::hcw_state_dimension;
                     ++column) {
                    scalar_entries.emplace(
                        layout_.state_index(interval, column),
                        row_base + row
                    );
                }
                scalar_entries.emplace(
                    layout_.state_index(interval + 1U, row),
                    row_base + row
                );
                for (std::size_t column = 0;
                     column < dynamics::hcw_control_dimension;
                     ++column) {
                    scalar_entries.emplace(
                        layout_.control_index(interval, column),
                        row_base + row
                    );
                }
            }
            if (layout_.control_set == HcwControlSet::box) {
                for (std::size_t component = 0;
                     component < dynamics::hcw_control_dimension;
                     ++component) {
                    scalar_entries.emplace(
                        layout_.control_index(interval, component),
                        layout_.control_row()
                            + interval * dynamics::hcw_control_dimension
                            + component
                    );
                }
            } else {
                const auto row = 4U * interval;
                for (std::size_t component = 0;
                     component < dynamics::hcw_control_dimension;
                     ++component) {
                    affine_entries.emplace(
                        layout_.control_index(interval, component),
                        row + component
                    );
                }
            }
        }

        core::FixedStructure result{};
        result.quadratic = hcw_detail::make_pattern(
            layout_.variables(),
            layout_.variables(),
            quadratic_entries
        );
        result.scalar_constraint = hcw_detail::make_pattern(
            layout_.scalar_rows(),
            layout_.variables(),
            scalar_entries
        );
        if (layout_.control_set == HcwControlSet::second_order_cone) {
            result.affine_cone = hcw_detail::make_pattern(
                layout_.affine_rows(),
                layout_.variables(),
                affine_entries
            );
            result.affine_cones.reserve(layout_.intervals);
            for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
                result.affine_cones.push_back(
                    ConeBlockDescriptor{
                        ConeKind::second_order,
                        static_cast<Index>(4U * interval),
                        2,
                        0.0,
                    }
                );
            }
        }
        return result;
    }

    void build_position_maps() {
        const auto q_lookup = hcw_detail::lookup(structure_.quadratic);
        const auto a_lookup = hcw_detail::lookup(structure_.scalar_constraint);
        quadratic_positions_.reserve(layout_.variables());
        for (std::size_t variable = 0; variable < layout_.variables(); ++variable) {
            quadratic_positions_.push_back(q_lookup.at({variable, variable}));
        }
        scalar_positions_.resize(structure_.scalar_constraint.nonzeros());
        for (std::size_t position = 0; position < scalar_positions_.size(); ++position) {
            scalar_positions_[position] = position;
        }
        if (structure_.affine_cone.has_value()) {
            affine_positions_.resize(structure_.affine_cone->nonzeros());
            for (std::size_t position = 0; position < affine_positions_.size(); ++position) {
                affine_positions_[position] = position;
            }
        }
    }

    [[nodiscard]] core::NumericValues build_fixed_values() const {
        core::NumericValues values{};
        values.quadratic.assign(structure_.quadratic.nonzeros(), 0.0);
        values.scalar_constraint.assign(structure_.scalar_constraint.nonzeros(), 0.0);
        values.affine_cone.assign(
            structure_.affine_cone.has_value() ? structure_.affine_cone->nonzeros() : 0U,
            0.0
        );
        values.linear_objective.assign(layout_.variables(), 0.0);
        values.scalar_lower.assign(layout_.scalar_rows(), 0.0);
        values.scalar_upper.assign(layout_.scalar_rows(), 0.0);
        values.affine_offset.assign(layout_.affine_rows(), 0.0);
        values.variable_lower.assign(
            layout_.variables(),
            -std::numeric_limits<double>::infinity()
        );
        values.variable_upper.assign(
            layout_.variables(),
            std::numeric_limits<double>::infinity()
        );

        const auto q_lookup = hcw_detail::lookup(structure_.quadratic);
        for (std::size_t node = 0; node <= layout_.intervals; ++node) {
            for (std::size_t component = 0;
                 component < dynamics::hcw_state_dimension;
                 ++component) {
                values.quadratic[q_lookup.at(
                    {
                        layout_.state_index(node, component),
                        layout_.state_index(node, component),
                    }
                )] = 2.0 * config_.state_weights[component];
            }
        }
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            for (std::size_t component = 0;
                 component < dynamics::hcw_control_dimension;
                 ++component) {
                values.quadratic[q_lookup.at(
                    {
                        layout_.control_index(interval, component),
                        layout_.control_index(interval, component),
                    }
                )] = 2.0 * config_.control_weights[component];
            }
        }

        const auto a_lookup = hcw_detail::lookup(structure_.scalar_constraint);
        for (std::size_t component = 0;
             component < dynamics::hcw_state_dimension;
             ++component) {
            values.scalar_constraint[a_lookup.at(
                {layout_.initial_row() + component, layout_.state_index(0U, component)}
            )] = 1.0;
            values.scalar_constraint[a_lookup.at(
                {
                    layout_.terminal_row() + component,
                    layout_.state_index(layout_.intervals, component),
                }
            )] = 1.0;
        }
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto row_base = layout_.dynamics_row()
                                  + interval * dynamics::hcw_state_dimension;
            for (std::size_t row = 0; row < dynamics::hcw_state_dimension; ++row) {
                for (std::size_t column = 0;
                     column < dynamics::hcw_state_dimension;
                     ++column) {
                    values.scalar_constraint[a_lookup.at(
                        {row_base + row, layout_.state_index(interval, column)}
                    )] = -dynamics_.state[
                        row * dynamics::hcw_state_dimension + column
                    ];
                }
                values.scalar_constraint[a_lookup.at(
                    {row_base + row, layout_.state_index(interval + 1U, row)}
                )] = 1.0;
                for (std::size_t column = 0;
                     column < dynamics::hcw_control_dimension;
                     ++column) {
                    values.scalar_constraint[a_lookup.at(
                        {row_base + row, layout_.control_index(interval, column)}
                    )] = -dynamics_.control[
                        row * dynamics::hcw_control_dimension + column
                    ];
                }
            }
            if (layout_.control_set == HcwControlSet::box) {
                for (std::size_t component = 0;
                     component < dynamics::hcw_control_dimension;
                     ++component) {
                    const auto row = layout_.control_row()
                                     + interval * dynamics::hcw_control_dimension
                                     + component;
                    values.scalar_constraint[a_lookup.at(
                        {row, layout_.control_index(interval, component)}
                    )] = 1.0;
                }
            }
        }

        if (structure_.affine_cone.has_value()) {
            const auto f_lookup = hcw_detail::lookup(*structure_.affine_cone);
            for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
                const auto row = 4U * interval;
                for (std::size_t component = 0;
                     component < dynamics::hcw_control_dimension;
                     ++component) {
                    values.affine_cone[f_lookup.at(
                        {row + component, layout_.control_index(interval, component)}
                    )] = 1.0;
                }
                values.affine_offset[row + 3U] = config_.maximum_acceleration;
            }
        }
        values.validate(structure_);
        return values;
    }

    static void validate_state(const dynamics::HcwState& state) {
        if (!std::all_of(state.begin(), state.end(), [](double value) {
                return std::isfinite(value);
            })) {
            throw std::invalid_argument("HCW boundary state must be finite");
        }
    }
};

}  // namespace spacepdhcg::transcription
