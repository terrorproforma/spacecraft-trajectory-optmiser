#define SPACEPDHCG_C_API_EXPORTS
#include "spacepdhcg/c_api.h"

#include "spacepdhcg/dynamics/powered_descent_3dof.hpp"
#include "spacepdhcg/dynamics/powered_descent_6dof.hpp"
#include "spacepdhcg/orbitweaver/lambert.hpp"
#include "spacepdhcg/orbitweaver/lambert_family.hpp"
#include "spacepdhcg/planner/describe.hpp"
#include "spacepdhcg/planner/families.hpp"
#include "spacepdhcg/planner/json.hpp"
#include "spacepdhcg/planner/problem.hpp"
#include "spacepdhcg/transcription/powered_descent_3dof_free_time.hpp"
#include "spacepdhcg/transcription/powered_descent_6dof_free_time.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct spacepdhcg_planner {
    std::unique_ptr<spacepdhcg::planner::FamilyAdapter> adapter{};
};

/// Opaque owner of one free-final-time transcription (exactly one of pd3/pd6 is set).
struct spacepdhcg_free_time_handle {
    std::unique_ptr<spacepdhcg::transcription::PoweredDescent3DofFreeTimeSubproblem> pd3{};
    std::unique_ptr<spacepdhcg::transcription::PoweredDescent6DofFreeTimeSubproblem> pd6{};

    [[nodiscard]] uint64_t state_dimension() const noexcept { return pd3 ? 7U : 14U; }
    [[nodiscard]] uint64_t control_dimension() const noexcept { return pd3 ? 4U : 7U; }
    [[nodiscard]] uint64_t intervals() const noexcept {
        return pd3 ? pd3->layout().intervals : pd6->layout().intervals;
    }
    [[nodiscard]] uint64_t control_offset() const noexcept {
        return pd3 ? pd3->layout().control_offset() : pd6->layout().control_offset();
    }
    [[nodiscard]] uint64_t sigma_index() const noexcept {
        return pd3 ? pd3->layout().sigma_index() : pd6->layout().sigma_index();
    }
    [[nodiscard]] uint64_t virtual_offset() const noexcept {
        return pd3 ? pd3->layout().virtual_offset() : pd6->layout().virtual_offset();
    }
    [[nodiscard]] uint64_t epigraph_offset() const noexcept {
        return pd3 ? pd3->layout().epigraph_offset() : pd6->layout().epigraph_offset();
    }
    [[nodiscard]] const spacepdhcg::core::FixedStructure& structure() const noexcept {
        return pd3 ? pd3->structure() : pd6->structure();
    }

    template <std::size_t S, std::size_t C>
    static void unpack(
        const double* states,
        const double* controls,
        std::size_t intervals,
        std::vector<std::array<double, S>>& out_states,
        std::vector<std::array<double, C>>& out_controls
    ) {
        out_states.resize(intervals + 1U);
        out_controls.resize(intervals);
        for (std::size_t node = 0; node <= intervals; ++node) {
            std::copy_n(states + node * S, S, out_states[node].begin());
        }
        for (std::size_t interval = 0; interval < intervals; ++interval) {
            std::copy_n(controls + interval * C, C, out_controls[interval].begin());
        }
    }

    [[nodiscard]] spacepdhcg::core::NumericValues values(
        const double* states,
        const double* controls,
        double sigma,
        const double* initial,
        const double* target,
        double trust_radius,
        double sigma_trust_radius
    ) const {
        if (pd3) {
            std::vector<std::array<double, 7U>> ref_states{};
            std::vector<std::array<double, 4U>> ref_controls{};
            unpack<7U, 4U>(states, controls, pd3->layout().intervals, ref_states, ref_controls);
            std::array<double, 7U> initial_state{};
            std::copy_n(initial, 7U, initial_state.begin());
            const std::array<double, 3U> position{target[0], target[1], target[2]};
            const std::array<double, 3U> velocity{target[3], target[4], target[5]};
            return pd3->values(
                ref_states, ref_controls, sigma, initial_state, position, velocity, trust_radius,
                sigma_trust_radius
            );
        }
        std::vector<std::array<double, 14U>> ref_states{};
        std::vector<std::array<double, 7U>> ref_controls{};
        unpack<14U, 7U>(states, controls, pd6->layout().intervals, ref_states, ref_controls);
        std::array<double, 14U> initial_state{};
        std::array<double, 14U> target_state{};
        std::copy_n(initial, 14U, initial_state.begin());
        std::copy_n(target, 14U, target_state.begin());
        return pd6->values(
            ref_states, ref_controls, sigma, initial_state, target_state, trust_radius,
            sigma_trust_radius
        );
    }

    void replay(const double* states, const double* controls, double sigma, double* next) const {
        if (pd3) {
            const auto k = pd3->layout().intervals;
            const auto d_tau = pd3->config().d_tau();
            for (std::size_t interval = 0; interval < k; ++interval) {
                std::array<double, 7U> state{};
                std::array<double, 4U> control{};
                std::copy_n(states + interval * 7U, 7U, state.begin());
                std::copy_n(controls + interval * 4U, 4U, control.begin());
                const auto result = spacepdhcg::transcription::time_dilated_step<7U, 4U>(
                    pd3->model(), state, control, sigma, d_tau, pd3->config().substeps
                );
                std::copy(result.begin(), result.end(), next + interval * 7U);
            }
            return;
        }
        const auto k = pd6->layout().intervals;
        const auto d_tau = pd6->config().d_tau();
        for (std::size_t interval = 0; interval < k; ++interval) {
            std::array<double, 14U> state{};
            std::array<double, 7U> control{};
            std::copy_n(states + interval * 14U, 14U, state.begin());
            std::copy_n(controls + interval * 7U, 7U, control.begin());
            const auto result = spacepdhcg::transcription::time_dilated_step<14U, 7U>(
                pd6->model(), state, control, sigma, d_tau, pd6->config().substeps
            );
            std::copy(result.begin(), result.end(), next + interval * 14U);
        }
    }

    void project_control(const double* control, double* projected) const {
        if (pd3) {
            std::copy_n(control, 4U, projected);
            return;
        }
        std::array<double, 7U> native{};
        std::copy_n(control, 7U, native.begin());
        const auto result = pd6->project_control(native);
        std::copy(result.begin(), result.end(), projected);
    }
};

namespace {

thread_local std::string last_error{};

spacepdhcg::dynamics::PoweredDescent3DofConfig convert_config(
    const spacepdhcg_powered_descent_3dof_config* config
) {
    if (config == nullptr) {
        throw std::invalid_argument("powered-descent config pointer may not be null");
    }
    return spacepdhcg::dynamics::PoweredDescent3DofConfig{
        {config->gravity[0], config->gravity[1], config->gravity[2]},
        config->mass_flow_coefficient,
        config->minimum_mass,
        config->maximum_thrust,
        config->minimum_sigma,
        config->maximum_tilt_radians,
        config->glide_slope_radians,
    };
}

template <typename Function>
spacepdhcg_status_code guard(Function&& function) noexcept {
    try {
        function();
        last_error.clear();
        return SPACEPDHCG_STATUS_OK;
    } catch (const std::invalid_argument& error) {
        last_error = error.what();
        return SPACEPDHCG_STATUS_INVALID_ARGUMENT;
    } catch (const std::runtime_error& error) {
        last_error = error.what();
        return SPACEPDHCG_STATUS_RUNTIME_ERROR;
    } catch (const std::exception& error) {
        last_error = error.what();
        return SPACEPDHCG_STATUS_INTERNAL_ERROR;
    } catch (...) {
        last_error = "unknown native exception";
        return SPACEPDHCG_STATUS_INTERNAL_ERROR;
    }
}

void require_pointer(const void* pointer, const char* name) {
    if (pointer == nullptr) {
        throw std::invalid_argument(std::string(name) + " pointer may not be null");
    }
}

const spacepdhcg::planner::FamilyAdapter& adapter_of(const spacepdhcg_planner* planner) {
    require_pointer(planner, "planner");
    if (!planner->adapter) {
        throw std::invalid_argument("planner handle has no transcription");
    }
    return *planner->adapter;
}

template <typename Source, typename Destination>
void copy_indices(const Source& source, Destination* destination, const char* name) {
    require_pointer(destination, name);
    for (std::size_t index = 0U; index < source.size(); ++index) {
        destination[index] = static_cast<Destination>(source[index]);
    }
}

void copy_doubles(const std::vector<double>& source, double* destination, const char* name) {
    require_pointer(destination, name);
    std::copy(source.begin(), source.end(), destination);
}

void copy_cones(
    const std::vector<spacepdhcg::ConeBlockDescriptor>& cones,
    spacepdhcg_planner_cone* destination,
    const char* name
) {
    if (cones.empty()) {
        return;
    }
    require_pointer(destination, name);
    for (std::size_t index = 0U; index < cones.size(); ++index) {
        destination[index].kind = static_cast<int32_t>(cones[index].kind);
        destination[index].start = static_cast<int32_t>(cones[index].start);
        destination[index].vector_dimension = static_cast<int32_t>(cones[index].vector_dimension);
        destination[index].power_alpha = cones[index].power_alpha;
    }
}

void write_path_components(
    const std::vector<spacepdhcg::planner::PathComponent>& components,
    spacepdhcg_planner_evaluation& evaluation
) {
    if (components.size() > SPACEPDHCG_PLANNER_MAX_PATH_COMPONENTS) {
        throw std::runtime_error("too many path components for the planner ABI");
    }
    evaluation.path_component_count = components.size();
    evaluation.path_violation = 0.0;
    for (std::size_t index = 0U; index < components.size(); ++index) {
        evaluation.path_normalised[index] = components[index].normalised;
        evaluation.path_physical[index] = components[index].physical;
        evaluation.path_violation =
            std::max(evaluation.path_violation, components[index].normalised);
        std::memset(evaluation.path_names[index], 0, sizeof(evaluation.path_names[index]));
        std::strncpy(
            evaluation.path_names[index],
            components[index].name.c_str(),
            sizeof(evaluation.path_names[index]) - 1U
        );
    }
}

void write_string(
    const std::string& text,
    char* buffer,
    const size_t capacity,
    size_t* required
) {
    require_pointer(required, "required size");
    *required = text.size() + 1U;
    if (buffer == nullptr || capacity == 0U) {
        return;
    }
    const std::size_t count = std::min(capacity - 1U, text.size());
    std::memcpy(buffer, text.data(), count);
    buffer[count] = '\0';
}

void copy_lambert_solution(
    const spacepdhcg::orbitweaver::LambertSolution& solution,
    spacepdhcg_lambert_result& result
) {
    std::copy(
        solution.departure_velocity.begin(),
        solution.departure_velocity.end(),
        result.departure_velocity
    );
    std::copy(
        solution.arrival_velocity.begin(),
        solution.arrival_velocity.end(),
        result.arrival_velocity
    );
    result.universal_parameter = solution.universal_parameter;
    result.transfer_angle_radians = solution.transfer_angle_radians;
    result.iterations = static_cast<uint64_t>(solution.iterations);
    result.time_of_flight_residual = solution.time_of_flight_residual;
}

}  // namespace

extern "C" {

uint32_t spacepdhcg_c_api_version(void) { return 1U; }

const char* spacepdhcg_native_version(void) { return "0.1.0.dev0"; }

const char* spacepdhcg_last_error(void) { return last_error.c_str(); }

void spacepdhcg_default_powered_descent_3dof_config(
    spacepdhcg_powered_descent_3dof_config* config
) {
    if (config == nullptr) {
        last_error = "powered-descent config pointer may not be null";
        return;
    }
    const spacepdhcg::dynamics::PoweredDescent3DofConfig defaults{};
    std::copy(defaults.gravity.begin(), defaults.gravity.end(), config->gravity);
    config->mass_flow_coefficient = defaults.mass_flow_coefficient;
    config->minimum_mass = defaults.minimum_mass;
    config->maximum_thrust = defaults.maximum_thrust;
    config->minimum_sigma = defaults.minimum_sigma;
    config->maximum_tilt_radians = defaults.maximum_tilt_radians;
    config->glide_slope_radians = defaults.glide_slope_radians;
    last_error.clear();
}

spacepdhcg_status_code spacepdhcg_powered_descent_3dof_dynamics(
    const spacepdhcg_powered_descent_3dof_config* config,
    const double state[7],
    const double control[4],
    double derivative[7]
) {
    return guard([&] {
        require_pointer(state, "state");
        require_pointer(control, "control");
        require_pointer(derivative, "derivative");
        const spacepdhcg::dynamics::PoweredDescent3DofModel model(convert_config(config));
        spacepdhcg::dynamics::PoweredDescentState native_state{};
        spacepdhcg::dynamics::PoweredDescentControl native_control{};
        std::copy_n(state, native_state.size(), native_state.begin());
        std::copy_n(control, native_control.size(), native_control.begin());
        const auto result = model.dynamics(native_state, native_control);
        std::copy(result.begin(), result.end(), derivative);
    });
}

spacepdhcg_status_code spacepdhcg_powered_descent_3dof_jacobians(
    const spacepdhcg_powered_descent_3dof_config* config,
    const double state[7],
    const double control[4],
    double state_jacobian[49],
    double control_jacobian[28]
) {
    return guard([&] {
        require_pointer(state, "state");
        require_pointer(control, "control");
        require_pointer(state_jacobian, "state Jacobian");
        require_pointer(control_jacobian, "control Jacobian");
        const spacepdhcg::dynamics::PoweredDescent3DofModel model(convert_config(config));
        spacepdhcg::dynamics::PoweredDescentState native_state{};
        spacepdhcg::dynamics::PoweredDescentControl native_control{};
        std::copy_n(state, native_state.size(), native_state.begin());
        std::copy_n(control, native_control.size(), native_control.begin());
        const auto result = model.jacobians(native_state, native_control);
        std::copy(result.state.begin(), result.state.end(), state_jacobian);
        std::copy(result.control.begin(), result.control.end(), control_jacobian);
    });
}

spacepdhcg_status_code spacepdhcg_lambert_zero_revolution(
    const double departure_position[3],
    const double arrival_position[3],
    double time_of_flight,
    double gravitational_parameter,
    int long_way,
    double time_tolerance,
    uint64_t maximum_iterations,
    spacepdhcg_lambert_result* result
) {
    return guard([&] {
        require_pointer(departure_position, "departure position");
        require_pointer(arrival_position, "arrival position");
        require_pointer(result, "Lambert result");
        const spacepdhcg::orbitweaver::Vector3 departure{
            departure_position[0], departure_position[1], departure_position[2]
        };
        const spacepdhcg::orbitweaver::Vector3 arrival{
            arrival_position[0], arrival_position[1], arrival_position[2]
        };
        const auto solution = spacepdhcg::orbitweaver::solve_lambert_zero_revolution(
            departure,
            arrival,
            time_of_flight,
            gravitational_parameter,
            long_way != 0,
            time_tolerance,
            static_cast<std::size_t>(maximum_iterations)
        );
        std::copy(
            solution.departure_velocity.begin(),
            solution.departure_velocity.end(),
            result->departure_velocity
        );
        std::copy(
            solution.arrival_velocity.begin(),
            solution.arrival_velocity.end(),
            result->arrival_velocity
        );
        result->universal_parameter = solution.universal_parameter;
        result->transfer_angle_radians = solution.transfer_angle_radians;
        result->iterations = static_cast<uint64_t>(solution.iterations);
        result->time_of_flight_residual = solution.time_of_flight_residual;
    });
}

size_t spacepdhcg_lambert_family_result_stride(
    const uint64_t supported_maximum_revolutions
) {
    if (supported_maximum_revolutions
        > (std::numeric_limits<size_t>::max() / 4U) - 1U) {
        return 0U;
    }
    return 2U * (1U + 2U * static_cast<size_t>(supported_maximum_revolutions));
}

spacepdhcg_status_code spacepdhcg_lambert_family_batch_cpu(
    const spacepdhcg_lambert_family_request* requests,
    const size_t request_count,
    const uint64_t supported_maximum_revolutions,
    spacepdhcg_lambert_family_result* results,
    const size_t result_capacity
) {
    return guard([&] {
        require_pointer(requests, "Lambert family requests");
        require_pointer(results, "Lambert family results");
        if (request_count == 0U) {
            throw std::invalid_argument("Lambert family batch may not be empty");
        }
        const auto stride =
            spacepdhcg_lambert_family_result_stride(supported_maximum_revolutions);
        if (stride == 0U || request_count > result_capacity / stride) {
            throw std::invalid_argument("Lambert family result capacity is insufficient");
        }
        for (size_t input = 0U; input < request_count; ++input) {
            const auto& request = requests[input];
            auto* output = results + input * stride;
            for (size_t slot = 0U; slot < stride; ++slot) {
                output[slot] = {};
                output[slot].deterministic_id = request.deterministic_id;
                output[slot].input_index = input;
                output[slot].family_index = slot;
                output[slot].status = SPACEPDHCG_LAMBERT_FAMILY_UNSUPPORTED;
            }
            for (uint64_t direction = 0U; direction < 2U; ++direction) {
                const auto included = direction == 0U ? request.include_short_way
                                                      : request.include_long_way;
                if (included == 0) {
                    continue;
                }
                const auto base =
                    static_cast<size_t>(direction)
                    * (1U + 2U * static_cast<size_t>(supported_maximum_revolutions));
                const auto requested_revolutions = std::min(
                    request.maximum_revolutions,
                    supported_maximum_revolutions
                );
                for (uint64_t revolution = 0U;
                     revolution <= requested_revolutions;
                     ++revolution) {
                    if (revolution == 0U) {
                        output[base].status = SPACEPDHCG_LAMBERT_FAMILY_NO_SOLUTION;
                        output[base].long_way = static_cast<int>(direction);
                        continue;
                    }
                    const auto first =
                        base + 1U + 2U * static_cast<size_t>(revolution - 1U);
                    output[first].status = SPACEPDHCG_LAMBERT_FAMILY_NO_SOLUTION;
                    output[first + 1U].status = SPACEPDHCG_LAMBERT_FAMILY_NO_SOLUTION;
                    output[first].long_way = output[first + 1U].long_way =
                        static_cast<int>(direction);
                    output[first].revolutions = output[first + 1U].revolutions =
                        revolution;
                    output[first].parameter_branch = 1;
                    output[first + 1U].parameter_branch = 2;
                }
            }
            try {
                const auto departure = spacepdhcg::orbitweaver::Vector3{
                    request.departure_position[0],
                    request.departure_position[1],
                    request.departure_position[2],
                };
                const auto arrival = spacepdhcg::orbitweaver::Vector3{
                    request.arrival_position[0],
                    request.arrival_position[1],
                    request.arrival_position[2],
                };
                const auto family =
                    spacepdhcg::orbitweaver::enumerate_lambert_families(
                        departure,
                        arrival,
                        request.time_of_flight,
                        request.gravitational_parameter,
                        static_cast<size_t>(std::min(
                            request.maximum_revolutions,
                            supported_maximum_revolutions
                        )),
                        request.include_short_way != 0,
                        request.include_long_way != 0,
                        request.time_tolerance,
                        static_cast<size_t>(request.maximum_iterations),
                        static_cast<size_t>(request.scan_samples_per_band)
                    );
                for (const auto& member : family) {
                    const auto base =
                        static_cast<size_t>(member.long_way)
                        * (1U + 2U * static_cast<size_t>(supported_maximum_revolutions));
                    size_t slot = base;
                    if (member.revolutions > 0U) {
                        slot += 1U + 2U * (member.revolutions - 1U);
                        if (member.branch
                            == spacepdhcg::orbitweaver::LambertParameterBranch::
                                higher_parameter) {
                            ++slot;
                        }
                    }
                    auto& destination = output[slot];
                    destination.status = SPACEPDHCG_LAMBERT_FAMILY_FEASIBLE;
                    destination.revolutions = member.revolutions;
                    destination.long_way = member.long_way ? 1 : 0;
                    destination.parameter_branch = static_cast<int>(member.branch);
                    copy_lambert_solution(member.solution, destination.solution);
                }
            } catch (const std::invalid_argument&) {
                for (size_t slot = 0U; slot < stride; ++slot) {
                    output[slot].status = SPACEPDHCG_LAMBERT_FAMILY_INVALID_INPUT;
                }
            } catch (const std::exception&) {
                for (size_t slot = 0U; slot < stride; ++slot) {
                    if (output[slot].status
                        == SPACEPDHCG_LAMBERT_FAMILY_NO_SOLUTION) {
                        output[slot].status =
                            SPACEPDHCG_LAMBERT_FAMILY_NUMERICAL_FAILURE;
                    }
                }
            }
        }
    });
}

spacepdhcg_status_code spacepdhcg_planner_create(
    const char* problem_json,
    spacepdhcg_planner** planner
) {
    return guard([&] {
        require_pointer(problem_json, "problem JSON");
        require_pointer(planner, "planner handle");
        *planner = nullptr;
        auto problem = spacepdhcg::planner::parse_problem_text(problem_json);
        auto handle = std::make_unique<spacepdhcg_planner>();
        handle->adapter = spacepdhcg::planner::make_adapter(std::move(problem));
        *planner = handle.release();
    });
}

void spacepdhcg_planner_destroy(spacepdhcg_planner* planner) {
    delete planner;  // NOLINT(cppcoreguidelines-owning-memory)
}

spacepdhcg_status_code spacepdhcg_planner_get_dimensions(
    const spacepdhcg_planner* planner,
    spacepdhcg_planner_dimensions* dimensions
) {
    return guard([&] {
        const auto& adapter = adapter_of(planner);
        require_pointer(dimensions, "dimensions");
        const auto& info = adapter.layout();
        const auto& structure = adapter.structure();
        *dimensions = {};
        dimensions->state_dimension = info.state_dimension;
        dimensions->control_dimension = info.control_dimension;
        dimensions->intervals = info.intervals;
        dimensions->terminal_dimension = info.terminal_dimension;
        dimensions->variables = info.variables;
        dimensions->scalar_rows = info.scalar_rows;
        dimensions->affine_rows = info.affine_rows;
        dimensions->quadratic_nonzeros = structure.quadratic.nonzeros();
        dimensions->scalar_nonzeros = structure.scalar_constraint.nonzeros();
        dimensions->affine_nonzeros =
            structure.affine_cone.has_value() ? structure.affine_cone->nonzeros() : 0U;
        dimensions->affine_cone_count = structure.affine_cones.size();
        dimensions->variable_cone_count = structure.variable_cones.size();
        dimensions->virtual_variable_count = info.virtual_variables.size();
        dimensions->step_seconds = adapter.problem().step_seconds;
        dimensions->initial_trust_radius = adapter.problem().solver.trust.initial_radius;
    });
}

spacepdhcg_status_code spacepdhcg_planner_structure(
    const spacepdhcg_planner* planner,
    int32_t* quadratic_offsets,
    int32_t* quadratic_indices,
    int32_t* scalar_offsets,
    int32_t* scalar_indices,
    int32_t* affine_offsets,
    int32_t* affine_indices,
    spacepdhcg_planner_cone* affine_cones,
    spacepdhcg_planner_cone* variable_cones,
    int32_t* state_variables,
    int32_t* control_variables,
    int32_t* virtual_variables
) {
    return guard([&] {
        const auto& adapter = adapter_of(planner);
        const auto& structure = adapter.structure();
        const auto& info = adapter.layout();
        copy_indices(structure.quadratic.offsets, quadratic_offsets, "quadratic offsets");
        copy_indices(structure.quadratic.indices, quadratic_indices, "quadratic indices");
        copy_indices(structure.scalar_constraint.offsets, scalar_offsets, "scalar offsets");
        copy_indices(structure.scalar_constraint.indices, scalar_indices, "scalar indices");
        require_pointer(affine_offsets, "affine offsets");
        if (structure.affine_cone.has_value()) {
            copy_indices(structure.affine_cone->offsets, affine_offsets, "affine offsets");
            if (!structure.affine_cone->indices.empty()) {
                copy_indices(structure.affine_cone->indices, affine_indices, "affine indices");
            }
        } else {
            for (std::size_t column = 0U; column <= info.variables; ++column) {
                affine_offsets[column] = 0;
            }
        }
        copy_cones(structure.affine_cones, affine_cones, "affine cones");
        copy_cones(structure.variable_cones, variable_cones, "variable cones");
        copy_indices(info.state_variables, state_variables, "state variables");
        copy_indices(info.control_variables, control_variables, "control variables");
        if (!info.virtual_variables.empty()) {
            copy_indices(info.virtual_variables, virtual_variables, "virtual variables");
        }
    });
}

spacepdhcg_status_code spacepdhcg_planner_values(
    const spacepdhcg_planner* planner,
    const double* reference_states,
    const double* reference_controls,
    const double trust_radius,
    double* quadratic,
    double* scalar_constraint,
    double* affine_cone,
    double* linear_objective,
    double* scalar_lower,
    double* scalar_upper,
    double* affine_offset,
    double* variable_lower,
    double* variable_upper
) {
    return guard([&] {
        const auto& adapter = adapter_of(planner);
        require_pointer(reference_states, "reference states");
        require_pointer(reference_controls, "reference controls");
        const auto& info = adapter.layout();
        spacepdhcg::planner::Trajectory reference{};
        reference.states.assign(
            reference_states,
            reference_states + (info.intervals + 1U) * info.state_dimension
        );
        reference.controls.assign(
            reference_controls,
            reference_controls + info.intervals * info.control_dimension
        );
        const auto values = adapter.values(reference, trust_radius);
        copy_doubles(values.quadratic, quadratic, "quadratic values");
        copy_doubles(values.scalar_constraint, scalar_constraint, "scalar values");
        if (!values.affine_cone.empty()) {
            copy_doubles(values.affine_cone, affine_cone, "affine values");
        }
        copy_doubles(values.linear_objective, linear_objective, "linear objective");
        copy_doubles(values.scalar_lower, scalar_lower, "scalar lower");
        copy_doubles(values.scalar_upper, scalar_upper, "scalar upper");
        if (!values.affine_offset.empty()) {
            copy_doubles(values.affine_offset, affine_offset, "affine offset");
        }
        copy_doubles(values.variable_lower, variable_lower, "variable lower");
        copy_doubles(values.variable_upper, variable_upper, "variable upper");
    });
}

spacepdhcg_status_code spacepdhcg_planner_initial_reference(
    const spacepdhcg_planner* planner,
    double* states,
    double* controls
) {
    return guard([&] {
        const auto& adapter = adapter_of(planner);
        const auto reference = adapter.initial_reference();
        copy_doubles(reference.states, states, "reference states");
        copy_doubles(reference.controls, controls, "reference controls");
    });
}

spacepdhcg_status_code spacepdhcg_planner_rollout(
    const spacepdhcg_planner* planner,
    const double* initial_state,
    const double* controls,
    const uint64_t intervals,
    const uint64_t substeps,
    double* states
) {
    return guard([&] {
        const auto& adapter = adapter_of(planner);
        require_pointer(initial_state, "initial state");
        require_pointer(controls, "controls");
        require_pointer(states, "states");
        if (intervals == 0U || substeps == 0U) {
            throw std::invalid_argument("rollout requires positive intervals and substeps");
        }
        const auto& info = adapter.layout();
        const std::vector<double> initial(initial_state, initial_state + info.state_dimension);
        const std::vector<double> control_vector(
            controls, controls + static_cast<std::size_t>(intervals) * info.control_dimension
        );
        const auto replay = adapter.rollout(
            initial, control_vector, static_cast<std::size_t>(substeps)
        );
        std::copy(replay.begin(), replay.end(), states);
    });
}

spacepdhcg_status_code spacepdhcg_planner_evaluate(
    const spacepdhcg_planner* planner,
    const double* states,
    const double* controls,
    spacepdhcg_planner_evaluation* evaluation
) {
    return guard([&] {
        const auto& adapter = adapter_of(planner);
        require_pointer(states, "states");
        require_pointer(controls, "controls");
        require_pointer(evaluation, "evaluation");
        const auto& info = adapter.layout();
        const std::vector<double> state_vector(
            states, states + (info.intervals + 1U) * info.state_dimension
        );
        const std::vector<double> control_vector(
            controls, controls + info.intervals * info.control_dimension
        );
        const auto result = adapter.evaluate(state_vector, control_vector);
        *evaluation = {};
        evaluation->objective = result.objective;
        evaluation->terminal_residual = result.terminal_residual;
        evaluation->terminal_position_error = result.terminal_position_error;
        evaluation->terminal_velocity_error = result.terminal_velocity_error;
        evaluation->propellant_used = result.propellant_used;
        evaluation->final_mass = result.final_mass;
        write_path_components(result.path, *evaluation);
    });
}

spacepdhcg_status_code spacepdhcg_planner_path_components(
    const spacepdhcg_planner* planner,
    const double* states,
    const double* controls,
    const uint64_t intervals,
    spacepdhcg_planner_evaluation* evaluation
) {
    return guard([&] {
        const auto& adapter = adapter_of(planner);
        require_pointer(states, "states");
        require_pointer(controls, "controls");
        require_pointer(evaluation, "evaluation");
        if (intervals == 0U) {
            throw std::invalid_argument("path evaluation requires at least one interval");
        }
        const auto& info = adapter.layout();
        const std::vector<double> state_vector(
            states, states + (static_cast<std::size_t>(intervals) + 1U) * info.state_dimension
        );
        const std::vector<double> control_vector(
            controls, controls + static_cast<std::size_t>(intervals) * info.control_dimension
        );
        *evaluation = {};
        write_path_components(adapter.path_components(state_vector, control_vector), *evaluation);
    });
}

spacepdhcg_status_code spacepdhcg_planner_describe(
    const spacepdhcg_planner* planner,
    char* buffer,
    const size_t capacity,
    size_t* required
) {
    return guard([&] {
        const auto& adapter = adapter_of(planner);
        write_string(
            spacepdhcg::planner::json::dump(
                spacepdhcg::planner::describe_problem(adapter.problem())
            ),
            buffer,
            capacity,
            required
        );
    });
}

spacepdhcg_status_code spacepdhcg_planner_default_document(
    const char* family,
    char* buffer,
    const size_t capacity,
    size_t* required
) {
    return guard([&] {
        require_pointer(family, "family");
        write_string(
            spacepdhcg::planner::json::dump(
                spacepdhcg::planner::default_document(spacepdhcg::planner::parse_family(family))
            ),
            buffer,
            capacity,
            required
        );
    });
}

/* ---------------------------------- free-final-time ---------------------------------- */

void spacepdhcg_default_pd3_fft_config(spacepdhcg_pd3_fft_config* config) {
    if (config == nullptr) {
        last_error = "pd3_fft config pointer may not be null";
        return;
    }
    spacepdhcg_default_powered_descent_3dof_config(&config->model);
    const spacepdhcg::transcription::PoweredDescent3DofFreeTimeConfig defaults{};
    config->intervals = defaults.intervals;
    config->substeps = defaults.substeps;
    config->sigma_minimum = defaults.sigma_minimum;
    config->sigma_maximum = defaults.sigma_maximum;
    config->trust_radius = defaults.trust_radius;
    config->sigma_trust_radius = defaults.sigma_trust_radius;
    config->virtual_l1_weight = defaults.virtual_l1_weight;
    config->virtual_quadratic_weight = defaults.virtual_quadratic_weight;
    config->virtual_epigraph_regularisation = defaults.virtual_epigraph_regularisation;
    config->fuel_weight = defaults.fuel_weight;
    config->time_weight = defaults.time_weight;
    config->sigma_tracking_weight = defaults.sigma_tracking_weight;
    std::copy(
        defaults.state_tracking_weights.begin(), defaults.state_tracking_weights.end(),
        config->state_tracking_weights
    );
    std::copy(
        defaults.control_tracking_weights.begin(), defaults.control_tracking_weights.end(),
        config->control_tracking_weights
    );
    std::copy(
        defaults.state_trust_scales.begin(), defaults.state_trust_scales.end(),
        config->state_trust_scales
    );
    std::copy(
        defaults.control_trust_scales.begin(), defaults.control_trust_scales.end(),
        config->control_trust_scales
    );
    last_error.clear();
}

void spacepdhcg_default_pd6_fft_config(spacepdhcg_pd6_fft_config* config) {
    if (config == nullptr) {
        last_error = "pd6_fft config pointer may not be null";
        return;
    }
    const spacepdhcg::dynamics::PoweredDescent6DofConfig model{};
    std::copy(model.gravity.begin(), model.gravity.end(), config->gravity);
    std::copy(
        model.principal_inertia.begin(), model.principal_inertia.end(), config->principal_inertia
    );
    config->mass_flow_coefficient = model.mass_flow_coefficient;
    config->minimum_mass = model.minimum_mass;
    config->maximum_thrust = model.maximum_thrust;
    config->minimum_sigma = model.minimum_sigma;
    config->maximum_torque = model.maximum_torque;
    config->maximum_angular_rate = model.maximum_angular_rate;
    config->maximum_tilt_radians = model.maximum_tilt_radians;
    config->glide_slope_radians = model.glide_slope_radians;
    const spacepdhcg::transcription::PoweredDescent6DofFreeTimeConfig defaults{};
    config->intervals = defaults.intervals;
    config->substeps = defaults.substeps;
    config->sigma_minimum = defaults.sigma_minimum;
    config->sigma_maximum = defaults.sigma_maximum;
    config->trust_radius = defaults.trust_radius;
    config->sigma_trust_radius = defaults.sigma_trust_radius;
    config->virtual_l1_weight = defaults.virtual_l1_weight;
    config->virtual_quadratic_weight = defaults.virtual_quadratic_weight;
    config->virtual_epigraph_regularisation = defaults.virtual_epigraph_regularisation;
    config->fuel_weight = defaults.fuel_weight;
    config->time_weight = defaults.time_weight;
    config->sigma_tracking_weight = defaults.sigma_tracking_weight;
    config->maximum_attitude_tilt_radians = defaults.maximum_attitude_tilt_radians;
    config->thrust_norm_mode = static_cast<int32_t>(defaults.thrust_norm_mode);
    config->torque_mode = static_cast<int32_t>(defaults.torque_mode);
    config->terminal_thrust_axial = defaults.terminal_thrust_axial ? 1 : 0;
    config->reserved = 0;
    std::copy(defaults.thrust_arm.begin(), defaults.thrust_arm.end(), config->thrust_arm);
    for (std::size_t component = 0; component < 14U; ++component) {
        config->initial_fixed[component] = defaults.initial_fixed[component] ? 1U : 0U;
        config->terminal_fixed[component] = defaults.terminal_fixed[component] ? 1U : 0U;
    }
    std::copy(
        defaults.state_tracking_weights.begin(), defaults.state_tracking_weights.end(),
        config->state_tracking_weights
    );
    std::copy(
        defaults.control_tracking_weights.begin(), defaults.control_tracking_weights.end(),
        config->control_tracking_weights
    );
    std::copy(
        defaults.state_trust_scales.begin(), defaults.state_trust_scales.end(),
        config->state_trust_scales
    );
    std::copy(
        defaults.control_trust_scales.begin(), defaults.control_trust_scales.end(),
        config->control_trust_scales
    );
    last_error.clear();
}

spacepdhcg_status_code spacepdhcg_pd3_fft_create(
    const spacepdhcg_pd3_fft_config* config,
    spacepdhcg_free_time_handle** handle
) {
    return guard([&] {
        require_pointer(config, "pd3_fft config");
        require_pointer(handle, "pd3_fft handle");
        spacepdhcg::transcription::PoweredDescent3DofFreeTimeConfig native{};
        native.intervals = static_cast<std::size_t>(config->intervals);
        native.substeps = static_cast<std::size_t>(config->substeps);
        native.sigma_minimum = config->sigma_minimum;
        native.sigma_maximum = config->sigma_maximum;
        native.trust_radius = config->trust_radius;
        native.sigma_trust_radius = config->sigma_trust_radius;
        native.virtual_l1_weight = config->virtual_l1_weight;
        native.virtual_quadratic_weight = config->virtual_quadratic_weight;
        native.virtual_epigraph_regularisation = config->virtual_epigraph_regularisation;
        native.fuel_weight = config->fuel_weight;
        native.time_weight = config->time_weight;
        native.sigma_tracking_weight = config->sigma_tracking_weight;
        std::copy_n(config->state_tracking_weights, 7U, native.state_tracking_weights.begin());
        std::copy_n(config->control_tracking_weights, 4U, native.control_tracking_weights.begin());
        std::copy_n(config->state_trust_scales, 7U, native.state_trust_scales.begin());
        std::copy_n(config->control_trust_scales, 4U, native.control_trust_scales.begin());
        auto owned = std::make_unique<spacepdhcg_free_time_handle>();
        owned->pd3 = std::make_unique<spacepdhcg::transcription::PoweredDescent3DofFreeTimeSubproblem>(
            spacepdhcg::dynamics::PoweredDescent3DofModel(convert_config(&config->model)), native
        );
        *handle = owned.release();
    });
}

spacepdhcg_status_code spacepdhcg_pd6_fft_create(
    const spacepdhcg_pd6_fft_config* config,
    spacepdhcg_free_time_handle** handle
) {
    return guard([&] {
        require_pointer(config, "pd6_fft config");
        require_pointer(handle, "pd6_fft handle");
        spacepdhcg::dynamics::PoweredDescent6DofConfig model{};
        std::copy_n(config->gravity, 3U, model.gravity.begin());
        std::copy_n(config->principal_inertia, 3U, model.principal_inertia.begin());
        model.mass_flow_coefficient = config->mass_flow_coefficient;
        model.minimum_mass = config->minimum_mass;
        model.maximum_thrust = config->maximum_thrust;
        model.minimum_sigma = config->minimum_sigma;
        model.maximum_torque = config->maximum_torque;
        model.maximum_angular_rate = config->maximum_angular_rate;
        model.maximum_tilt_radians = config->maximum_tilt_radians;
        model.glide_slope_radians = config->glide_slope_radians;
        spacepdhcg::transcription::PoweredDescent6DofFreeTimeConfig native{};
        native.intervals = static_cast<std::size_t>(config->intervals);
        native.substeps = static_cast<std::size_t>(config->substeps);
        native.sigma_minimum = config->sigma_minimum;
        native.sigma_maximum = config->sigma_maximum;
        native.trust_radius = config->trust_radius;
        native.sigma_trust_radius = config->sigma_trust_radius;
        native.virtual_l1_weight = config->virtual_l1_weight;
        native.virtual_quadratic_weight = config->virtual_quadratic_weight;
        native.virtual_epigraph_regularisation = config->virtual_epigraph_regularisation;
        native.fuel_weight = config->fuel_weight;
        native.time_weight = config->time_weight;
        native.sigma_tracking_weight = config->sigma_tracking_weight;
        native.maximum_attitude_tilt_radians = config->maximum_attitude_tilt_radians;
        if (config->thrust_norm_mode < 0 || config->thrust_norm_mode > 1) {
            throw std::invalid_argument("pd6_fft thrust_norm_mode must be 0 or 1");
        }
        if (config->torque_mode < 0 || config->torque_mode > 1) {
            throw std::invalid_argument("pd6_fft torque_mode must be 0 or 1");
        }
        native.thrust_norm_mode =
            static_cast<spacepdhcg::transcription::FreeTimeThrustNormMode>(config->thrust_norm_mode);
        native.torque_mode =
            static_cast<spacepdhcg::transcription::FreeTimeTorqueMode>(config->torque_mode);
        native.terminal_thrust_axial = config->terminal_thrust_axial != 0;
        std::copy_n(config->thrust_arm, 3U, native.thrust_arm.begin());
        for (std::size_t component = 0; component < 14U; ++component) {
            native.initial_fixed[component] = config->initial_fixed[component] != 0U;
            native.terminal_fixed[component] = config->terminal_fixed[component] != 0U;
        }
        std::copy_n(config->state_tracking_weights, 14U, native.state_tracking_weights.begin());
        std::copy_n(config->control_tracking_weights, 7U, native.control_tracking_weights.begin());
        std::copy_n(config->state_trust_scales, 14U, native.state_trust_scales.begin());
        std::copy_n(config->control_trust_scales, 7U, native.control_trust_scales.begin());
        auto owned = std::make_unique<spacepdhcg_free_time_handle>();
        owned->pd6 = std::make_unique<spacepdhcg::transcription::PoweredDescent6DofFreeTimeSubproblem>(
            spacepdhcg::dynamics::PoweredDescent6DofModel(model), native
        );
        *handle = owned.release();
    });
}

void spacepdhcg_free_time_destroy(spacepdhcg_free_time_handle* handle) { delete handle; }

spacepdhcg_status_code spacepdhcg_free_time_sizes_of(
    const spacepdhcg_free_time_handle* handle,
    spacepdhcg_free_time_sizes* sizes
) {
    return guard([&] {
        require_pointer(handle, "free-time handle");
        require_pointer(sizes, "free-time sizes");
        const auto& structure = handle->structure();
        sizes->state_dimension = handle->state_dimension();
        sizes->control_dimension = handle->control_dimension();
        sizes->intervals = handle->intervals();
        sizes->variables = static_cast<uint64_t>(structure.variables());
        sizes->scalar_rows = static_cast<uint64_t>(structure.scalar_constraint.rows);
        sizes->affine_rows = static_cast<uint64_t>(structure.affine_cone->rows);
        sizes->quadratic_nonzeros = structure.quadratic.nonzeros();
        sizes->scalar_nonzeros = structure.scalar_constraint.nonzeros();
        sizes->affine_nonzeros = structure.affine_cone->nonzeros();
        sizes->cone_count = structure.affine_cones.size();
        sizes->control_offset = handle->control_offset();
        sizes->sigma_index = handle->sigma_index();
        sizes->virtual_offset = handle->virtual_offset();
        sizes->epigraph_offset = handle->epigraph_offset();
        sizes->dynamics_row_start = handle->state_dimension();
        sizes->topology_fingerprint = structure.fingerprint();
    });
}

spacepdhcg_status_code spacepdhcg_free_time_structure(
    const spacepdhcg_free_time_handle* handle,
    int32_t* quadratic_offsets,
    int32_t* quadratic_indices,
    int32_t* scalar_offsets,
    int32_t* scalar_indices,
    int32_t* affine_offsets,
    int32_t* affine_indices,
    int32_t* cone_starts,
    int32_t* cone_vector_dimensions
) {
    return guard([&] {
        require_pointer(handle, "free-time handle");
        const auto& structure = handle->structure();
        const auto copy_pattern = [](const spacepdhcg::core::CscPattern& pattern,
                                     int32_t* offsets,
                                     int32_t* indices,
                                     const char* name) {
            require_pointer(offsets, name);
            require_pointer(indices, name);
            std::copy(pattern.offsets.begin(), pattern.offsets.end(), offsets);
            std::copy(pattern.indices.begin(), pattern.indices.end(), indices);
        };
        copy_pattern(structure.quadratic, quadratic_offsets, quadratic_indices, "quadratic pattern");
        copy_pattern(structure.scalar_constraint, scalar_offsets, scalar_indices, "scalar pattern");
        copy_pattern(*structure.affine_cone, affine_offsets, affine_indices, "affine pattern");
        require_pointer(cone_starts, "cone starts");
        require_pointer(cone_vector_dimensions, "cone dimensions");
        for (std::size_t cone = 0; cone < structure.affine_cones.size(); ++cone) {
            cone_starts[cone] = structure.affine_cones[cone].start;
            cone_vector_dimensions[cone] = structure.affine_cones[cone].vector_dimension;
        }
    });
}

spacepdhcg_status_code spacepdhcg_free_time_values(
    const spacepdhcg_free_time_handle* handle,
    const double* states,
    const double* controls,
    double sigma,
    const double* initial,
    const double* target,
    double trust_radius,
    double sigma_trust_radius,
    double* quadratic,
    double* scalar_values,
    double* linear_objective,
    double* scalar_lower,
    double* scalar_upper,
    double* affine_values,
    double* affine_offset,
    double* variable_lower,
    double* variable_upper
) {
    return guard([&] {
        require_pointer(handle, "free-time handle");
        require_pointer(states, "reference states");
        require_pointer(controls, "reference controls");
        require_pointer(initial, "initial state");
        require_pointer(target, "target state");
        const double* outputs[] = {
            quadratic, scalar_values, linear_objective, scalar_lower, scalar_upper,
            affine_values, affine_offset, variable_lower, variable_upper,
        };
        for (const auto* output : outputs) {
            require_pointer(output, "free-time values output");
        }
        const auto values = handle->values(
            states, controls, sigma, initial, target, trust_radius, sigma_trust_radius
        );
        std::copy(values.quadratic.begin(), values.quadratic.end(), quadratic);
        std::copy(values.scalar_constraint.begin(), values.scalar_constraint.end(), scalar_values);
        std::copy(values.linear_objective.begin(), values.linear_objective.end(), linear_objective);
        std::copy(values.scalar_lower.begin(), values.scalar_lower.end(), scalar_lower);
        std::copy(values.scalar_upper.begin(), values.scalar_upper.end(), scalar_upper);
        std::copy(values.affine_cone.begin(), values.affine_cone.end(), affine_values);
        std::copy(values.affine_offset.begin(), values.affine_offset.end(), affine_offset);
        std::copy(values.variable_lower.begin(), values.variable_lower.end(), variable_lower);
        std::copy(values.variable_upper.begin(), values.variable_upper.end(), variable_upper);
    });
}

spacepdhcg_status_code spacepdhcg_free_time_replay(
    const spacepdhcg_free_time_handle* handle,
    const double* states,
    const double* controls,
    double sigma,
    double* next_states
) {
    return guard([&] {
        require_pointer(handle, "free-time handle");
        require_pointer(states, "reference states");
        require_pointer(controls, "reference controls");
        require_pointer(next_states, "next states");
        handle->replay(states, controls, sigma, next_states);
    });
}

spacepdhcg_status_code spacepdhcg_free_time_project_control(
    const spacepdhcg_free_time_handle* handle,
    const double* control,
    double* projected
) {
    return guard([&] {
        require_pointer(handle, "free-time handle");
        require_pointer(control, "control");
        require_pointer(projected, "projected control");
        handle->project_control(control, projected);
    });
}

}  // extern "C"
