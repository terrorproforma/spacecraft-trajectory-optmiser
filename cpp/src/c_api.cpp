#define SPACEPDHCG_C_API_EXPORTS
#include "spacepdhcg/c_api.h"

#include "spacepdhcg/dynamics/powered_descent_3dof.hpp"
#include "spacepdhcg/orbitweaver/lambert.hpp"
#include "spacepdhcg/orbitweaver/lambert_family.hpp"
#include "spacepdhcg/planner/describe.hpp"
#include "spacepdhcg/planner/families.hpp"
#include "spacepdhcg/planner/json.hpp"
#include "spacepdhcg/planner/problem.hpp"

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

}  // extern "C"
