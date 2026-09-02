#define SPACEPDHCG_C_API_EXPORTS
#include "spacepdhcg/c_api.h"

#include "spacepdhcg/dynamics/powered_descent_3dof.hpp"
#include "spacepdhcg/dynamics/powered_descent_6dof.hpp"
#include "spacepdhcg/orbitweaver/lambert.hpp"
#include "spacepdhcg/orbitweaver/lambert_family.hpp"
#include "spacepdhcg/transcription/powered_descent_3dof_free_time.hpp"
#include "spacepdhcg/transcription/powered_descent_6dof_free_time.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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
