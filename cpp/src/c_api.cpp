#define SPACEPDHCG_C_API_EXPORTS
#include "spacepdhcg/c_api.h"

#include "spacepdhcg/dynamics/powered_descent_3dof.hpp"
#include "spacepdhcg/orbitweaver/lambert.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <string>

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

}  // extern "C"
