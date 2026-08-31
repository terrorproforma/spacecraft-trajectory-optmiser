#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(SPACEPDHCG_C_API_EXPORTS)
#    define SPACEPDHCG_C_API __declspec(dllexport)
#  else
#    define SPACEPDHCG_C_API __declspec(dllimport)
#  endif
#else
#  define SPACEPDHCG_C_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum spacepdhcg_status_code {
    SPACEPDHCG_STATUS_OK = 0,
    SPACEPDHCG_STATUS_INVALID_ARGUMENT = 1,
    SPACEPDHCG_STATUS_RUNTIME_ERROR = 2,
    SPACEPDHCG_STATUS_INTERNAL_ERROR = 3
} spacepdhcg_status_code;

typedef struct spacepdhcg_powered_descent_3dof_config {
    double gravity[3];
    double mass_flow_coefficient;
    double minimum_mass;
    double maximum_thrust;
    double minimum_sigma;
    double maximum_tilt_radians;
    double glide_slope_radians;
} spacepdhcg_powered_descent_3dof_config;

typedef struct spacepdhcg_lambert_result {
    double departure_velocity[3];
    double arrival_velocity[3];
    double universal_parameter;
    double transfer_angle_radians;
    uint64_t iterations;
    double time_of_flight_residual;
} spacepdhcg_lambert_result;

/// ABI version, incremented only for an incompatible C interface change.
SPACEPDHCG_C_API uint32_t spacepdhcg_c_api_version(void);

/// Human-readable native core version string with static storage duration.
SPACEPDHCG_C_API const char* spacepdhcg_native_version(void);

/// Thread-local diagnostic for the most recent failing C API call.
SPACEPDHCG_C_API const char* spacepdhcg_last_error(void);

SPACEPDHCG_C_API void spacepdhcg_default_powered_descent_3dof_config(
    spacepdhcg_powered_descent_3dof_config* config
);

/// Evaluate the 7-state derivative for [r, v, mass] and [thrust, sigma].
SPACEPDHCG_C_API spacepdhcg_status_code spacepdhcg_powered_descent_3dof_dynamics(
    const spacepdhcg_powered_descent_3dof_config* config,
    const double state[7],
    const double control[4],
    double derivative[7]
);

/// Evaluate row-major df/dx (7x7) and df/du (7x4).
SPACEPDHCG_C_API spacepdhcg_status_code spacepdhcg_powered_descent_3dof_jacobians(
    const spacepdhcg_powered_descent_3dof_config* config,
    const double state[7],
    const double control[4],
    double state_jacobian[49],
    double control_jacobian[28]
);

SPACEPDHCG_C_API spacepdhcg_status_code spacepdhcg_lambert_zero_revolution(
    const double departure_position[3],
    const double arrival_position[3],
    double time_of_flight,
    double gravitational_parameter,
    int long_way,
    double time_tolerance,
    uint64_t maximum_iterations,
    spacepdhcg_lambert_result* result
);

#ifdef __cplusplus
}
#endif
