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

typedef enum spacepdhcg_lambert_family_status {
    SPACEPDHCG_LAMBERT_FAMILY_FEASIBLE = 0,
    SPACEPDHCG_LAMBERT_FAMILY_NO_SOLUTION = 1,
    SPACEPDHCG_LAMBERT_FAMILY_INVALID_INPUT = 2,
    SPACEPDHCG_LAMBERT_FAMILY_UNSUPPORTED = 3,
    SPACEPDHCG_LAMBERT_FAMILY_NUMERICAL_FAILURE = 4
} spacepdhcg_lambert_family_status;

typedef struct spacepdhcg_lambert_family_request {
    uint64_t deterministic_id;
    double departure_position[3];
    double arrival_position[3];
    double time_of_flight;
    double gravitational_parameter;
    double time_tolerance;
    uint64_t maximum_iterations;
    uint64_t maximum_revolutions;
    uint64_t scan_samples_per_band;
    int include_short_way;
    int include_long_way;
} spacepdhcg_lambert_family_request;

typedef struct spacepdhcg_lambert_family_result {
    uint64_t deterministic_id;
    uint64_t input_index;
    uint64_t family_index;
    uint64_t revolutions;
    int long_way;
    int parameter_branch;
    spacepdhcg_lambert_family_status status;
    spacepdhcg_lambert_result solution;
} spacepdhcg_lambert_family_result;

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

/// Fixed output stride matching the CUDA batch layout for CPU/GPU parity.
SPACEPDHCG_C_API size_t spacepdhcg_lambert_family_result_stride(
    uint64_t supported_maximum_revolutions
);

/// Deterministic independent CPU truth path. Every input retains a fixed result
/// region including unsupported and no-solution slots.
SPACEPDHCG_C_API spacepdhcg_status_code spacepdhcg_lambert_family_batch_cpu(
    const spacepdhcg_lambert_family_request* requests,
    size_t request_count,
    uint64_t supported_maximum_revolutions,
    spacepdhcg_lambert_family_result* results,
    size_t result_capacity
);

/* ------------------------------------------------------------------------------------------
 * Free-final-time (time-dilated) transcriptions: pd3_fft and pd6_fft.
 *
 * These export the native fixed-pattern CQP (structure + numeric values) so a host conic
 * solver (Clarabel through the Python package, QOCO on the device) can drive the outer SCvx
 * loop against the exact native linearisation.  Structure arrays use the native 32-bit index
 * type.  Decision layout: states | controls | sigma | virtual | epigraph (see
 * `transcription/powered_descent_*_free_time.hpp`).
 * ---------------------------------------------------------------------------------------- */

typedef struct spacepdhcg_free_time_handle spacepdhcg_free_time_handle;

typedef struct spacepdhcg_pd3_fft_config {
    spacepdhcg_powered_descent_3dof_config model;
    uint64_t intervals;
    uint64_t substeps;
    double sigma_minimum;
    double sigma_maximum;
    double trust_radius;
    double sigma_trust_radius;
    double virtual_l1_weight;
    double virtual_quadratic_weight;
    double virtual_epigraph_regularisation;
    double fuel_weight;
    double time_weight;
    double sigma_tracking_weight;
    double state_tracking_weights[7];
    double control_tracking_weights[4];
    double state_trust_scales[7];
    double control_trust_scales[4];
} spacepdhcg_pd3_fft_config;

typedef struct spacepdhcg_pd6_fft_config {
    double gravity[3];
    double principal_inertia[3];
    double mass_flow_coefficient;
    double minimum_mass;
    double maximum_thrust;
    double minimum_sigma;
    double maximum_torque;
    double maximum_angular_rate;
    double maximum_tilt_radians;
    double glide_slope_radians;
    uint64_t intervals;
    uint64_t substeps;
    double sigma_minimum;
    double sigma_maximum;
    double trust_radius;
    double sigma_trust_radius;
    double virtual_l1_weight;
    double virtual_quadratic_weight;
    double virtual_epigraph_regularisation;
    double fuel_weight;
    double time_weight;
    double sigma_tracking_weight;
    double maximum_attitude_tilt_radians; /* body-z vs inertial-z tilt bound; pi disables */
    int32_t thrust_norm_mode;      /* 0 = epigraph Gamma >= |T|, 1 = linearised norm */
    int32_t torque_mode;           /* 0 = direct torque control, 1 = tau = thrust_arm x T */
    int32_t terminal_thrust_axial; /* nonzero: T_x = T_y = 0 on the last interval */
    int32_t reserved;
    double thrust_arm[3];
    uint8_t initial_fixed[14];
    uint8_t terminal_fixed[14];
    double state_tracking_weights[14];
    double control_tracking_weights[7];
    double state_trust_scales[14];
    double control_trust_scales[7];
} spacepdhcg_pd6_fft_config;

typedef struct spacepdhcg_free_time_sizes {
    uint64_t state_dimension;
    uint64_t control_dimension;
    uint64_t intervals;
    uint64_t variables;
    uint64_t scalar_rows;
    uint64_t affine_rows;
    uint64_t quadratic_nonzeros;
    uint64_t scalar_nonzeros;
    uint64_t affine_nonzeros;
    uint64_t cone_count;
    uint64_t control_offset;
    uint64_t sigma_index;
    uint64_t virtual_offset;
    uint64_t epigraph_offset;
    uint64_t dynamics_row_start;
    uint64_t topology_fingerprint;
} spacepdhcg_free_time_sizes;

SPACEPDHCG_C_API void spacepdhcg_default_pd3_fft_config(spacepdhcg_pd3_fft_config* config);
SPACEPDHCG_C_API void spacepdhcg_default_pd6_fft_config(spacepdhcg_pd6_fft_config* config);

SPACEPDHCG_C_API spacepdhcg_status_code spacepdhcg_pd3_fft_create(
    const spacepdhcg_pd3_fft_config* config,
    spacepdhcg_free_time_handle** handle
);
SPACEPDHCG_C_API spacepdhcg_status_code spacepdhcg_pd6_fft_create(
    const spacepdhcg_pd6_fft_config* config,
    spacepdhcg_free_time_handle** handle
);
SPACEPDHCG_C_API void spacepdhcg_free_time_destroy(spacepdhcg_free_time_handle* handle);

SPACEPDHCG_C_API spacepdhcg_status_code spacepdhcg_free_time_sizes_of(
    const spacepdhcg_free_time_handle* handle,
    spacepdhcg_free_time_sizes* sizes
);

/// CSC patterns (offsets have `columns + 1` entries) and cone blocks (`start`, `vector_dimension`
/// per second-order cone in affine-row order).
SPACEPDHCG_C_API spacepdhcg_status_code spacepdhcg_free_time_structure(
    const spacepdhcg_free_time_handle* handle,
    int32_t* quadratic_offsets,
    int32_t* quadratic_indices,
    int32_t* scalar_offsets,
    int32_t* scalar_indices,
    int32_t* affine_offsets,
    int32_t* affine_indices,
    int32_t* cone_starts,
    int32_t* cone_vector_dimensions
);

/// Numeric values about a reference `(states[(K+1)*S], controls[K*C], sigma)`.
/// `initial` has S entries.  `target` has S entries for pd6 (masked by `terminal_fixed`) and
/// 6 entries (position, velocity) for pd3.  Non-positive radii fall back to the config.
SPACEPDHCG_C_API spacepdhcg_status_code spacepdhcg_free_time_values(
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
);

/// Nonlinear time-dilated map of every interval: `next[k] = F(states[k], controls[k], sigma)`.
SPACEPDHCG_C_API spacepdhcg_status_code spacepdhcg_free_time_replay(
    const spacepdhcg_free_time_handle* handle,
    const double* states,
    const double* controls,
    double sigma,
    double* next_states
);

/// Project a control onto the configured control model (pd6: Gamma := |T| in linearised
/// mode, tau := r_T x T in thrust-arm mode; pd3: identity).
SPACEPDHCG_C_API spacepdhcg_status_code spacepdhcg_free_time_project_control(
    const spacepdhcg_free_time_handle* handle,
    const double* control,
    double* projected
);

#ifdef __cplusplus
}
#endif
