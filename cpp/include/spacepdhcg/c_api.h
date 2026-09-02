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

/*
 * Planner transcription ABI (schema 1.0.0).
 *
 * A planner handle owns one frozen transcription built from a normalised planner
 * problem document.  The CPU reference solver in Python uses it to obtain the
 * exact CQP topology/values, dynamics-consistent references, independent RK4
 * replays, and device-equivalent nonlinear quality metrics for every family.
 * All functions are exception-free and report failures through the status code
 * plus `spacepdhcg_last_error`.
 */
typedef struct spacepdhcg_planner spacepdhcg_planner;

typedef struct spacepdhcg_planner_dimensions {
    uint64_t state_dimension;
    uint64_t control_dimension;
    uint64_t intervals;
    uint64_t terminal_dimension;
    uint64_t variables;
    uint64_t scalar_rows;
    uint64_t affine_rows;
    uint64_t quadratic_nonzeros;
    uint64_t scalar_nonzeros;
    uint64_t affine_nonzeros;
    uint64_t affine_cone_count;
    uint64_t variable_cone_count;
    uint64_t virtual_variable_count;
    double step_seconds;
    double initial_trust_radius;
} spacepdhcg_planner_dimensions;

typedef struct spacepdhcg_planner_cone {
    int32_t kind;  /* 0 second-order, 1 rotated second-order, 2 exp, 3 power, 4 PSD */
    int32_t start;
    int32_t vector_dimension;
    double power_alpha;
} spacepdhcg_planner_cone;

#define SPACEPDHCG_PLANNER_MAX_PATH_COMPONENTS 8

typedef struct spacepdhcg_planner_evaluation {
    double objective;
    double path_violation;
    double terminal_residual;
    double terminal_position_error;
    double terminal_velocity_error;
    double propellant_used;
    double final_mass;
    uint64_t path_component_count;
    double path_normalised[SPACEPDHCG_PLANNER_MAX_PATH_COMPONENTS];
    double path_physical[SPACEPDHCG_PLANNER_MAX_PATH_COMPONENTS];
    char path_names[SPACEPDHCG_PLANNER_MAX_PATH_COMPONENTS][32];
} spacepdhcg_planner_evaluation;

/// Parse a normalised planner problem document and build its transcription.
SPACEPDHCG_C_API spacepdhcg_status_code spacepdhcg_planner_create(
    const char* problem_json,
    spacepdhcg_planner** planner
);

SPACEPDHCG_C_API void spacepdhcg_planner_destroy(spacepdhcg_planner* planner);

SPACEPDHCG_C_API spacepdhcg_status_code spacepdhcg_planner_get_dimensions(
    const spacepdhcg_planner* planner,
    spacepdhcg_planner_dimensions* dimensions
);

/// Fill caller-allocated topology arrays sized from `spacepdhcg_planner_dimensions`.
/// `virtual_variables` may be NULL when `virtual_variable_count` is zero.
SPACEPDHCG_C_API spacepdhcg_status_code spacepdhcg_planner_structure(
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
);

/// Numeric CQP coefficients linearised about the supplied reference trajectory.
SPACEPDHCG_C_API spacepdhcg_status_code spacepdhcg_planner_values(
    const spacepdhcg_planner* planner,
    const double* reference_states,
    const double* reference_controls,
    double trust_radius,
    double* quadratic,
    double* scalar_constraint,
    double* affine_cone,
    double* linear_objective,
    double* scalar_lower,
    double* scalar_upper,
    double* affine_offset,
    double* variable_lower,
    double* variable_upper
);

/// Dynamics-consistent initial reference (warm start when the document supplies one).
SPACEPDHCG_C_API spacepdhcg_status_code spacepdhcg_planner_initial_reference(
    const spacepdhcg_planner* planner,
    double* states,
    double* controls
);

/// Independent replay; `states` receives (intervals * substeps + 1) * nx entries.
SPACEPDHCG_C_API spacepdhcg_status_code spacepdhcg_planner_rollout(
    const spacepdhcg_planner* planner,
    const double* initial_state,
    const double* controls,
    uint64_t intervals,
    uint64_t substeps,
    double* states
);

/// Device-equivalent nonlinear quality for node states/controls of the full horizon.
SPACEPDHCG_C_API spacepdhcg_status_code spacepdhcg_planner_evaluate(
    const spacepdhcg_planner* planner,
    const double* states,
    const double* controls,
    spacepdhcg_planner_evaluation* evaluation
);

/// Path-violation components for an arbitrary (intervals + 1)-state sequence
/// (dense continuous-time checks). Only the path fields are populated.
SPACEPDHCG_C_API spacepdhcg_status_code spacepdhcg_planner_path_components(
    const spacepdhcg_planner* planner,
    const double* states,
    const double* controls,
    uint64_t intervals,
    spacepdhcg_planner_evaluation* evaluation
);

/// JSON description of the parsed problem (resolved defaults, units, orders,
/// solver policy). Writes up to `capacity` bytes including the terminator and
/// reports the full required size through `required`.
SPACEPDHCG_C_API spacepdhcg_status_code spacepdhcg_planner_describe(
    const spacepdhcg_planner* planner,
    char* buffer,
    size_t capacity,
    size_t* required
);

/// JSON document of the native family defaults (vehicle, environment,
/// constraints, transcription weights, terminal pattern, units, orders).
SPACEPDHCG_C_API spacepdhcg_status_code spacepdhcg_planner_default_document(
    const char* family,
    char* buffer,
    size_t capacity,
    size_t* required
);

#ifdef __cplusplus
}
#endif
