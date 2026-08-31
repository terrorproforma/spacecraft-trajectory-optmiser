#pragma once

#include <stddef.h>

#ifdef _WIN32
#    ifdef SPACEPDHCG_NATIVE_EXPORTS
#        define SPACEPDHCG_NATIVE_API __declspec(dllexport)
#    else
#        define SPACEPDHCG_NATIVE_API __declspec(dllimport)
#    endif
#else
#    define SPACEPDHCG_NATIVE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum spacepdhcg_native_status {
    SPACEPDHCG_NATIVE_SUCCESS = 0,
    SPACEPDHCG_NATIVE_INVALID_ARGUMENT = 1,
    SPACEPDHCG_NATIVE_INTERNAL_ERROR = 2,
};

SPACEPDHCG_NATIVE_API int spacepdhcg_native_abi_version(void);

SPACEPDHCG_NATIVE_API const char* spacepdhcg_native_last_error(void);

/**
 * Compute the exact zero-order-hold HCW matrices in row-major order.
 *
 * `state_matrix` must provide 36 doubles and `control_matrix` 18 doubles.
 */
SPACEPDHCG_NATIVE_API int spacepdhcg_cw_discretise(
    double mean_motion,
    double step_seconds,
    double* state_matrix,
    size_t state_matrix_length,
    double* control_matrix,
    size_t control_matrix_length
);

#ifdef __cplusplus
}
#endif
