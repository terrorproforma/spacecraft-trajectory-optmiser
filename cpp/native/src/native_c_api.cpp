#include "spacepdhcg/native_c_api.h"

#include "spacepdhcg/native/cw.hpp"

#include <algorithm>
#include <exception>
#include <string>

namespace {
thread_local std::string last_error;

int fail(int code, const char* message) {
    last_error = message == nullptr ? "unknown native error" : message;
    return code;
}
}  // namespace

extern "C" int spacepdhcg_native_abi_version(void) {
    return 1;
}

extern "C" const char* spacepdhcg_native_last_error(void) {
    return last_error.c_str();
}

extern "C" int spacepdhcg_cw_discretise(
    double mean_motion,
    double step_seconds,
    double* state_matrix,
    size_t state_matrix_length,
    double* control_matrix,
    size_t control_matrix_length
) {
    if (state_matrix == nullptr || control_matrix == nullptr) {
        return fail(SPACEPDHCG_NATIVE_INVALID_ARGUMENT, "output matrix pointer is null");
    }
    if (state_matrix_length != spacepdhcg::native::cw_state_dimension *
                                   spacepdhcg::native::cw_state_dimension ||
        control_matrix_length != spacepdhcg::native::cw_state_dimension *
                                     spacepdhcg::native::cw_control_dimension) {
        return fail(SPACEPDHCG_NATIVE_INVALID_ARGUMENT, "output matrix length is incorrect");
    }

    try {
        const auto matrices = spacepdhcg::native::discretise_cw(mean_motion, step_seconds);
        std::copy(matrices.state.begin(), matrices.state.end(), state_matrix);
        std::copy(matrices.control.begin(), matrices.control.end(), control_matrix);
        last_error.clear();
        return SPACEPDHCG_NATIVE_SUCCESS;
    } catch (const std::invalid_argument& error) {
        return fail(SPACEPDHCG_NATIVE_INVALID_ARGUMENT, error.what());
    } catch (const std::exception& error) {
        return fail(SPACEPDHCG_NATIVE_INTERNAL_ERROR, error.what());
    } catch (...) {
        return fail(SPACEPDHCG_NATIVE_INTERNAL_ERROR, "unknown native exception");
    }
}
