/*
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace spacepdhcg::cuda {

struct DeviceTopology {
    std::int32_t* q_offsets{nullptr};
    std::int32_t* q_indices{nullptr};
    std::int32_t* a_offsets{nullptr};
    std::int32_t* a_indices{nullptr};
    std::int32_t* f_offsets{nullptr};
    std::int32_t* f_indices{nullptr};
};

struct DeviceNumeric {
    double* q{nullptr};
    double* a{nullptr};
    double* f{nullptr};
    double* c{nullptr};
    double* scalar_lower{nullptr};
    double* scalar_upper{nullptr};
    double* affine_offset{nullptr};
    double* variable_lower{nullptr};
    double* variable_upper{nullptr};
};

struct DeviceState {
    double* primal{nullptr};
    double* dual{nullptr};
    double* previous_primal{nullptr};
    double* extrapolated_primal{nullptr};
    double* primal_product{nullptr};
    double* affine_product{nullptr};
    double* gradient{nullptr};
    double* cone_scratch{nullptr};
    double* average_primal{nullptr};
    double* average_dual{nullptr};
    double* recovery_direction_dual{nullptr};
    double* recovery_coefficients{nullptr};
    double* recovery_row_values{nullptr};
    double* recovery_scalars{nullptr};
    double* recovery_backup_primal{nullptr};
    double* recovery_backup_dual{nullptr};
    double* scaling{nullptr};
    void* control{nullptr};
    void* diagnostics{nullptr};
    std::int32_t* cancellation{nullptr};
};

}  // namespace spacepdhcg::cuda
