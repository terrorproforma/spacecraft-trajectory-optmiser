/*
 * Persistent single-GPU CUDA workspace for SpacePDHCG.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "spacepdhcg/cuda/persistent_pdhcg_c_api.h"
#include "spacepdhcg/cuda/persistent_pdhcg_dlpack_c_api.h"

#include "spacepdhcg/cuda/allocation_ledger.hpp"
#include "spacepdhcg/cuda/device_buffers.hpp"
#include "spacepdhcg/cuda/stream_event.hpp"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace {

using spacepdhcg::cuda::AllocationCategory;
using spacepdhcg::cuda::AllocationLedger;
using spacepdhcg::cuda::DeviceNumeric;
using spacepdhcg::cuda::DeviceState;
using spacepdhcg::cuda::DeviceTopology;
using spacepdhcg::cuda::StreamEvent;
using spacepdhcg::cuda::TimingEvents;
using spacepdhcg::cuda::native_stream;
using spacepdhcg::cuda::same_stream;

constexpr int kThreads = 256;
constexpr std::uint32_t kDLPackMajorVersion = 1U;
constexpr std::uint64_t kDLPackReadOnly = 1ULL;

struct DLPackDevice {
    std::int32_t type;
    std::int32_t id;
};

struct DLPackDataType {
    std::uint8_t code;
    std::uint8_t bits;
    std::uint16_t lanes;
};

struct DLPackTensor {
    void* data;
    DLPackDevice device;
    std::int32_t ndim;
    DLPackDataType dtype;
    std::int64_t* shape;
    std::int64_t* strides;
    std::uint64_t byte_offset;
};

struct DLPackLegacyManaged {
    DLPackTensor tensor;
    void* manager_context;
    void (*deleter)(DLPackLegacyManaged*);
};

struct DLPackVersion {
    std::uint32_t major;
    std::uint32_t minor;
};

struct DLPackVersionedManaged {
    DLPackVersion version;
    void* manager_context;
    void (*deleter)(DLPackVersionedManaged*);
    std::uint64_t flags;
    DLPackTensor tensor;
};

class DLPackOwner {
  public:
    DLPackOwner() = default;
    explicit DLPackOwner(spacepdhcg_dlpack_managed_tensor managed)
        : pointer_(managed.managed_tensor), kind_(managed.kind) {}

    DLPackOwner(const DLPackOwner&) = delete;
    DLPackOwner& operator=(const DLPackOwner&) = delete;

    DLPackOwner(DLPackOwner&& other) noexcept
        : pointer_(std::exchange(other.pointer_, nullptr)), kind_(other.kind_) {}

    DLPackOwner& operator=(DLPackOwner&& other) noexcept {
        if (this != &other) {
            reset();
            pointer_ = std::exchange(other.pointer_, nullptr);
            kind_ = other.kind_;
        }
        return *this;
    }

    ~DLPackOwner() { reset(); }

    void reset() noexcept {
        void* pointer = std::exchange(pointer_, nullptr);
        if (pointer == nullptr) {
            return;
        }
        if (kind_ == SPACEPDHCG_DLPACK_VERSIONED) {
            auto* managed = static_cast<DLPackVersionedManaged*>(pointer);
            if (managed->deleter != nullptr) {
                managed->deleter(managed);
            }
        } else {
            auto* managed = static_cast<DLPackLegacyManaged*>(pointer);
            if (managed->deleter != nullptr) {
                managed->deleter(managed);
            }
        }
    }

  private:
    void* pointer_{nullptr};
    spacepdhcg_dlpack_managed_kind kind_{SPACEPDHCG_DLPACK_LEGACY};
};

struct DeviceCone {
    int kind;
    int start;
    int vector_dimension;
    double power_alpha;
};

struct DeviceProblem {
    int variables;
    int scalar_rows;
    int affine_rows;
    int q_nonzeros;
    int a_nonzeros;
    int f_nonzeros;
    const int* q_offsets;
    const int* q_indices;
    const int* a_offsets;
    const int* a_indices;
    const int* f_offsets;
    const int* f_indices;
    const double* q;
    const double* a;
    const double* f;
    const double* c;
    const double* scalar_lower;
    const double* scalar_upper;
    const double* affine_offset;
    const double* variable_lower;
    const double* variable_upper;
    double* primal;
    double* dual;
    double* previous_primal;
    double* extrapolated_primal;
    double* scalar_product;
    double* affine_product;
    double* gradient;
    double* cone_scratch;
    double* average_primal;
    double* average_dual;
    double* recovery_direction_dual;
    double* recovery_coefficients;
    double* recovery_row_values;
    double* recovery_scalars;
    double* recovery_backup_primal;
    double* recovery_backup_dual;
    double* scaling;
    const DeviceCone* affine_cones;
    int affine_cone_count;
    const DeviceCone* variable_cones;
    int variable_cone_count;
};

struct DeviceControl {
    double optimality_tolerance;
    double feasibility_tolerance;
    std::uint64_t iteration_limit;
    unsigned int residual_check_frequency;
    int scaling_mode;
    double matrix_change_threshold;
    double vector_change_threshold;
    std::uint64_t maximum_reuse_updates;
    std::uint64_t scaling_reuse_count;
    double coefficient_change_max;
    double coefficient_change_norm;
    double primal_step;
    double dual_step;
    int force_scaling_refresh;
    int scaling_refreshed;
    std::uint64_t recovery_count;
    std::uint64_t recovery_rejected_count;
    std::uint64_t recovery_attempt_count;
};

struct DeviceReport {
    int termination;
    std::uint64_t iterations;
    double objective;
    double scalar_primal_violation_inf;
    double box_violation_inf;
    double affine_cone_distance_inf;
    double stationarity_inf;
    double natural_residual_inf;
    double complementarity_inf;
    double relative_primal_residual;
    double relative_dual_residual;
    double coefficient_change_max;
    double coefficient_change_norm;
    double scaling_min;
    double scaling_max;
    std::uint64_t scaling_reuse_count;
    int scaling_refreshed;
    std::uint64_t recovery_count;
    std::uint64_t recovery_rejected_count;
    std::uint64_t recovery_iterations;
    std::uint64_t recovery_attempt_count;
    int recovery_trigger_reason;
    int recovery_outcome_reason;
    double recovery_initial_residual;
    double recovery_final_residual;
    double recovery_final_primal_residual;
    double recovery_final_stationarity;
    double recovery_final_complementarity;
    int recovery_stationarity_index;
    double recovery_stationarity_value;
};

struct NumericPointers {
    const double* q;
    const double* a;
    const double* f;
    const double* c;
    const double* scalar_lower;
    const double* scalar_upper;
    const double* affine_offset;
    const double* variable_lower;
    const double* variable_upper;
};

enum class LastOperation {
    none,
    create,
    update,
    warm_start,
    solve,
    reset,
    checkpoint,
    restore,
    refresh,
    residual,
};

template <typename T>
T* offset_pointer(const spacepdhcg_accelerator_buffer_view& view) {
    auto* bytes = static_cast<unsigned char*>(view.data);
    return reinterpret_cast<T*>(bytes + view.byte_offset);
}

template <typename T>
const T* offset_pointer_const(const spacepdhcg_accelerator_buffer_view& view) {
    const auto* bytes = static_cast<const unsigned char*>(view.data);
    return reinterpret_cast<const T*>(bytes + view.byte_offset);
}

std::size_t checked_bytes(std::size_t elements, std::size_t scalar_bytes) {
    if (elements > std::numeric_limits<std::size_t>::max() / scalar_bytes) {
        throw std::overflow_error("buffer size overflows size_t");
    }
    return elements * scalar_bytes;
}

void set_error(spacepdhcg_cuda_workspace* workspace, const char* message);

const char* cuda_message(cudaError_t error) {
    return cudaGetErrorString(error);
}

__device__ double device_abs(const double value) {
    return value < 0.0 ? -value : value;
}

__device__ double project_interval(const double value, const double lower, const double upper) {
    return fmin(upper, fmax(lower, value));
}

__device__ void project_standard_soc(double* values, const int start, const int length) {
    const int radius_index = start + length - 1;
    double norm_squared = 0.0;
    for (int index = start; index < radius_index; ++index) {
        norm_squared += values[index] * values[index];
    }
    const double norm = sqrt(norm_squared);
    const double radius = values[radius_index];
    if (norm <= radius) {
        return;
    }
    if (norm <= -radius) {
        for (int index = start; index <= radius_index; ++index) {
            values[index] = 0.0;
        }
        return;
    }
    const double projected_radius = 0.5 * (norm + radius);
    const double scale = norm > 0.0 ? projected_radius / norm : 0.0;
    for (int index = start; index < radius_index; ++index) {
        values[index] *= scale;
    }
    values[radius_index] = projected_radius;
}

__device__ void project_rotated_soc(
    double* values,
    const int start,
    const int vector_dimension
) {
    const int first_scalar = start + vector_dimension;
    const int second_scalar = first_scalar + 1;
    const double sqrt_two = 1.4142135623730950488;
    for (int index = 0; index < vector_dimension; ++index) {
        values[start + index] *= sqrt_two;
    }
    const double first = values[first_scalar];
    const double second = values[second_scalar];
    values[first_scalar] = first - second;
    values[second_scalar] = first + second;
    project_standard_soc(values, start, vector_dimension + 2);
    const double difference = values[first_scalar];
    const double sum = values[second_scalar];
    for (int index = 0; index < vector_dimension; ++index) {
        values[start + index] /= sqrt_two;
    }
    values[first_scalar] = 0.5 * (sum + difference);
    values[second_scalar] = 0.5 * (sum - difference);
}

__device__ void project_cone_blocks(
    double* values,
    const DeviceCone* cones,
    const int cone_count
) {
    for (int cone_index = 0; cone_index < cone_count; ++cone_index) {
        const DeviceCone cone = cones[cone_index];
        if (cone.kind == SPACEPDHCG_CUDA_CONE_SECOND_ORDER) {
            project_standard_soc(values, cone.start, cone.vector_dimension + 2);
        } else if (cone.kind == SPACEPDHCG_CUDA_CONE_ROTATED_SECOND_ORDER) {
            project_rotated_soc(values, cone.start, cone.vector_dimension);
        }
    }
}

__device__ void zero_vector(double* vector, const int count) {
    for (int index = threadIdx.x; index < count; index += blockDim.x) {
        vector[index] = 0.0;
    }
}

__device__ void csc_multiply(
    const int columns,
    const int* offsets,
    const int* indices,
    const double* values,
    const double* vector,
    double* result
) {
    if (offsets == nullptr || indices == nullptr || values == nullptr) {
        return;
    }
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const double x = vector[column];
        for (int position = offsets[column]; position < offsets[column + 1]; ++position) {
            atomicAdd(result + indices[position], values[position] * x);
        }
    }
}

__device__ void csc_transpose_multiply(
    const int columns,
    const int* offsets,
    const int* indices,
    const double* values,
    const double* vector,
    double* result
) {
    if (offsets == nullptr || indices == nullptr || values == nullptr) {
        return;
    }
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        double sum = 0.0;
        for (int position = offsets[column]; position < offsets[column + 1]; ++position) {
            sum += values[position] * vector[indices[position]];
        }
        result[column] += sum;
    }
}

__device__ double positive_relative_change(const double before, const double after) {
    return device_abs(after - before) / fmax(1.0, device_abs(before));
}

__global__ void coefficient_change_kernel(
    DeviceControl* control,
    NumericPointers source,
    DeviceProblem* problem
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    double maximum = 0.0;
    double norm_squared = 0.0;
#define SPACEPDHCG_ACCUMULATE_CHANGE(SOURCE, TARGET, COUNT) \
    for (int i = 0; i < (COUNT); ++i) { \
        const double difference = (SOURCE)[i] - (TARGET)[i]; \
        maximum = fmax(maximum, positive_relative_change((TARGET)[i], (SOURCE)[i])); \
        norm_squared += difference * difference; \
    }
    SPACEPDHCG_ACCUMULATE_CHANGE(source.q, problem->q, problem->q_nonzeros)
    SPACEPDHCG_ACCUMULATE_CHANGE(source.a, problem->a, problem->a_nonzeros)
    SPACEPDHCG_ACCUMULATE_CHANGE(source.f, problem->f, problem->f_nonzeros)
    SPACEPDHCG_ACCUMULATE_CHANGE(source.c, problem->c, problem->variables)
    SPACEPDHCG_ACCUMULATE_CHANGE(
        source.scalar_lower,
        problem->scalar_lower,
        problem->scalar_rows
    )
    SPACEPDHCG_ACCUMULATE_CHANGE(
        source.scalar_upper,
        problem->scalar_upper,
        problem->scalar_rows
    )
    SPACEPDHCG_ACCUMULATE_CHANGE(
        source.affine_offset,
        problem->affine_offset,
        problem->affine_rows
    )
    SPACEPDHCG_ACCUMULATE_CHANGE(
        source.variable_lower,
        problem->variable_lower,
        problem->variables
    )
    SPACEPDHCG_ACCUMULATE_CHANGE(
        source.variable_upper,
        problem->variable_upper,
        problem->variables
    )
#undef SPACEPDHCG_ACCUMULATE_CHANGE
    control->coefficient_change_max = maximum;
    control->coefficient_change_norm = sqrt(norm_squared);
}

__global__ void initialise_control_kernel(DeviceControl* control, DeviceProblem* problem) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const bool threshold_refresh =
        control->coefficient_change_max > control->matrix_change_threshold;
    const bool budget_refresh =
        control->scaling_reuse_count >= control->maximum_reuse_updates;
    const bool refresh =
        control->force_scaling_refresh != 0
        || control->scaling_mode == SPACEPDHCG_CUDA_SCALING_ALWAYS_REFRESH
        || (control->scaling_mode == SPACEPDHCG_CUDA_SCALING_REFRESH_IF_NEEDED
            && (threshold_refresh || budget_refresh));
    if (refresh || !(control->primal_step > 0.0) || !(control->dual_step > 0.0)) {
        double* const variable_scale = problem->scaling;
        double* const row_scale = problem->scaling + problem->variables;
        for (int variable = 0; variable < problem->variables; ++variable) {
            variable_scale[variable] = 1.0;
        }
        for (int row = 0; row < problem->scalar_rows + problem->affine_rows; ++row) {
            row_scale[row] = 1.0;
        }
        // Ten cone-preserving Ruiz passes match the fixed-pattern upstream policy.
        // The existing product buffers are safe create-time scratch before solve.
        for (int pass = 0; pass < 10; ++pass) {
            for (int variable = 0; variable < problem->variables; ++variable) {
                problem->previous_primal[variable] = 0.0;
            }
            for (int row = 0; row < problem->scalar_rows; ++row) {
                problem->scalar_product[row] = 0.0;
            }
            for (int row = 0; row < problem->affine_rows; ++row) {
                problem->affine_product[row] = 0.0;
            }
            for (int variable = 0; variable < problem->variables; ++variable) {
                for (int index = problem->scalar_rows > 0
                         ? problem->a_offsets[variable]
                         : 0;
                     index < (problem->scalar_rows > 0
                         ? problem->a_offsets[variable + 1]
                         : 0);
                     ++index) {
                    const int row = problem->a_indices[index];
                    const double value = device_abs(
                        problem->a[index]
                        / (row_scale[row] * variable_scale[variable])
                    );
                    problem->previous_primal[variable] =
                        fmax(problem->previous_primal[variable], value);
                    problem->scalar_product[row] =
                        fmax(problem->scalar_product[row], value);
                }
                for (int index = problem->affine_rows > 0
                         ? problem->f_offsets[variable]
                         : 0;
                     index < (problem->affine_rows > 0
                         ? problem->f_offsets[variable + 1]
                         : 0);
                     ++index) {
                    const int row = problem->f_indices[index];
                    const double value = device_abs(
                        problem->f[index]
                        / (
                            row_scale[problem->scalar_rows + row]
                            * variable_scale[variable]
                        )
                    );
                    problem->previous_primal[variable] =
                        fmax(problem->previous_primal[variable], value);
                    problem->affine_product[row] =
                        fmax(problem->affine_product[row], value);
                }
            }
            for (int cone_index = 0;
                 cone_index < problem->variable_cone_count;
                 ++cone_index) {
                const DeviceCone cone = problem->variable_cones[cone_index];
                const int length = cone.vector_dimension + 2;
                double block_maximum = 0.0;
                for (int slot = cone.start; slot < cone.start + length; ++slot) {
                    block_maximum =
                        fmax(block_maximum, problem->previous_primal[slot]);
                }
                for (int slot = cone.start; slot < cone.start + length; ++slot) {
                    problem->previous_primal[slot] = block_maximum;
                }
            }
            for (int cone_index = 0;
                 cone_index < problem->affine_cone_count;
                 ++cone_index) {
                const DeviceCone cone = problem->affine_cones[cone_index];
                const int length = cone.vector_dimension + 2;
                double block_maximum = 0.0;
                for (int slot = cone.start; slot < cone.start + length; ++slot) {
                    block_maximum =
                        fmax(block_maximum, problem->affine_product[slot]);
                }
                for (int slot = cone.start; slot < cone.start + length; ++slot) {
                    problem->affine_product[slot] = block_maximum;
                }
            }
            for (int variable = 0; variable < problem->variables; ++variable) {
                const double factor = problem->previous_primal[variable] > 1.0e-12
                    ? sqrt(problem->previous_primal[variable])
                    : 1.0;
                variable_scale[variable] *= factor;
            }
            for (int row = 0; row < problem->scalar_rows; ++row) {
                const double factor = problem->scalar_product[row] > 1.0e-12
                    ? sqrt(problem->scalar_product[row])
                    : 1.0;
                row_scale[row] *= factor;
            }
            for (int row = 0; row < problem->affine_rows; ++row) {
                const double factor = problem->affine_product[row] > 1.0e-12
                    ? sqrt(problem->affine_product[row])
                    : 1.0;
                row_scale[problem->scalar_rows + row] *= factor;
            }
        }
        double bound_norm_squared = 0.0;
        for (int row = 0; row < problem->scalar_rows; ++row) {
            const double scale = row_scale[row];
            if (isfinite(problem->scalar_lower[row])
                && problem->scalar_lower[row] != problem->scalar_upper[row]) {
                const double value = problem->scalar_lower[row] / scale;
                bound_norm_squared += value * value;
            }
            if (isfinite(problem->scalar_upper[row])) {
                const double value = problem->scalar_upper[row] / scale;
                bound_norm_squared += value * value;
            }
        }
        for (int row = 0; row < problem->affine_rows; ++row) {
            const double value =
                problem->affine_offset[row]
                / row_scale[problem->scalar_rows + row];
            bound_norm_squared += value * value;
        }
        double objective_norm_squared = 0.0;
        for (int variable = 0; variable < problem->variables; ++variable) {
            const double value = problem->c[variable] / variable_scale[variable];
            objective_norm_squared += value * value;
        }
        const double bound_scale = 1.0 / (sqrt(bound_norm_squared) + 1.0);
        const double objective_scale = 1.0 / (sqrt(objective_norm_squared) + 1.0);
        for (int variable = 0; variable < problem->variables; ++variable) {
            problem->gradient[variable] = 0.0;
        }
        double operator_norm_squared = 0.0;
        for (int variable = 0; variable < problem->variables; ++variable) {
            for (int index = problem->q_offsets[variable];
                 index < problem->q_offsets[variable + 1];
                 ++index) {
                const int row = problem->q_indices[index];
                const double value =
                    problem->q[index] * objective_scale
                    / (
                        variable_scale[row] * variable_scale[variable]
                        * bound_scale
                    );
                problem->gradient[row] += device_abs(value);
            }
            for (int index = problem->scalar_rows > 0
                     ? problem->a_offsets[variable]
                     : 0;
                 index < (problem->scalar_rows > 0
                     ? problem->a_offsets[variable + 1]
                     : 0);
                 ++index) {
                const int row = problem->a_indices[index];
                const double value =
                    problem->a[index]
                    / (row_scale[row] * variable_scale[variable]);
                operator_norm_squared += value * value;
            }
            for (int index = problem->affine_rows > 0
                     ? problem->f_offsets[variable]
                     : 0;
                 index < (problem->affine_rows > 0
                     ? problem->f_offsets[variable + 1]
                     : 0);
                 ++index) {
                const int row = problem->f_indices[index];
                const double value =
                    problem->f[index]
                    / (
                        row_scale[problem->scalar_rows + row]
                        * variable_scale[variable]
                    );
                operator_norm_squared += value * value;
            }
        }
        double q_norm = 0.0;
        for (int variable = 0; variable < problem->variables; ++variable) {
            q_norm = fmax(q_norm, problem->gradient[variable]);
            problem->previous_primal[variable] =
                1.0 / sqrt(static_cast<double>(problem->variables));
        }
        double operator_norm = sqrt(operator_norm_squared);
        for (int pass = 0; pass < 20; ++pass) {
            for (int row = 0; row < problem->scalar_rows; ++row) {
                problem->scalar_product[row] = 0.0;
            }
            for (int row = 0; row < problem->affine_rows; ++row) {
                problem->affine_product[row] = 0.0;
            }
            for (int variable = 0; variable < problem->variables; ++variable) {
                const double x = problem->previous_primal[variable];
                for (int index = problem->scalar_rows > 0
                         ? problem->a_offsets[variable]
                         : 0;
                     index < (problem->scalar_rows > 0
                         ? problem->a_offsets[variable + 1]
                         : 0);
                     ++index) {
                    const int row = problem->a_indices[index];
                    problem->scalar_product[row] +=
                        problem->a[index] * x
                        / (row_scale[row] * variable_scale[variable]);
                }
                for (int index = problem->affine_rows > 0
                         ? problem->f_offsets[variable]
                         : 0;
                     index < (problem->affine_rows > 0
                         ? problem->f_offsets[variable + 1]
                         : 0);
                     ++index) {
                    const int row = problem->f_indices[index];
                    problem->affine_product[row] +=
                        problem->f[index] * x
                        / (
                            row_scale[problem->scalar_rows + row]
                            * variable_scale[variable]
                        );
                }
            }
            double row_norm_squared = 0.0;
            for (int row = 0; row < problem->scalar_rows; ++row) {
                row_norm_squared +=
                    problem->scalar_product[row] * problem->scalar_product[row];
            }
            for (int row = 0; row < problem->affine_rows; ++row) {
                row_norm_squared +=
                    problem->affine_product[row] * problem->affine_product[row];
            }
            const double row_norm = sqrt(row_norm_squared);
            if (!(row_norm > 1.0e-12)) {
                operator_norm = 0.0;
                break;
            }
            for (int row = 0; row < problem->scalar_rows; ++row) {
                problem->scalar_product[row] /= row_norm;
            }
            for (int row = 0; row < problem->affine_rows; ++row) {
                problem->affine_product[row] /= row_norm;
            }
            for (int variable = 0; variable < problem->variables; ++variable) {
                double value = 0.0;
                for (int index = problem->scalar_rows > 0
                         ? problem->a_offsets[variable]
                         : 0;
                     index < (problem->scalar_rows > 0
                         ? problem->a_offsets[variable + 1]
                         : 0);
                     ++index) {
                    const int row = problem->a_indices[index];
                    value +=
                        problem->a[index] * problem->scalar_product[row]
                        / (row_scale[row] * variable_scale[variable]);
                }
                for (int index = problem->affine_rows > 0
                         ? problem->f_offsets[variable]
                         : 0;
                     index < (problem->affine_rows > 0
                         ? problem->f_offsets[variable + 1]
                         : 0);
                     ++index) {
                    const int row = problem->f_indices[index];
                    value +=
                        problem->f[index] * problem->affine_product[row]
                        / (
                            row_scale[problem->scalar_rows + row]
                            * variable_scale[variable]
                        );
                }
                problem->gradient[variable] = value;
            }
            double variable_norm_squared = 0.0;
            for (int variable = 0; variable < problem->variables; ++variable) {
                variable_norm_squared +=
                    problem->gradient[variable] * problem->gradient[variable];
            }
            operator_norm = sqrt(variable_norm_squared);
            if (!(operator_norm > 1.0e-12)) {
                break;
            }
            for (int variable = 0; variable < problem->variables; ++variable) {
                problem->previous_primal[variable] =
                    problem->gradient[variable] / operator_norm;
            }
        }
        const double denominator = fmax(1.0, q_norm + operator_norm);
        control->primal_step = 0.9 / denominator;
        control->dual_step = 0.9 / fmax(1.0, operator_norm);
        for (int variable = 0; variable < problem->variables; ++variable) {
            problem->scaling[variable] =
                objective_scale
                / (
                    variable_scale[variable] * variable_scale[variable]
                    * bound_scale
                );
        }
        for (int row = 0; row < problem->scalar_rows + problem->affine_rows; ++row) {
            problem->scaling[problem->variables + row] =
                bound_scale
                / (
                    row_scale[row] * row_scale[row]
                    * objective_scale
                );
        }
        control->scaling_reuse_count = 0;
        control->scaling_refreshed = 1;
    } else {
        ++control->scaling_reuse_count;
        control->scaling_refreshed = 0;
    }
    control->force_scaling_refresh = 0;
}

__global__ void set_solve_options_kernel(
    DeviceControl* control,
    const double optimality_tolerance,
    const double feasibility_tolerance,
    const std::uint64_t iteration_limit,
    const unsigned int residual_check_frequency
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        control->optimality_tolerance = optimality_tolerance;
        control->feasibility_tolerance = feasibility_tolerance;
        control->iteration_limit = iteration_limit;
        control->residual_check_frequency =
            residual_check_frequency == 0U ? 25U : residual_check_frequency;
    }
}

__device__ void compute_products(DeviceProblem* problem, const double* primal) {
    zero_vector(problem->scalar_product, problem->scalar_rows);
    zero_vector(problem->affine_product, problem->affine_rows);
    zero_vector(problem->gradient, problem->variables);
    __syncthreads();
    csc_multiply(
        problem->variables,
        problem->a_offsets,
        problem->a_indices,
        problem->a,
        primal,
        problem->scalar_product
    );
    csc_multiply(
        problem->variables,
        problem->f_offsets,
        problem->f_indices,
        problem->f,
        primal,
        problem->affine_product
    );
    csc_multiply(
        problem->variables,
        problem->q_offsets,
        problem->q_indices,
        problem->q,
        primal,
        problem->gradient
    );
    __syncthreads();
}

__device__ void add_transpose_dual(DeviceProblem* problem) {
    csc_transpose_multiply(
        problem->variables,
        problem->a_offsets,
        problem->a_indices,
        problem->a,
        problem->dual,
        problem->gradient
    );
    csc_transpose_multiply(
        problem->variables,
        problem->f_offsets,
        problem->f_indices,
        problem->f,
        problem->dual + problem->scalar_rows,
        problem->gradient
    );
    __syncthreads();
}

__device__ void evaluate_report(
    DeviceProblem* problem,
    DeviceControl* control,
    DeviceReport* report,
    const std::uint64_t iteration
) {
    compute_products(problem, problem->primal);
    add_transpose_dual(problem);
    if (threadIdx.x == 0) {
        double scalar_violation = 0.0;
        double scalar_natural = 0.0;
        double scalar_scale = 1.0;
        double complementarity = 0.0;
        for (int row = 0; row < problem->scalar_rows; ++row) {
            const double value = problem->scalar_product[row];
            const double projection =
                project_interval(value, problem->scalar_lower[row], problem->scalar_upper[row]);
            scalar_violation = fmax(scalar_violation, device_abs(value - projection));
            const double natural_projection = project_interval(
                value + problem->dual[row],
                problem->scalar_lower[row],
                problem->scalar_upper[row]
            );
            scalar_natural = fmax(
                scalar_natural,
                device_abs(value - natural_projection)
            );
            if (isfinite(problem->scalar_lower[row])) {
                scalar_scale = fmax(scalar_scale, device_abs(problem->scalar_lower[row]));
            }
            if (isfinite(problem->scalar_upper[row])) {
                scalar_scale = fmax(scalar_scale, device_abs(problem->scalar_upper[row]));
            }
            complementarity =
                fmax(complementarity, device_abs(problem->dual[row] * (value - projection)));
        }

        double box_violation = 0.0;
        double stationarity = 0.0;
        double objective = 0.0;
        for (int variable = 0; variable < problem->variables; ++variable) {
            const double x = problem->primal[variable];
            const double projection =
                project_interval(x, problem->variable_lower[variable], problem->variable_upper[variable]);
            box_violation = fmax(box_violation, device_abs(x - projection));
            const double gradient = problem->gradient[variable] + problem->c[variable];
            problem->average_primal[variable] = project_interval(
                x - gradient,
                problem->variable_lower[variable],
                problem->variable_upper[variable]
            );
            objective += problem->c[variable] * x
                + 0.5 * x * problem->gradient[variable];
        }
        project_cone_blocks(
            problem->average_primal,
            problem->variable_cones,
            problem->variable_cone_count
        );
        for (int variable = 0; variable < problem->variables; ++variable) {
            stationarity = fmax(
                stationarity,
                device_abs(problem->primal[variable] - problem->average_primal[variable])
            );
            problem->average_primal[variable] = problem->primal[variable];
        }
        project_cone_blocks(
            problem->average_primal,
            problem->variable_cones,
            problem->variable_cone_count
        );
        for (int variable = 0; variable < problem->variables; ++variable) {
            box_violation = fmax(
                box_violation,
                device_abs(problem->primal[variable] - problem->average_primal[variable])
            );
        }

        for (int row = 0; row < problem->affine_rows; ++row) {
            problem->cone_scratch[row] =
                problem->affine_product[row] + problem->affine_offset[row];
        }
        project_cone_blocks(
            problem->cone_scratch,
            problem->affine_cones,
            problem->affine_cone_count
        );
        double affine_distance = 0.0;
        for (int row = 0; row < problem->affine_rows; ++row) {
            const double original =
                problem->affine_product[row] + problem->affine_offset[row];
            affine_distance =
                fmax(affine_distance, device_abs(original - problem->cone_scratch[row]));
        }
        for (int row = 0; row < problem->affine_rows; ++row) {
            problem->cone_scratch[row] =
                problem->affine_product[row] + problem->affine_offset[row]
                + problem->dual[problem->scalar_rows + row];
        }
        project_cone_blocks(
            problem->cone_scratch,
            problem->affine_cones,
            problem->affine_cone_count
        );
        double affine_natural = 0.0;
        for (int row = 0; row < problem->affine_rows; ++row) {
            const double original =
                problem->affine_product[row] + problem->affine_offset[row];
            affine_natural = fmax(
                affine_natural,
                device_abs(original - problem->cone_scratch[row])
            );
        }
        double affine_complementarity = 0.0;
        for (int cone_index = 0;
             cone_index < problem->affine_cone_count;
             ++cone_index) {
            const DeviceCone cone = problem->affine_cones[cone_index];
            const int length = cone.vector_dimension + 2;
            double inner_product = 0.0;
            for (int slot = cone.start; slot < cone.start + length; ++slot) {
                const double original =
                    problem->affine_product[slot] + problem->affine_offset[slot];
                inner_product +=
                    problem->dual[problem->scalar_rows + slot] * original;
            }
            affine_complementarity =
                fmax(affine_complementarity, device_abs(inner_product));
        }
        complementarity = fmax(complementarity, affine_complementarity);
        const double primal_residual = fmax(scalar_violation, fmax(box_violation, affine_distance));
        const double natural_residual = fmax(
            fmax(primal_residual, stationarity),
            fmax(scalar_natural, affine_natural)
        );
        const double dual_residual = fmax(
            fmax(stationarity, complementarity),
            fmax(scalar_natural, affine_natural)
        );
        double scaling_min = INFINITY;
        double scaling_max = 0.0;
        const int scaling_count =
            problem->variables + problem->scalar_rows + problem->affine_rows;
        for (int index = 0; index < scaling_count; ++index) {
            scaling_min = fmin(scaling_min, problem->scaling[index]);
            scaling_max = fmax(scaling_max, problem->scaling[index]);
        }

        report->iterations = iteration;
        report->objective = objective;
        report->scalar_primal_violation_inf = scalar_violation;
        report->box_violation_inf = box_violation;
        report->affine_cone_distance_inf = affine_distance;
        report->stationarity_inf = stationarity;
        report->natural_residual_inf = natural_residual;
        report->complementarity_inf = complementarity;
        report->relative_primal_residual = primal_residual / scalar_scale;
        report->relative_dual_residual = dual_residual / fmax(1.0, device_abs(objective));
        report->coefficient_change_max = control->coefficient_change_max;
        report->coefficient_change_norm = control->coefficient_change_norm;
        report->scaling_min = scaling_min;
        report->scaling_max = scaling_max;
        report->scaling_reuse_count = control->scaling_reuse_count;
        report->scaling_refreshed = control->scaling_refreshed;
        report->recovery_count = control->recovery_count;
        report->recovery_rejected_count = control->recovery_rejected_count;
        report->recovery_attempt_count = control->recovery_attempt_count;
        report->recovery_final_residual = natural_residual;
    }
    __syncthreads();
}

__global__ void solve_kernel(
    DeviceProblem* problem,
    DeviceControl* control,
    DeviceReport* report,
    volatile int* cancellation
) {
    __shared__ int should_stop;
    __shared__ int cancelled;
    if (threadIdx.x == 0) {
        report->termination = SPACEPDHCG_CUDA_TERMINATION_ITERATION_LIMIT;
        report->iterations = 0;
        report->recovery_iterations = 0U;
        report->recovery_trigger_reason = SPACEPDHCG_CUDA_RECOVERY_NOT_TRIGGERED;
        report->recovery_outcome_reason = SPACEPDHCG_CUDA_RECOVERY_NOT_TRIGGERED;
        report->recovery_initial_residual = 0.0;
        report->recovery_final_residual = 0.0;
        report->recovery_final_primal_residual = 0.0;
        report->recovery_final_stationarity = 0.0;
        report->recovery_final_complementarity = 0.0;
        report->recovery_stationarity_index = -1;
        report->recovery_stationarity_value = 0.0;
        atomicExch(&should_stop, 0);
        atomicExch(&cancelled, 0);
    }
    __syncthreads();

    const double primal_step = control->primal_step;
    const double dual_step = control->dual_step;
    const unsigned int check_frequency =
        control->residual_check_frequency == 0U ? 1U : control->residual_check_frequency;
    const bool recovery_enabled =
        control->iteration_limit >= 350'000U
        && fmin(control->feasibility_tolerance, control->optimality_tolerance)
            <= 1.0e-6;
    const std::uint64_t pdhg_limit =
        recovery_enabled ? 300'000U : control->iteration_limit;

    for (std::uint64_t iteration = 1; iteration <= pdhg_limit; ++iteration) {
        if (threadIdx.x == 0 && *cancellation != 0) {
            atomicExch(&cancelled, 1);
            atomicExch(&should_stop, 1);
        }
        __syncthreads();
        if (atomicAdd(&should_stop, 0) != 0) {
            break;
        }

        zero_vector(problem->scalar_product, problem->scalar_rows);
        zero_vector(problem->affine_product, problem->affine_rows);
        __syncthreads();
        csc_multiply(
            problem->variables,
            problem->a_offsets,
            problem->a_indices,
            problem->a,
            problem->extrapolated_primal,
            problem->scalar_product
        );
        csc_multiply(
            problem->variables,
            problem->f_offsets,
            problem->f_indices,
            problem->f,
            problem->extrapolated_primal,
            problem->affine_product
        );
        __syncthreads();

        for (int row = threadIdx.x; row < problem->scalar_rows; row += blockDim.x) {
            const double value =
                problem->dual[row]
                + dual_step
                    * problem->scaling[problem->variables + row]
                    * problem->scalar_product[row];
            const double row_step =
                dual_step * problem->scaling[problem->variables + row];
            const double projected = project_interval(
                value / row_step,
                problem->scalar_lower[row],
                problem->scalar_upper[row]
            );
            problem->dual[row] = value - row_step * projected;
        }
        for (int row = threadIdx.x; row < problem->affine_rows; row += blockDim.x) {
            const int dual_row = problem->scalar_rows + row;
            const double row_step =
                dual_step * problem->scaling[problem->variables + dual_row];
            const double value =
                problem->dual[dual_row] + row_step * problem->affine_product[row];
            problem->cone_scratch[row] =
                value / row_step + problem->affine_offset[row];
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            project_cone_blocks(
                problem->cone_scratch,
                problem->affine_cones,
                problem->affine_cone_count
            );
        }
        __syncthreads();
        for (int row = threadIdx.x; row < problem->affine_rows; row += blockDim.x) {
            const int dual_row = problem->scalar_rows + row;
            const double row_step =
                dual_step * problem->scaling[problem->variables + dual_row];
            const double value =
                problem->dual[dual_row] + row_step * problem->affine_product[row];
            problem->dual[dual_row] =
                value
                - row_step
                    * (problem->cone_scratch[row] - problem->affine_offset[row]);
        }
        __syncthreads();

        zero_vector(problem->gradient, problem->variables);
        __syncthreads();
        csc_multiply(
            problem->variables,
            problem->q_offsets,
            problem->q_indices,
            problem->q,
            problem->primal,
            problem->gradient
        );
        __syncthreads();
        add_transpose_dual(problem);
        for (int variable = threadIdx.x; variable < problem->variables; variable += blockDim.x) {
            const double previous = problem->primal[variable];
            problem->previous_primal[variable] = previous;
            const double variable_step =
                primal_step * problem->scaling[variable];
            problem->primal[variable] = project_interval(
                previous
                    - variable_step
                        * (problem->gradient[variable] + problem->c[variable]),
                problem->variable_lower[variable],
                problem->variable_upper[variable]
            );
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            project_cone_blocks(
                problem->primal,
                problem->variable_cones,
                problem->variable_cone_count
            );
        }
        __syncthreads();
        for (int variable = threadIdx.x; variable < problem->variables; variable += blockDim.x) {
            problem->extrapolated_primal[variable] =
                2.0 * problem->primal[variable] - problem->previous_primal[variable];
        }
        __syncthreads();

        if (iteration == 1U || iteration % check_frequency == 0U
            || iteration == pdhg_limit) {
            evaluate_report(problem, control, report, iteration);
            if (threadIdx.x == 0) {
                if (!isfinite(report->objective)
                    || !isfinite(report->relative_primal_residual)
                    || !isfinite(report->relative_dual_residual)) {
                    report->termination = SPACEPDHCG_CUDA_TERMINATION_NUMERICAL_FAILURE;
                    atomicExch(&should_stop, 1);
                } else if (
                    report->natural_residual_inf
                        <= fmin(
                            control->feasibility_tolerance,
                            control->optimality_tolerance
                        )
                ) {
                    report->termination = SPACEPDHCG_CUDA_TERMINATION_OPTIMAL;
                    atomicExch(&should_stop, 1);
                }
            }
            __syncthreads();
            if (atomicAdd(&should_stop, 0) != 0) {
                break;
            }
        }
    }
    if (threadIdx.x == 0 && atomicAdd(&cancelled, 0) != 0) {
        report->termination = SPACEPDHCG_CUDA_TERMINATION_CANCELLED;
    }
}

__device__ void recovery_project_image(
    DeviceProblem* problem,
    const bool affine,
    const int cgls_iterations
) {
    const int rows = affine ? problem->affine_rows : problem->scalar_rows;
    const int* offsets = affine ? problem->f_offsets : problem->a_offsets;
    const int* indices = affine ? problem->f_indices : problem->a_indices;
    const double* values = affine ? problem->f : problem->a;
    double* product = affine ? problem->affine_product : problem->scalar_product;
    double* residual = problem->average_dual;
    zero_vector(problem->average_primal, problem->variables);
    zero_vector(problem->gradient, problem->variables);
    __syncthreads();
    csc_transpose_multiply(
        problem->variables,
        offsets,
        indices,
        values,
        residual,
        problem->gradient
    );
    __syncthreads();
    for (int variable = threadIdx.x;
         variable < problem->variables;
         variable += blockDim.x) {
        problem->previous_primal[variable] = problem->gradient[variable];
    }
    if (threadIdx.x == 0) {
        double gamma = 0.0;
        for (int variable = 0; variable < problem->variables; ++variable) {
            gamma += problem->gradient[variable] * problem->gradient[variable];
        }
        problem->recovery_scalars[0] = gamma;
    }
    __syncthreads();
    for (int iteration = 0; iteration < cgls_iterations; ++iteration) {
        zero_vector(product, rows);
        __syncthreads();
        csc_multiply(
            problem->variables,
            offsets,
            indices,
            values,
            problem->previous_primal,
            product
        );
        __syncthreads();
        if (threadIdx.x == 0) {
            double denominator = 0.0;
            for (int row = 0; row < rows; ++row) {
                denominator += product[row] * product[row];
            }
            problem->recovery_scalars[1] =
                denominator > 1.0e-30
                ? problem->recovery_scalars[0] / denominator
                : 0.0;
        }
        __syncthreads();
        const double alpha = problem->recovery_scalars[1];
        for (int variable = threadIdx.x;
             variable < problem->variables;
             variable += blockDim.x) {
            problem->average_primal[variable] +=
                alpha * problem->previous_primal[variable];
        }
        for (int row = threadIdx.x; row < rows; row += blockDim.x) {
            residual[row] -= alpha * product[row];
        }
        zero_vector(problem->gradient, problem->variables);
        __syncthreads();
        csc_transpose_multiply(
            problem->variables,
            offsets,
            indices,
            values,
            residual,
            problem->gradient
        );
        __syncthreads();
        if (threadIdx.x == 0) {
            double next_gamma = 0.0;
            for (int variable = 0; variable < problem->variables; ++variable) {
                next_gamma +=
                    problem->gradient[variable] * problem->gradient[variable];
            }
            problem->recovery_scalars[2] =
                problem->recovery_scalars[0] > 1.0e-30
                ? next_gamma / problem->recovery_scalars[0]
                : 0.0;
            problem->recovery_scalars[0] = next_gamma;
        }
        __syncthreads();
        const double beta = problem->recovery_scalars[2];
        for (int variable = threadIdx.x;
             variable < problem->variables;
             variable += blockDim.x) {
            problem->previous_primal[variable] =
                problem->gradient[variable]
                + beta * problem->previous_primal[variable];
        }
        __syncthreads();
        if (problem->recovery_scalars[0] <= 1.0e-28) {
            break;
        }
    }
    for (int variable = threadIdx.x;
         variable < problem->variables;
         variable += blockDim.x) {
        problem->primal[variable] += problem->average_primal[variable];
        problem->primal[variable] = project_interval(
            problem->primal[variable],
            problem->variable_lower[variable],
            problem->variable_upper[variable]
        );
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        project_cone_blocks(
            problem->primal,
            problem->variable_cones,
            problem->variable_cone_count
        );
    }
    __syncthreads();
}

__device__ void recovery_scalar_projection(DeviceProblem* problem) {
    zero_vector(problem->scalar_product, problem->scalar_rows);
    __syncthreads();
    csc_multiply(
        problem->variables,
        problem->a_offsets,
        problem->a_indices,
        problem->a,
        problem->primal,
        problem->scalar_product
    );
    __syncthreads();
    for (int row = threadIdx.x; row < problem->scalar_rows; row += blockDim.x) {
        const double value = problem->scalar_product[row];
        problem->average_dual[row] =
            project_interval(
                value,
                problem->scalar_lower[row],
                problem->scalar_upper[row]
            )
            - value;
    }
    __syncthreads();
    recovery_project_image(problem, false, 48);
}

__device__ void recovery_affine_projection(DeviceProblem* problem) {
    zero_vector(problem->affine_product, problem->affine_rows);
    __syncthreads();
    csc_multiply(
        problem->variables,
        problem->f_offsets,
        problem->f_indices,
        problem->f,
        problem->primal,
        problem->affine_product
    );
    __syncthreads();
    for (int row = threadIdx.x; row < problem->affine_rows; row += blockDim.x) {
        const double value =
            problem->affine_product[row] + problem->affine_offset[row];
        problem->cone_scratch[row] = value;
        problem->average_dual[row] = value;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        project_cone_blocks(
            problem->cone_scratch,
            problem->affine_cones,
            problem->affine_cone_count
        );
    }
    __syncthreads();
    for (int row = threadIdx.x; row < problem->affine_rows; row += blockDim.x) {
        problem->average_dual[row] =
            problem->cone_scratch[row] - problem->average_dual[row];
    }
    __syncthreads();
    recovery_project_image(problem, true, 48);
}

__device__ void recovery_active_cone_projection(DeviceProblem* problem) {
    zero_vector(problem->affine_product, problem->affine_rows);
    __syncthreads();
    csc_multiply(
        problem->variables,
        problem->f_offsets,
        problem->f_indices,
        problem->f,
        problem->primal,
        problem->affine_product
    );
    __syncthreads();
    zero_vector(problem->average_dual, problem->affine_rows);
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int cone_index = 0;
             cone_index < problem->affine_cone_count;
             ++cone_index) {
            const DeviceCone cone = problem->affine_cones[cone_index];
            if (cone.kind != SPACEPDHCG_CUDA_CONE_SECOND_ORDER) {
                continue;
            }
            const int last = cone.start + cone.vector_dimension + 1;
            double spatial_norm_squared = 0.0;
            double scale = 1.0;
            for (int slot = cone.start; slot < last; ++slot) {
                const double value =
                    problem->affine_product[slot]
                    + problem->affine_offset[slot];
                spatial_norm_squared += value * value;
                scale = fmax(scale, device_abs(value));
            }
            const double radius =
                problem->affine_product[last] + problem->affine_offset[last];
            const double spatial_norm = sqrt(spatial_norm_squared);
            scale = fmax(scale, device_abs(radius));
            if (radius - spatial_norm <= 1.0e-6 * scale) {
                problem->average_dual[last] = spatial_norm - radius;
            }
        }
    }
    __syncthreads();
    recovery_project_image(problem, true, 48);
}

__device__ bool recovery_scalar_active(
    const DeviceProblem* problem,
    const int row,
    int* sign
) {
    const double value = problem->recovery_row_values[row];
    const double lower = problem->scalar_lower[row];
    const double upper = problem->scalar_upper[row];
    if (isfinite(lower) && isfinite(upper) && lower == upper) {
        *sign = 1;
        return true;
    }
    double scale = fmax(1.0, device_abs(value));
    if (isfinite(lower)) {
        scale = fmax(scale, device_abs(lower));
    }
    if (isfinite(upper)) {
        scale = fmax(scale, device_abs(upper));
    }
    if (isfinite(upper) && upper - value <= 1.0e-6 * scale) {
        *sign = 1;
        return true;
    }
    if (isfinite(lower) && value - lower <= 1.0e-6 * scale) {
        *sign = -1;
        return true;
    }
    *sign = 0;
    return false;
}

__device__ bool recovery_soc_normal(
    const DeviceProblem* problem,
    const DeviceCone cone,
    double* spatial_norm
) {
    if (cone.kind != SPACEPDHCG_CUDA_CONE_SECOND_ORDER) {
        return false;
    }
    const int row_offset = problem->scalar_rows;
    const int last = cone.start + cone.vector_dimension + 1;
    double norm_squared = 0.0;
    double scale = 1.0;
    for (int slot = cone.start; slot < last; ++slot) {
        const double value = problem->recovery_row_values[row_offset + slot];
        norm_squared += value * value;
        scale = fmax(scale, device_abs(value));
    }
    const double radius = problem->recovery_row_values[row_offset + last];
    *spatial_norm = sqrt(norm_squared);
    scale = fmax(scale, device_abs(radius));
    return *spatial_norm > 1.0e-12
        && radius - *spatial_norm <= 1.0e-6 * scale;
}

__device__ bool recovery_soc_apex(
    const DeviceProblem* problem,
    const DeviceCone cone
) {
    if (cone.kind != SPACEPDHCG_CUDA_CONE_SECOND_ORDER) {
        return false;
    }
    const int row_offset = problem->scalar_rows;
    const int end = cone.start + cone.vector_dimension + 2;
    double maximum = 0.0;
    for (int slot = cone.start; slot < end; ++slot) {
        maximum = fmax(
            maximum,
            device_abs(problem->recovery_row_values[row_offset + slot])
        );
    }
    return maximum <= 1.0e-9;
}

__device__ bool recovery_variable_active(
    const DeviceProblem* problem,
    const int variable,
    int* sign,
    bool* fixed
) {
    const double value = problem->primal[variable];
    const double lower = problem->variable_lower[variable];
    const double upper = problem->variable_upper[variable];
    *fixed = isfinite(lower) && isfinite(upper) && lower == upper;
    if (*fixed) {
        *sign = 1;
        return true;
    }
    double scale = fmax(1.0, device_abs(value));
    if (isfinite(lower)) {
        scale = fmax(scale, device_abs(lower));
    }
    if (isfinite(upper)) {
        scale = fmax(scale, device_abs(upper));
    }
    if (isfinite(upper) && upper - value <= 1.0e-6 * scale) {
        *sign = 1;
        return true;
    }
    if (isfinite(lower) && value - lower <= 1.0e-6 * scale) {
        *sign = -1;
        return true;
    }
    *sign = 0;
    return false;
}

__device__ bool recovery_variable_in_cone(
    const DeviceProblem* problem,
    const int variable
) {
    for (int cone_index = 0;
         cone_index < problem->variable_cone_count;
         ++cone_index) {
        const DeviceCone cone = problem->variable_cones[cone_index];
        const int end = cone.start + cone.vector_dimension + 2;
        if (variable >= cone.start && variable < end) {
            return true;
        }
    }
    return false;
}

__device__ bool recovery_variable_soc_normal(
    const DeviceProblem* problem,
    const DeviceCone cone,
    const int variable,
    double* normal
) {
    if (cone.kind != SPACEPDHCG_CUDA_CONE_SECOND_ORDER) {
        return false;
    }
    const int last = cone.start + cone.vector_dimension + 1;
    double norm_squared = 0.0;
    double scale = 1.0;
    for (int slot = cone.start; slot < last; ++slot) {
        const double value = problem->primal[slot];
        norm_squared += value * value;
        scale = fmax(scale, device_abs(value));
    }
    const double spatial_norm = sqrt(norm_squared);
    const double radius = problem->primal[last];
    scale = fmax(scale, device_abs(radius));
    if (!(spatial_norm > 1.0e-12)
        || radius - spatial_norm > 1.0e-6 * scale) {
        return false;
    }
    *normal = variable < last
        ? problem->primal[variable] / spatial_norm
        : -1.0;
    return true;
}

__device__ void recovery_dual_map_to_stationarity(
    DeviceProblem* problem,
    const double* coefficients,
    double* result
) {
    for (int row = threadIdx.x; row < problem->scalar_rows; row += blockDim.x) {
        int sign = 0;
        problem->scalar_product[row] =
            recovery_scalar_active(problem, row, &sign)
            ? static_cast<double>(sign) * coefficients[row]
            : 0.0;
    }
    for (int row = threadIdx.x; row < problem->affine_rows; row += blockDim.x) {
        problem->affine_product[row] = 0.0;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int cone_index = 0;
             cone_index < problem->affine_cone_count;
             ++cone_index) {
            const DeviceCone cone = problem->affine_cones[cone_index];
            double spatial_norm = 0.0;
            if (!recovery_soc_normal(problem, cone, &spatial_norm)) {
                continue;
            }
            const int last = cone.start + cone.vector_dimension + 1;
            const double coefficient =
                coefficients[problem->scalar_rows + cone.start];
            for (int slot = cone.start; slot < last; ++slot) {
                problem->affine_product[slot] =
                    coefficient
                    * problem->recovery_row_values[problem->scalar_rows + slot]
                    / spatial_norm;
            }
            problem->affine_product[last] = -coefficient;
        }
    }
    zero_vector(result, problem->variables);
    __syncthreads();
    csc_transpose_multiply(
        problem->variables,
        problem->a_offsets,
        problem->a_indices,
        problem->a,
        problem->scalar_product,
        result
    );
    csc_transpose_multiply(
        problem->variables,
        problem->f_offsets,
        problem->f_indices,
        problem->f,
        problem->affine_product,
        result
    );
    __syncthreads();
    const int variable_offset = problem->scalar_rows + problem->affine_rows;
    for (int variable = threadIdx.x;
         variable < problem->variables;
         variable += blockDim.x) {
        int sign = 0;
        bool fixed = false;
        if (!recovery_variable_in_cone(problem, variable)
            && recovery_variable_active(problem, variable, &sign, &fixed)) {
            result[variable] +=
                static_cast<double>(sign)
                * coefficients[variable_offset + variable];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int cone_index = 0;
             cone_index < problem->variable_cone_count;
             ++cone_index) {
            const DeviceCone cone = problem->variable_cones[cone_index];
            const double coefficient =
                coefficients[variable_offset + cone.start];
            for (int variable = cone.start;
                 variable < cone.start + cone.vector_dimension + 2;
                 ++variable) {
                double normal = 0.0;
                if (recovery_variable_soc_normal(
                        problem, cone, variable, &normal
                    )) {
                    result[variable] += coefficient * normal;
                }
            }
        }
    }
    __syncthreads();
}

__device__ void recovery_dual_adjoint(
    DeviceProblem* problem,
    const double* residual,
    double* result
) {
    zero_vector(problem->scalar_product, problem->scalar_rows);
    zero_vector(problem->affine_product, problem->affine_rows);
    __syncthreads();
    csc_multiply(
        problem->variables,
        problem->a_offsets,
        problem->a_indices,
        problem->a,
        residual,
        problem->scalar_product
    );
    csc_multiply(
        problem->variables,
        problem->f_offsets,
        problem->f_indices,
        problem->f,
        residual,
        problem->affine_product
    );
    __syncthreads();
    for (int row = threadIdx.x; row < problem->scalar_rows; row += blockDim.x) {
        int sign = 0;
        result[row] =
            recovery_scalar_active(problem, row, &sign)
            ? static_cast<double>(sign) * problem->scalar_product[row]
            : 0.0;
    }
    for (int row = threadIdx.x; row < problem->affine_rows; row += blockDim.x) {
        result[problem->scalar_rows + row] = 0.0;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int cone_index = 0;
             cone_index < problem->affine_cone_count;
             ++cone_index) {
            const DeviceCone cone = problem->affine_cones[cone_index];
            double spatial_norm = 0.0;
            if (!recovery_soc_normal(problem, cone, &spatial_norm)) {
                continue;
            }
            const int last = cone.start + cone.vector_dimension + 1;
            double value = -problem->affine_product[last];
            for (int slot = cone.start; slot < last; ++slot) {
                value +=
                    problem->recovery_row_values[problem->scalar_rows + slot]
                    * problem->affine_product[slot]
                    / spatial_norm;
            }
            result[problem->scalar_rows + cone.start] = value;
        }
    }
    __syncthreads();
    const int variable_offset = problem->scalar_rows + problem->affine_rows;
    for (int variable = threadIdx.x;
         variable < problem->variables;
         variable += blockDim.x) {
        int sign = 0;
        bool fixed = false;
        result[variable_offset + variable] =
            !recovery_variable_in_cone(problem, variable)
                && recovery_variable_active(problem, variable, &sign, &fixed)
            ? static_cast<double>(sign) * residual[variable]
            : 0.0;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int cone_index = 0;
             cone_index < problem->variable_cone_count;
             ++cone_index) {
            const DeviceCone cone = problem->variable_cones[cone_index];
            double value = 0.0;
            bool active = false;
            for (int variable = cone.start;
                 variable < cone.start + cone.vector_dimension + 2;
                 ++variable) {
                double normal = 0.0;
                if (recovery_variable_soc_normal(
                        problem, cone, variable, &normal
                    )) {
                    active = true;
                    value += normal * residual[variable];
                }
            }
            if (active) {
                result[variable_offset + cone.start] = value;
            }
        }
    }
    __syncthreads();
}

__device__ bool recovery_reconstruct_dual(DeviceProblem* problem) {
    zero_vector(problem->scalar_product, problem->scalar_rows);
    zero_vector(problem->affine_product, problem->affine_rows);
    zero_vector(problem->gradient, problem->variables);
    __syncthreads();
    csc_multiply(
        problem->variables,
        problem->a_offsets,
        problem->a_indices,
        problem->a,
        problem->primal,
        problem->scalar_product
    );
    csc_multiply(
        problem->variables,
        problem->f_offsets,
        problem->f_indices,
        problem->f,
        problem->primal,
        problem->affine_product
    );
    csc_multiply(
        problem->variables,
        problem->q_offsets,
        problem->q_indices,
        problem->q,
        problem->primal,
        problem->gradient
    );
    __syncthreads();
    for (int row = threadIdx.x; row < problem->scalar_rows; row += blockDim.x) {
        problem->recovery_row_values[row] = problem->scalar_product[row];
    }
    for (int row = threadIdx.x; row < problem->affine_rows; row += blockDim.x) {
        problem->recovery_row_values[problem->scalar_rows + row] =
            problem->affine_product[row] + problem->affine_offset[row];
    }
    for (int row = threadIdx.x; row < problem->affine_rows; row += blockDim.x) {
        problem->cone_scratch[row] =
            -problem->dual[problem->scalar_rows + row];
        problem->dual[problem->scalar_rows + row] = 0.0;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int cone_index = 0;
             cone_index < problem->affine_cone_count;
             ++cone_index) {
            const DeviceCone cone = problem->affine_cones[cone_index];
            if (!recovery_soc_apex(problem, cone)) {
                continue;
            }
            project_standard_soc(
                problem->cone_scratch,
                cone.start,
                cone.vector_dimension + 2
            );
            const int end = cone.start + cone.vector_dimension + 2;
            for (int slot = cone.start; slot < end; ++slot) {
                problem->dual[problem->scalar_rows + slot] =
                    -problem->cone_scratch[slot];
            }
        }
    }
    __syncthreads();
    csc_transpose_multiply(
        problem->variables,
        problem->f_offsets,
        problem->f_indices,
        problem->f,
        problem->dual + problem->scalar_rows,
        problem->gradient
    );
    __syncthreads();
    for (int variable = threadIdx.x;
         variable < problem->variables;
         variable += blockDim.x) {
        problem->average_primal[variable] =
            -problem->gradient[variable] - problem->c[variable];
    }
    const int duals = problem->scalar_rows + problem->affine_rows;
    const int coefficients = duals + problem->variables;
    zero_vector(problem->recovery_coefficients, coefficients);
    __syncthreads();
    // Restart CGLS from the current stationarity residual.  Fixed-pattern
    // trajectory CQPs can be strongly rank-deficient, so finite-precision
    // conjugacy may be exhausted well before the nominal dimension bound.
    for (int restart = 0; restart < 8; ++restart) {
        recovery_dual_adjoint(
            problem,
            problem->average_primal,
            problem->average_dual
        );
        for (int index = threadIdx.x; index < coefficients; index += blockDim.x) {
            problem->recovery_direction_dual[index] = problem->average_dual[index];
        }
        if (threadIdx.x == 0) {
            double gamma = 0.0;
            for (int index = 0; index < coefficients; ++index) {
                gamma +=
                    problem->average_dual[index] * problem->average_dual[index];
            }
            problem->recovery_scalars[0] = gamma;
        }
        __syncthreads();
        for (int iteration = 0; iteration < 2 * coefficients; ++iteration) {
            recovery_dual_map_to_stationarity(
                problem,
                problem->recovery_direction_dual,
                problem->gradient
            );
            if (threadIdx.x == 0) {
                double denominator = 0.0;
                for (int variable = 0; variable < problem->variables; ++variable) {
                    denominator +=
                        problem->gradient[variable] * problem->gradient[variable];
                }
                problem->recovery_scalars[1] =
                    denominator > 1.0e-30
                    ? problem->recovery_scalars[0] / denominator
                    : 0.0;
            }
            __syncthreads();
            const double alpha = problem->recovery_scalars[1];
            for (int index = threadIdx.x;
                 index < coefficients;
                 index += blockDim.x) {
                problem->recovery_coefficients[index] +=
                    alpha * problem->recovery_direction_dual[index];
            }
            for (int variable = threadIdx.x;
                 variable < problem->variables;
                 variable += blockDim.x) {
                problem->average_primal[variable] -=
                    alpha * problem->gradient[variable];
            }
            __syncthreads();
            recovery_dual_adjoint(
                problem,
                problem->average_primal,
                problem->average_dual
            );
            if (threadIdx.x == 0) {
                double next_gamma = 0.0;
                for (int index = 0; index < coefficients; ++index) {
                    next_gamma +=
                        problem->average_dual[index]
                        * problem->average_dual[index];
                }
                problem->recovery_scalars[2] =
                    problem->recovery_scalars[0] > 1.0e-30
                    ? next_gamma / problem->recovery_scalars[0]
                    : 0.0;
                problem->recovery_scalars[0] = next_gamma;
            }
            __syncthreads();
            const double beta = problem->recovery_scalars[2];
            for (int index = threadIdx.x;
                 index < coefficients;
                 index += blockDim.x) {
                problem->recovery_direction_dual[index] =
                    problem->average_dual[index]
                    + beta * problem->recovery_direction_dual[index];
            }
            __syncthreads();
            if (problem->recovery_scalars[0] <= 1.0e-28) {
                break;
            }
        }
    }
    if (threadIdx.x == 0) {
        problem->recovery_scalars[3] = 1.0;
        for (int row = 0; row < problem->scalar_rows; ++row) {
            int sign = 0;
            if (!recovery_scalar_active(problem, row, &sign)) {
                problem->dual[row] = 0.0;
            } else if (
                !(isfinite(problem->scalar_lower[row])
                    && isfinite(problem->scalar_upper[row])
                    && problem->scalar_lower[row] == problem->scalar_upper[row])
                && problem->recovery_coefficients[row] < -1.0e-8
            ) {
                problem->recovery_scalars[3] = 0.0;
            } else {
                problem->dual[row] =
                    problem->recovery_coefficients[row]
                    * static_cast<double>(sign);
            }
        }
        for (int variable = 0; variable < problem->variables; ++variable) {
            int sign = 0;
            bool fixed = false;
            if (!recovery_variable_in_cone(problem, variable)
                && recovery_variable_active(problem, variable, &sign, &fixed)
                && !fixed
                && problem->recovery_coefficients[duals + variable] < -1.0e-8) {
                problem->recovery_scalars[3] = 0.0;
            }
        }
        for (int cone_index = 0;
             cone_index < problem->variable_cone_count;
             ++cone_index) {
            const DeviceCone cone = problem->variable_cones[cone_index];
            double normal = 0.0;
            if (recovery_variable_soc_normal(
                    problem, cone, cone.start, &normal
                )
                && problem->recovery_coefficients[duals + cone.start] < -1.0e-8) {
                problem->recovery_scalars[3] = 0.0;
            }
        }
        for (int row = 0; row < problem->affine_rows; ++row) {
            problem->affine_product[row] =
                problem->dual[problem->scalar_rows + row];
        }
        for (int cone_index = 0;
             cone_index < problem->affine_cone_count;
             ++cone_index) {
            const DeviceCone cone = problem->affine_cones[cone_index];
            double spatial_norm = 0.0;
            if (!recovery_soc_normal(problem, cone, &spatial_norm)) {
                continue;
            }
            const int coefficient_index = problem->scalar_rows + cone.start;
            const double coefficient =
                problem->recovery_coefficients[coefficient_index];
            if (coefficient < -1.0e-8) {
                problem->recovery_scalars[3] = 0.0;
                continue;
            }
            const int last = cone.start + cone.vector_dimension + 1;
            for (int slot = cone.start; slot < last; ++slot) {
                problem->affine_product[slot] +=
                    fmax(0.0, coefficient)
                    * problem->recovery_row_values[problem->scalar_rows + slot]
                    / spatial_norm;
            }
            problem->affine_product[last] -= fmax(0.0, coefficient);
        }
        for (int row = 0; row < problem->affine_rows; ++row) {
            problem->dual[problem->scalar_rows + row] =
                problem->affine_product[row];
        }
    }
    __syncthreads();
    return problem->recovery_scalars[3] != 0.0;
}

__global__ void recovery_kernel(
    DeviceProblem* problem,
    DeviceControl* control,
    DeviceReport* report,
    volatile int* cancellation
) {
    if (report->termination != SPACEPDHCG_CUDA_TERMINATION_ITERATION_LIMIT
        || control->iteration_limit < 350'000U
        || fmin(control->feasibility_tolerance, control->optimality_tolerance)
            > 1.0e-6
        || *cancellation != 0) {
        return;
    }
    if (threadIdx.x == 0) {
        ++control->recovery_attempt_count;
        report->recovery_attempt_count = control->recovery_attempt_count;
        report->recovery_trigger_reason =
            SPACEPDHCG_CUDA_RECOVERY_TIGHT_ITERATION_LIMIT;
        report->recovery_outcome_reason =
            SPACEPDHCG_CUDA_RECOVERY_EXHAUSTED;
        report->recovery_initial_residual = report->natural_residual_inf;
        bool supported = true;
        for (int cone_index = 0;
             cone_index < problem->affine_cone_count;
             ++cone_index) {
            const int kind = problem->affine_cones[cone_index].kind;
            supported = supported
                && (kind == SPACEPDHCG_CUDA_CONE_SECOND_ORDER
                    || kind == SPACEPDHCG_CUDA_CONE_ROTATED_SECOND_ORDER);
        }
        for (int cone_index = 0;
             cone_index < problem->variable_cone_count;
             ++cone_index) {
            const int kind = problem->variable_cones[cone_index].kind;
            supported = supported
                && kind == SPACEPDHCG_CUDA_CONE_SECOND_ORDER;
        }
        if (!supported) {
            report->recovery_outcome_reason =
                SPACEPDHCG_CUDA_RECOVERY_UNSUPPORTED_CONE;
        }
        double q_row_maximum = 0.0;
        for (int variable = 0; variable < problem->variables; ++variable) {
            double row_sum = 0.0;
            for (int index = problem->q_offsets[variable];
                 index < problem->q_offsets[variable + 1];
                 ++index) {
                row_sum += device_abs(problem->q[index]);
            }
            q_row_maximum = fmax(q_row_maximum, row_sum);
        }
        problem->recovery_scalars[3] =
            supported && q_row_maximum > 1.0e-12
            ? 0.9 / q_row_maximum
            : 0.0;
        if (supported && !(q_row_maximum > 1.0e-12)) {
            report->recovery_outcome_reason =
                SPACEPDHCG_CUDA_RECOVERY_ZERO_CURVATURE;
        }
    }
    __syncthreads();
    if (!(problem->recovery_scalars[3] > 0.0)) {
        return;
    }
    for (int variable = threadIdx.x;
         variable < problem->variables;
         variable += blockDim.x) {
        problem->recovery_backup_primal[variable] = problem->primal[variable];
    }
    const int duals = problem->scalar_rows + problem->affine_rows;
    for (int row = threadIdx.x; row < duals; row += blockDim.x) {
        problem->recovery_backup_dual[row] = problem->dual[row];
    }
    __syncthreads();
    for (int outer = 0; outer < 50'000; ++outer) {
        if (*cancellation != 0) {
            break;
        }
        zero_vector(problem->gradient, problem->variables);
        __syncthreads();
        csc_multiply(
            problem->variables,
            problem->q_offsets,
            problem->q_indices,
            problem->q,
            problem->primal,
            problem->gradient
        );
        __syncthreads();
        for (int variable = threadIdx.x;
             variable < problem->variables;
             variable += blockDim.x) {
            double diagonal = 0.0;
            for (int index = problem->q_offsets[variable];
                 index < problem->q_offsets[variable + 1];
                 ++index) {
                if (problem->q_indices[index] == variable) {
                    diagonal += device_abs(problem->q[index]);
                }
            }
            const double step = 0.9 / sqrt(
                fmax(1.0e-12, diagonal)
                / fmax(1.0e-12, problem->recovery_scalars[3] / 0.9)
            );
            problem->primal[variable] = project_interval(
                problem->primal[variable]
                    - step * (problem->gradient[variable] + problem->c[variable]),
                problem->variable_lower[variable],
                problem->variable_upper[variable]
            );
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            project_cone_blocks(
                problem->primal,
                problem->variable_cones,
                problem->variable_cone_count
            );
        }
        __syncthreads();
        recovery_scalar_projection(problem);
        recovery_affine_projection(problem);
    }
    if (*cancellation != 0) {
        for (int variable = threadIdx.x;
             variable < problem->variables;
             variable += blockDim.x) {
            problem->primal[variable] = problem->recovery_backup_primal[variable];
        }
        for (int row = threadIdx.x; row < duals; row += blockDim.x) {
            problem->dual[row] = problem->recovery_backup_dual[row];
        }
        __syncthreads();
        evaluate_report(problem, control, report, control->iteration_limit);
        if (threadIdx.x == 0) {
            report->recovery_count = control->recovery_count;
            report->recovery_rejected_count = control->recovery_rejected_count;
            report->recovery_iterations = 0U;
            report->recovery_outcome_reason =
                SPACEPDHCG_CUDA_RECOVERY_CANCELLED;
            report->termination = SPACEPDHCG_CUDA_TERMINATION_CANCELLED;
        }
        return;
    }
    for (int iteration = 0; iteration < 100; ++iteration) {
        recovery_scalar_projection(problem);
        recovery_active_cone_projection(problem);
    }
    bool accepted = false;
    __shared__ int kkt_converged;
    for (int refinement = 0; refinement < 32; ++refinement) {
        accepted = recovery_reconstruct_dual(problem);
        evaluate_report(problem, control, report, control->iteration_limit);
        if (threadIdx.x == 0) {
            kkt_converged = accepted
                && isfinite(report->natural_residual_inf)
                && report->natural_residual_inf
                    <= fmin(
                        control->feasibility_tolerance,
                        control->optimality_tolerance
                    );
        }
        __syncthreads();
        if (kkt_converged != 0) {
            break;
        }
        for (int variable = threadIdx.x;
             variable < problem->variables;
             variable += blockDim.x) {
            double diagonal = 0.0;
            for (int index = problem->q_offsets[variable];
                 index < problem->q_offsets[variable + 1];
                 ++index) {
                if (problem->q_indices[index] == variable) {
                    diagonal += device_abs(problem->q[index]);
                }
            }
            const double reduced_gradient =
                problem->gradient[variable] + problem->c[variable];
            const double maximum_change =
                0.05 * fmax(1.0, device_abs(problem->primal[variable]));
            const double change = fmin(
                maximum_change,
                fmax(
                    -maximum_change,
                    -reduced_gradient / fmax(1.0e-6, diagonal)
                )
            );
            problem->primal[variable] = project_interval(
                problem->primal[variable] + change,
                problem->variable_lower[variable],
                problem->variable_upper[variable]
            );
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            project_cone_blocks(
                problem->primal,
                problem->variable_cones,
                problem->variable_cone_count
            );
        }
        __syncthreads();
        for (int projection = 0; projection < 100; ++projection) {
            recovery_scalar_projection(problem);
            recovery_active_cone_projection(problem);
        }
    }
    accepted = recovery_reconstruct_dual(problem);
    evaluate_report(problem, control, report, control->iteration_limit);
    __shared__ int qualified;
    if (threadIdx.x == 0) {
        problem->recovery_scalars[2] = report->natural_residual_inf;
        report->recovery_final_primal_residual = fmax(
            report->scalar_primal_violation_inf,
            fmax(report->box_violation_inf, report->affine_cone_distance_inf)
        );
        report->recovery_final_stationarity = report->stationarity_inf;
        report->recovery_final_complementarity = report->complementarity_inf;
        report->recovery_stationarity_index = -1;
        report->recovery_stationarity_value = 0.0;
        for (int variable = 0; variable < problem->variables; ++variable) {
            const double reduced_gradient =
                problem->gradient[variable] + problem->c[variable];
            const double natural = problem->primal[variable]
                - project_interval(
                    problem->primal[variable] - reduced_gradient,
                    problem->variable_lower[variable],
                    problem->variable_upper[variable]
                );
            if (device_abs(natural)
                > device_abs(report->recovery_stationarity_value)) {
                report->recovery_stationarity_index = variable;
                report->recovery_stationarity_value = natural;
            }
        }
        qualified = accepted
            && isfinite(report->natural_residual_inf)
            && report->natural_residual_inf
                <= fmin(
                    control->feasibility_tolerance,
                    control->optimality_tolerance
                );
        if (qualified != 0) {
            ++control->recovery_count;
            report->recovery_outcome_reason =
                SPACEPDHCG_CUDA_RECOVERY_QUALIFIED;
        } else {
            ++control->recovery_rejected_count;
            report->recovery_outcome_reason = accepted
                ? SPACEPDHCG_CUDA_RECOVERY_EXHAUSTED
                : SPACEPDHCG_CUDA_RECOVERY_DUAL_INFEASIBLE;
        }
    }
    __syncthreads();
    if (qualified == 0) {
        for (int variable = threadIdx.x;
             variable < problem->variables;
             variable += blockDim.x) {
            problem->primal[variable] = problem->recovery_backup_primal[variable];
        }
        for (int row = threadIdx.x; row < duals; row += blockDim.x) {
            problem->dual[row] = problem->recovery_backup_dual[row];
        }
        __syncthreads();
        evaluate_report(problem, control, report, control->iteration_limit);
    }
    if (threadIdx.x == 0) {
        report->recovery_count = control->recovery_count;
        report->recovery_rejected_count = control->recovery_rejected_count;
        report->recovery_iterations = 50'000U;
        report->recovery_attempt_count = control->recovery_attempt_count;
        report->recovery_final_residual = problem->recovery_scalars[2];
        report->termination =
            qualified != 0
            ? SPACEPDHCG_CUDA_TERMINATION_OPTIMAL
            : SPACEPDHCG_CUDA_TERMINATION_ITERATION_LIMIT;
    }
}

__global__ void residual_kernel(
    DeviceProblem* problem,
    DeviceControl* control,
    DeviceReport* report
) {
    evaluate_report(problem, control, report, report->iterations);
}

__global__ void set_constant_kernel(double* values, const int count, const double value) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < count;
         index += blockDim.x * gridDim.x) {
        values[index] = value;
    }
}

__global__ void prepare_warm_start_kernel(
    double* previous,
    double* extrapolated,
    const double* primal,
    const int count
) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < count;
         index += blockDim.x * gridDim.x) {
        previous[index] = primal[index];
        extrapolated[index] = primal[index];
    }
}

__global__ void checkpoint_steps_kernel(double* destination, const DeviceControl* control) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        destination[0] = control->primal_step;
        destination[1] = control->dual_step;
    }
}

__global__ void restore_steps_kernel(DeviceControl* control, const double* source) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        control->primal_step = source[0];
        control->dual_step = source[1];
        control->force_scaling_refresh = 0;
    }
}

struct ViewExpectation {
    std::size_t elements;
    spacepdhcg_accelerator_scalar_type scalar_type;
    spacepdhcg_accelerator_access access;
    const char* name;
};

spacepdhcg_cuda_status dlpack_view(
    const spacepdhcg_dlpack_managed_tensor& source,
    DLPackOwner& owner,
    spacepdhcg_accelerator_buffer_view& destination,
    std::string& error
) {
    if (source.managed_tensor == nullptr
        || (source.kind != SPACEPDHCG_DLPACK_LEGACY
            && source.kind != SPACEPDHCG_DLPACK_VERSIONED)
        || (source.access != SPACEPDHCG_ACCESS_READ_ONLY
            && source.access != SPACEPDHCG_ACCESS_READ_WRITE)) {
        error = "invalid DLPack managed tensor wrapper";
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    owner = DLPackOwner(source);

    const DLPackTensor* tensor = nullptr;
    if (source.kind == SPACEPDHCG_DLPACK_VERSIONED) {
        const auto* managed =
            static_cast<const DLPackVersionedManaged*>(source.managed_tensor);
        if (managed->version.major != kDLPackMajorVersion) {
            error = "incompatible DLPack major ABI version";
            return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
        }
        if (source.access == SPACEPDHCG_ACCESS_READ_WRITE
            && (managed->flags & kDLPackReadOnly) != 0U) {
            error = "read-only DLPack tensor cannot satisfy a writable slot";
            return SPACEPDHCG_CUDA_POINTER_CONTRACT;
        }
        tensor = &managed->tensor;
    } else {
        tensor = &static_cast<const DLPackLegacyManaged*>(
                      source.managed_tensor
        )->tensor;
    }

    if (tensor->ndim != 1 || tensor->shape == nullptr
        || tensor->shape[0] < 0) {
        error = "DLPack tensor must have rank one and a non-negative shape";
        return SPACEPDHCG_CUDA_POINTER_CONTRACT;
    }
    if (tensor->shape[0] > 0 && tensor->strides != nullptr
        && tensor->strides[0] != 1) {
        error = "DLPack tensor must have compact unit stride";
        return SPACEPDHCG_CUDA_POINTER_CONTRACT;
    }
    if (tensor->dtype.lanes != 1U
        || (tensor->dtype.code != 0U && tensor->dtype.code != 2U)
        || (tensor->dtype.bits != 32U && tensor->dtype.bits != 64U)) {
        error = "DLPack tensor has an unsupported scalar dtype";
        return SPACEPDHCG_CUDA_POINTER_CONTRACT;
    }
    if (static_cast<std::uint64_t>(tensor->shape[0])
        > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        error = "DLPack tensor element count exceeds size_t";
        return SPACEPDHCG_CUDA_POINTER_CONTRACT;
    }

    spacepdhcg_accelerator_device device{};
    if (tensor->device.type == 2 && tensor->device.id >= 0) {
        device = {SPACEPDHCG_DEVICE_CUDA, tensor->device.id};
    } else if (tensor->device.type == 13 && tensor->device.id == 0) {
        device = {SPACEPDHCG_DEVICE_CUDA_MANAGED, 0};
    } else {
        error = "DLPack storage must be CUDA device or CUDA managed memory";
        return SPACEPDHCG_CUDA_POINTER_CONTRACT;
    }

    spacepdhcg_accelerator_scalar_type scalar_type{};
    if (tensor->dtype.code == 0U) {
        scalar_type = tensor->dtype.bits == 32U
            ? SPACEPDHCG_SCALAR_INT32
            : SPACEPDHCG_SCALAR_INT64;
    } else {
        scalar_type = tensor->dtype.bits == 32U
            ? SPACEPDHCG_SCALAR_FLOAT32
            : SPACEPDHCG_SCALAR_FLOAT64;
    }
    const auto elements = static_cast<std::size_t>(tensor->shape[0]);
    const auto scalar_bytes = static_cast<std::size_t>(tensor->dtype.bits / 8U);
    if (elements > 0U && tensor->data == nullptr) {
        error = "non-empty DLPack tensor has a null data pointer";
        return SPACEPDHCG_CUDA_POINTER_CONTRACT;
    }
    const auto pointer_value = reinterpret_cast<std::uintptr_t>(tensor->data);
    if (tensor->byte_offset > std::numeric_limits<std::size_t>::max()
        || (elements > 0U
            && (tensor->byte_offset
                    > std::numeric_limits<std::uintptr_t>::max() - pointer_value
                || (pointer_value + static_cast<std::size_t>(tensor->byte_offset))
                        % scalar_bytes
                    != 0U))) {
        error = "DLPack tensor byte offset is not scalar aligned";
        return SPACEPDHCG_CUDA_POINTER_CONTRACT;
    }
    destination = {
        elements == 0U ? nullptr : tensor->data,
        device,
        scalar_type,
        elements,
        elements == 0U ? 0U : static_cast<std::size_t>(tensor->byte_offset),
        1,
        source.access,
    };
    return SPACEPDHCG_CUDA_SUCCESS;
}

spacepdhcg_cuda_status validate_view(
    const spacepdhcg_accelerator_buffer_view& view,
    const spacepdhcg_accelerator_device storage,
    const ViewExpectation expectation,
    const int execution_device,
    std::string& error
) {
    if (view.device.type != storage.type || view.device.id != storage.id) {
        error = std::string(expectation.name) + " has a different storage device";
        return SPACEPDHCG_CUDA_POINTER_CONTRACT;
    }
    if (view.scalar_type != expectation.scalar_type
        || view.elements != expectation.elements
        || view.access != expectation.access) {
        error = std::string(expectation.name) + " has an invalid dtype, length, or access";
        return SPACEPDHCG_CUDA_POINTER_CONTRACT;
    }
    if (view.element_stride != 1) {
        error = std::string(expectation.name) + " must be contiguous";
        return SPACEPDHCG_CUDA_POINTER_CONTRACT;
    }
    const std::size_t scalar_bytes =
        view.scalar_type == SPACEPDHCG_SCALAR_INT32
            || view.scalar_type == SPACEPDHCG_SCALAR_FLOAT32
        ? 4U
        : 8U;
    if (view.byte_offset % scalar_bytes != 0U) {
        error = std::string(expectation.name) + " has an unaligned byte offset";
        return SPACEPDHCG_CUDA_POINTER_CONTRACT;
    }
    if (view.elements == 0U) {
        if (view.data != nullptr || view.byte_offset != 0U) {
            error = std::string(expectation.name) + " zero view is not canonical";
            return SPACEPDHCG_CUDA_POINTER_CONTRACT;
        }
        return SPACEPDHCG_CUDA_SUCCESS;
    }
    if (view.data == nullptr) {
        error = std::string(expectation.name) + " is null";
        return SPACEPDHCG_CUDA_POINTER_CONTRACT;
    }
    cudaPointerAttributes attributes{};
    const auto pointer_status = cudaPointerGetAttributes(&attributes, view.data);
    if (pointer_status != cudaSuccess) {
        error = std::string(expectation.name) + " CUDA pointer query failed: "
            + cuda_message(pointer_status);
        static_cast<void>(cudaGetLastError());
        return SPACEPDHCG_CUDA_POINTER_CONTRACT;
    }
    if (storage.type == SPACEPDHCG_DEVICE_CUDA) {
        if (attributes.type != cudaMemoryTypeDevice || attributes.device != execution_device) {
            error = std::string(expectation.name) + " is not device memory on the stream GPU";
            return SPACEPDHCG_CUDA_POINTER_CONTRACT;
        }
    } else if (storage.type == SPACEPDHCG_DEVICE_CUDA_MANAGED) {
        if (attributes.type != cudaMemoryTypeManaged) {
            error = std::string(expectation.name) + " is not CUDA managed memory";
            return SPACEPDHCG_CUDA_POINTER_CONTRACT;
        }
    } else {
        error = "persistent storage must be CUDA device or CUDA managed";
        return SPACEPDHCG_CUDA_POINTER_CONTRACT;
    }
    return SPACEPDHCG_CUDA_SUCCESS;
}

std::pair<std::uintptr_t, std::uintptr_t> address_interval(
    const spacepdhcg_accelerator_buffer_view& view
) {
    if (view.elements == 0U) {
        return {0U, 0U};
    }
    const std::size_t scalar_bytes =
        view.scalar_type == SPACEPDHCG_SCALAR_INT32
            || view.scalar_type == SPACEPDHCG_SCALAR_FLOAT32
        ? 4U
        : 8U;
    const auto begin = reinterpret_cast<std::uintptr_t>(view.data) + view.byte_offset;
    return {begin, begin + view.elements * scalar_bytes};
}

bool intervals_overlap(
    const std::pair<std::uintptr_t, std::uintptr_t>& left,
    const std::pair<std::uintptr_t, std::uintptr_t>& right
) {
    return left.first != left.second && right.first != right.second
        && left.first < right.second && right.first < left.second;
}

spacepdhcg_cuda_status validate_cones(
    const spacepdhcg_cuda_cone_descriptor* cones,
    const std::size_t count,
    const int ambient,
    const bool require_cover,
    std::string& error
) {
    if (count > 0U && cones == nullptr) {
        error = "cone descriptor pointer is null";
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    std::size_t previous_stop = 0U;
    for (std::size_t index = 0; index < count; ++index) {
        const auto& cone = cones[index];
        if (cone.start < 0 || cone.vector_dimension <= 0) {
            error = "cone start or dimension is invalid";
            return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
        }
        if (cone.kind != SPACEPDHCG_CUDA_CONE_SECOND_ORDER
            && cone.kind != SPACEPDHCG_CUDA_CONE_ROTATED_SECOND_ORDER) {
            error = "G2 persistent kernel supports SOC and rotated SOC cones";
            return SPACEPDHCG_CUDA_UNSUPPORTED;
        }
        const auto start = static_cast<std::size_t>(cone.start);
        const auto stop = start + static_cast<std::size_t>(cone.vector_dimension) + 2U;
        if (start < previous_stop || stop > static_cast<std::size_t>(ambient)
            || (require_cover && start != previous_stop)) {
            error = "cone blocks overlap, exceed, or do not cover their ambient space";
            return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
        }
        previous_stop = stop;
    }
    if (require_cover && previous_stop != static_cast<std::size_t>(ambient)) {
        error = "affine cone blocks do not cover all affine rows";
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    return SPACEPDHCG_CUDA_SUCCESS;
}

}  // namespace

struct spacepdhcg_cuda_workspace {
    std::mutex mutex{};
    spacepdhcg_cuda_structure structure{};
    spacepdhcg_cuda_create_options options{};
    spacepdhcg_cqp_iterate_accelerator_views external_iterates{};
    spacepdhcg_accelerator_stream consumer_stream{};
    spacepdhcg_cuda_workspace_state state{SPACEPDHCG_CUDA_UNINITIALISED};
    spacepdhcg_cuda_termination termination{SPACEPDHCG_CUDA_TERMINATION_UNSPECIFIED};
    spacepdhcg_cuda_warm_start_mode warm_mode{SPACEPDHCG_CUDA_WARM_START_NONE};
    bool warm_accepted{false};
    bool external_retained{false};
    std::string error{};
    std::vector<DLPackOwner> persistent_dlpack_borrows{};
    std::vector<DLPackOwner> pending_dlpack_borrows{};
    AllocationLedger ledger{};
    DeviceTopology topology{};
    DeviceNumeric numeric{};
    DeviceState solver{};
    DeviceCone* affine_cones{nullptr};
    DeviceCone* variable_cones{nullptr};
    DeviceProblem* device_problem{nullptr};
    DeviceControl* control{nullptr};
    DeviceReport* report{nullptr};
    DeviceReport* host_report{nullptr};
    int* host_cancellation{nullptr};
    cusparseHandle_t sparse{nullptr};
    cusparseSpMatDescr_t q_descriptor{nullptr};
    cusparseSpMatDescr_t a_descriptor{nullptr};
    cusparseSpMatDescr_t f_descriptor{nullptr};
    cudaStream_t control_stream{nullptr};
    StreamEvent completion{};
    TimingEvents update_timer{};
    TimingEvents solve_timer{};
    TimingEvents recovery_timer{};
    LastOperation last_operation{LastOperation::none};
    std::uint64_t topology_index_copy_count{0};
    std::uint64_t total_copy_count{0};
    std::uint64_t total_copy_bytes{0};
    std::uint64_t update_epoch{0};
    std::uint64_t solve_epoch{0};
    std::uint64_t scaling_epoch{0};
    std::uint64_t graph_epoch{0};
    std::uint64_t allocation_delta_last_update{0};
    std::uint64_t topology_allocation_delta_last_update{0};
    std::uint64_t topology_index_copy_delta_last_update{0};
    double update_seconds{0.0};
    double scaling_seconds{0.0};
    double solve_seconds{0.0};
    double recovery_seconds{0.0};
};

namespace {

void set_error(spacepdhcg_cuda_workspace* workspace, const char* message) {
    if (workspace != nullptr) {
        workspace->error = message == nullptr ? "unknown error" : message;
    }
}

spacepdhcg_cuda_status append_dlpack_view(
    const spacepdhcg_dlpack_managed_tensor& source,
    spacepdhcg_accelerator_buffer_view& destination,
    std::vector<DLPackOwner>& owners,
    std::string& error
) {
    DLPackOwner owner{};
    const auto status = dlpack_view(source, owner, destination, error);
    if (status == SPACEPDHCG_CUDA_SUCCESS) {
        owners.push_back(std::move(owner));
    }
    return status;
}

spacepdhcg_cuda_status cuda_failure(
    spacepdhcg_cuda_workspace* workspace,
    const cudaError_t error,
    const char* operation
) {
    if (workspace != nullptr) {
        workspace->error = std::string(operation) + ": " + cuda_message(error);
        workspace->state = SPACEPDHCG_CUDA_FAILED;
    }
    return error == cudaErrorMemoryAllocation
        ? SPACEPDHCG_CUDA_OUT_OF_MEMORY
        : SPACEPDHCG_CUDA_RUNTIME_ERROR;
}

spacepdhcg_cuda_status allocate_device(
    spacepdhcg_cuda_workspace* workspace,
    void** pointer,
    const std::size_t bytes,
    const AllocationCategory category
) {
    const auto status =
        workspace->ledger.allocate(pointer, bytes, category, workspace->update_epoch + 1U);
    return status == cudaSuccess
        ? SPACEPDHCG_CUDA_SUCCESS
        : cuda_failure(workspace, status, "cudaMalloc");
}

spacepdhcg_cuda_status copy_async(
    spacepdhcg_cuda_workspace* workspace,
    void* destination,
    const void* source,
    const std::size_t bytes,
    const cudaMemcpyKind kind,
    const cudaStream_t stream,
    const bool topology_index
) {
    if (bytes == 0U) {
        return SPACEPDHCG_CUDA_SUCCESS;
    }
    const auto status = cudaMemcpyAsync(destination, source, bytes, kind, stream);
    if (status != cudaSuccess) {
        return cuda_failure(workspace, status, "cudaMemcpyAsync");
    }
    ++workspace->total_copy_count;
    workspace->total_copy_bytes += bytes;
    if (topology_index) {
        ++workspace->topology_index_copy_count;
    }
    return SPACEPDHCG_CUDA_SUCCESS;
}

spacepdhcg_cuda_status check_stream(
    spacepdhcg_cuda_workspace* workspace,
    const spacepdhcg_accelerator_stream stream
) {
    if (stream.device.type != SPACEPDHCG_DEVICE_CUDA
        || stream.device.id != workspace->consumer_stream.device.id) {
        set_error(workspace, "stream execution device does not match the workspace");
        return SPACEPDHCG_CUDA_POINTER_CONTRACT;
    }
    if (!same_stream(stream, workspace->consumer_stream)) {
        set_error(workspace, "workspace operations must use the declared consumer stream");
        return SPACEPDHCG_CUDA_POINTER_CONTRACT;
    }
    return SPACEPDHCG_CUDA_SUCCESS;
}

std::vector<spacepdhcg_accelerator_buffer_view> topology_views(
    const spacepdhcg_cqp_accelerator_exchange& exchange
) {
    return {
        exchange.topology.quadratic_offsets,
        exchange.topology.quadratic_indices,
        exchange.topology.scalar_offsets,
        exchange.topology.scalar_indices,
        exchange.topology.affine_offsets,
        exchange.topology.affine_indices,
    };
}

std::vector<spacepdhcg_accelerator_buffer_view> numeric_views(
    const spacepdhcg_cqp_numeric_accelerator_views& values
) {
    return {
        values.quadratic,
        values.scalar_constraint,
        values.affine_cone,
        values.linear_objective,
        values.scalar_lower,
        values.scalar_upper,
        values.affine_offset,
        values.variable_lower,
        values.variable_upper,
    };
}

spacepdhcg_cuda_status validate_numeric_views(
    spacepdhcg_cuda_workspace* workspace,
    const spacepdhcg_cqp_numeric_accelerator_views& values,
    const spacepdhcg_accelerator_device storage
) {
    const auto& structure = workspace->structure;
    const ViewExpectation expectations[] = {
        {structure.quadratic_nonzeros, SPACEPDHCG_SCALAR_FLOAT64,
         SPACEPDHCG_ACCESS_READ_WRITE, "quadratic values"},
        {structure.scalar_nonzeros, SPACEPDHCG_SCALAR_FLOAT64,
         SPACEPDHCG_ACCESS_READ_WRITE, "scalar values"},
        {structure.affine_nonzeros, SPACEPDHCG_SCALAR_FLOAT64,
         SPACEPDHCG_ACCESS_READ_WRITE, "affine values"},
        {static_cast<std::size_t>(structure.variables), SPACEPDHCG_SCALAR_FLOAT64,
         SPACEPDHCG_ACCESS_READ_WRITE, "linear objective"},
        {static_cast<std::size_t>(structure.scalar_rows), SPACEPDHCG_SCALAR_FLOAT64,
         SPACEPDHCG_ACCESS_READ_WRITE, "scalar lower"},
        {static_cast<std::size_t>(structure.scalar_rows), SPACEPDHCG_SCALAR_FLOAT64,
         SPACEPDHCG_ACCESS_READ_WRITE, "scalar upper"},
        {static_cast<std::size_t>(structure.affine_rows), SPACEPDHCG_SCALAR_FLOAT64,
         SPACEPDHCG_ACCESS_READ_WRITE, "affine offset"},
        {static_cast<std::size_t>(structure.variables), SPACEPDHCG_SCALAR_FLOAT64,
         SPACEPDHCG_ACCESS_READ_WRITE, "variable lower"},
        {static_cast<std::size_t>(structure.variables), SPACEPDHCG_SCALAR_FLOAT64,
         SPACEPDHCG_ACCESS_READ_WRITE, "variable upper"},
    };
    const auto views = numeric_views(values);
    for (std::size_t index = 0; index < views.size(); ++index) {
        std::string error;
        const auto status = validate_view(
            views[index],
            storage,
            expectations[index],
            workspace->consumer_stream.device.id,
            error
        );
        if (status != SPACEPDHCG_CUDA_SUCCESS) {
            workspace->error = std::move(error);
            return status;
        }
    }
    return SPACEPDHCG_CUDA_SUCCESS;
}

spacepdhcg_cuda_status enqueue_numeric_copy(
    spacepdhcg_cuda_workspace* workspace,
    const spacepdhcg_cqp_numeric_accelerator_views& values,
    const cudaStream_t stream
) {
    struct Copy {
        double* destination;
        const spacepdhcg_accelerator_buffer_view* source;
    };
    const Copy copies[] = {
        {workspace->numeric.q, &values.quadratic},
        {workspace->numeric.a, &values.scalar_constraint},
        {workspace->numeric.f, &values.affine_cone},
        {workspace->numeric.c, &values.linear_objective},
        {workspace->numeric.scalar_lower, &values.scalar_lower},
        {workspace->numeric.scalar_upper, &values.scalar_upper},
        {workspace->numeric.affine_offset, &values.affine_offset},
        {workspace->numeric.variable_lower, &values.variable_lower},
        {workspace->numeric.variable_upper, &values.variable_upper},
    };
    for (const auto& copy : copies) {
        const auto bytes = checked_bytes(copy.source->elements, sizeof(double));
        const auto status = copy_async(
            workspace,
            copy.destination,
            offset_pointer_const<double>(*copy.source),
            bytes,
            cudaMemcpyDefault,
            stream,
            false
        );
        if (status != SPACEPDHCG_CUDA_SUCCESS) {
            return status;
        }
    }
    return SPACEPDHCG_CUDA_SUCCESS;
}

NumericPointers source_pointers(
    const spacepdhcg_cqp_numeric_accelerator_views& values
) {
    return NumericPointers{
        offset_pointer_const<double>(values.quadratic),
        offset_pointer_const<double>(values.scalar_constraint),
        offset_pointer_const<double>(values.affine_cone),
        offset_pointer_const<double>(values.linear_objective),
        offset_pointer_const<double>(values.scalar_lower),
        offset_pointer_const<double>(values.scalar_upper),
        offset_pointer_const<double>(values.affine_offset),
        offset_pointer_const<double>(values.variable_lower),
        offset_pointer_const<double>(values.variable_upper),
    };
}

spacepdhcg_cuda_status record_completion(
    spacepdhcg_cuda_workspace* workspace,
    const cudaStream_t stream,
    const LastOperation operation
) {
    const auto status = workspace->completion.record(stream);
    if (status != cudaSuccess) {
        return cuda_failure(workspace, status, "cudaEventRecord");
    }
    workspace->last_operation = operation;
    return SPACEPDHCG_CUDA_SUCCESS;
}

void finish_solve(spacepdhcg_cuda_workspace* workspace) {
    workspace->termination =
        static_cast<spacepdhcg_cuda_termination>(workspace->host_report->termination);
    if (workspace->termination == SPACEPDHCG_CUDA_TERMINATION_CANCELLED) {
        workspace->state = SPACEPDHCG_CUDA_CANCELLED;
    } else if (workspace->termination == SPACEPDHCG_CUDA_TERMINATION_NUMERICAL_FAILURE) {
        workspace->state = SPACEPDHCG_CUDA_FAILED;
        workspace->error = "persistent CUDA iteration produced non-finite diagnostics";
    } else {
        workspace->state = SPACEPDHCG_CUDA_SOLVED;
    }
    workspace->solve_seconds = workspace->solve_timer.elapsed_seconds();
    workspace->recovery_seconds =
        workspace->host_report->recovery_attempt_count > 0U
        ? workspace->recovery_timer.elapsed_seconds()
        : 0.0;
    if (workspace->host_report->scaling_refreshed != 0) {
        ++workspace->scaling_epoch;
    }
}

spacepdhcg_cuda_status finalize_if_complete(spacepdhcg_cuda_workspace* workspace) {
    const auto event_status = workspace->completion.query();
    if (event_status == cudaErrorNotReady) {
        return SPACEPDHCG_CUDA_BUSY;
    }
    if (event_status != cudaSuccess) {
        return cuda_failure(workspace, event_status, "cudaEventQuery");
    }
    if (workspace->last_operation == LastOperation::solve
        && workspace->state == SPACEPDHCG_CUDA_SOLVING) {
        finish_solve(workspace);
    } else if (workspace->last_operation == LastOperation::update) {
        workspace->update_seconds = workspace->update_timer.elapsed_seconds();
    }
    workspace->pending_dlpack_borrows.clear();
    return SPACEPDHCG_CUDA_SUCCESS;
}

spacepdhcg_cuda_status create_sparse_descriptor(
    spacepdhcg_cuda_workspace* workspace,
    cusparseSpMatDescr_t* descriptor,
    const int rows,
    const int columns,
    const std::size_t nonzeros,
    int* offsets,
    int* indices,
    double* values
) {
    if (nonzeros == 0U) {
        *descriptor = nullptr;
        return SPACEPDHCG_CUDA_SUCCESS;
    }
    const auto status = cusparseCreateCsc(
        descriptor,
        rows,
        columns,
        static_cast<std::int64_t>(nonzeros),
        offsets,
        indices,
        values,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F
    );
    if (status != CUSPARSE_STATUS_SUCCESS) {
        workspace->error = "cusparseCreateCsc failed";
        workspace->state = SPACEPDHCG_CUDA_FAILED;
        return SPACEPDHCG_CUDA_RUNTIME_ERROR;
    }
    return SPACEPDHCG_CUDA_SUCCESS;
}

void cleanup_workspace(spacepdhcg_cuda_workspace* workspace) noexcept {
    if (workspace == nullptr) {
        return;
    }
    if (workspace->q_descriptor != nullptr) {
        static_cast<void>(cusparseDestroySpMat(workspace->q_descriptor));
    }
    if (workspace->a_descriptor != nullptr) {
        static_cast<void>(cusparseDestroySpMat(workspace->a_descriptor));
    }
    if (workspace->f_descriptor != nullptr) {
        static_cast<void>(cusparseDestroySpMat(workspace->f_descriptor));
    }
    if (workspace->sparse != nullptr) {
        static_cast<void>(cusparseDestroy(workspace->sparse));
    }
    if (workspace->control_stream != nullptr) {
        static_cast<void>(cudaStreamDestroy(workspace->control_stream));
    }
    const auto& records = workspace->ledger.records();
    std::uint64_t epoch = workspace->update_epoch + workspace->solve_epoch + 2U;
    for (auto iterator = records.rbegin(); iterator != records.rend(); ++iterator) {
        if (iterator->free_epoch == 0U) {
            static_cast<void>(workspace->ledger.release(iterator->pointer, epoch));
        }
    }
    if (workspace->external_retained && workspace->options.release_external != nullptr) {
        workspace->options.release_external(workspace->options.external_lifetime_context);
        workspace->external_retained = false;
    }
}

}  // namespace

extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_workspace_create(
    const spacepdhcg_cuda_structure* structure,
    const spacepdhcg_cqp_accelerator_exchange* exchange,
    const spacepdhcg_cuda_create_options* options,
    spacepdhcg_cuda_workspace** workspace
) {
    if (workspace == nullptr) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    *workspace = nullptr;
    if (structure == nullptr || exchange == nullptr || options == nullptr) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    auto* result = new (std::nothrow) spacepdhcg_cuda_workspace;
    if (result == nullptr) {
        return SPACEPDHCG_CUDA_OUT_OF_MEMORY;
    }
    try {
        result->structure = *structure;
        result->options = *options;
        result->consumer_stream = exchange->consumer_stream;
        result->external_iterates = exchange->iterates;
        if (structure->abi_version != SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION
            || options->abi_version != SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION
            || exchange->abi_version != SPACEPDHCG_ACCELERATOR_EXCHANGE_ABI_VERSION) {
            set_error(result, "unsupported workspace or exchange ABI version");
            cleanup_workspace(result);
            delete result;
            return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
        }
        if (structure->topology_fingerprint != exchange->topology_fingerprint) {
            set_error(result, "exchange topology fingerprint mismatch");
            cleanup_workspace(result);
            delete result;
            return SPACEPDHCG_CUDA_TOPOLOGY_MISMATCH;
        }
        if (structure->variables <= 0 || structure->scalar_rows < 0
            || structure->affine_rows < 0
            || structure->quadratic_nonzeros > static_cast<std::size_t>(INT_MAX)
            || structure->scalar_nonzeros > static_cast<std::size_t>(INT_MAX)
            || structure->affine_nonzeros > static_cast<std::size_t>(INT_MAX)) {
            set_error(result, "invalid workspace dimensions");
            cleanup_workspace(result);
            delete result;
            return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
        }
        std::string cone_error;
        auto status = validate_cones(
            structure->affine_cones,
            structure->affine_cone_count,
            structure->affine_rows,
            structure->affine_rows > 0,
            cone_error
        );
        if (status == SPACEPDHCG_CUDA_SUCCESS) {
            status = validate_cones(
                structure->variable_cones,
                structure->variable_cone_count,
                structure->variables,
                false,
                cone_error
            );
        }
        if (status != SPACEPDHCG_CUDA_SUCCESS) {
            result->error = std::move(cone_error);
            cleanup_workspace(result);
            delete result;
            return status;
        }
        if (exchange->consumer_stream.device.type != SPACEPDHCG_DEVICE_CUDA
            || exchange->consumer_stream.device.id < 0) {
            set_error(result, "consumer stream must identify a CUDA execution device");
            cleanup_workspace(result);
            delete result;
            return SPACEPDHCG_CUDA_POINTER_CONTRACT;
        }
        auto cuda_status = cudaSetDevice(exchange->consumer_stream.device.id);
        if (cuda_status != cudaSuccess) {
            status = cuda_failure(result, cuda_status, "cudaSetDevice");
            cleanup_workspace(result);
            delete result;
            return status;
        }
        const auto storage = exchange->topology.quadratic_offsets.device;
        if ((storage.type != SPACEPDHCG_DEVICE_CUDA
             && storage.type != SPACEPDHCG_DEVICE_CUDA_MANAGED)
            || (storage.type == SPACEPDHCG_DEVICE_CUDA
                && storage.id != exchange->consumer_stream.device.id)
            || (storage.type == SPACEPDHCG_DEVICE_CUDA_MANAGED && storage.id != 0)) {
            set_error(result, "invalid persistent storage device");
            cleanup_workspace(result);
            delete result;
            return SPACEPDHCG_CUDA_POINTER_CONTRACT;
        }
        const ViewExpectation topology_expectations[] = {
            {static_cast<std::size_t>(structure->variables) + 1U,
             SPACEPDHCG_SCALAR_INT32, SPACEPDHCG_ACCESS_READ_ONLY, "quadratic offsets"},
            {structure->quadratic_nonzeros,
             SPACEPDHCG_SCALAR_INT32, SPACEPDHCG_ACCESS_READ_ONLY, "quadratic indices"},
            {static_cast<std::size_t>(structure->variables) + 1U,
             SPACEPDHCG_SCALAR_INT32, SPACEPDHCG_ACCESS_READ_ONLY, "scalar offsets"},
            {structure->scalar_nonzeros,
             SPACEPDHCG_SCALAR_INT32, SPACEPDHCG_ACCESS_READ_ONLY, "scalar indices"},
            {structure->affine_rows > 0
                 ? static_cast<std::size_t>(structure->variables) + 1U
                 : 0U,
             SPACEPDHCG_SCALAR_INT32, SPACEPDHCG_ACCESS_READ_ONLY, "affine offsets"},
            {structure->affine_nonzeros,
             SPACEPDHCG_SCALAR_INT32, SPACEPDHCG_ACCESS_READ_ONLY, "affine indices"},
        };
        const auto input_topology_views = topology_views(*exchange);
        for (std::size_t index = 0; index < input_topology_views.size(); ++index) {
            std::string error;
            status = validate_view(
                input_topology_views[index],
                storage,
                topology_expectations[index],
                exchange->consumer_stream.device.id,
                error
            );
            if (status != SPACEPDHCG_CUDA_SUCCESS) {
                result->error = std::move(error);
                cleanup_workspace(result);
                delete result;
                return status;
            }
        }
        status = validate_numeric_views(result, exchange->numeric, storage);
        if (status != SPACEPDHCG_CUDA_SUCCESS) {
            cleanup_workspace(result);
            delete result;
            return status;
        }
        const ViewExpectation iterate_expectations[] = {
            {static_cast<std::size_t>(structure->variables),
             SPACEPDHCG_SCALAR_FLOAT64, SPACEPDHCG_ACCESS_READ_WRITE, "primal iterate"},
            {static_cast<std::size_t>(structure->scalar_rows + structure->affine_rows),
             SPACEPDHCG_SCALAR_FLOAT64, SPACEPDHCG_ACCESS_READ_WRITE, "dual iterate"},
        };
        const spacepdhcg_accelerator_buffer_view iterate_views[] = {
            exchange->iterates.primal,
            exchange->iterates.dual,
        };
        for (std::size_t index = 0; index < 2U; ++index) {
            std::string error;
            status = validate_view(
                iterate_views[index],
                storage,
                iterate_expectations[index],
                exchange->consumer_stream.device.id,
                error
            );
            if (status != SPACEPDHCG_CUDA_SUCCESS) {
                result->error = std::move(error);
                cleanup_workspace(result);
                delete result;
                return status;
            }
        }
        if (options->debug_validate_aliases != 0) {
            const auto write_views = numeric_views(exchange->numeric);
            std::vector<spacepdhcg_accelerator_buffer_view> all_writes = write_views;
            all_writes.push_back(exchange->iterates.primal);
            all_writes.push_back(exchange->iterates.dual);
            for (std::size_t left = 0; left < all_writes.size(); ++left) {
                for (std::size_t right = left + 1U; right < all_writes.size(); ++right) {
                    if (intervals_overlap(
                            address_interval(all_writes[left]),
                            address_interval(all_writes[right])
                        )) {
                        set_error(result, "writable accelerator views overlap");
                        cleanup_workspace(result);
                        delete result;
                        return SPACEPDHCG_CUDA_POINTER_CONTRACT;
                    }
                }
                for (const auto& topology_view : input_topology_views) {
                    if (intervals_overlap(
                            address_interval(all_writes[left]),
                            address_interval(topology_view)
                        )) {
                        set_error(result, "topology and writable accelerator views overlap");
                        cleanup_workspace(result);
                        delete result;
                        return SPACEPDHCG_CUDA_POINTER_CONTRACT;
                    }
                }
            }
        }

        if (options->retain_external != nullptr) {
            options->retain_external(options->external_lifetime_context);
            result->external_retained = true;
        }

        const int variables = structure->variables;
        const int scalar_rows = structure->scalar_rows;
        const int affine_rows = structure->affine_rows;
        const int duals = scalar_rows + affine_rows;
        const auto alloc = [&](void** pointer, std::size_t bytes, AllocationCategory category) {
            return allocate_device(result, pointer, bytes, category);
        };
#define SPACEPDHCG_ALLOC(PTR, COUNT, TYPE, CATEGORY) \
        status = alloc(reinterpret_cast<void**>(&(PTR)), \
                       checked_bytes(static_cast<std::size_t>(COUNT), sizeof(TYPE)), CATEGORY); \
        if (status != SPACEPDHCG_CUDA_SUCCESS) { \
            cleanup_workspace(result); \
            delete result; \
            return status; \
        }
        SPACEPDHCG_ALLOC(result->topology.q_offsets, variables + 1, int, AllocationCategory::topology)
        SPACEPDHCG_ALLOC(result->topology.q_indices, structure->quadratic_nonzeros, int, AllocationCategory::topology)
        SPACEPDHCG_ALLOC(result->topology.a_offsets, variables + 1, int, AllocationCategory::topology)
        SPACEPDHCG_ALLOC(result->topology.a_indices, structure->scalar_nonzeros, int, AllocationCategory::topology)
        SPACEPDHCG_ALLOC(
            result->topology.f_offsets,
            affine_rows > 0 ? variables + 1 : 0,
            int,
            AllocationCategory::topology
        )
        SPACEPDHCG_ALLOC(result->topology.f_indices, structure->affine_nonzeros, int, AllocationCategory::topology)
        SPACEPDHCG_ALLOC(result->numeric.q, structure->quadratic_nonzeros, double, AllocationCategory::numeric)
        SPACEPDHCG_ALLOC(result->numeric.a, structure->scalar_nonzeros, double, AllocationCategory::numeric)
        SPACEPDHCG_ALLOC(result->numeric.f, structure->affine_nonzeros, double, AllocationCategory::numeric)
        SPACEPDHCG_ALLOC(result->numeric.c, variables, double, AllocationCategory::numeric)
        SPACEPDHCG_ALLOC(result->numeric.scalar_lower, scalar_rows, double, AllocationCategory::numeric)
        SPACEPDHCG_ALLOC(result->numeric.scalar_upper, scalar_rows, double, AllocationCategory::numeric)
        SPACEPDHCG_ALLOC(result->numeric.affine_offset, affine_rows, double, AllocationCategory::numeric)
        SPACEPDHCG_ALLOC(result->numeric.variable_lower, variables, double, AllocationCategory::numeric)
        SPACEPDHCG_ALLOC(result->numeric.variable_upper, variables, double, AllocationCategory::numeric)
        SPACEPDHCG_ALLOC(result->solver.primal, variables, double, AllocationCategory::iterate)
        SPACEPDHCG_ALLOC(result->solver.dual, duals, double, AllocationCategory::iterate)
        SPACEPDHCG_ALLOC(result->solver.previous_primal, variables, double, AllocationCategory::iterate)
        SPACEPDHCG_ALLOC(result->solver.extrapolated_primal, variables, double, AllocationCategory::iterate)
        SPACEPDHCG_ALLOC(result->solver.primal_product, scalar_rows, double, AllocationCategory::residual)
        SPACEPDHCG_ALLOC(result->solver.affine_product, affine_rows, double, AllocationCategory::residual)
        SPACEPDHCG_ALLOC(result->solver.gradient, variables, double, AllocationCategory::residual)
        SPACEPDHCG_ALLOC(result->solver.cone_scratch, affine_rows, double, AllocationCategory::cone)
        SPACEPDHCG_ALLOC(result->solver.average_primal, variables, double, AllocationCategory::iterate)
        SPACEPDHCG_ALLOC(result->solver.average_dual, duals + variables, double, AllocationCategory::iterate)
        SPACEPDHCG_ALLOC(result->solver.recovery_direction_dual, duals + variables, double, AllocationCategory::iterate)
        SPACEPDHCG_ALLOC(result->solver.recovery_coefficients, duals + variables, double, AllocationCategory::iterate)
        SPACEPDHCG_ALLOC(result->solver.recovery_row_values, duals, double, AllocationCategory::residual)
        SPACEPDHCG_ALLOC(result->solver.recovery_scalars, 4, double, AllocationCategory::residual)
        SPACEPDHCG_ALLOC(result->solver.recovery_backup_primal, variables, double, AllocationCategory::iterate)
        SPACEPDHCG_ALLOC(result->solver.recovery_backup_dual, duals, double, AllocationCategory::iterate)
        SPACEPDHCG_ALLOC(result->solver.scaling, variables + duals, double, AllocationCategory::scaling)
        SPACEPDHCG_ALLOC(result->affine_cones, structure->affine_cone_count, DeviceCone, AllocationCategory::cone)
        SPACEPDHCG_ALLOC(result->variable_cones, structure->variable_cone_count, DeviceCone, AllocationCategory::cone)
        SPACEPDHCG_ALLOC(result->device_problem, 1, DeviceProblem, AllocationCategory::descriptor_scratch)
        SPACEPDHCG_ALLOC(result->control, 1, DeviceControl, AllocationCategory::scaling)
        SPACEPDHCG_ALLOC(result->report, 1, DeviceReport, AllocationCategory::diagnostics)
#undef SPACEPDHCG_ALLOC
        cuda_status = result->ledger.allocate_pinned(
            reinterpret_cast<void**>(&result->host_report),
            sizeof(DeviceReport),
            AllocationCategory::diagnostics,
            1U
        );
        if (cuda_status != cudaSuccess) {
            status = cuda_failure(result, cuda_status, "cudaHostAlloc");
            cleanup_workspace(result);
            delete result;
            return status;
        }
        cuda_status = result->ledger.allocate_mapped(
            reinterpret_cast<void**>(&result->host_cancellation),
            reinterpret_cast<void**>(&result->solver.cancellation),
            sizeof(int),
            AllocationCategory::diagnostics,
            1U
        );
        if (cuda_status != cudaSuccess) {
            status = cuda_failure(result, cuda_status, "cudaHostAllocMapped");
            cleanup_workspace(result);
            delete result;
            return status;
        }
        std::memset(result->host_report, 0, sizeof(DeviceReport));
        *result->host_cancellation = 0;

        const cudaStream_t stream = native_stream(exchange->consumer_stream);
        cuda_status = cudaMemsetAsync(result->report, 0, sizeof(DeviceReport), stream);
        if (cuda_status != cudaSuccess) {
            status = cuda_failure(result, cuda_status, "diagnostics initialization");
            cleanup_workspace(result);
            delete result;
            return status;
        }
        const struct TopologyCopy {
            int* destination;
            const spacepdhcg_accelerator_buffer_view* source;
        } topology_copies[] = {
            {result->topology.q_offsets, &exchange->topology.quadratic_offsets},
            {result->topology.q_indices, &exchange->topology.quadratic_indices},
            {result->topology.a_offsets, &exchange->topology.scalar_offsets},
            {result->topology.a_indices, &exchange->topology.scalar_indices},
            {result->topology.f_offsets, &exchange->topology.affine_offsets},
            {result->topology.f_indices, &exchange->topology.affine_indices},
        };
        for (const auto& copy : topology_copies) {
            status = copy_async(
                result,
                copy.destination,
                offset_pointer_const<int>(*copy.source),
                checked_bytes(copy.source->elements, sizeof(int)),
                cudaMemcpyDefault,
                stream,
                true
            );
            if (status != SPACEPDHCG_CUDA_SUCCESS) {
                cleanup_workspace(result);
                delete result;
                return status;
            }
        }
        status = enqueue_numeric_copy(result, exchange->numeric, stream);
        if (status != SPACEPDHCG_CUDA_SUCCESS) {
            cleanup_workspace(result);
            delete result;
            return status;
        }
        status = copy_async(
            result,
            result->solver.primal,
            offset_pointer_const<double>(exchange->iterates.primal),
            checked_bytes(exchange->iterates.primal.elements, sizeof(double)),
            cudaMemcpyDefault,
            stream,
            false
        );
        if (status == SPACEPDHCG_CUDA_SUCCESS) {
            status = copy_async(
                result,
                result->solver.dual,
                offset_pointer_const<double>(exchange->iterates.dual),
                checked_bytes(exchange->iterates.dual.elements, sizeof(double)),
                cudaMemcpyDefault,
                stream,
                false
            );
        }
        if (status != SPACEPDHCG_CUDA_SUCCESS) {
            cleanup_workspace(result);
            delete result;
            return status;
        }

        std::vector<DeviceCone> host_affine(structure->affine_cone_count);
        std::vector<DeviceCone> host_variable(structure->variable_cone_count);
        for (std::size_t index = 0; index < host_affine.size(); ++index) {
            const auto& source = structure->affine_cones[index];
            host_affine[index] = DeviceCone{
                static_cast<int>(source.kind),
                source.start,
                source.vector_dimension,
                source.power_alpha,
            };
        }
        for (std::size_t index = 0; index < host_variable.size(); ++index) {
            const auto& source = structure->variable_cones[index];
            host_variable[index] = DeviceCone{
                static_cast<int>(source.kind),
                source.start,
                source.vector_dimension,
                source.power_alpha,
            };
        }
        if (!host_affine.empty()) {
            status = copy_async(
                result,
                result->affine_cones,
                host_affine.data(),
                host_affine.size() * sizeof(DeviceCone),
                cudaMemcpyHostToDevice,
                stream,
                false
            );
        }
        if (status == SPACEPDHCG_CUDA_SUCCESS && !host_variable.empty()) {
            status = copy_async(
                result,
                result->variable_cones,
                host_variable.data(),
                host_variable.size() * sizeof(DeviceCone),
                cudaMemcpyHostToDevice,
                stream,
                false
            );
        }
        if (status != SPACEPDHCG_CUDA_SUCCESS) {
            cleanup_workspace(result);
            delete result;
            return status;
        }

        DeviceProblem host_problem{
            variables,
            scalar_rows,
            affine_rows,
            static_cast<int>(structure->quadratic_nonzeros),
            static_cast<int>(structure->scalar_nonzeros),
            static_cast<int>(structure->affine_nonzeros),
            result->topology.q_offsets,
            result->topology.q_indices,
            result->topology.a_offsets,
            result->topology.a_indices,
            result->topology.f_offsets,
            result->topology.f_indices,
            result->numeric.q,
            result->numeric.a,
            result->numeric.f,
            result->numeric.c,
            result->numeric.scalar_lower,
            result->numeric.scalar_upper,
            result->numeric.affine_offset,
            result->numeric.variable_lower,
            result->numeric.variable_upper,
            result->solver.primal,
            result->solver.dual,
            result->solver.previous_primal,
            result->solver.extrapolated_primal,
            result->solver.primal_product,
            result->solver.affine_product,
            result->solver.gradient,
            result->solver.cone_scratch,
            result->solver.average_primal,
            result->solver.average_dual,
            result->solver.recovery_direction_dual,
            result->solver.recovery_coefficients,
            result->solver.recovery_row_values,
            result->solver.recovery_scalars,
            result->solver.recovery_backup_primal,
            result->solver.recovery_backup_dual,
            result->solver.scaling,
            result->affine_cones,
            static_cast<int>(structure->affine_cone_count),
            result->variable_cones,
            static_cast<int>(structure->variable_cone_count),
        };
        DeviceControl host_control{
            1.0e-6,
            1.0e-6,
            1U,
            1U,
            static_cast<int>(options->scaling_mode),
            options->maximum_relative_matrix_change > 0.0
                ? options->maximum_relative_matrix_change
                : 0.25,
            options->maximum_relative_vector_change > 0.0
                ? options->maximum_relative_vector_change
                : 0.5,
            options->maximum_scaling_reuse_updates,
            0U,
            0.0,
            0.0,
            0.0,
            0.0,
            1,
            1,
            0U,
            0U,
            0U,
        };
        status = copy_async(
            result,
            result->device_problem,
            &host_problem,
            sizeof(host_problem),
            cudaMemcpyHostToDevice,
            stream,
            false
        );
        if (status == SPACEPDHCG_CUDA_SUCCESS) {
            status = copy_async(
                result,
                result->control,
                &host_control,
                sizeof(host_control),
                cudaMemcpyHostToDevice,
                stream,
                false
            );
        }
        if (status != SPACEPDHCG_CUDA_SUCCESS) {
            cleanup_workspace(result);
            delete result;
            return status;
        }
        const int blocks = std::max(1, (variables + duals + kThreads - 1) / kThreads);
        set_constant_kernel<<<blocks, kThreads, 0, stream>>>(
            result->solver.scaling,
            variables + duals,
            1.0
        );
        prepare_warm_start_kernel<<<blocks, kThreads, 0, stream>>>(
            result->solver.previous_primal,
            result->solver.extrapolated_primal,
            result->solver.primal,
            variables
        );
        if (cusparseCreate(&result->sparse) != CUSPARSE_STATUS_SUCCESS
            || cusparseSetStream(result->sparse, stream) != CUSPARSE_STATUS_SUCCESS) {
            set_error(result, "failed to create or bind CUDA library handles");
            cleanup_workspace(result);
            delete result;
            return SPACEPDHCG_CUDA_RUNTIME_ERROR;
        }
        status = create_sparse_descriptor(
            result,
            &result->q_descriptor,
            variables,
            variables,
            structure->quadratic_nonzeros,
            result->topology.q_offsets,
            result->topology.q_indices,
            result->numeric.q
        );
        if (status == SPACEPDHCG_CUDA_SUCCESS) {
            status = create_sparse_descriptor(
                result,
                &result->a_descriptor,
                scalar_rows,
                variables,
                structure->scalar_nonzeros,
                result->topology.a_offsets,
                result->topology.a_indices,
                result->numeric.a
            );
        }
        if (status == SPACEPDHCG_CUDA_SUCCESS) {
            status = create_sparse_descriptor(
                result,
                &result->f_descriptor,
                affine_rows,
                variables,
                structure->affine_nonzeros,
                result->topology.f_offsets,
                result->topology.f_indices,
                result->numeric.f
            );
        }
        if (status != SPACEPDHCG_CUDA_SUCCESS) {
            cleanup_workspace(result);
            delete result;
            return status;
        }
        cuda_status = cudaStreamCreateWithFlags(&result->control_stream, cudaStreamNonBlocking);
        if (cuda_status != cudaSuccess
            || result->completion.create() != cudaSuccess
            || result->update_timer.create() != cudaSuccess
            || result->solve_timer.create() != cudaSuccess
            || result->recovery_timer.create() != cudaSuccess) {
            status = cuda_failure(result, cudaGetLastError(), "CUDA stream/event creation");
            cleanup_workspace(result);
            delete result;
            return status;
        }
        status = record_completion(result, stream, LastOperation::create);
        if (status == SPACEPDHCG_CUDA_SUCCESS) {
            cuda_status = result->completion.wait();
            if (cuda_status != cudaSuccess) {
                status = cuda_failure(result, cuda_status, "workspace creation completion");
            }
        }
        if (status != SPACEPDHCG_CUDA_SUCCESS) {
            cleanup_workspace(result);
            delete result;
            return status;
        }
        result->state = SPACEPDHCG_CUDA_CREATED;
        result->warm_mode = SPACEPDHCG_CUDA_WARM_START_PRIMAL_DUAL;
        result->warm_accepted = true;
        *workspace = result;
        return SPACEPDHCG_CUDA_SUCCESS;
    } catch (const std::bad_alloc&) {
        cleanup_workspace(result);
        delete result;
        return SPACEPDHCG_CUDA_OUT_OF_MEMORY;
    } catch (const std::exception& exception) {
        set_error(result, exception.what());
        cleanup_workspace(result);
        delete result;
        return SPACEPDHCG_CUDA_INTERNAL_ERROR;
    } catch (...) {
        cleanup_workspace(result);
        delete result;
        return SPACEPDHCG_CUDA_INTERNAL_ERROR;
    }
}

extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_workspace_update_async(
    spacepdhcg_cuda_workspace* workspace,
    const std::uint64_t topology_fingerprint,
    const spacepdhcg_cqp_numeric_accelerator_views* values,
    const spacepdhcg_accelerator_stream stream
) {
    if (workspace == nullptr || values == nullptr) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    std::lock_guard lock(workspace->mutex);
    if (topology_fingerprint != workspace->structure.topology_fingerprint) {
        set_error(workspace, "numerical update topology fingerprint mismatch");
        return SPACEPDHCG_CUDA_TOPOLOGY_MISMATCH;
    }
    if (workspace->state == SPACEPDHCG_CUDA_SOLVING) {
        return SPACEPDHCG_CUDA_BUSY;
    }
    auto status = check_stream(workspace, stream);
    if (status != SPACEPDHCG_CUDA_SUCCESS) {
        return status;
    }
    const auto storage = values->quadratic.device;
    status = validate_numeric_views(workspace, *values, storage);
    if (status != SPACEPDHCG_CUDA_SUCCESS) {
        return status;
    }
    const auto allocations_before = workspace->ledger.allocation_count();
    const auto topology_allocations_before = workspace->ledger.topology_allocation_count();
    const auto topology_copies_before = workspace->topology_index_copy_count;
    const auto cuda_stream = native_stream(stream);
    auto cuda_status = workspace->update_timer.begin(cuda_stream);
    if (cuda_status != cudaSuccess) {
        return cuda_failure(workspace, cuda_status, "update timing start");
    }
    coefficient_change_kernel<<<1, 1, 0, cuda_stream>>>(
        workspace->control,
        source_pointers(*values),
        workspace->device_problem
    );
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        return cuda_failure(workspace, cuda_status, "coefficient change kernel");
    }
    status = enqueue_numeric_copy(workspace, *values, cuda_stream);
    if (status != SPACEPDHCG_CUDA_SUCCESS) {
        return status;
    }
    cuda_status = workspace->update_timer.end(cuda_stream);
    if (cuda_status != cudaSuccess) {
        return cuda_failure(workspace, cuda_status, "update timing stop");
    }
    ++workspace->update_epoch;
    ++workspace->graph_epoch;
    workspace->allocation_delta_last_update =
        workspace->ledger.allocation_count() - allocations_before;
    workspace->topology_allocation_delta_last_update =
        workspace->ledger.topology_allocation_count() - topology_allocations_before;
    workspace->topology_index_copy_delta_last_update =
        workspace->topology_index_copy_count - topology_copies_before;
    workspace->state = SPACEPDHCG_CUDA_VALUES_UPDATED;
    workspace->termination = SPACEPDHCG_CUDA_TERMINATION_UNSPECIFIED;
    return record_completion(workspace, cuda_stream, LastOperation::update);
}

extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_workspace_warm_start_async(
    spacepdhcg_cuda_workspace* workspace,
    const spacepdhcg_cuda_warm_start_mode mode,
    const spacepdhcg_cqp_iterate_accelerator_views* iterates,
    const spacepdhcg_accelerator_stream stream
) {
    if (workspace == nullptr) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    std::lock_guard lock(workspace->mutex);
    if (workspace->state == SPACEPDHCG_CUDA_SOLVING) {
        return SPACEPDHCG_CUDA_BUSY;
    }
    auto status = check_stream(workspace, stream);
    if (status != SPACEPDHCG_CUDA_SUCCESS) {
        return status;
    }
    if (mode < SPACEPDHCG_CUDA_WARM_START_NONE
        || mode > SPACEPDHCG_CUDA_WARM_START_FULL_RETAINED) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    const int variables = workspace->structure.variables;
    const int duals = workspace->structure.scalar_rows + workspace->structure.affine_rows;
    const auto cuda_stream = native_stream(stream);
    if (mode == SPACEPDHCG_CUDA_WARM_START_FULL_RETAINED) {
        if (workspace->state != SPACEPDHCG_CUDA_SOLVED
            && workspace->state != SPACEPDHCG_CUDA_VALUES_UPDATED) {
            set_error(workspace, "full retained warm start requires prior retained solver state");
            workspace->warm_accepted = false;
            return SPACEPDHCG_CUDA_INVALID_STATE;
        }
    } else if (mode == SPACEPDHCG_CUDA_WARM_START_NONE) {
        auto cuda_status =
            cudaMemsetAsync(workspace->solver.primal, 0, variables * sizeof(double), cuda_stream);
        if (cuda_status == cudaSuccess) {
            cuda_status =
                cudaMemsetAsync(workspace->solver.dual, 0, duals * sizeof(double), cuda_stream);
        }
        if (cuda_status != cudaSuccess) {
            return cuda_failure(workspace, cuda_status, "warm-start reset");
        }
    } else {
        if (iterates == nullptr) {
            return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
        }
        const auto storage = iterates->primal.device;
        std::string error;
        status = validate_view(
            iterates->primal,
            storage,
            ViewExpectation{
                static_cast<std::size_t>(variables),
                SPACEPDHCG_SCALAR_FLOAT64,
                SPACEPDHCG_ACCESS_READ_WRITE,
                "warm primal",
            },
            workspace->consumer_stream.device.id,
            error
        );
        if (status != SPACEPDHCG_CUDA_SUCCESS) {
            workspace->error = std::move(error);
            workspace->warm_accepted = false;
            return status;
        }
        status = copy_async(
            workspace,
            workspace->solver.primal,
            offset_pointer_const<double>(iterates->primal),
            variables * sizeof(double),
            cudaMemcpyDefault,
            cuda_stream,
            false
        );
        if (status != SPACEPDHCG_CUDA_SUCCESS) {
            return status;
        }
        if (mode == SPACEPDHCG_CUDA_WARM_START_PRIMAL_DUAL) {
            status = validate_view(
                iterates->dual,
                storage,
                ViewExpectation{
                    static_cast<std::size_t>(duals),
                    SPACEPDHCG_SCALAR_FLOAT64,
                    SPACEPDHCG_ACCESS_READ_WRITE,
                    "warm dual",
                },
                workspace->consumer_stream.device.id,
                error
            );
            if (status != SPACEPDHCG_CUDA_SUCCESS) {
                workspace->error = std::move(error);
                workspace->warm_accepted = false;
                return status;
            }
            status = copy_async(
                workspace,
                workspace->solver.dual,
                offset_pointer_const<double>(iterates->dual),
                duals * sizeof(double),
                cudaMemcpyDefault,
                cuda_stream,
                false
            );
            if (status != SPACEPDHCG_CUDA_SUCCESS) {
                return status;
            }
        } else {
            const auto cuda_status =
                cudaMemsetAsync(workspace->solver.dual, 0, duals * sizeof(double), cuda_stream);
            if (cuda_status != cudaSuccess) {
                return cuda_failure(workspace, cuda_status, "dual warm-start reset");
            }
        }
    }
    const int blocks = std::max(1, (variables + kThreads - 1) / kThreads);
    prepare_warm_start_kernel<<<blocks, kThreads, 0, cuda_stream>>>(
        workspace->solver.previous_primal,
        workspace->solver.extrapolated_primal,
        workspace->solver.primal,
        variables
    );
    const auto cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        return cuda_failure(workspace, cuda_status, "warm-start preparation kernel");
    }
    workspace->warm_mode = mode;
    workspace->warm_accepted = true;
    workspace->state = SPACEPDHCG_CUDA_WARM_STARTED;
    return record_completion(workspace, cuda_stream, LastOperation::warm_start);
}

extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_workspace_solve_async(
    spacepdhcg_cuda_workspace* workspace,
    const spacepdhcg_cuda_solve_options* options,
    const spacepdhcg_accelerator_stream stream
) {
    if (workspace == nullptr || options == nullptr
        || options->abi_version != SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION
        || !std::isfinite(options->optimality_tolerance)
        || !std::isfinite(options->feasibility_tolerance)
        || options->optimality_tolerance <= 0.0
        || options->feasibility_tolerance <= 0.0
        || options->iteration_limit == 0U) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    std::lock_guard lock(workspace->mutex);
    if (workspace->state == SPACEPDHCG_CUDA_SOLVING) {
        return SPACEPDHCG_CUDA_BUSY;
    }
    auto status = check_stream(workspace, stream);
    if (status != SPACEPDHCG_CUDA_SUCCESS) {
        return status;
    }
    const auto cuda_stream = native_stream(stream);
    set_solve_options_kernel<<<1, 1, 0, cuda_stream>>>(
        workspace->control,
        options->optimality_tolerance,
        options->feasibility_tolerance,
        options->iteration_limit,
        options->residual_check_frequency
    );
    auto cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        return cuda_failure(workspace, cuda_status, "solve options kernel");
    }
    std::atomic_ref<int>(*workspace->host_cancellation).store(
        0,
        std::memory_order_release
    );
    cuda_status = workspace->solve_timer.begin(cuda_stream);
    if (cuda_status != cudaSuccess) {
        return cuda_failure(workspace, cuda_status, "solve timing start");
    }
    initialise_control_kernel<<<1, 1, 0, cuda_stream>>>(
        workspace->control,
        workspace->device_problem
    );
    solve_kernel<<<1, kThreads, 0, cuda_stream>>>(
        workspace->device_problem,
        workspace->control,
        workspace->report,
        workspace->solver.cancellation
    );
    cuda_status = workspace->recovery_timer.begin(cuda_stream);
    if (cuda_status != cudaSuccess) {
        return cuda_failure(workspace, cuda_status, "recovery timing start");
    }
    recovery_kernel<<<1, kThreads, 0, cuda_stream>>>(
        workspace->device_problem,
        workspace->control,
        workspace->report,
        workspace->solver.cancellation
    );
    cuda_status = workspace->recovery_timer.end(cuda_stream);
    if (cuda_status != cudaSuccess) {
        return cuda_failure(workspace, cuda_status, "recovery timing stop");
    }
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        return cuda_failure(workspace, cuda_status, "persistent solve kernel launch");
    }
    cuda_status = workspace->solve_timer.end(cuda_stream);
    if (cuda_status != cudaSuccess) {
        return cuda_failure(workspace, cuda_status, "solve timing stop");
    }
    status = copy_async(
        workspace,
        workspace->host_report,
        workspace->report,
        sizeof(DeviceReport),
        cudaMemcpyDeviceToHost,
        cuda_stream,
        false
    );
    if (status == SPACEPDHCG_CUDA_SUCCESS) {
        status = copy_async(
            workspace,
            offset_pointer<double>(workspace->external_iterates.primal),
            workspace->solver.primal,
            workspace->structure.variables * sizeof(double),
            cudaMemcpyDefault,
            cuda_stream,
            false
        );
    }
    if (status == SPACEPDHCG_CUDA_SUCCESS) {
        status = copy_async(
            workspace,
            offset_pointer<double>(workspace->external_iterates.dual),
            workspace->solver.dual,
            (workspace->structure.scalar_rows + workspace->structure.affine_rows)
                * sizeof(double),
            cudaMemcpyDefault,
            cuda_stream,
            false
        );
    }
    if (status != SPACEPDHCG_CUDA_SUCCESS) {
        return status;
    }
    ++workspace->solve_epoch;
    workspace->state = SPACEPDHCG_CUDA_SOLVING;
    workspace->termination = SPACEPDHCG_CUDA_TERMINATION_UNSPECIFIED;
    return record_completion(workspace, cuda_stream, LastOperation::solve);
}

extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_workspace_query(
    spacepdhcg_cuda_workspace* workspace,
    int32_t* complete
) {
    if (workspace == nullptr || complete == nullptr) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    std::lock_guard lock(workspace->mutex);
    const auto status = finalize_if_complete(workspace);
    if (status == SPACEPDHCG_CUDA_BUSY) {
        *complete = 0;
        return SPACEPDHCG_CUDA_SUCCESS;
    }
    *complete = status == SPACEPDHCG_CUDA_SUCCESS ? 1 : 0;
    return status;
}

extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_workspace_wait(
    spacepdhcg_cuda_workspace* workspace
) {
    if (workspace == nullptr) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    std::lock_guard lock(workspace->mutex);
    const auto status = workspace->completion.wait();
    if (status != cudaSuccess) {
        return cuda_failure(workspace, status, "cudaEventSynchronize");
    }
    if (workspace->last_operation == LastOperation::solve
        && workspace->state == SPACEPDHCG_CUDA_SOLVING) {
        finish_solve(workspace);
    } else if (workspace->last_operation == LastOperation::update) {
        workspace->update_seconds = workspace->update_timer.elapsed_seconds();
    }
    workspace->pending_dlpack_borrows.clear();
    return SPACEPDHCG_CUDA_SUCCESS;
}

extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_workspace_diagnostics(
    spacepdhcg_cuda_workspace* workspace,
    spacepdhcg_cuda_diagnostics* diagnostics
) {
    if (workspace == nullptr || diagnostics == nullptr) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    std::lock_guard lock(workspace->mutex);
    if (workspace->state == SPACEPDHCG_CUDA_SOLVING) {
        return SPACEPDHCG_CUDA_BUSY;
    }
    std::memset(diagnostics, 0, sizeof(*diagnostics));
    diagnostics->abi_version = SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION;
    diagnostics->state = workspace->state;
    diagnostics->termination = workspace->termination;
    if (workspace->solve_epoch > 0U) {
        diagnostics->iterations = workspace->host_report->iterations;
        diagnostics->objective = workspace->host_report->objective;
        diagnostics->scalar_primal_violation_inf =
            workspace->host_report->scalar_primal_violation_inf;
        diagnostics->box_violation_inf = workspace->host_report->box_violation_inf;
        diagnostics->affine_cone_distance_inf =
            workspace->host_report->affine_cone_distance_inf;
        diagnostics->stationarity_inf = workspace->host_report->stationarity_inf;
        diagnostics->natural_residual_inf = workspace->host_report->natural_residual_inf;
        diagnostics->complementarity_inf = workspace->host_report->complementarity_inf;
        diagnostics->relative_primal_residual =
            workspace->host_report->relative_primal_residual;
        diagnostics->relative_dual_residual =
            workspace->host_report->relative_dual_residual;
        diagnostics->coefficient_change_max =
            workspace->host_report->coefficient_change_max;
        diagnostics->coefficient_change_norm =
            workspace->host_report->coefficient_change_norm;
        diagnostics->scaling_min = workspace->host_report->scaling_min;
        diagnostics->scaling_max = workspace->host_report->scaling_max;
        diagnostics->scaling_reuse_count =
            workspace->host_report->scaling_reuse_count;
        diagnostics->scaling_refreshed = workspace->host_report->scaling_refreshed;
        diagnostics->recovery_count = workspace->host_report->recovery_count;
        diagnostics->recovery_rejected_count =
            workspace->host_report->recovery_rejected_count;
        diagnostics->recovery_iterations =
            workspace->host_report->recovery_iterations;
        diagnostics->recovery_attempt_count =
            workspace->host_report->recovery_attempt_count;
        diagnostics->recovery_trigger_reason =
            static_cast<spacepdhcg_cuda_recovery_reason>(
                workspace->host_report->recovery_trigger_reason
            );
        diagnostics->recovery_outcome_reason =
            static_cast<spacepdhcg_cuda_recovery_reason>(
                workspace->host_report->recovery_outcome_reason
            );
        diagnostics->recovery_seconds = workspace->recovery_seconds;
        diagnostics->recovery_initial_residual =
            workspace->host_report->recovery_initial_residual;
        diagnostics->recovery_final_residual =
            workspace->host_report->recovery_final_residual;
        diagnostics->recovery_final_primal_residual =
            workspace->host_report->recovery_final_primal_residual;
        diagnostics->recovery_final_stationarity =
            workspace->host_report->recovery_final_stationarity;
        diagnostics->recovery_final_complementarity =
            workspace->host_report->recovery_final_complementarity;
        diagnostics->recovery_stationarity_index =
            workspace->host_report->recovery_stationarity_index;
        diagnostics->recovery_stationarity_value =
            workspace->host_report->recovery_stationarity_value;
    }
    diagnostics->update_seconds = workspace->update_seconds;
    diagnostics->scaling_seconds = workspace->scaling_seconds;
    diagnostics->solve_seconds = workspace->solve_seconds;
    diagnostics->allocation_count = workspace->ledger.allocation_count();
    diagnostics->free_count = workspace->ledger.free_count();
    diagnostics->active_allocation_count = workspace->ledger.active_count();
    diagnostics->active_bytes = workspace->ledger.active_bytes();
    diagnostics->peak_active_bytes = workspace->ledger.peak_active_bytes();
    diagnostics->topology_allocation_count =
        workspace->ledger.topology_allocation_count();
    diagnostics->topology_index_copy_count = workspace->topology_index_copy_count;
    diagnostics->total_copy_count = workspace->total_copy_count;
    diagnostics->total_copy_bytes = workspace->total_copy_bytes;
    diagnostics->update_epoch = workspace->update_epoch;
    diagnostics->solve_epoch = workspace->solve_epoch;
    diagnostics->scaling_epoch = workspace->scaling_epoch;
    diagnostics->graph_epoch = workspace->graph_epoch;
    diagnostics->allocation_delta_last_update = workspace->allocation_delta_last_update;
    diagnostics->topology_allocation_delta_last_update =
        workspace->topology_allocation_delta_last_update;
    diagnostics->topology_index_copy_delta_last_update =
        workspace->topology_index_copy_delta_last_update;
    diagnostics->warm_start_mode = workspace->warm_mode;
    diagnostics->warm_start_accepted = workspace->warm_accepted ? 1 : 0;
    diagnostics->used_declared_stream = 1;
    diagnostics->hidden_cpu_fallback = 0;
    return SPACEPDHCG_CUDA_SUCCESS;
}

extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_workspace_residuals_async(
    spacepdhcg_cuda_workspace* workspace,
    const spacepdhcg_accelerator_stream stream
) {
    if (workspace == nullptr) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    std::lock_guard lock(workspace->mutex);
    if (workspace->state == SPACEPDHCG_CUDA_SOLVING) {
        return SPACEPDHCG_CUDA_BUSY;
    }
    auto status = check_stream(workspace, stream);
    if (status != SPACEPDHCG_CUDA_SUCCESS) {
        return status;
    }
    const auto cuda_stream = native_stream(stream);
    residual_kernel<<<1, kThreads, 0, cuda_stream>>>(
        workspace->device_problem,
        workspace->control,
        workspace->report
    );
    const auto cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        return cuda_failure(workspace, cuda_status, "independent residual kernel");
    }
    status = copy_async(
        workspace,
        workspace->host_report,
        workspace->report,
        sizeof(DeviceReport),
        cudaMemcpyDeviceToHost,
        cuda_stream,
        false
    );
    if (status != SPACEPDHCG_CUDA_SUCCESS) {
        return status;
    }
    return record_completion(workspace, cuda_stream, LastOperation::residual);
}

extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_workspace_pointer_snapshot(
    spacepdhcg_cuda_workspace* workspace,
    spacepdhcg_cuda_pointer_snapshot* snapshot
) {
    if (workspace == nullptr || snapshot == nullptr) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    std::lock_guard lock(workspace->mutex);
    *snapshot = spacepdhcg_cuda_pointer_snapshot{
        reinterpret_cast<uintptr_t>(workspace->topology.q_offsets),
        reinterpret_cast<uintptr_t>(workspace->topology.q_indices),
        reinterpret_cast<uintptr_t>(workspace->topology.a_offsets),
        reinterpret_cast<uintptr_t>(workspace->topology.a_indices),
        reinterpret_cast<uintptr_t>(workspace->topology.f_offsets),
        reinterpret_cast<uintptr_t>(workspace->topology.f_indices),
        reinterpret_cast<uintptr_t>(workspace->numeric.q),
        reinterpret_cast<uintptr_t>(workspace->numeric.a),
        reinterpret_cast<uintptr_t>(workspace->numeric.f),
        reinterpret_cast<uintptr_t>(workspace->solver.primal),
        reinterpret_cast<uintptr_t>(workspace->solver.dual),
        reinterpret_cast<uintptr_t>(workspace->solver.scaling),
    };
    return SPACEPDHCG_CUDA_SUCCESS;
}

extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_workspace_reset_async(
    spacepdhcg_cuda_workspace* workspace,
    const uint32_t reset_flags,
    const spacepdhcg_accelerator_stream stream
) {
    if (workspace == nullptr || reset_flags == 0U
        || (reset_flags & ~static_cast<uint32_t>(SPACEPDHCG_CUDA_RESET_FULL)) != 0U) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    std::lock_guard lock(workspace->mutex);
    if (workspace->state == SPACEPDHCG_CUDA_SOLVING) {
        return SPACEPDHCG_CUDA_BUSY;
    }
    auto status = check_stream(workspace, stream);
    if (status != SPACEPDHCG_CUDA_SUCCESS) {
        return status;
    }
    const auto cuda_stream = native_stream(stream);
    if ((reset_flags & SPACEPDHCG_CUDA_RESET_ITERATES) != 0U) {
        auto cuda_status = cudaMemsetAsync(
            workspace->solver.primal,
            0,
            workspace->structure.variables * sizeof(double),
            cuda_stream
        );
        if (cuda_status == cudaSuccess) {
            cuda_status = cudaMemsetAsync(
                workspace->solver.dual,
                0,
                (workspace->structure.scalar_rows + workspace->structure.affine_rows)
                    * sizeof(double),
                cuda_stream
            );
        }
        if (cuda_status != cudaSuccess) {
            return cuda_failure(workspace, cuda_status, "iterate reset");
        }
        const int blocks =
            std::max(1, (workspace->structure.variables + kThreads - 1) / kThreads);
        prepare_warm_start_kernel<<<blocks, kThreads, 0, cuda_stream>>>(
            workspace->solver.previous_primal,
            workspace->solver.extrapolated_primal,
            workspace->solver.primal,
            workspace->structure.variables
        );
        workspace->warm_mode = SPACEPDHCG_CUDA_WARM_START_NONE;
        workspace->warm_accepted = true;
    }
    if ((reset_flags & SPACEPDHCG_CUDA_RESET_SCALING) != 0U) {
        DeviceControl host_control{};
        auto cuda_status = cudaMemcpy(
            &host_control,
            workspace->control,
            sizeof(host_control),
            cudaMemcpyDeviceToHost
        );
        if (cuda_status != cudaSuccess) {
            return cuda_failure(workspace, cuda_status, "scaling reset read");
        }
        host_control.force_scaling_refresh = 1;
        host_control.scaling_reuse_count = 0U;
        cuda_status = cudaMemcpyAsync(
            workspace->control,
            &host_control,
            sizeof(host_control),
            cudaMemcpyHostToDevice,
            cuda_stream
        );
        if (cuda_status != cudaSuccess) {
            return cuda_failure(workspace, cuda_status, "scaling reset write");
        }
    }
    workspace->state = SPACEPDHCG_CUDA_CREATED;
    workspace->termination = SPACEPDHCG_CUDA_TERMINATION_UNSPECIFIED;
    return record_completion(workspace, cuda_stream, LastOperation::reset);
}

extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_workspace_refresh_scaling_async(
    spacepdhcg_cuda_workspace* workspace,
    const spacepdhcg_accelerator_stream stream
) {
    if (workspace == nullptr) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    std::lock_guard lock(workspace->mutex);
    if (workspace->state == SPACEPDHCG_CUDA_SOLVING) {
        return SPACEPDHCG_CUDA_BUSY;
    }
    auto status = check_stream(workspace, stream);
    if (status != SPACEPDHCG_CUDA_SUCCESS) {
        return status;
    }
    DeviceControl host_control{};
    auto cuda_status =
        cudaMemcpy(&host_control, workspace->control, sizeof(host_control), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        return cuda_failure(workspace, cuda_status, "scaling refresh read");
    }
    host_control.force_scaling_refresh = 1;
    const auto cuda_stream = native_stream(stream);
    cuda_status = cudaMemcpyAsync(
        workspace->control,
        &host_control,
        sizeof(host_control),
        cudaMemcpyHostToDevice,
        cuda_stream
    );
    if (cuda_status != cudaSuccess) {
        return cuda_failure(workspace, cuda_status, "scaling refresh write");
    }
    initialise_control_kernel<<<1, 1, 0, cuda_stream>>>(
        workspace->control,
        workspace->device_problem
    );
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        return cuda_failure(workspace, cuda_status, "scaling refresh kernel");
    }
    ++workspace->scaling_epoch;
    return record_completion(workspace, cuda_stream, LastOperation::refresh);
}

extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_workspace_checkpoint_bytes(
    const spacepdhcg_cuda_workspace* workspace,
    size_t* bytes
) {
    if (workspace == nullptr || bytes == nullptr) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    const std::size_t variables = static_cast<std::size_t>(workspace->structure.variables);
    const std::size_t duals =
        static_cast<std::size_t>(
            workspace->structure.scalar_rows + workspace->structure.affine_rows
        );
    *bytes = (4U * variables + 2U * duals + 2U) * sizeof(double);
    return SPACEPDHCG_CUDA_SUCCESS;
}

extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_workspace_checkpoint_async(
    spacepdhcg_cuda_workspace* workspace,
    const spacepdhcg_accelerator_buffer_view checkpoint,
    const spacepdhcg_accelerator_stream stream
) {
    if (workspace == nullptr) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    std::lock_guard lock(workspace->mutex);
    if (workspace->state == SPACEPDHCG_CUDA_SOLVING) {
        return SPACEPDHCG_CUDA_BUSY;
    }
    auto status = check_stream(workspace, stream);
    if (status != SPACEPDHCG_CUDA_SUCCESS) {
        return status;
    }
    std::size_t bytes{0U};
    static_cast<void>(spacepdhcg_cuda_workspace_checkpoint_bytes(workspace, &bytes));
    std::string error;
    status = validate_view(
        checkpoint,
        checkpoint.device,
        ViewExpectation{
            bytes / sizeof(double),
            SPACEPDHCG_SCALAR_FLOAT64,
            SPACEPDHCG_ACCESS_READ_WRITE,
            "checkpoint",
        },
        workspace->consumer_stream.device.id,
        error
    );
    if (status != SPACEPDHCG_CUDA_SUCCESS) {
        workspace->error = std::move(error);
        return status;
    }
    auto* destination = offset_pointer<double>(checkpoint);
    const int variables = workspace->structure.variables;
    const int duals = workspace->structure.scalar_rows + workspace->structure.affine_rows;
    const auto cuda_stream = native_stream(stream);
    const struct Segment {
        const double* source;
        std::size_t elements;
    } segments[] = {
        {workspace->solver.primal, static_cast<std::size_t>(variables)},
        {workspace->solver.dual, static_cast<std::size_t>(duals)},
        {workspace->solver.previous_primal, static_cast<std::size_t>(variables)},
        {workspace->solver.extrapolated_primal, static_cast<std::size_t>(variables)},
        {workspace->solver.scaling, static_cast<std::size_t>(variables + duals)},
    };
    std::size_t offset = 0U;
    for (const auto& segment : segments) {
        status = copy_async(
            workspace,
            destination + offset,
            segment.source,
            segment.elements * sizeof(double),
            cudaMemcpyDefault,
            cuda_stream,
            false
        );
        if (status != SPACEPDHCG_CUDA_SUCCESS) {
            return status;
        }
        offset += segment.elements;
    }
    checkpoint_steps_kernel<<<1, 1, 0, cuda_stream>>>(
        destination + offset,
        workspace->control
    );
    const auto cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        return cuda_failure(workspace, cuda_status, "checkpoint scaling kernel");
    }
    return record_completion(workspace, cuda_stream, LastOperation::checkpoint);
}

extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_workspace_restore_async(
    spacepdhcg_cuda_workspace* workspace,
    const uint64_t topology_fingerprint,
    const spacepdhcg_accelerator_buffer_view checkpoint,
    const spacepdhcg_accelerator_stream stream
) {
    if (workspace == nullptr) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    std::lock_guard lock(workspace->mutex);
    if (topology_fingerprint != workspace->structure.topology_fingerprint) {
        return SPACEPDHCG_CUDA_TOPOLOGY_MISMATCH;
    }
    if (workspace->state == SPACEPDHCG_CUDA_SOLVING) {
        return SPACEPDHCG_CUDA_BUSY;
    }
    auto status = check_stream(workspace, stream);
    if (status != SPACEPDHCG_CUDA_SUCCESS) {
        return status;
    }
    std::size_t bytes{0U};
    static_cast<void>(spacepdhcg_cuda_workspace_checkpoint_bytes(workspace, &bytes));
    std::string error;
    status = validate_view(
        checkpoint,
        checkpoint.device,
        ViewExpectation{
            bytes / sizeof(double),
            SPACEPDHCG_SCALAR_FLOAT64,
            SPACEPDHCG_ACCESS_READ_WRITE,
            "checkpoint",
        },
        workspace->consumer_stream.device.id,
        error
    );
    if (status != SPACEPDHCG_CUDA_SUCCESS) {
        workspace->error = std::move(error);
        return status;
    }
    const auto* source = offset_pointer_const<double>(checkpoint);
    const int variables = workspace->structure.variables;
    const int duals = workspace->structure.scalar_rows + workspace->structure.affine_rows;
    const auto cuda_stream = native_stream(stream);
    const struct Segment {
        double* destination;
        std::size_t elements;
    } segments[] = {
        {workspace->solver.primal, static_cast<std::size_t>(variables)},
        {workspace->solver.dual, static_cast<std::size_t>(duals)},
        {workspace->solver.previous_primal, static_cast<std::size_t>(variables)},
        {workspace->solver.extrapolated_primal, static_cast<std::size_t>(variables)},
        {workspace->solver.scaling, static_cast<std::size_t>(variables + duals)},
    };
    std::size_t offset = 0U;
    for (const auto& segment : segments) {
        status = copy_async(
            workspace,
            segment.destination,
            source + offset,
            segment.elements * sizeof(double),
            cudaMemcpyDefault,
            cuda_stream,
            false
        );
        if (status != SPACEPDHCG_CUDA_SUCCESS) {
            return status;
        }
        offset += segment.elements;
    }
    restore_steps_kernel<<<1, 1, 0, cuda_stream>>>(
        workspace->control,
        source + offset
    );
    const auto cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        return cuda_failure(workspace, cuda_status, "checkpoint scaling restore kernel");
    }
    workspace->state = SPACEPDHCG_CUDA_WARM_STARTED;
    workspace->warm_mode = SPACEPDHCG_CUDA_WARM_START_FULL_RETAINED;
    workspace->warm_accepted = true;
    return record_completion(workspace, cuda_stream, LastOperation::restore);
}

extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_workspace_cancel(
    spacepdhcg_cuda_workspace* workspace
) {
    if (workspace == nullptr) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    std::lock_guard lock(workspace->mutex);
    if (workspace->state != SPACEPDHCG_CUDA_SOLVING) {
        return SPACEPDHCG_CUDA_INVALID_STATE;
    }
    std::atomic_ref<int>(*workspace->host_cancellation).store(
        1,
        std::memory_order_release
    );
    return SPACEPDHCG_CUDA_SUCCESS;
}

extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_workspace_last_error(
    const spacepdhcg_cuda_workspace* workspace,
    char* destination,
    const size_t destination_bytes
) {
    if (workspace == nullptr || destination == nullptr || destination_bytes == 0U) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    const std::size_t length =
        std::min(destination_bytes - 1U, workspace->error.size());
    std::memcpy(destination, workspace->error.data(), length);
    destination[length] = '\0';
    return SPACEPDHCG_CUDA_SUCCESS;
}

extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_workspace_create_from_dlpack(
    const spacepdhcg_cuda_structure* structure,
    const spacepdhcg_cqp_dlpack_exchange* exchange,
    const spacepdhcg_cuda_create_options* options,
    spacepdhcg_cuda_workspace** workspace
) {
    if (workspace == nullptr) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    *workspace = nullptr;
    try {
    if (structure == nullptr || exchange == nullptr || options == nullptr
        || exchange->abi_version != SPACEPDHCG_ACCELERATOR_EXCHANGE_ABI_VERSION) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    spacepdhcg_cqp_accelerator_exchange views{};
    views.abi_version = exchange->abi_version;
    views.topology_fingerprint = exchange->topology_fingerprint;
    views.consumer_stream = exchange->consumer_stream;
    std::vector<DLPackOwner> owners{};
    owners.reserve(17U);
    std::string error{};
    spacepdhcg_cuda_status status = SPACEPDHCG_CUDA_SUCCESS;
#define SPACEPDHCG_DLPACK_CREATE_VIEW(PATH) \
    { \
        const auto current_status = \
            append_dlpack_view(exchange->PATH, views.PATH, owners, error); \
        if (status == SPACEPDHCG_CUDA_SUCCESS) { \
            status = current_status; \
        } \
    }
    SPACEPDHCG_DLPACK_CREATE_VIEW(topology.quadratic_offsets)
    SPACEPDHCG_DLPACK_CREATE_VIEW(topology.quadratic_indices)
    SPACEPDHCG_DLPACK_CREATE_VIEW(topology.scalar_offsets)
    SPACEPDHCG_DLPACK_CREATE_VIEW(topology.scalar_indices)
    SPACEPDHCG_DLPACK_CREATE_VIEW(topology.affine_offsets)
    SPACEPDHCG_DLPACK_CREATE_VIEW(topology.affine_indices)
    SPACEPDHCG_DLPACK_CREATE_VIEW(numeric.quadratic)
    SPACEPDHCG_DLPACK_CREATE_VIEW(numeric.scalar_constraint)
    SPACEPDHCG_DLPACK_CREATE_VIEW(numeric.affine_cone)
    SPACEPDHCG_DLPACK_CREATE_VIEW(numeric.linear_objective)
    SPACEPDHCG_DLPACK_CREATE_VIEW(numeric.scalar_lower)
    SPACEPDHCG_DLPACK_CREATE_VIEW(numeric.scalar_upper)
    SPACEPDHCG_DLPACK_CREATE_VIEW(numeric.affine_offset)
    SPACEPDHCG_DLPACK_CREATE_VIEW(numeric.variable_lower)
    SPACEPDHCG_DLPACK_CREATE_VIEW(numeric.variable_upper)
    SPACEPDHCG_DLPACK_CREATE_VIEW(iterates.primal)
    SPACEPDHCG_DLPACK_CREATE_VIEW(iterates.dual)
#undef SPACEPDHCG_DLPACK_CREATE_VIEW
    if (status != SPACEPDHCG_CUDA_SUCCESS) {
        return status;
    }
    status = spacepdhcg_cuda_workspace_create(structure, &views, options, workspace);
    if (status == SPACEPDHCG_CUDA_SUCCESS) {
        std::lock_guard lock((*workspace)->mutex);
        (*workspace)->persistent_dlpack_borrows = std::move(owners);
    }
    return status;
    } catch (const std::bad_alloc&) {
        if (*workspace != nullptr) {
            static_cast<void>(spacepdhcg_cuda_workspace_destroy(workspace));
        }
        return SPACEPDHCG_CUDA_OUT_OF_MEMORY;
    } catch (...) {
        if (*workspace != nullptr) {
            static_cast<void>(spacepdhcg_cuda_workspace_destroy(workspace));
        }
        return SPACEPDHCG_CUDA_INTERNAL_ERROR;
    }
}

extern "C" spacepdhcg_cuda_status
spacepdhcg_cuda_workspace_update_from_dlpack_async(
    spacepdhcg_cuda_workspace* workspace,
    const uint64_t topology_fingerprint,
    const spacepdhcg_cqp_numeric_dlpack_tensors* values,
    const spacepdhcg_accelerator_stream stream
) {
    if (workspace == nullptr || values == nullptr) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    try {
    spacepdhcg_cqp_numeric_accelerator_views views{};
    std::vector<DLPackOwner> owners{};
    owners.reserve(9U);
    std::string error{};
    spacepdhcg_cuda_status status = SPACEPDHCG_CUDA_SUCCESS;
#define SPACEPDHCG_DLPACK_UPDATE_VIEW(NAME) \
    { \
        const auto current_status = \
            append_dlpack_view(values->NAME, views.NAME, owners, error); \
        if (status == SPACEPDHCG_CUDA_SUCCESS) { \
            status = current_status; \
        } \
    }
    SPACEPDHCG_DLPACK_UPDATE_VIEW(quadratic)
    SPACEPDHCG_DLPACK_UPDATE_VIEW(scalar_constraint)
    SPACEPDHCG_DLPACK_UPDATE_VIEW(affine_cone)
    SPACEPDHCG_DLPACK_UPDATE_VIEW(linear_objective)
    SPACEPDHCG_DLPACK_UPDATE_VIEW(scalar_lower)
    SPACEPDHCG_DLPACK_UPDATE_VIEW(scalar_upper)
    SPACEPDHCG_DLPACK_UPDATE_VIEW(affine_offset)
    SPACEPDHCG_DLPACK_UPDATE_VIEW(variable_lower)
    SPACEPDHCG_DLPACK_UPDATE_VIEW(variable_upper)
#undef SPACEPDHCG_DLPACK_UPDATE_VIEW
    if (status != SPACEPDHCG_CUDA_SUCCESS) {
        return status;
    }
    status = spacepdhcg_cuda_workspace_update_async(
        workspace,
        topology_fingerprint,
        &views,
        stream
    );
    if (status == SPACEPDHCG_CUDA_SUCCESS) {
        std::lock_guard lock(workspace->mutex);
        for (auto& owner : owners) {
            workspace->pending_dlpack_borrows.push_back(std::move(owner));
        }
    }
    return status;
    } catch (const std::bad_alloc&) {
        return SPACEPDHCG_CUDA_OUT_OF_MEMORY;
    } catch (...) {
        return SPACEPDHCG_CUDA_INTERNAL_ERROR;
    }
}

extern "C" spacepdhcg_cuda_status
spacepdhcg_cuda_workspace_warm_start_from_dlpack_async(
    spacepdhcg_cuda_workspace* workspace,
    const spacepdhcg_cuda_warm_start_mode mode,
    const spacepdhcg_cqp_iterate_dlpack_tensors* iterates,
    const spacepdhcg_accelerator_stream stream
) {
    if (workspace == nullptr) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    try {
    if (mode == SPACEPDHCG_CUDA_WARM_START_NONE
        || mode == SPACEPDHCG_CUDA_WARM_START_FULL_RETAINED) {
        if (iterates != nullptr) {
            return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
        }
        return spacepdhcg_cuda_workspace_warm_start_async(
            workspace,
            mode,
            nullptr,
            stream
        );
    }
    if (iterates == nullptr) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    spacepdhcg_cqp_iterate_accelerator_views views{};
    std::vector<DLPackOwner> owners{};
    owners.reserve(2U);
    std::string error{};
    auto status = append_dlpack_view(
        iterates->primal,
        views.primal,
        owners,
        error
    );
    const auto dual_status = append_dlpack_view(
        iterates->dual,
        views.dual,
        owners,
        error
    );
    if (status == SPACEPDHCG_CUDA_SUCCESS) {
        status = dual_status;
    }
    if (status != SPACEPDHCG_CUDA_SUCCESS) {
        return status;
    }
    status = spacepdhcg_cuda_workspace_warm_start_async(
        workspace,
        mode,
        &views,
        stream
    );
    if (status == SPACEPDHCG_CUDA_SUCCESS) {
        std::lock_guard lock(workspace->mutex);
        for (auto& owner : owners) {
            workspace->pending_dlpack_borrows.push_back(std::move(owner));
        }
    }
    return status;
    } catch (const std::bad_alloc&) {
        return SPACEPDHCG_CUDA_OUT_OF_MEMORY;
    } catch (...) {
        return SPACEPDHCG_CUDA_INTERNAL_ERROR;
    }
}

extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_workspace_destroy(
    spacepdhcg_cuda_workspace** workspace
) {
    if (workspace == nullptr) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    auto* target = *workspace;
    if (target == nullptr) {
        return SPACEPDHCG_CUDA_SUCCESS;
    }
    {
        std::lock_guard lock(target->mutex);
        if (target->state == SPACEPDHCG_CUDA_SOLVING) {
            std::atomic_ref<int>(*target->host_cancellation).store(
                1,
                std::memory_order_release
            );
        }
        static_cast<void>(target->completion.wait());
        target->state = SPACEPDHCG_CUDA_DESTROYED;
        cleanup_workspace(target);
    }
    delete target;
    *workspace = nullptr;
    return SPACEPDHCG_CUDA_SUCCESS;
}
