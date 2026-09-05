// SPDX-License-Identifier: Apache-2.0

// Cooperative implementation shares the legacy problem representation and numerical policy.
// Every grid barrier must be reached by every resident thread. Launch only through
// cudaLaunchCooperativeKernel with an occupancy-bounded grid.
__device__ int grid_rank() { return blockIdx.x * blockDim.x + threadIdx.x; }
__device__ int grid_stride() { return blockDim.x * gridDim.x; }
__device__ void grid_barrier() { cooperative_groups::this_grid().sync(); }

enum class GridReduction { sum, maximum, minimum };
template <GridReduction Operation>
__device__ double grid_reduce(double value, DeviceProblem* problem) {
    __shared__ double partial[kThreads];
    partial[threadIdx.x] = value;
    __syncthreads();
    for (unsigned int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (threadIdx.x < step) {
            const double other = partial[threadIdx.x + step];
            if constexpr (Operation == GridReduction::sum) partial[threadIdx.x] += other;
            if constexpr (Operation == GridReduction::maximum) partial[threadIdx.x] = fmax(partial[threadIdx.x], other);
            if constexpr (Operation == GridReduction::minimum) partial[threadIdx.x] = fmin(partial[threadIdx.x], other);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) problem->grid_partials[blockIdx.x] = partial[0];
    grid_barrier();
    if (grid_rank() == 0) {
        double result = problem->grid_partials[0];
        for (unsigned int block = 1; block < gridDim.x; ++block) {
            const double other = problem->grid_partials[block];
            if constexpr (Operation == GridReduction::sum) result += other;
            if constexpr (Operation == GridReduction::maximum) result = fmax(result, other);
            if constexpr (Operation == GridReduction::minimum) result = fmin(result, other);
        }
        problem->grid_partials[0] = result;
    }
    grid_barrier();
    const double result = problem->grid_partials[0];
    grid_barrier(); // All readers finish before another reduction reuses the buffer.
    return result;
}

__device__ bool grid_cancelled(DeviceProblem* problem, volatile int* cancellation) {
    grid_barrier();
    if (grid_rank() == 0) problem->grid_flags[1] = cancellation != nullptr && *cancellation != 0;
    grid_barrier();
    return problem->grid_flags[1] != 0;
}

__device__ void atomic_max_positive(double* address, double value) {
    // Positive IEEE-754 doubles have the same ordering as unsigned bits. Structural
    // CSC zeros can be -0.0 (for example -A in the dynamics rows): canonicalize the
    // sign first, or its sign bit would outrank every positive row norm. Match fmax's
    // treatment of NaNs as well, rather than turning a NaN payload into a maximum.
    if (!isnan(value)) {
        atomicMax(reinterpret_cast<unsigned long long*>(address), __double_as_longlong(fabs(value)));
    }
}

__device__ void grid_zero_vector(double* vector, const int count) {
    for (int index = grid_rank(); index < count; index += grid_stride()) {
        vector[index] = 0.0;
    }
}

__device__ void grid_csc_multiply(
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
    for (int column = grid_rank(); column < columns; column += grid_stride()) {
        const double x = vector[column];
        for (int position = offsets[column]; position < offsets[column + 1]; ++position) {
            atomicAdd(result + indices[position], values[position] * x);
        }
    }
}

__device__ void grid_csc_transpose_multiply(
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
    for (int column = grid_rank(); column < columns; column += grid_stride()) {
        double sum = 0.0;
        for (int position = offsets[column]; position < offsets[column + 1]; ++position) {
            sum += values[position] * vector[indices[position]];
        }
        result[column] += sum;
    }
}

__device__ void grid_project_cone_blocks(
    double* values,
    const DeviceCone* cones,
    const int cone_count
) {
    for (int cone_index = grid_rank(); cone_index < cone_count; cone_index += grid_stride()) {
        const DeviceCone cone = cones[cone_index];
        if (cone.kind == SPACEPDHCG_CUDA_CONE_SECOND_ORDER) {
            project_standard_soc(values, cone.start, cone.vector_dimension + 2);
        } else if (cone.kind == SPACEPDHCG_CUDA_CONE_ROTATED_SECOND_ORDER) {
            project_rotated_soc(values, cone.start, cone.vector_dimension);
        }
    }
}

__device__ void grid_compute_products(DeviceProblem* problem, const double* primal) {
    grid_zero_vector(problem->scalar_product, problem->scalar_rows);
    grid_zero_vector(problem->affine_product, problem->affine_rows);
    grid_zero_vector(problem->gradient, problem->variables);
    grid_barrier();
    grid_csc_multiply(
        problem->variables,
        problem->a_offsets,
        problem->a_indices,
        problem->a,
        primal,
        problem->scalar_product
    );
    grid_csc_multiply(
        problem->variables,
        problem->f_offsets,
        problem->f_indices,
        problem->f,
        primal,
        problem->affine_product
    );
    grid_csc_multiply(
        problem->variables,
        problem->q_offsets,
        problem->q_indices,
        problem->q,
        primal,
        problem->gradient
    );
    grid_barrier();
}

__device__ void grid_add_transpose_dual(DeviceProblem* problem) {
    grid_csc_transpose_multiply(
        problem->variables,
        problem->a_offsets,
        problem->a_indices,
        problem->a,
        problem->dual,
        problem->gradient
    );
    grid_csc_transpose_multiply(
        problem->variables,
        problem->f_offsets,
        problem->f_indices,
        problem->f,
        problem->dual + problem->scalar_rows,
        problem->gradient
    );
    grid_barrier();
}

__global__ void cooperative_initialise_kernel(
    DeviceControl* control,
    DeviceProblem* problem,
    volatile int* cancellation
) {
    const bool threshold_refresh =
        control->coefficient_change_max > control->matrix_change_threshold;
    const bool budget_refresh =
        control->scaling_reuse_count >= control->maximum_reuse_updates;
    const bool refresh =
        control->force_scaling_refresh != 0
        || control->scaling_mode == SPACEPDHCG_CUDA_SCALING_ALWAYS_REFRESH
        || (control->scaling_mode == SPACEPDHCG_CUDA_SCALING_REFRESH_IF_NEEDED
            && (threshold_refresh || budget_refresh));
    const bool initialise = refresh || !(control->primal_step > 0.0) || !(control->dual_step > 0.0);
    grid_barrier(); // Freeze the branch decision before the leader updates reuse metadata.
    if (initialise) {
        // The equilibration factors accumulate in recovery scratch (idle before a solve) and
        // problem->scaling is written only once the refresh completes, so a cancel between
        // passes leaves the previous scaling and steps mutually consistent.
        double* const variable_scale = problem->recovery_backup_primal;
        double* const row_scale = problem->recovery_backup_dual;
        grid_barrier();
        for (int variable = grid_rank(); variable < problem->variables; variable += grid_stride()) {
            variable_scale[variable] = 1.0;
        }
        grid_barrier();
        for (int row = grid_rank(); row < problem->scalar_rows + problem->affine_rows; row += grid_stride()) {
            row_scale[row] = 1.0;
        }
        // Ten cone-preserving Ruiz passes match the fixed-pattern upstream policy.
        // The existing product buffers are safe create-time scratch before solve.
        for (int pass = 0; pass < 10; ++pass) {
            if (grid_cancelled(problem, cancellation)) {
                if (grid_rank() == 0) {
                    control->force_scaling_refresh = 1;
                    control->scaling_refreshed = 0;
                }
                return;
            }
            grid_barrier();
            for (int variable = grid_rank(); variable < problem->variables; variable += grid_stride()) {
                problem->previous_primal[variable] = 0.0;
            }
            grid_barrier();
            for (int row = grid_rank(); row < problem->scalar_rows; row += grid_stride()) {
                problem->scalar_product[row] = 0.0;
            }
            grid_barrier();
            for (int row = grid_rank(); row < problem->affine_rows; row += grid_stride()) {
                problem->affine_product[row] = 0.0;
            }
            grid_barrier();
            for (int variable = grid_rank(); variable < problem->variables; variable += grid_stride()) {
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
                    atomic_max_positive(problem->scalar_product + row, value);
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
                    atomic_max_positive(problem->affine_product + row, value);
                }
            }
            grid_barrier();
            for (int cone_index = grid_rank();
                 cone_index < problem->variable_cone_count; cone_index += grid_stride()) {
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
            grid_barrier();
            for (int cone_index = grid_rank();
                 cone_index < problem->affine_cone_count; cone_index += grid_stride()) {
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
            grid_barrier();
            for (int variable = grid_rank(); variable < problem->variables; variable += grid_stride()) {
                const double factor = problem->previous_primal[variable] > 1.0e-12
                    ? sqrt(problem->previous_primal[variable])
                    : 1.0;
                variable_scale[variable] *= factor;
            }
            grid_barrier();
            for (int row = grid_rank(); row < problem->scalar_rows; row += grid_stride()) {
                const double factor = problem->scalar_product[row] > 1.0e-12
                    ? sqrt(problem->scalar_product[row])
                    : 1.0;
                row_scale[row] *= factor;
            }
            grid_barrier();
            for (int row = grid_rank(); row < problem->affine_rows; row += grid_stride()) {
                const double factor = problem->affine_product[row] > 1.0e-12
                    ? sqrt(problem->affine_product[row])
                    : 1.0;
                row_scale[problem->scalar_rows + row] *= factor;
            }
        }
        double bound_norm_squared = 0.0;
        grid_barrier();
        for (int row = grid_rank(); row < problem->scalar_rows; row += grid_stride()) {
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
        grid_barrier();
        for (int row = grid_rank(); row < problem->affine_rows; row += grid_stride()) {
            const double value =
                problem->affine_offset[row]
                / row_scale[problem->scalar_rows + row];
            bound_norm_squared += value * value;
        }
        double objective_norm_squared = 0.0;
        grid_barrier();
        for (int variable = grid_rank(); variable < problem->variables; variable += grid_stride()) {
            const double value = problem->c[variable] / variable_scale[variable];
            objective_norm_squared += value * value;
        }
        bound_norm_squared = grid_reduce<GridReduction::sum>(bound_norm_squared, problem);
        objective_norm_squared = grid_reduce<GridReduction::sum>(objective_norm_squared, problem);
        const double bound_scale = 1.0 / (sqrt(bound_norm_squared) + 1.0);
        const double objective_scale = 1.0 / (sqrt(objective_norm_squared) + 1.0);
        grid_barrier();
        for (int variable = grid_rank(); variable < problem->variables; variable += grid_stride()) {
            problem->gradient[variable] = 0.0;
        }
        double operator_norm_squared = 0.0;
        grid_barrier();
        for (int variable = grid_rank(); variable < problem->variables; variable += grid_stride()) {
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
                atomicAdd(problem->gradient + row, device_abs(value));
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
        grid_barrier();
        for (int variable = grid_rank(); variable < problem->variables; variable += grid_stride()) {
            q_norm = fmax(q_norm, problem->gradient[variable]);
            problem->previous_primal[variable] =
                1.0 / sqrt(static_cast<double>(problem->variables));
        }
        q_norm = grid_reduce<GridReduction::maximum>(q_norm, problem);
        operator_norm_squared = grid_reduce<GridReduction::sum>(operator_norm_squared, problem);
        double operator_norm = sqrt(operator_norm_squared);
        for (int pass = 0; pass < 20; ++pass) {
            if (grid_cancelled(problem, cancellation)) {
                if (grid_rank() == 0) {
                    control->force_scaling_refresh = 1;
                    control->scaling_refreshed = 0;
                }
                return;
            }
            grid_barrier();
            for (int row = grid_rank(); row < problem->scalar_rows; row += grid_stride()) {
                problem->scalar_product[row] = 0.0;
            }
            grid_barrier();
            for (int row = grid_rank(); row < problem->affine_rows; row += grid_stride()) {
                problem->affine_product[row] = 0.0;
            }
            grid_barrier();
            for (int variable = grid_rank(); variable < problem->variables; variable += grid_stride()) {
                const double x = problem->previous_primal[variable];
                for (int index = problem->scalar_rows > 0
                         ? problem->a_offsets[variable]
                         : 0;
                     index < (problem->scalar_rows > 0
                         ? problem->a_offsets[variable + 1]
                         : 0);
                     ++index) {
                    const int row = problem->a_indices[index];
                    atomicAdd(problem->scalar_product + row,
                        problem->a[index] * x
                        / (row_scale[row] * variable_scale[variable]));
                }
                for (int index = problem->affine_rows > 0
                         ? problem->f_offsets[variable]
                         : 0;
                     index < (problem->affine_rows > 0
                         ? problem->f_offsets[variable + 1]
                         : 0);
                     ++index) {
                    const int row = problem->f_indices[index];
                    atomicAdd(problem->affine_product + row,
                        problem->f[index] * x
                        / (
                            row_scale[problem->scalar_rows + row]
                            * variable_scale[variable]
                        ));
                }
            }
            double row_norm_squared = 0.0;
            grid_barrier();
            for (int row = grid_rank(); row < problem->scalar_rows; row += grid_stride()) {
                row_norm_squared +=
                    problem->scalar_product[row] * problem->scalar_product[row];
            }
            grid_barrier();
            for (int row = grid_rank(); row < problem->affine_rows; row += grid_stride()) {
                row_norm_squared +=
                    problem->affine_product[row] * problem->affine_product[row];
            }
            row_norm_squared = grid_reduce<GridReduction::sum>(row_norm_squared, problem);
            const double row_norm = sqrt(row_norm_squared);
            if (!(row_norm > 1.0e-12)) {
                operator_norm = 0.0;
                break;
            }
            grid_barrier();
            for (int row = grid_rank(); row < problem->scalar_rows; row += grid_stride()) {
                problem->scalar_product[row] /= row_norm;
            }
            grid_barrier();
            for (int row = grid_rank(); row < problem->affine_rows; row += grid_stride()) {
                problem->affine_product[row] /= row_norm;
            }
            grid_barrier();
            for (int variable = grid_rank(); variable < problem->variables; variable += grid_stride()) {
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
            grid_barrier();
            for (int variable = grid_rank(); variable < problem->variables; variable += grid_stride()) {
                variable_norm_squared +=
                    problem->gradient[variable] * problem->gradient[variable];
            }
            variable_norm_squared = grid_reduce<GridReduction::sum>(variable_norm_squared, problem);
            operator_norm = sqrt(variable_norm_squared);
            if (!(operator_norm > 1.0e-12)) {
                break;
            }
            grid_barrier();
            for (int variable = grid_rank(); variable < problem->variables; variable += grid_stride()) {
                problem->previous_primal[variable] =
                    problem->gradient[variable] / operator_norm;
            }
        }
        const double denominator = fmax(1.0, q_norm + operator_norm);
        if (grid_rank() == 0) {
            control->primal_step = 0.9 / denominator;
            control->dual_step = 0.9 / fmax(1.0, operator_norm);
        }
        grid_barrier();
        for (int variable = grid_rank(); variable < problem->variables; variable += grid_stride()) {
            problem->scaling[variable] =
                objective_scale
                / (
                    variable_scale[variable] * variable_scale[variable]
                    * bound_scale
                );
        }
        grid_barrier();
        for (int row = grid_rank(); row < problem->scalar_rows + problem->affine_rows; row += grid_stride()) {
            problem->scaling[problem->variables + row] =
                bound_scale
                / (
                    row_scale[row] * row_scale[row]
                    * objective_scale
                );
        }
        grid_barrier();
        if (grid_rank() == 0) {
            control->scaling_reuse_count = 0;
            control->scaling_refreshed = 1;
        }
    } else if (grid_rank() == 0) {
        ++control->scaling_reuse_count;
        control->scaling_refreshed = 0;
    }
    if (grid_rank() == 0) control->force_scaling_refresh = 0;
}

__device__ void grid_evaluate_report(
    DeviceProblem* problem,
    DeviceControl* control,
    DeviceReport* report,
    const std::uint64_t iteration
) {
    grid_compute_products(problem, problem->primal);
    grid_add_transpose_dual(problem);
    double scalar_violation = 0.0;
    double scalar_natural = 0.0;
    double scalar_scale = 1.0;
    double complementarity = 0.0;
    grid_barrier();
    for (int row = grid_rank(); row < problem->scalar_rows; row += grid_stride()) {
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
    grid_barrier();
    for (int variable = grid_rank(); variable < problem->variables; variable += grid_stride()) {
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
    grid_barrier();
    grid_project_cone_blocks(
        problem->average_primal,
        problem->variable_cones,
        problem->variable_cone_count
    );
    grid_barrier();
    for (int variable = grid_rank(); variable < problem->variables; variable += grid_stride()) {
        stationarity = fmax(
            stationarity,
            device_abs(problem->primal[variable] - problem->average_primal[variable])
        );
        problem->average_primal[variable] = problem->primal[variable];
    }
    grid_barrier();
    grid_project_cone_blocks(
        problem->average_primal,
        problem->variable_cones,
        problem->variable_cone_count
    );
    grid_barrier();
    for (int variable = grid_rank(); variable < problem->variables; variable += grid_stride()) {
        box_violation = fmax(
            box_violation,
            device_abs(problem->primal[variable] - problem->average_primal[variable])
        );
    }

    grid_barrier();
    for (int row = grid_rank(); row < problem->affine_rows; row += grid_stride()) {
        problem->cone_scratch[row] =
            problem->affine_product[row] + problem->affine_offset[row];
    }
    grid_barrier();
    grid_project_cone_blocks(
        problem->cone_scratch,
        problem->affine_cones,
        problem->affine_cone_count
    );
    double affine_distance = 0.0;
    grid_barrier();
    for (int row = grid_rank(); row < problem->affine_rows; row += grid_stride()) {
        const double original =
            problem->affine_product[row] + problem->affine_offset[row];
        affine_distance =
            fmax(affine_distance, device_abs(original - problem->cone_scratch[row]));
    }
    grid_barrier();
    for (int row = grid_rank(); row < problem->affine_rows; row += grid_stride()) {
        problem->cone_scratch[row] =
            problem->affine_product[row] + problem->affine_offset[row]
            + problem->dual[problem->scalar_rows + row];
    }
    grid_barrier();
    grid_project_cone_blocks(
        problem->cone_scratch,
        problem->affine_cones,
        problem->affine_cone_count
    );
    double affine_natural = 0.0;
    grid_barrier();
    for (int row = grid_rank(); row < problem->affine_rows; row += grid_stride()) {
        const double original =
            problem->affine_product[row] + problem->affine_offset[row];
        affine_natural = fmax(
            affine_natural,
            device_abs(original - problem->cone_scratch[row])
        );
    }
    double affine_complementarity = 0.0;
    grid_barrier();
    for (int cone_index = grid_rank();
         cone_index < problem->affine_cone_count; cone_index += grid_stride()) {
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
    scalar_violation = grid_reduce<GridReduction::maximum>(scalar_violation, problem);
    scalar_natural = grid_reduce<GridReduction::maximum>(scalar_natural, problem);
    scalar_scale = grid_reduce<GridReduction::maximum>(scalar_scale, problem);
    complementarity = grid_reduce<GridReduction::maximum>(complementarity, problem);
    box_violation = grid_reduce<GridReduction::maximum>(box_violation, problem);
    stationarity = grid_reduce<GridReduction::maximum>(stationarity, problem);
    objective = grid_reduce<GridReduction::sum>(objective, problem);
    affine_distance = grid_reduce<GridReduction::maximum>(affine_distance, problem);
    affine_natural = grid_reduce<GridReduction::maximum>(affine_natural, problem);
    affine_complementarity = grid_reduce<GridReduction::maximum>(affine_complementarity, problem);
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
    grid_barrier();
    for (int index = grid_rank(); index < scaling_count; index += grid_stride()) {
        scaling_min = fmin(scaling_min, problem->scaling[index]);
        scaling_max = fmax(scaling_max, problem->scaling[index]);
    }

    scaling_min = grid_reduce<GridReduction::minimum>(scaling_min, problem);
    scaling_max = grid_reduce<GridReduction::maximum>(scaling_max, problem);
    if (grid_rank() == 0) {
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
    grid_barrier();
}

__global__ void cooperative_solve_kernel(
    DeviceProblem* problem,
    DeviceControl* control,
    DeviceReport* report,
    volatile int* cancellation
) {
    int& should_stop = problem->grid_flags[0];
    int& cancelled = problem->grid_flags[1];
    if (grid_rank() == 0) {
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
    grid_barrier();

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
        grid_barrier(); // Finish every reader of the preceding stop decision before polling again.
        if (grid_rank() == 0 && *cancellation != 0) {
            atomicExch(&cancelled, 1);
            atomicExch(&should_stop, 1);
        }
        grid_barrier();
        if (atomicAdd(&should_stop, 0) != 0) {
            break;
        }

        grid_zero_vector(problem->scalar_product, problem->scalar_rows);
        grid_zero_vector(problem->affine_product, problem->affine_rows);
        grid_barrier();
        grid_csc_multiply(
            problem->variables,
            problem->a_offsets,
            problem->a_indices,
            problem->a,
            problem->extrapolated_primal,
            problem->scalar_product
        );
        grid_csc_multiply(
            problem->variables,
            problem->f_offsets,
            problem->f_indices,
            problem->f,
            problem->extrapolated_primal,
            problem->affine_product
        );
        grid_barrier();

        for (int row = grid_rank(); row < problem->scalar_rows; row += grid_stride()) {
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
        for (int row = grid_rank(); row < problem->affine_rows; row += grid_stride()) {
            const int dual_row = problem->scalar_rows + row;
            const double row_step =
                dual_step * problem->scaling[problem->variables + dual_row];
            const double value =
                problem->dual[dual_row] + row_step * problem->affine_product[row];
            problem->cone_scratch[row] =
                value / row_step + problem->affine_offset[row];
        }
        grid_barrier();
        grid_project_cone_blocks(
            problem->cone_scratch,
            problem->affine_cones,
            problem->affine_cone_count
        );
        grid_barrier();
        for (int row = grid_rank(); row < problem->affine_rows; row += grid_stride()) {
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
        grid_barrier();

        grid_zero_vector(problem->gradient, problem->variables);
        grid_barrier();
        grid_csc_multiply(
            problem->variables,
            problem->q_offsets,
            problem->q_indices,
            problem->q,
            problem->primal,
            problem->gradient
        );
        grid_barrier();
        grid_add_transpose_dual(problem);
        for (int variable = grid_rank(); variable < problem->variables; variable += grid_stride()) {
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
        grid_barrier();
        grid_project_cone_blocks(
            problem->primal,
            problem->variable_cones,
            problem->variable_cone_count
        );
        grid_barrier();
        for (int variable = grid_rank(); variable < problem->variables; variable += grid_stride()) {
            problem->extrapolated_primal[variable] =
                2.0 * problem->primal[variable] - problem->previous_primal[variable];
        }
        grid_barrier();

        if (iteration == 1U || iteration % check_frequency == 0U
            || iteration == pdhg_limit) {
            grid_evaluate_report(problem, control, report, iteration);
            if (grid_rank() == 0) {
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
            grid_barrier();
            if (atomicAdd(&should_stop, 0) != 0) {
                break;
            }
        }
    }
    if (grid_rank() == 0 && atomicAdd(&cancelled, 0) != 0) {
        report->termination = SPACEPDHCG_CUDA_TERMINATION_CANCELLED;
    }
}
