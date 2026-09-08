// SPDX-License-Identifier: Apache-2.0
// Isolated diagnostic: full-layout storage, exact masked LP + L1 proximal map.
#include "spacepdhcg/cuda/l1_epigraph_arithmetic.hpp"
namespace lm=spacepdhcg::cuda::l1;
struct L1State {
    spacepdhcg_cuda_l1_options options;
    spacepdhcg_cuda_l1_diagnostics result;
    spacepdhcg_cuda_l1_pair* pairs;
    int *inactive_variable, *removed_scalar;
    double *masked_a, *smooth_c, *lambda;
    DeviceProblem* working;
    int weight_mode;
    double omega;
};
__global__ void l1_prepare(DeviceProblem* p,L1State* l) {
    const int rank=blockIdx.x*blockDim.x+threadIdx.x,stride=blockDim.x*gridDim.x;
    if(!rank) { *l->working=*p;l->working->a=l->masked_a;l->working->c=l->smooth_c; }
    for(int j=rank;j<p->variables;j+=stride) {
        l->smooth_c[j]=l->inactive_variable[j]?0.0:p->c[j];l->lambda[j]=0.0;
        for(int k=p->a_offsets[j];k<p->a_offsets[j+1];++k)
            l->masked_a[k]=l->removed_scalar[p->a_indices[k]]?0.0:p->a[k];
    }
}
__global__ void l1_validate_pairs(DeviceProblem* p,L1State* l,int* invalid) {
    const int rank=blockIdx.x*blockDim.x+threadIdx.x,stride=blockDim.x*gridDim.x;
    for(int k=rank;k<p->q_nonzeros;k+=stride)if(p->q[k]!=0.0)atomicExch(invalid,1);
    for(int i=rank;i<l->options.pair_count;i+=stride) {
        const auto pair=l->pairs[i];const int t=pair.epigraph_variable,v=pair.absolute_variable;
        bool ok=isfinite(p->c[t]) && p->c[t]>0.0;
        int uses=0;
        for(int k=p->a_offsets[t];k<p->a_offsets[t+1];++k)if(p->a[k]!=0.0) {
            ++uses;ok=ok && p->a[k]==-1.0 && (p->a_indices[k]==pair.positive_scalar_row || p->a_indices[k]==pair.negative_scalar_row);
        }
        if(p->affine_rows)for(int k=p->f_offsets[t];k<p->f_offsets[t+1];++k)if(p->f[k]!=0.0)ok=false;
        ok=ok && uses==2;
        for(int side=0;side<2;++side) {
            const int row=side?pair.negative_scalar_row:pair.positive_scalar_row;
            ok=ok && row>=p->common_kkt->options.equality_rows
                && p->scalar_lower[row]==-CUDART_INF && p->scalar_upper[row]==0.0;
            int actual=0,found_t=0,found_v=0;
            for(int k=p->common_kkt->row_offsets[row];k<p->common_kkt->row_offsets[row+1];++k) {
                const int position=p->common_kkt->row_positions[k];const double a=p->a[position];
                if(a==0.0)continue;
                ++actual;const int column=static_cast<int>(p->common_kkt->row_keys[k]&0xffffffffULL);
                if(column==t && a==-1.0)++found_t;
                if(column==v && a==(side?-1.0:1.0))++found_v;
            }
            ok=ok && actual==2 && found_t==1 && found_v==1;
        }
        if(!ok)atomicExch(invalid,1);
        l->lambda[v]=p->c[t];
    }
}

// The independent initializer below intentionally duplicates the frozen
// equilibration policy so no default kernel instantiation acquires L1 state.

__global__ void cooperative_l1_initialise_kernel(
    DeviceControl* control,
    DeviceProblem* original,
    volatile int* cancellation,
    L1State* l
) {
    (void)original;
    DeviceProblem* problem=l->working;
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
                if(l->inactive_variable[variable]) {variable_scale[variable]=1.0;problem->previous_primal[variable]=0.0;problem->gradient[variable]=0.0;continue;}
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
                if(l->inactive_variable[variable]) {variable_scale[variable]=1.0;problem->previous_primal[variable]=0.0;problem->gradient[variable]=0.0;continue;}
                problem->previous_primal[variable] = 0.0;
            }
            grid_barrier();
            for (int row = grid_rank(); row < problem->scalar_rows; row += grid_stride()) {
                if(l->removed_scalar[row]) {row_scale[row]=1.0;problem->scalar_product[row]=0.0;continue;}
                problem->scalar_product[row] = 0.0;
            }
            grid_barrier();
            for (int row = grid_rank(); row < problem->affine_rows; row += grid_stride()) {
                problem->affine_product[row] = 0.0;
            }
            grid_barrier();
            for (int variable = grid_rank(); variable < problem->variables; variable += grid_stride()) {
                if(l->inactive_variable[variable]) {variable_scale[variable]=1.0;problem->previous_primal[variable]=0.0;problem->gradient[variable]=0.0;continue;}
                for (int index = problem->scalar_rows > 0
                         ? problem->a_offsets[variable]
                         : 0;
                     index < (problem->scalar_rows > 0
                         ? problem->a_offsets[variable + 1]
                         : 0);
                     ++index) {
                    const int row = problem->a_indices[index];
                    if(l->removed_scalar[row])continue;
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
                if(l->inactive_variable[variable]) {variable_scale[variable]=1.0;problem->previous_primal[variable]=0.0;problem->gradient[variable]=0.0;continue;}
                const double factor = problem->previous_primal[variable] > 1.0e-12
                    ? sqrt(problem->previous_primal[variable])
                    : 1.0;
                variable_scale[variable] *= factor;
            }
            grid_barrier();
            for (int row = grid_rank(); row < problem->scalar_rows; row += grid_stride()) {
                if(l->removed_scalar[row]) {row_scale[row]=1.0;problem->scalar_product[row]=0.0;continue;}
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
                if(l->removed_scalar[row]) {row_scale[row]=1.0;problem->scalar_product[row]=0.0;continue;}
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
                if(l->inactive_variable[variable]) {variable_scale[variable]=1.0;problem->previous_primal[variable]=0.0;problem->gradient[variable]=0.0;continue;}
            const double value = problem->c[variable] / variable_scale[variable];
            objective_norm_squared += value * value;
            const double penalty=l->lambda[variable]/variable_scale[variable];
            objective_norm_squared += penalty*penalty;
        }
        bound_norm_squared = grid_reduce<GridReduction::sum>(bound_norm_squared, problem);
        objective_norm_squared = grid_reduce<GridReduction::sum>(objective_norm_squared, problem);
        const double bound_scale = 1.0 / (sqrt(bound_norm_squared) + 1.0);
        const double objective_scale = 1.0 / (sqrt(objective_norm_squared) + 1.0);
        grid_barrier();
        for (int variable = grid_rank(); variable < problem->variables; variable += grid_stride()) {
                if(l->inactive_variable[variable]) {variable_scale[variable]=1.0;problem->previous_primal[variable]=0.0;problem->gradient[variable]=0.0;continue;}
            problem->gradient[variable] = 0.0;
        }
        double operator_norm_squared = 0.0;
        grid_barrier();
        for (int variable = grid_rank(); variable < problem->variables; variable += grid_stride()) {
                if(l->inactive_variable[variable]) {variable_scale[variable]=1.0;problem->previous_primal[variable]=0.0;problem->gradient[variable]=0.0;continue;}
            // Exact Q=0 is validated before enable. Do not traverse even its
            // structural zeros: an atomic zero write to an inactive gradient
            // would race that slot's ordinary dummy initialization.
            for (int index = problem->scalar_rows > 0
                     ? problem->a_offsets[variable]
                     : 0;
                 index < (problem->scalar_rows > 0
                     ? problem->a_offsets[variable + 1]
                     : 0);
                 ++index) {
                const int row = problem->a_indices[index];
                    if(l->removed_scalar[row])continue;
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
        const double q_norm = 0.0;
        grid_barrier();
        for (int variable = grid_rank(); variable < problem->variables; variable += grid_stride()) {
                if(l->inactive_variable[variable]) {variable_scale[variable]=1.0;problem->previous_primal[variable]=0.0;problem->gradient[variable]=0.0;continue;}
            problem->previous_primal[variable] =
                1.0 / sqrt(static_cast<double>(problem->variables-l->options.pair_count));
        }
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
                if(l->removed_scalar[row]) {row_scale[row]=1.0;problem->scalar_product[row]=0.0;continue;}
                problem->scalar_product[row] = 0.0;
            }
            grid_barrier();
            for (int row = grid_rank(); row < problem->affine_rows; row += grid_stride()) {
                problem->affine_product[row] = 0.0;
            }
            grid_barrier();
            for (int variable = grid_rank(); variable < problem->variables; variable += grid_stride()) {
                if(l->inactive_variable[variable]) {variable_scale[variable]=1.0;problem->previous_primal[variable]=0.0;problem->gradient[variable]=0.0;continue;}
                const double x = problem->previous_primal[variable];
                for (int index = problem->scalar_rows > 0
                         ? problem->a_offsets[variable]
                         : 0;
                     index < (problem->scalar_rows > 0
                         ? problem->a_offsets[variable + 1]
                         : 0);
                     ++index) {
                    const int row = problem->a_indices[index];
                    if(l->removed_scalar[row])continue;
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
                if(l->removed_scalar[row]) {row_scale[row]=1.0;problem->scalar_product[row]=0.0;continue;}
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
                if(l->removed_scalar[row]) {row_scale[row]=1.0;problem->scalar_product[row]=0.0;continue;}
                problem->scalar_product[row] /= row_norm;
            }
            grid_barrier();
            for (int row = grid_rank(); row < problem->affine_rows; row += grid_stride()) {
                problem->affine_product[row] /= row_norm;
            }
            grid_barrier();
            for (int variable = grid_rank(); variable < problem->variables; variable += grid_stride()) {
                if(l->inactive_variable[variable]) {variable_scale[variable]=1.0;problem->previous_primal[variable]=0.0;problem->gradient[variable]=0.0;continue;}
                double value = 0.0;
                for (int index = problem->scalar_rows > 0
                         ? problem->a_offsets[variable]
                         : 0;
                     index < (problem->scalar_rows > 0
                         ? problem->a_offsets[variable + 1]
                         : 0);
                     ++index) {
                    const int row = problem->a_indices[index];
                    if(l->removed_scalar[row])continue;
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
                if(l->inactive_variable[variable]) {variable_scale[variable]=1.0;problem->previous_primal[variable]=0.0;problem->gradient[variable]=0.0;continue;}
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
                if(l->inactive_variable[variable]) {variable_scale[variable]=1.0;problem->previous_primal[variable]=0.0;problem->gradient[variable]=0.0;continue;}
                problem->previous_primal[variable] =
                    problem->gradient[variable] / operator_norm;
            }
        }
        const double denominator = fmax(1.0, q_norm + operator_norm);
        if (grid_rank() == 0) {
            control->primal_step = 0.9 / denominator;
            control->dual_step = 0.9 / fmax(1.0, operator_norm);
            control->halpern_bound_scale = bound_scale;
            control->halpern_objective_scale = objective_scale;
        }
        grid_barrier();
        for (int variable = grid_rank(); variable < problem->variables; variable += grid_stride()) {
                if(l->inactive_variable[variable]) {variable_scale[variable]=1.0;problem->previous_primal[variable]=0.0;problem->gradient[variable]=0.0;continue;}
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
    grid_barrier();
    // Inactive full-layout slots are never part of the working metric. Retain
    // finite positive placeholders for original natural-residual telemetry.
    for(int j=grid_rank();j<problem->variables;j+=grid_stride())if(l->inactive_variable[j])problem->scaling[j]=1.0;
    for(int r=grid_rank();r<problem->scalar_rows;r+=grid_stride())if(l->removed_scalar[r])problem->scaling[problem->variables+r]=1.0;
    grid_barrier();
    if (grid_rank() == 0) control->force_scaling_refresh = 0;
}


__device__ __forceinline__ void l1_forward(DeviceProblem* p,L1State* l) {
    grid_zero_vector(p->scalar_product,p->scalar_rows);
    grid_zero_vector(p->affine_product,p->affine_rows);grid_barrier();
    for(int j=grid_rank();j<p->variables;j+=grid_stride())if(!l->inactive_variable[j]) {
        const double x=p->extrapolated_primal[j];
        for(int k=p->a_offsets[j];k<p->a_offsets[j+1];++k)if(!l->removed_scalar[p->a_indices[k]])
            atomicAdd(p->scalar_product+p->a_indices[k],p->a[k]*x);
        if(p->affine_rows)for(int k=p->f_offsets[j];k<p->f_offsets[j+1];++k)
            atomicAdd(p->affine_product+p->f_indices[k],p->f[k]*x);
    }
    grid_barrier();
}
template<bool Weighted=false>
__device__ __forceinline__ void l1_update(DeviceProblem* p,DeviceControl* c,L1State* l) {
    double primal_step=c->primal_step,dual_step=c->dual_step;
    if constexpr(Weighted) {primal_step/=l->omega;dual_step*=l->omega;}
    l1_forward(p,l);
    for(int row=grid_rank();row<p->scalar_rows;row+=grid_stride()) {
        if(l->removed_scalar[row]) {p->dual[row]=0.0;continue;}
        const double step=dual_step*p->scaling[p->variables+row];
        const double value=p->dual[row]+step*p->scalar_product[row];
        const double projected=project_interval(value/step,p->scalar_lower[row],p->scalar_upper[row]);
        p->dual[row]=value-step*projected;
        if(!isfinite(step) || !(step>0.0) || !isfinite(value) || !isfinite(p->dual[row]))atomicExch(&l->result.finite,0);
    }
    for(int row=grid_rank();row<p->affine_rows;row+=grid_stride()) {
        const int r=p->scalar_rows+row;const double step=dual_step*p->scaling[p->variables+r];
        const double value=p->dual[r]+step*p->affine_product[row];
        p->cone_scratch[row]=value/step+p->affine_offset[row];
        if(!isfinite(step)||!(step>0.0)||!isfinite(p->cone_scratch[row]))atomicExch(&l->result.finite,0);
    }
    grid_barrier();grid_project_cone_blocks(p->cone_scratch,p->affine_cones,p->affine_cone_count);grid_barrier();
    for(int row=grid_rank();row<p->affine_rows;row+=grid_stride()) {
        const int r=p->scalar_rows+row;const double step=dual_step*p->scaling[p->variables+r];
        const double value=p->dual[r]+step*p->affine_product[row];
        p->dual[r]=value-step*(p->cone_scratch[row]-p->affine_offset[row]);
        if(!isfinite(p->dual[r]))atomicExch(&l->result.finite,0);
    }
    grid_barrier();
    for(int j=grid_rank();j<p->variables;j+=grid_stride())if(!l->inactive_variable[j]) {
        double gradient=p->c[j];
        for(int k=p->a_offsets[j];k<p->a_offsets[j+1];++k)if(!l->removed_scalar[p->a_indices[k]])
            gradient+=p->a[k]*p->dual[p->a_indices[k]];
        if(p->affine_rows)for(int k=p->f_offsets[j];k<p->f_offsets[j+1];++k)
            gradient+=p->f[k]*p->dual[p->scalar_rows+p->f_indices[k]];
        const double previous=p->primal[j],step=primal_step*p->scaling[j];
        const double argument=previous-step*gradient,threshold=step*l->lambda[j];
        p->gradient[j]=gradient;p->previous_primal[j]=previous;
        p->primal[j]=l->lambda[j]>0.0?lm::soft_threshold(argument,threshold):argument;
        p->extrapolated_primal[j]=2.0*p->primal[j]-previous;
        if(!isfinite(gradient)||!isfinite(step)||!(step>0.0)||!isfinite(argument)
            ||!isfinite(threshold)||(l->lambda[j]>0.0 && !(threshold>0.0))
            ||!isfinite(p->primal[j])||!isfinite(p->extrapolated_primal[j]))atomicExch(&l->result.finite,0);
    }
    grid_barrier();
    // Recover original output coordinates every completed step. These pair
    // rows/columns are excluded from all subsequent working operations.
    for(int i=grid_rank();i<l->options.pair_count;i+=grid_stride()) {
        const auto pair=l->pairs[i];const int v=pair.absolute_variable,t=pair.epigraph_variable;
        const auto z=lm::complete_dual(p->primal[v],p->gradient[v],l->lambda[v]);
        p->primal[t]=fabs(p->primal[v]);
        p->previous_primal[t]=p->extrapolated_primal[t]=p->primal[t];
        p->dual[pair.positive_scalar_row]=z.positive;p->dual[pair.negative_scalar_row]=z.negative;
        if(!isfinite(z.positive)||!isfinite(z.negative)||!isfinite(p->primal[t]))atomicExch(&l->result.finite,0);
    }
    grid_barrier();
}
__device__ __forceinline__ bool l1_check(DeviceProblem* original,DeviceControl* c,DeviceReport* report,
        L1State* l,volatile int* cancellation,std::uint64_t iteration) {
    grid_evaluate_report<1>(original,c,report,iteration);
    common_kkt_evaluate<true>(original,iteration);
    if(!grid_rank()) {
        if(!original->common_kkt->result.finite)l->result.finite=0;
        if(*cancellation)report->termination=SPACEPDHCG_CUDA_TERMINATION_CANCELLED;
        else if(!l->result.finite)report->termination=SPACEPDHCG_CUDA_TERMINATION_NUMERICAL_FAILURE;
        else if(original->common_kkt->result.passes)report->termination=SPACEPDHCG_CUDA_TERMINATION_OPTIMAL;
    }
    grid_barrier();return report->termination!=SPACEPDHCG_CUDA_TERMINATION_ITERATION_LIMIT;
}
template<bool Weighted=false>
__global__ void cooperative_l1_kernel(DeviceProblem* original,DeviceControl* c,DeviceReport* report,
        volatile int* cancellation,L1State* l) {
    DeviceProblem* p=l->working;
    if(!grid_rank()) {
        *report={};report->termination=SPACEPDHCG_CUDA_TERMINATION_ITERATION_LIMIT;report->recovery_stationarity_index=-1;
        original->common_kkt->result={};l->result={};
        l->result.abi_version=SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION;
        l->result.enabled=l->result.valid=l->result.finite=1;
        l->result.pairs=l->options.pair_count;l->result.retained_variables=p->variables;
        l->result.retained_rows=p->scalar_rows+p->affine_rows;
        l->result.active_variables=p->variables-l->options.pair_count;
        l->result.active_rows=l->result.retained_rows-2*l->options.pair_count;
        l->result.eta=c->primal_step;l->result.bound_scale=c->halpern_bound_scale;
        l->result.objective_scale=c->halpern_objective_scale;
        if constexpr(Weighted)if(l->weight_mode==SPACEPDHCG_CUDA_L1_WEIGHT_CANCEL_GLOBAL)
            l->omega=c->halpern_objective_scale/c->halpern_bound_scale;
    }
    grid_barrier();
    if(grid_cancelled(original,cancellation)) {
        grid_evaluate_report<1>(original,c,report,0);
        if(!grid_rank())report->termination=SPACEPDHCG_CUDA_TERMINATION_CANCELLED;return;
    }
    double primal_step=c->primal_step;
    if constexpr(Weighted) {
        const auto steps=lm::reciprocal_steps(c->primal_step,l->omega);
        primal_step=steps.primal;
        if(!grid_rank() && (!(l->omega>0.0)||!isfinite(l->omega)
            ||!(1.0/l->omega>0.0)||!isfinite(1.0/l->omega)
            ||!(steps.primal>0.0)||!isfinite(steps.primal)
            ||!(steps.dual>0.0)||!isfinite(steps.dual)))l->result.finite=0;
        grid_barrier();
        // Check every actual diagonal step, including a zero-step seed solve.
        // Inactive full-layout dummy entries are excluded from this metric.
        for(int j=grid_rank();j<p->variables;j+=grid_stride())if(!l->inactive_variable[j]) {
            const double step=steps.primal*p->scaling[j];
            if(!(step>0.0)||!isfinite(step))atomicExch(&l->result.finite,0);
        }
        for(int row=grid_rank();row<p->scalar_rows+p->affine_rows;row+=grid_stride()) {
            if(row<p->scalar_rows && l->removed_scalar[row])continue;
            const double step=steps.dual*p->scaling[p->variables+row];
            if(!(step>0.0)||!isfinite(step))atomicExch(&l->result.finite,0);
        }
        grid_barrier();
    }
    double minimum=CUDART_INF,maximum=0.0;
    for(int i=grid_rank();i<l->options.pair_count;i+=grid_stride()) {
        const int v=l->pairs[i].absolute_variable;
        const double threshold=primal_step*p->scaling[v]*l->lambda[v];
        if(!isfinite(threshold)||!(threshold>0.0))atomicExch(&l->result.finite,0);
        minimum=fmin(minimum,threshold);maximum=fmax(maximum,threshold);
    }
    minimum=grid_reduce<GridReduction::minimum>(minimum,p);maximum=grid_reduce<GridReduction::maximum>(maximum,p);
    if(!grid_rank()) {
        l->result.minimum_threshold=minimum;l->result.maximum_threshold=maximum;
        if(!(c->primal_step>0.0)||!isfinite(c->primal_step)||c->primal_step!=c->dual_step
            ||!(c->halpern_bound_scale>0.0)||!isfinite(c->halpern_bound_scale)
            ||!(c->halpern_objective_scale>0.0)||!isfinite(c->halpern_objective_scale))l->result.finite=0;
    }
    grid_barrier();
    // Crucial: this sees the entire original supplied x,t,y,z unchanged.
    if(l1_check(original,c,report,l,cancellation,0))return;
    const unsigned int frequency=c->residual_check_frequency?c->residual_check_frequency:1U;
    for(std::uint64_t iteration=1;iteration<=c->iteration_limit;++iteration) {
        if(grid_cancelled(original,cancellation)) {
            grid_evaluate_report<1>(original,c,report,iteration-1);
            if(!grid_rank())report->termination=SPACEPDHCG_CUDA_TERMINATION_CANCELLED;return;
        }
        l1_update<Weighted>(p,c,l);
        if(!grid_rank()) {l->result.updates=iteration;l->result.completions=iteration;}
        grid_barrier();
        if(iteration==1||iteration%frequency==0||iteration==c->iteration_limit||!l->result.finite)
            if(l1_check(original,c,report,l,cancellation,iteration))return;
    }
}
