// SPDX-License-Identifier: Apache-2.0
// Diagnostic cooperative path. Included after cooperative_pdhg.cuh. The default
// kernels do not reference this state or instantiate any of this iteration.
#include "spacepdhcg/cuda/halpern_arithmetic.hpp"
namespace hm = spacepdhcg::cuda::halpern;
struct HalpernState {
    spacepdhcg_cuda_halpern_options options;
    spacepdhcg_cuda_halpern_diagnostics result;
    double *working_x, *working_y, *anchor_x, *anchor_y;
    double last_trial_error, best_weight, best_residual_gap, integral_error;
    int needs_epoch_reference;
};

__global__ void halpern_validate_zero_q(const DeviceProblem* p, int* invalid) {
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < p->q_nonzeros;
         k += blockDim.x * gridDim.x)
        if (p->q[k] != 0.0) atomicExch(invalid, 1); // no epsilon or zero dropping
}
__global__ void halpern_force_refresh(DeviceControl* c) {
    if (!threadIdx.x && !blockIdx.x) c->force_scaling_refresh = 1;
}
__global__ void halpern_restore_default_history(DeviceProblem* p) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < p->variables;
         j += blockDim.x * gridDim.x)
        p->previous_primal[j] = p->extrapolated_primal[j] = p->primal[j];
}

__device__ void halpern_forward(DeviceProblem* p, const double* x) {
    grid_zero_vector(p->scalar_product, p->scalar_rows);
    grid_zero_vector(p->affine_product, p->affine_rows);
    grid_barrier();
    grid_csc_multiply(p->variables, p->a_offsets, p->a_indices, p->a, x, p->scalar_product);
    grid_csc_multiply(p->variables, p->f_offsets, p->f_indices, p->f, x, p->affine_product);
    grid_barrier();
}

__device__ void halpern_proximal_point(DeviceProblem* p, DeviceControl* c, HalpernState* h) {
    // Free primal, exactly Q=0: the primal prox is explicit. Current working
    // duals may lie outside their cones; only the returned T dual is projected.
    const double tau = c->primal_step / h->result.primal_weight;
    const double sigma = c->dual_step * h->result.primal_weight;
    for (int j = grid_rank(); j < p->variables; j += grid_stride()) {
        double gradient = p->c[j];
        if (p->scalar_rows) for (int k = p->a_offsets[j]; k < p->a_offsets[j+1]; ++k)
            gradient += p->a[k] * h->working_y[p->a_indices[k]];
        if (p->affine_rows) for (int k = p->f_offsets[j]; k < p->f_offsets[j+1]; ++k)
            gradient += p->f[k] * h->working_y[p->scalar_rows + p->f_indices[k]];
        p->primal[j] = h->working_x[j] - tau * p->scaling[j] * gradient;
        p->extrapolated_primal[j] = 2.0 * p->primal[j] - h->working_x[j];
        if (!isfinite(gradient) || !isfinite(p->primal[j]) || !isfinite(p->extrapolated_primal[j])
            || !isfinite(tau) || !(tau * p->scaling[j] > 0.0)) atomicExch(&h->result.finite, 0);
    }
    grid_barrier();
    halpern_forward(p, p->extrapolated_primal);
    const int equality_rows = p->common_kkt->options.equality_rows;
    for (int i = grid_rank(); i < p->scalar_rows; i += grid_stride()) {
        const double step = sigma * p->scaling[p->variables + i];
        const double value = h->working_y[i] + step * (p->scalar_product[i] - p->scalar_upper[i]);
        // Direct conjugate prox avoids value/step cancellation for equalities.
        p->dual[i] = i < equality_rows ? value : fmax(0.0, value);
        if (!isfinite(value) || !isfinite(step) || !(step > 0.0)) atomicExch(&h->result.finite, 0);
    }
    for (int i = grid_rank(); i < p->affine_rows; i += grid_stride()) {
        const int row = p->scalar_rows + i;
        const double step = sigma * p->scaling[p->variables + row];
        p->cone_scratch[i] = h->working_y[row] / step + p->affine_product[i] + p->affine_offset[i];
        if (!isfinite(p->cone_scratch[i]) || !isfinite(step) || !(step > 0.0)) atomicExch(&h->result.finite, 0);
    }
    grid_barrier();
    grid_project_cone_blocks(p->cone_scratch, p->affine_cones, p->affine_cone_count);
    grid_barrier();
    for (int i = grid_rank(); i < p->affine_rows; i += grid_stride()) {
        const int row = p->scalar_rows + i;
        const double step = sigma * p->scaling[p->variables + row];
        p->dual[row] = h->working_y[row] + step *
            (p->affine_product[i] + p->affine_offset[i] - p->cone_scratch[i]);
        if (!isfinite(p->dual[row])) atomicExch(&h->result.finite, 0);
    }
    grid_barrier();
}

__device__ void halpern_fixed_point_error(DeviceProblem* p, DeviceControl* c, HalpernState* h,
                                        std::uint64_t iteration) {
    // delta=R(z)-z=2(T(z)-z), BEFORE replacing the working point with its blend.
    // Scaling identities: ||dx_tilde||^2=B*O*sum(dx^2/scaling_x), likewise y.
    const double bo = c->halpern_bound_scale * c->halpern_objective_scale;
    double xx = 0.0, yy = 0.0, cross = 0.0, invalid = 0.0;
    for (int j = grid_rank(); j < p->variables; j += grid_stride()) {
        const double delta = 2.0 * (p->primal[j] - h->working_x[j]);
        p->previous_primal[j] = delta;
        xx += delta * delta / p->scaling[j];
        if (!isfinite(delta) || !isfinite(xx)) invalid = 1.0;
    }
    grid_barrier();
    halpern_forward(p, p->previous_primal);
    for (int i = grid_rank(); i < p->scalar_rows + p->affine_rows; i += grid_stride()) {
        const double delta = 2.0 * (p->dual[i] - h->working_y[i]);
        const double product = i < p->scalar_rows ? p->scalar_product[i] : p->affine_product[i-p->scalar_rows];
        yy += delta * delta / p->scaling[p->variables + i];
        cross += delta * product;
        if (!isfinite(delta) || !isfinite(yy) || !isfinite(cross)) invalid = 1.0;
    }
    xx = grid_reduce<GridReduction::sum>(xx, p);
    yy = grid_reduce<GridReduction::sum>(yy, p);
    cross = grid_reduce<GridReduction::sum>(cross, p);
    invalid = grid_reduce<GridReduction::maximum>(invalid, p);
    if (!grid_rank()) {
        const double metric = hm::metric_squared(bo * xx, bo * yy, bo * cross,
                                                  c->primal_step, h->result.primal_weight);
        ++h->result.metric_evaluations;
        if (invalid || !isfinite(metric) || metric < 0.0) h->result.finite = 0;
        h->result.fixed_point_error = sqrt(metric);
        if (h->needs_epoch_reference) {
            h->result.epoch_initial_error = h->result.fixed_point_error;
            h->result.epoch_reference_iteration = iteration;
            h->needs_epoch_reference = 0;
        }
    }
    grid_barrier();
}

__device__ void halpern_restart(DeviceProblem* p, DeviceControl* c, HalpernState* h,
                               std::uint64_t iteration) {
    const double bo = c->halpern_bound_scale * c->halpern_objective_scale;
    double xx = 0.0, yy = 0.0, invalid = 0.0;
    for (int j = grid_rank(); j < p->variables; j += grid_stride()) {
        const double delta = p->primal[j] - h->anchor_x[j];
        xx += delta * delta / p->scaling[j];
        if (!isfinite(delta) || !isfinite(xx)) invalid = 1.0;
    }
    for (int i = grid_rank(); i < p->scalar_rows + p->affine_rows; i += grid_stride()) {
        const double delta = p->dual[i] - h->anchor_y[i];
        yy += delta * delta / p->scaling[p->variables+i];
        if (!isfinite(delta) || !isfinite(yy)) invalid = 1.0;
    }
    xx = grid_reduce<GridReduction::sum>(xx, p);
    yy = grid_reduce<GridReduction::sum>(yy, p);
    invalid = grid_reduce<GridReduction::maximum>(invalid, p);
    if (!grid_rank()) {
        const double dx = sqrt(bo * xx), dy = sqrt(bo * yy);
        const auto& kkt = p->common_kkt->result;
        // Documented variant: common original-equation normalized residuals,
        // rather than the upstream implementation's own relative residuals.
        const double ratio = kkt.dual_relative / kkt.primal_relative;
        const bool guarded = !invalid && isfinite(dx) && isfinite(dy) && isfinite(ratio)
            && dx > 1e-16 && dy > 1e-16 && dx < 1e12 && dy < 1e12 && ratio > 1e-8 && ratio < 1e8;
        if (guarded) {
            const double error = log(dy) - log(dx) - log(h->result.primal_weight);
            const double integral = 0.3 * h->integral_error + error;
            const double weight = h->result.primal_weight * exp(0.99 * error + 0.01 * integral);
            if (!isfinite(weight) || !(weight > 0.0) || !isfinite(integral)) {
                h->result.finite = 0; // never clamp an overflowing adaptation
            } else {
                h->integral_error = integral;
                h->result.primal_weight = weight;
                ++h->result.weight_updates;
            }
        } else {
            h->result.primal_weight = h->best_weight;
            h->integral_error = 0.0;
            ++h->result.weight_fallbacks;
            if (invalid || !isfinite(dx) || !isfinite(dy)) h->result.finite = 0;
        }
        if (isfinite(ratio) && ratio > 0.0) {
            const double gap = fabs(log10(ratio));
            if (gap < h->best_residual_gap) {
                h->best_residual_gap = gap;
                h->best_weight = h->result.primal_weight;
            }
        }
        h->result.minimum_primal_weight = fmin(h->result.minimum_primal_weight, h->result.primal_weight);
        h->result.maximum_primal_weight = fmax(h->result.maximum_primal_weight, h->result.primal_weight);
        ++h->result.restarts;
        h->result.last_restart_iteration = iteration;
        h->result.inner_iterations = 0;
        h->last_trial_error = CUDART_INF;
        h->needs_epoch_reference = 1;
    }
    for (int j = grid_rank(); j < p->variables; j += grid_stride())
        h->anchor_x[j] = h->working_x[j] = p->primal[j];
    for (int i = grid_rank(); i < p->scalar_rows + p->affine_rows; i += grid_stride())
        h->anchor_y[i] = h->working_y[i] = p->dual[i];
    grid_barrier();
}

__device__ bool halpern_check(DeviceProblem* p, DeviceControl* c, DeviceReport* report,
                             HalpernState* h, volatile int* cancellation, std::uint64_t iteration) {
    grid_evaluate_report<1>(p, c, report, iteration);
    common_kkt_evaluate<true>(p, iteration);
    if (!grid_rank()) {
        if (!p->common_kkt->result.finite) h->result.finite = 0;
        if (*cancellation) report->termination = SPACEPDHCG_CUDA_TERMINATION_CANCELLED;
        else if (!p->common_kkt->result.finite || !h->result.finite)
            report->termination = SPACEPDHCG_CUDA_TERMINATION_NUMERICAL_FAILURE;
        else if (p->common_kkt->result.passes) report->termination = SPACEPDHCG_CUDA_TERMINATION_OPTIMAL;
    }
    grid_barrier();
    return report->termination != SPACEPDHCG_CUDA_TERMINATION_ITERATION_LIMIT;
}

__global__ void cooperative_halpern_kernel(DeviceProblem* p, DeviceControl* c,
        DeviceReport* report, volatile int* cancellation, HalpernState* h) {
    if (!grid_rank()) {
        *report = {};
        report->termination = SPACEPDHCG_CUDA_TERMINATION_ITERATION_LIMIT;
        report->recovery_stationarity_index = -1;
        p->common_kkt->result = {};
        h->result = {};
        h->result.abi_version = SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION;
        h->result.mode = h->options.mode;
        h->result.valid = 1;
        h->result.finite = 1;
        h->result.primal_weight = h->result.minimum_primal_weight = h->result.maximum_primal_weight = 1.0;
        h->result.eta = c->primal_step;
        h->best_weight = 1.0;
        h->best_residual_gap = CUDART_INF;
        h->last_trial_error = CUDART_INF;
        h->integral_error = 0.0;
        h->needs_epoch_reference = 0;
    }
    grid_barrier();
    if (grid_cancelled(p, cancellation)) {
        grid_evaluate_report<1>(p, c, report, 0);
        if (!grid_rank()) report->termination = SPACEPDHCG_CUDA_TERMINATION_CANCELLED;
        return;
    }
    if (!grid_rank() && (!(c->primal_step > 0.0) || !isfinite(c->primal_step)
        || c->primal_step != c->dual_step || !(c->halpern_bound_scale > 0.0)
        || !(c->halpern_objective_scale > 0.0) || !isfinite(c->halpern_bound_scale)
        || !isfinite(c->halpern_objective_scale) || !isfinite(c->halpern_bound_scale*c->halpern_objective_scale)
        || !(c->halpern_bound_scale*c->halpern_objective_scale > 0.0))) h->result.finite = 0;
    grid_barrier();
    if (halpern_check(p, c, report, h, cancellation, 0)) return;
    for (int j = grid_rank(); j < p->variables; j += grid_stride())
        h->anchor_x[j] = h->working_x[j] = p->primal[j];
    for (int i = grid_rank(); i < p->scalar_rows + p->affine_rows; i += grid_stride())
        h->anchor_y[i] = h->working_y[i] = p->dual[i];
    grid_barrier();
    const auto frequency = c->residual_check_frequency ? c->residual_check_frequency : 1U;
    for (std::uint64_t iteration = 1; iteration <= c->iteration_limit; ++iteration) {
        if (grid_cancelled(p, cancellation)) {
            grid_evaluate_report<1>(p, c, report, iteration - 1);
            if (!grid_rank()) {
                p->common_kkt->result.valid = p->common_kkt->result.passes = 0;
                report->termination = SPACEPDHCG_CUDA_TERMINATION_CANCELLED;
            }
            return;
        }
        halpern_proximal_point(p, c, h);
        if (iteration % 200 == 0 || h->needs_epoch_reference)
            halpern_fixed_point_error(p, c, h, iteration);
        const auto inner = h->result.inner_iterations;
        for (int j = grid_rank(); j < p->variables; j += grid_stride()) {
            h->working_x[j] = hm::blend(h->anchor_x[j], 2.0*p->primal[j]-h->working_x[j], inner);
            if (!isfinite(h->working_x[j])) atomicExch(&h->result.finite, 0);
        }
        for (int i = grid_rank(); i < p->scalar_rows + p->affine_rows; i += grid_stride()) {
            h->working_y[i] = hm::blend(h->anchor_y[i], 2.0*p->dual[i]-h->working_y[i], inner);
            if (!isfinite(h->working_y[i])) atomicExch(&h->result.finite, 0);
        }
        grid_barrier();
        if (!grid_rank()) { ++h->result.inner_iterations; h->result.updates = iteration; }
        grid_barrier();
        if (iteration == 1 || iteration % frequency == 0 || iteration % 200 == 0
            || iteration == c->iteration_limit || !h->result.finite) {
            if (halpern_check(p, c, report, h, cancellation, iteration)) return;
        }
        if (h->options.mode == 2 && iteration % 200 == 0 && iteration < c->iteration_limit) {
            const bool restart = hm::restart(iteration, h->result.inner_iterations,
                h->result.fixed_point_error, h->result.epoch_initial_error, h->last_trial_error);
            grid_barrier(); // Every reader consumes the previous checkpoint error before its update.
            if (!grid_rank()) h->last_trial_error = h->result.fixed_point_error;
            grid_barrier();
            if (restart) halpern_restart(p, c, h, iteration);
            if (!h->result.finite) {
                if (!grid_rank()) report->termination = *cancellation ? SPACEPDHCG_CUDA_TERMINATION_CANCELLED
                    : SPACEPDHCG_CUDA_TERMINATION_NUMERICAL_FAILURE;
                return;
            }
        }
    }
}
