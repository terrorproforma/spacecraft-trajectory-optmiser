// SPDX-License-Identifier: Apache-2.0
// Isolated exact causal-mass + L1 experiment. Original public arrays are retained.
#include "spacepdhcg/cuda/mass_causal_arithmetic.hpp"
namespace mm=spacepdhcg::cuda::mass;
struct MassState {
    spacepdhcg_cuda_mass_options options;
    spacepdhcg_cuda_mass_diagnostics result;
    spacepdhcg_cuda_mass_node* nodes;
    int *variable_node,*equality_node,*gamma_node,*virtual_node,*row_node,*row_sign;
    double *g,*affine,*constant,*scan0,*scan1,*suffix;
    double *row_sum,*column_sum,*upper;
};
__device__ __forceinline__ void mass_finite(MassState* m,double value) {
    if(!isfinite(value))atomicExch(&m->result.finite,0);
}
__device__ __forceinline__ bool mass_active(const L1State* l,const MassState* m,int j) {
    return !l->inactive_variable[j] && m->variable_node[j]<0;
}
__device__ __forceinline__ bool mass_row_active(const L1State* l,const MassState* m,int r) {
    return !l->removed_scalar[r] && m->equality_node[r]<0;
}
// Inclusive parallel scan in original coefficient order; no expanded sparse fill.
// Reassociation changes floating-point roundoff, never the original KKT gates.
template<bool Reverse=false,bool Upward=false>
__device__ __forceinline__ void mass_scan(MassState* m) {
    double* a=m->scan0;double* b=m->scan1;const int n=m->options.node_count;
    grid_barrier();
    for(int distance=1;distance<n;distance*=2) {
        for(int i=grid_rank();i<n;i+=grid_stride()) {
            const int other=Reverse?i+distance:i-distance;
            double value=a[i];
            if(other>=0 && other<n)value=Upward?mm::add_up(value,a[other]):value+a[other];
            b[i]=value;mass_finite(m,value);
        }
        grid_barrier();double* temp=a;a=b;b=temp;
    }
    if(a!=m->scan0) {
        for(int i=grid_rank();i<n;i+=grid_stride())m->scan0[i]=a[i];
        grid_barrier();
    }
}
__global__ void mass_prepare(DeviceProblem* p,MassState* m) {
    const int rank=blockIdx.x*blockDim.x+threadIdx.x,stride=blockDim.x*gridDim.x;
    for(int r=rank;r<p->scalar_rows;r+=stride){m->row_node[r]=-1;m->row_sign[r]=0;}
}
__global__ void mass_validate(DeviceProblem* p,L1State* l,MassState* m,int* invalid) {
    const int rank=blockIdx.x*blockDim.x+threadIdx.x,stride=blockDim.x*gridDim.x;
    for(int k=rank;k<p->q_nonzeros;k+=stride)if(p->q[k]!=0.0)atomicExch(invalid,1);
    for(int i=rank;i<m->options.node_count;i+=stride) {
        const auto node=m->nodes[i];const int mass=node.mass_variable,row=node.equality_row;
        bool ok=p->c[mass]==0.0 && !l->inactive_variable[mass] && l->lambda[mass]==0.0
            && row<p->common_kkt->options.equality_rows && !l->removed_scalar[row]
            && isfinite(p->scalar_upper[row]) && p->scalar_lower[row]==p->scalar_upper[row];
        int count=0,found_mass=0,found_previous=0,found_gamma=0,found_virtual=0;
        double g=0.0;
        for(int k=p->common_kkt->row_offsets[row];k<p->common_kkt->row_offsets[row+1];++k) {
            const int pos=p->common_kkt->row_positions[k];const double value=p->a[pos];
            if(value==0.0)continue;
            ++count;const int j=static_cast<int>(p->common_kkt->row_keys[k]&0xffffffffULL);
            if(j==mass && value==1.0)++found_mass;
            if(i && j==m->nodes[i-1].mass_variable && value==-1.0)++found_previous;
            if(i && j==node.gamma_variable && value>0.0 && isfinite(value)){++found_gamma;g=value;}
            if(i && j==node.virtual_variable && value==-1.0)++found_virtual;
        }
        if(i)ok=ok && count==4 && found_mass==1 && found_previous==1 && found_gamma==1 && found_virtual==1
            && mass_active(l,m,node.gamma_variable) && mass_active(l,m,node.virtual_variable);
        else ok=ok && count==1 && found_mass==1 && p->scalar_upper[row]==1.0;
        int bounds=0;
        for(int k=p->a_offsets[mass];k<p->a_offsets[mass+1];++k) {
            const double value=p->a[k];if(value==0.0)continue;
            const int r=p->a_indices[k];
            if(r==row){ok=ok && value==1.0;continue;}
            if(i+1<m->options.node_count && r==m->nodes[i+1].equality_row){ok=ok && value==-1.0;continue;}
            bool bound=r>=p->common_kkt->options.equality_rows && !l->removed_scalar[r]
                && (value==1.0||value==-1.0) && p->scalar_lower[r]==-CUDART_INF && isfinite(p->scalar_upper[r]);
            int actual=0;
            for(int t=p->common_kkt->row_offsets[r];t<p->common_kkt->row_offsets[r+1];++t)
                if(p->a[p->common_kkt->row_positions[t]]!=0.0)++actual;
            bound=bound && actual==1;ok=ok && bound;
            if(bound){++bounds;m->row_node[r]=i;m->row_sign[r]=value>0.0?1:-1;}
        }
        if(p->affine_rows)for(int k=p->f_offsets[mass];k<p->f_offsets[mass+1];++k)
            if(p->f[k]!=0.0)ok=false;
        ok=ok && bounds==3;
        m->g[i]=g;m->affine[i]=p->scalar_upper[row];
        if(!ok)atomicExch(invalid,1);
    }
}
__global__ void cooperative_mass_initialise_kernel(DeviceControl* c,DeviceProblem* original,
        volatile int* cancellation,L1State* l,MassState* m) {
    DeviceProblem* p=l->working;const int n=m->options.node_count;
    if(!grid_rank()) {
        m->result={};auto& r=m->result;r.abi_version=SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION;
        r.enabled=r.finite=1;r.nodes=n;r.retained_variables=p->variables;r.retained_rows=p->scalar_rows+p->affine_rows;
        r.active_variables=p->variables-l->options.pair_count-n;
        r.active_rows=r.retained_rows-2*l->options.pair_count-n;r.theta=mm::theta;
    }
    grid_barrier();if(grid_cancelled(original,cancellation))return;
    // Affine constants are separate from the linear operator and its adjoint.
    for(int i=grid_rank();i<n;i+=grid_stride())m->scan0[i]=m->affine[i];
    mass_scan(m);
    for(int i=grid_rank();i<n;i+=grid_stride())m->constant[i]=m->scan0[i];
    grid_barrier();
    for(int i=grid_rank();i<n;i+=grid_stride())m->scan0[i]=i?mm::add_up(1.0,m->g[i]):0.0;
    mass_scan<false,true>(m);
    // Common CSR maps retain all original structural zeros. Only actual
    // eliminated entries are excluded from these positive coefficient sums.
    for(int r=grid_rank();r<p->scalar_rows+p->affine_rows;r+=grid_stride()) {
        double sum=0.0;
        if(r<p->scalar_rows && !mass_row_active(l,m,r)) {m->row_sum[r]=0.0;continue;}
        if(r<p->scalar_rows && m->row_node[r]>=0) {
            const int i=m->row_node[r];sum=m->scan0[i];
            m->upper[r]=p->scalar_upper[r]-m->row_sign[r]*m->constant[i];mass_finite(m,m->upper[r]);
        } else {
            for(int k=original->common_kkt->row_offsets[r];k<original->common_kkt->row_offsets[r+1];++k) {
                const int j=static_cast<int>(original->common_kkt->row_keys[k]&0xffffffffULL);
                if(!mass_active(l,m,j))continue;
                const int pos=original->common_kkt->row_positions[k];
                const double value=pos<p->a_nonzeros?p->a[pos]:p->f[pos-p->a_nonzeros];
                sum=mm::add_up(sum,fabs(value));
            }
            if(r<p->scalar_rows)m->upper[r]=p->scalar_upper[r];
        }
        m->row_sum[r]=sum;mass_finite(m,sum);
    }
    for(int j=grid_rank();j<p->variables;j+=grid_stride()) {
        double sum=0.0;
        if(mass_active(l,m,j)) {
            for(int k=p->a_offsets[j];k<p->a_offsets[j+1];++k)
                if(mass_row_active(l,m,p->a_indices[k]))sum=mm::add_up(sum,fabs(p->a[k]));
            if(p->affine_rows)for(int k=p->f_offsets[j];k<p->f_offsets[j+1];++k)sum=mm::add_up(sum,fabs(p->f[k]));
            const int gi=m->gamma_node[j],vi=m->virtual_node[j];
            if(gi>=0)sum=mm::add_up(sum,mm::multiply_up(3.0*(n-gi),m->g[gi]));
            if(vi>=0)sum=mm::add_up(sum,3.0*(n-vi));
        }
        m->column_sum[j]=sum;mass_finite(m,sum);
    }
    grid_barrier();
    for(int j=grid_rank();j<p->variables;j+=grid_stride())p->scaling[j]=mass_active(l,m,j)?mm::step_down(m->column_sum[j]):1.0;
    for(int r=grid_rank();r<p->scalar_rows+p->affine_rows;r+=grid_stride())
        p->scaling[p->variables+r]=(r<p->scalar_rows&&!mass_row_active(l,m,r))?1.0:mm::step_down(m->row_sum[r]);
    grid_barrier();
    for(int ci=grid_rank();ci<p->affine_cone_count;ci+=grid_stride()) {
        const auto cone=p->affine_cones[ci];double denominator=0.0;
        for(int k=0;k<cone.vector_dimension+2;++k)denominator=fmax(denominator,m->row_sum[p->scalar_rows+cone.start+k]);
        const double step=mm::step_down(denominator);
        for(int k=0;k<cone.vector_dimension+2;++k)p->scaling[p->variables+p->scalar_rows+cone.start+k]=step;
    }
    grid_barrier();
    double alpha=0.0,beta=0.0,tmin=CUDART_INF,tmax=0.0,smin=CUDART_INF,smax=0.0,lmin=CUDART_INF,lmax=0.0;
    for(int j=grid_rank();j<p->variables;j+=grid_stride())if(mass_active(l,m,j)) {
        const double step=p->scaling[j];mass_finite(m,step);if(!(step>0.0))atomicExch(&m->result.finite,0);
        tmin=fmin(tmin,step);tmax=fmax(tmax,step);const double factor=mm::multiply_up(step,m->column_sum[j]);
        mass_finite(m,factor);beta=fmax(beta,factor);
        if(l->lambda[j]>0.0) {const double threshold=step*l->lambda[j];mass_finite(m,threshold);
            if(!(threshold>0.0))atomicExch(&m->result.finite,0);lmin=fmin(lmin,threshold);lmax=fmax(lmax,threshold);}
    }
    for(int r=grid_rank();r<p->scalar_rows+p->affine_rows;r+=grid_stride()) {
        if(r<p->scalar_rows && !mass_row_active(l,m,r))continue;
        const double step=p->scaling[p->variables+r];mass_finite(m,step);if(!(step>0.0))atomicExch(&m->result.finite,0);
        smin=fmin(smin,step);smax=fmax(smax,step);const double factor=mm::multiply_up(step,m->row_sum[r]);
        mass_finite(m,factor);alpha=fmax(alpha,factor);
    }
    alpha=grid_reduce<GridReduction::maximum>(alpha,p);beta=grid_reduce<GridReduction::maximum>(beta,p);
    tmin=grid_reduce<GridReduction::minimum>(tmin,p);tmax=grid_reduce<GridReduction::maximum>(tmax,p);
    smin=grid_reduce<GridReduction::minimum>(smin,p);smax=grid_reduce<GridReduction::maximum>(smax,p);
    lmin=grid_reduce<GridReduction::minimum>(lmin,p);lmax=grid_reduce<GridReduction::maximum>(lmax,p);
    if(!grid_rank()) {
        auto& r=m->result;r.row_factor_upper=alpha;r.column_factor_upper=beta;r.norm_squared_upper=mm::multiply_up(alpha,beta);
        r.minimum_primal_step=tmin;r.maximum_primal_step=tmax;r.minimum_dual_step=smin;r.maximum_dual_step=smax;
        r.minimum_threshold=lmin;r.maximum_threshold=lmax;
        if(!isfinite(r.norm_squared_upper)||!(r.norm_squared_upper<1.0))r.finite=0;
        // Actual updates multiply these direct diagonal steps by one only.
        c->primal_step=c->dual_step=1.0;c->halpern_bound_scale=c->halpern_objective_scale=1.0;
        c->scaling_refreshed=1;c->scaling_reuse_count=0;c->force_scaling_refresh=0;r.valid=1;
    }
    grid_barrier();
}
__device__ __forceinline__ void mass_linear_prefix(DeviceProblem* p,MassState* m,const double* x) {
    for(int i=grid_rank();i<m->options.node_count;i+=grid_stride()) {
        const auto node=m->nodes[i];const double value=i?x[node.virtual_variable]-m->g[i]*x[node.gamma_variable]:0.0;
        m->scan0[i]=value;mass_finite(m,value);
    }
    mass_scan(m);
}
__device__ __forceinline__ void mass_update(DeviceProblem* p,L1State* l,MassState* m) {
    mass_linear_prefix(p,m,p->extrapolated_primal);
    grid_zero_vector(p->scalar_product,p->scalar_rows);grid_zero_vector(p->affine_product,p->affine_rows);grid_barrier();
    for(int j=grid_rank();j<p->variables;j+=grid_stride())if(mass_active(l,m,j)) {
        const double value=p->extrapolated_primal[j];
        for(int k=p->a_offsets[j];k<p->a_offsets[j+1];++k)if(mass_row_active(l,m,p->a_indices[k]))
            atomicAdd(p->scalar_product+p->a_indices[k],p->a[k]*value);
        if(p->affine_rows)for(int k=p->f_offsets[j];k<p->f_offsets[j+1];++k)atomicAdd(p->affine_product+p->f_indices[k],p->f[k]*value);
    }
    grid_barrier();
    for(int r=grid_rank();r<p->scalar_rows;r+=grid_stride()) {
        if(!mass_row_active(l,m,r)){p->dual[r]=0.0;continue;}
        if(m->row_node[r]>=0)p->scalar_product[r]=m->row_sign[r]*m->scan0[m->row_node[r]];
        const double step=p->scaling[p->variables+r],value=p->dual[r]+step*p->scalar_product[r];
        p->dual[r]=value-step*project_interval(value/step,p->scalar_lower[r],m->upper[r]);
        mass_finite(m,value);mass_finite(m,p->dual[r]);
    }
    for(int r=grid_rank();r<p->affine_rows;r+=grid_stride()) {
        const int row=p->scalar_rows+r;const double step=p->scaling[p->variables+row];
        p->cone_scratch[r]=p->dual[row]/step+p->affine_product[r]+p->affine_offset[r];mass_finite(m,p->cone_scratch[r]);
    }
    grid_barrier();grid_project_cone_blocks(p->cone_scratch,p->affine_cones,p->affine_cone_count);grid_barrier();
    for(int r=grid_rank();r<p->affine_rows;r+=grid_stride()) {
        const int row=p->scalar_rows+r;const double step=p->scaling[p->variables+row];
        p->dual[row]+=step*(p->affine_product[r]-p->cone_scratch[r]+p->affine_offset[r]);mass_finite(m,p->dual[row]);
    }
    grid_barrier();
    // Only retained mass-bound covectors enter the reverse map; no affine d.
    for(int i=grid_rank();i<m->options.node_count;i+=grid_stride()) {
        const int j=m->nodes[i].mass_variable;double w=0.0;
        for(int k=p->a_offsets[j];k<p->a_offsets[j+1];++k)if(mass_row_active(l,m,p->a_indices[k]))w+=p->a[k]*p->dual[p->a_indices[k]];
        m->scan0[i]=w;mass_finite(m,w);
    }
    mass_scan<true>(m);
    for(int i=grid_rank();i<m->options.node_count;i+=grid_stride())m->suffix[i]=m->scan0[i];
    grid_barrier();
    for(int j=grid_rank();j<p->variables;j+=grid_stride())if(mass_active(l,m,j)) {
        double gradient=p->c[j];
        for(int k=p->a_offsets[j];k<p->a_offsets[j+1];++k)if(mass_row_active(l,m,p->a_indices[k]))gradient+=p->a[k]*p->dual[p->a_indices[k]];
        if(p->affine_rows)for(int k=p->f_offsets[j];k<p->f_offsets[j+1];++k)gradient+=p->f[k]*p->dual[p->scalar_rows+p->f_indices[k]];
        if(m->gamma_node[j]>=0)gradient-=m->g[m->gamma_node[j]]*m->suffix[m->gamma_node[j]];
        if(m->virtual_node[j]>=0)gradient+=m->suffix[m->virtual_node[j]];
        const double previous=p->primal[j],step=p->scaling[j],argument=previous-step*gradient,threshold=step*l->lambda[j];
        p->gradient[j]=gradient;p->previous_primal[j]=previous;
        p->primal[j]=l->lambda[j]>0.0?lm::soft_threshold(argument,threshold):argument;
        p->extrapolated_primal[j]=2.0*p->primal[j]-previous;
        mass_finite(m,gradient);mass_finite(m,argument);mass_finite(m,p->primal[j]);mass_finite(m,p->extrapolated_primal[j]);
    }
    grid_barrier();mass_linear_prefix(p,m,p->primal);
    for(int i=grid_rank();i<m->options.node_count;i+=grid_stride()) {
        const auto node=m->nodes[i];const double value=m->constant[i]+m->scan0[i];
        p->primal[node.mass_variable]=p->previous_primal[node.mass_variable]=p->extrapolated_primal[node.mass_variable]=value;
        p->dual[node.equality_row]=-m->suffix[i];mass_finite(m,value);mass_finite(m,p->dual[node.equality_row]);
    }
    grid_barrier();
    // The reduced gradient equals original stationarity after the mass-dual
    // pullback, so pair completion includes the eliminated equality normal.
    for(int i=grid_rank();i<l->options.pair_count;i+=grid_stride()) {
        const auto pair=l->pairs[i];const int v=pair.absolute_variable,t=pair.epigraph_variable;
        const auto z=lm::complete_dual(p->primal[v],p->gradient[v],l->lambda[v]);
        p->primal[t]=p->previous_primal[t]=p->extrapolated_primal[t]=fabs(p->primal[v]);
        p->dual[pair.positive_scalar_row]=z.positive;p->dual[pair.negative_scalar_row]=z.negative;
        mass_finite(m,z.positive);mass_finite(m,z.negative);mass_finite(m,p->primal[t]);
    }
    grid_barrier();
}
__device__ __forceinline__ bool mass_check(DeviceProblem* original,DeviceControl* c,DeviceReport* report,
        L1State* l,MassState* m,volatile int* cancellation,std::uint64_t iteration) {
    grid_evaluate_report<1>(original,c,report,iteration);common_kkt_evaluate<true>(original,iteration);
    if(!grid_rank()) {
        if(!original->common_kkt->result.finite)m->result.finite=0;l->result.finite=m->result.finite;
        if(*cancellation)report->termination=SPACEPDHCG_CUDA_TERMINATION_CANCELLED;
        else if(!m->result.finite)report->termination=SPACEPDHCG_CUDA_TERMINATION_NUMERICAL_FAILURE;
        else if(original->common_kkt->result.passes)report->termination=SPACEPDHCG_CUDA_TERMINATION_OPTIMAL;
    }
    grid_barrier();return report->termination!=SPACEPDHCG_CUDA_TERMINATION_ITERATION_LIMIT;
}
__global__ void cooperative_mass_kernel(DeviceProblem* original,DeviceControl* c,DeviceReport* report,
        volatile int* cancellation,L1State* l,MassState* m) {
    if(!grid_rank()) {
        *report={};report->termination=SPACEPDHCG_CUDA_TERMINATION_ITERATION_LIMIT;report->recovery_stationarity_index=-1;
        original->common_kkt->result={};l->result={};auto& r=l->result;
        r.abi_version=SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION;r.enabled=1;r.valid=m->result.valid;r.finite=m->result.finite;
        r.pairs=l->options.pair_count;r.active_variables=m->result.active_variables;r.active_rows=m->result.active_rows;
        r.retained_variables=m->result.retained_variables;r.retained_rows=m->result.retained_rows;
        r.eta=r.bound_scale=r.objective_scale=1.0;r.minimum_threshold=m->result.minimum_threshold;r.maximum_threshold=m->result.maximum_threshold;
    }
    grid_barrier();
    if(grid_cancelled(original,cancellation)) {
        grid_evaluate_report<1>(original,c,report,0);if(!grid_rank())report->termination=SPACEPDHCG_CUDA_TERMINATION_CANCELLED;return;
    }
    // No canonical reconstruction precedes this original-point check.
    if(mass_check(original,c,report,l,m,cancellation,0))return;
    const unsigned int frequency=c->residual_check_frequency?c->residual_check_frequency:1U;
    for(std::uint64_t iteration=1;iteration<=c->iteration_limit;++iteration) {
        if(grid_cancelled(original,cancellation)) {
            grid_evaluate_report<1>(original,c,report,iteration-1);
            if(!grid_rank())report->termination=SPACEPDHCG_CUDA_TERMINATION_CANCELLED;return;
        }
        mass_update(l->working,l,m);
        if(!grid_rank()){m->result.updates=m->result.completions=iteration;l->result.updates=l->result.completions=iteration;}
        grid_barrier();
        if(iteration==1||iteration%frequency==0||iteration==c->iteration_limit||!m->result.finite)
            if(mass_check(original,c,report,l,m,cancellation,iteration))return;
    }
}
