// SPDX-License-Identifier: Apache-2.0
// Included after the private problem/control/report types. Optional diagnostic
// reductions never borrow or modify the iteration's products or gradient.
namespace ck=spacepdhcg::cuda::common_kkt;
struct CommonPartial {
    ck::Sum quadratic{}, linear{}, support{};
    double primal{}, equation{}, primal_scale{}, dual{}, dual_scale{};
    double primal_cone{}, dual_cone{}, block_complementarity{};
    int finite{1};
};
struct CommonKktState {
    spacepdhcg_cuda_common_kkt_options options{};
    spacepdhcg_cuda_common_kkt_diagnostics result{};
    const unsigned long long* row_keys{};
    const int* row_positions{};
    const int* row_offsets{};
    ck::Sum* row_products{};
    double* slack{};
    CommonPartial* partials{};
    unsigned long long begin_cycles{};
};
__device__ bool common_kkt_enabled(const DeviceProblem* p) {
    return p->common_kkt && p->common_kkt->options.enabled;
}
template<bool Grid> __device__ int common_rank() {
    if constexpr(Grid)return blockIdx.x*blockDim.x+threadIdx.x;
    else return threadIdx.x;
}
template<bool Grid> __device__ int common_stride() {
    if constexpr(Grid)return blockDim.x*gridDim.x;
    else return blockDim.x;
}
template<bool Grid> __device__ void common_barrier() {
    if constexpr(Grid)cooperative_groups::this_grid().sync();
    else __syncthreads();
}
__device__ void common_max(double& destination,double v,int& finite) {
    if(!isfinite(v))finite=0;
    destination=fmax(destination,v);
}
__device__ CommonPartial common_merge(CommonPartial a,const CommonPartial& b) {
    a.quadratic=ck::add(a.quadratic,b.quadratic);
    a.linear=ck::add(a.linear,b.linear);a.support=ck::add(a.support,b.support);
    a.finite=a.finite && b.finite;
    common_max(a.primal,b.primal,a.finite);common_max(a.equation,b.equation,a.finite);
    common_max(a.primal_scale,b.primal_scale,a.finite);common_max(a.dual,b.dual,a.finite);
    common_max(a.dual_scale,b.dual_scale,a.finite);common_max(a.primal_cone,b.primal_cone,a.finite);
    common_max(a.dual_cone,b.dual_cone,a.finite);common_max(a.block_complementarity,b.block_complementarity,a.finite);
    return a;
}
template<bool Grid> __device__ void common_kkt_evaluate(DeviceProblem* p,std::uint64_t iteration) {
    auto* k=p->common_kkt;
    const int rank=common_rank<Grid>(),stride=common_stride<Grid>();
    if(rank==0)k->begin_cycles=clock64();
    common_barrier<Grid>();
    CommonPartial part{};
    const int equalities=k->options.equality_rows;
    // CSC transpose gathers distinguish Aeq^T y from the combined original G^T z.
    for(int j=rank;j<p->variables;j+=stride) {
        ck::Sum qx{},aty{},gtz{};
        for(int t=p->q_offsets[j];t<p->q_offsets[j+1];++t)
            qx=ck::add(qx,ck::product(p->q[t],p->primal[p->q_indices[t]]));
        for(int t=p->a_offsets[j];t<p->a_offsets[j+1];++t) {
            const int r=p->a_indices[t];const auto v=ck::product(p->a[t],p->dual[r]);
            if(r<equalities)aty=ck::add(aty,v);else gtz=ck::add(gtz,v);
        }
        if(p->affine_rows)for(int t=p->f_offsets[j];t<p->f_offsets[j+1];++t)
            gtz=ck::add(gtz,ck::product(p->f[t],p->dual[p->scalar_rows+p->f_indices[t]]));
        const auto stationarity=ck::add(ck::add(qx,{p->c[j],0}),ck::add(aty,gtz));
        part.finite=part.finite && ck::finite(qx) && ck::finite(aty) && ck::finite(gtz)
            && ck::finite(stationarity) && isfinite(p->primal[j]);
        common_max(part.dual,ck::absolute(stationarity),part.finite);
        common_max(part.dual_scale,ck::absolute(qx),part.finite);
        common_max(part.dual_scale,fabs(p->c[j]),part.finite);
        common_max(part.dual_scale,ck::absolute(aty),part.finite);
        common_max(part.dual_scale,ck::absolute(gtz),part.finite);
        part.quadratic=ck::add(part.quadratic,ck::multiply(qx,p->primal[j]));
        part.linear=ck::add(part.linear,ck::product(p->c[j],p->primal[j]));
    }
    // Retained CSR entry maps give deterministic row gathers without atomics.
    for(int r=rank;r<p->scalar_rows+p->affine_rows;r+=stride) {
        ck::Sum product{};
        for(int t=k->row_offsets[r];t<k->row_offsets[r+1];++t) {
            const int pos=k->row_positions[t];
            const double coefficient=pos<p->a_nonzeros?p->a[pos]:p->f[pos-p->a_nonzeros];
            product=ck::add(product,ck::product(coefficient,p->primal[static_cast<unsigned int>(k->row_keys[t])]));
        }
        k->row_products[r]=product;
        part.finite=part.finite && ck::finite(product) && isfinite(p->dual[r]);
        const bool affine=r>=p->scalar_rows;
        const double rhs=affine?p->affine_offset[r-p->scalar_rows]:p->scalar_upper[r];
        const ck::Sum gx=affine?ck::negate(product):product;
        common_max(part.primal_scale,ck::absolute(gx),part.finite);
        common_max(part.primal_scale,fabs(rhs),part.finite);
        if(r<equalities) {
            common_max(part.primal,ck::absolute(ck::add(product,{-rhs,0})),part.finite);
            k->slack[r]=0;
        } else {
            const double slack=ck::value(ck::add({rhs,0},ck::negate(gx)));
            k->slack[r]=slack;
            const double equation=ck::absolute(ck::add(ck::add(gx,{slack,0}),{-rhs,0}));
            common_max(part.equation,equation,part.finite);common_max(part.primal,equation,part.finite);
            common_max(part.primal_scale,fabs(slack),part.finite);
            if(!affine) {
                common_max(part.primal_cone,-slack,part.finite);common_max(part.dual_cone,-p->dual[r],part.finite);
                common_max(part.block_complementarity,ck::absolute(ck::product(slack,p->dual[r])),part.finite);
            }
        }
        part.support=ck::add(part.support,ck::product(affine?rhs:-rhs,p->dual[r]));
    }
    common_barrier<Grid>();
    for(int cone_index=rank;cone_index<p->affine_cone_count;cone_index+=stride) {
        const auto cone=p->affine_cones[cone_index];
        const int first=p->scalar_rows+cone.start,last=first+cone.vector_dimension+1;
        ck::Sum snorm{},znorm{},dot{};
        for(int r=first;r<=last;++r) {
            dot=ck::add(dot,ck::product(k->slack[r],p->dual[r]));
            if(r!=last) {
                snorm=ck::add(snorm,ck::product(k->slack[r],k->slack[r]));
                znorm=ck::add(znorm,ck::product(p->dual[r],p->dual[r]));
            }
        }
        part.finite=part.finite && ck::finite(snorm) && ck::finite(znorm) && ck::finite(dot);
        common_max(part.primal_cone,ck::value(ck::add(ck::square_root(snorm),{-k->slack[last],0})),part.finite);
        common_max(part.dual_cone,ck::value(ck::add(ck::square_root(znorm),{p->dual[last],0})),part.finite);
        common_max(part.block_complementarity,ck::absolute(dot),part.finite);
    }
    part.finite=part.finite && ck::finite(part.quadratic) && ck::finite(part.linear) && ck::finite(part.support);
    k->partials[rank]=part;
    common_barrier<Grid>();
    for(int step=1;step<stride;step*=2) {
        if(rank%(2*step)==0 && rank+step<stride)k->partials[rank]=common_merge(k->partials[rank],k->partials[rank+step]);
        common_barrier<Grid>();
    }
    if(rank==0) {
        const auto a=k->partials[0];auto& result=k->result;
        const auto half_q=ck::multiply(a.quadratic,0.5);
        const auto primal=ck::add(half_q,a.linear),dual=ck::add(ck::negate(half_q),a.support);
        const double scale=fmax(1.0,fmax(ck::absolute(primal),ck::absolute(dual)));
        result.abi_version=SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION;result.enabled=1;result.valid=1;
        result.equality_rows=equalities;++result.evaluations;result.evaluated_iteration=iteration;
        result.primal_absolute=a.primal;result.conic_equation_absolute=a.equation;result.dual_absolute=a.dual;
        result.primal_relative=a.primal/(1.0+a.primal_scale);result.dual_relative=a.dual/(1.0+a.dual_scale);
        result.gap_relative=ck::absolute(ck::add(primal,ck::negate(dual)))/scale;
        result.block_complementarity_relative=a.block_complementarity/scale;
        result.primal_cone_violation=a.primal_cone;result.dual_cone_violation=a.dual_cone;
        result.objective=ck::value(primal);result.dual_objective=ck::value(dual);
        result.finite=a.finite && ck::finite(primal) && ck::finite(dual) && isfinite(scale)
            && isfinite(result.primal_relative) && isfinite(result.dual_relative) && isfinite(result.gap_relative)
            && isfinite(result.block_complementarity_relative);
        result.passes=result.finite && result.primal_relative<=k->options.relative_tolerance
            && result.dual_relative<=k->options.relative_tolerance && result.gap_relative<=k->options.relative_tolerance
            && result.block_complementarity_relative<=k->options.relative_tolerance
            && result.primal_cone_violation<=k->options.cone_tolerance && result.dual_cone_violation<=k->options.cone_tolerance;
        result.evaluation_clock_cycles+=clock64()-k->begin_cycles;
    }
    common_barrier<Grid>();
}

__global__ void common_validate_domain(const DeviceProblem* p,int equalities,int* invalid) {
    const int rank=blockIdx.x*blockDim.x+threadIdx.x,stride=blockDim.x*gridDim.x;
    for(int j=rank;j<p->variables;j+=stride) {
        if(p->variable_lower[j]!=-CUDART_INF || p->variable_upper[j]!=CUDART_INF || !isfinite(p->c[j]))atomicExch(invalid,1);
        if(p->q_offsets[j]<0 || p->q_offsets[j]>p->q_offsets[j+1] || p->q_offsets[j+1]>p->q_nonzeros
           || p->a_offsets[j]<0 || p->a_offsets[j]>p->a_offsets[j+1] || p->a_offsets[j+1]>p->a_nonzeros)atomicExch(invalid,1);
        if(p->affine_rows && (p->f_offsets[j]<0 || p->f_offsets[j]>p->f_offsets[j+1] || p->f_offsets[j+1]>p->f_nonzeros))atomicExch(invalid,1);
    }
    for(int r=rank;r<p->scalar_rows;r+=stride) {
        if(!isfinite(p->scalar_upper[r]) || (r<equalities?p->scalar_lower[r]!=p->scalar_upper[r]:p->scalar_lower[r]!=-CUDART_INF))atomicExch(invalid,1);
    }
    for(int r=rank;r<p->affine_rows;r+=stride)if(!isfinite(p->affine_offset[r]))atomicExch(invalid,1);
    for(int i=rank;i<p->q_nonzeros;i+=stride)if(!isfinite(p->q[i]) || p->q_indices[i]<0 || p->q_indices[i]>=p->variables)atomicExch(invalid,1);
    for(int i=rank;i<p->a_nonzeros;i+=stride)if(!isfinite(p->a[i]) || p->a_indices[i]<0 || p->a_indices[i]>=p->scalar_rows)atomicExch(invalid,1);
    for(int i=rank;i<p->f_nonzeros;i+=stride)if(!isfinite(p->f[i]) || p->f_indices[i]<0 || p->f_indices[i]>=p->affine_rows)atomicExch(invalid,1);
    for(int i=rank;i<p->affine_cone_count;i+=stride) {
        const auto c=p->affine_cones[i];const int expected=i?p->affine_cones[i-1].start+p->affine_cones[i-1].vector_dimension+2:0;
        if(c.kind!=SPACEPDHCG_CUDA_CONE_SECOND_ORDER || c.start!=expected || c.vector_dimension<=0
            || (i==p->affine_cone_count-1 && c.start+c.vector_dimension+2!=p->affine_rows))atomicExch(invalid,1);
    }
    if(rank==0 && (p->variable_cone_count || (p->affine_rows && !p->affine_cone_count)
        || p->q_offsets[0] || p->q_offsets[p->variables]!=p->q_nonzeros
        || p->a_offsets[0] || p->a_offsets[p->variables]!=p->a_nonzeros
        || (p->affine_rows && (p->f_offsets[0] || p->f_offsets[p->variables]!=p->f_nonzeros))))atomicExch(invalid,1);
}
__global__ void common_entry_keys(const DeviceProblem* p,unsigned long long* keys,int* positions) {
    for(int j=blockIdx.x*blockDim.x+threadIdx.x;j<p->variables;j+=blockDim.x*gridDim.x) {
        for(int t=p->a_offsets[j];t<p->a_offsets[j+1];++t) {keys[t]=(static_cast<unsigned long long>(p->a_indices[t])<<32)|static_cast<unsigned int>(j);positions[t]=t;}
        if(p->affine_rows)for(int t=p->f_offsets[j];t<p->f_offsets[j+1];++t) {
            const int index=p->a_nonzeros+t;keys[index]=(static_cast<unsigned long long>(p->scalar_rows+p->f_indices[t])<<32)|static_cast<unsigned int>(j);positions[index]=index;
        }
    }
}
__global__ void common_row_offsets(const unsigned long long* keys,int count,int rows,int* offsets) {
    for(int r=blockIdx.x*blockDim.x+threadIdx.x;r<=rows;r+=blockDim.x*gridDim.x) {
        int first=0,last=count;const auto target=static_cast<unsigned long long>(r)<<32;
        while(first<last) {const int mid=first+(last-first)/2;if(keys[mid]<target)first=mid+1;else last=mid;}
        offsets[r]=first;
    }
}
__global__ void common_bind(DeviceProblem* p,CommonKktState* state) {if(threadIdx.x==0)p->common_kkt=state;}
