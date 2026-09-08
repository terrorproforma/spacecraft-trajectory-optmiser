// Experimental small-batch hop operator. Preserve original scan order and bisection.
__device__ bool scan_warp(const double lower,const double upper,const uint32_t samples,
    const Geometry value,const spacepdhcg_orbitweaver_lambert_request& request,
    Root& root,const double* cached) {
    constexpr unsigned mask=0xffffffffU;
    const int lane=threadIdx.x&31;
    double prior_parameter=0,prior_residual=0;int prior_valid=0;
    for(uint64_t base=0;base<=samples;base+=32) {
        const uint64_t sample=base+lane;
        const double parameter=lower+static_cast<double>(sample)/static_cast<double>(samples)*(upper-lower);
        Evaluation current{};
        if(sample<=samples)current=evaluate(parameter,value,request,cached?cached+2*sample:nullptr);
        const double left_parameter=__shfl_up_sync(mask,parameter,1);
        const double left_residual=__shfl_up_sync(mask,current.residual,1);
        const int left_valid=__shfl_up_sync(mask,int(current.valid),1);
        const double previous_parameter=lane?left_parameter:prior_parameter;
        const double previous_residual=lane?left_residual:prior_residual;
        const int previous_valid=lane?left_valid:prior_valid;
        const bool exact=current.valid&&fabs(current.residual)<=request.time_tolerance;
        const bool bracket=current.valid&&previous_valid&&previous_residual*current.residual<0;
        unsigned candidates=__ballot_sync(mask,exact||bracket);
        while(candidates) {
            const int leader=__ffs(candidates)-1;
            Root local{};bool found=false;
            if(lane==leader) {
                if(exact){local={parameter,0U};found=true;}
                else found=bisect(previous_parameter,parameter,value,request,local);
            }
            const int accepted=__shfl_sync(mask,int(found),leader);
            const double selected=__shfl_sync(mask,local.parameter,leader);
            const unsigned iterations=__shfl_sync(mask,local.iterations,leader);
            if(accepted){root={selected,iterations};return true;}
            candidates&=candidates-1;
        }
        prior_parameter=__shfl_sync(mask,parameter,31);
        prior_residual=__shfl_sync(mask,current.residual,31);
        prior_valid=__shfl_sync(mask,int(current.valid),31);
    }
    return false;
}

__global__ void hop_warp_kernel(const spacepdhcg_orbitweaver_hop_request* requests,
    size_t count,uint32_t samples,spacepdhcg_orbitweaver_hop_result* results,const double* cached) {
    const size_t i=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)/32;
    const int lane=threadIdx.x&31;
    if(i>=count)return;
    const auto& hop=requests[i];auto& output=results[i];
    if(lane==0) {
        output={};output.departure_delta_v=output.arrival_delta_v=INFINITY;
        for(int k=0;k<3;++k)output.departure_velocity[k]=output.arrival_velocity[k]=NAN;
    }
    auto request=hop.lambert;
    request.include_short_way=request.include_long_way=1;request.maximum_revolutions=0;
    if(!valid(request)||!isfinite(hop.departure_allowance)||!isfinite(hop.arrival_allowance)
        ||hop.departure_allowance<0||hop.arrival_allowance<0)return;
    for(int k=0;k<3;++k)
        if(!isfinite(hop.departure_body_velocity[k])||!isfinite(hop.arrival_body_velocity[k]))return;
    for(int direction=0;direction<2;++direction) {
        const auto value=geometry(request,direction!=0);
        if(!value.valid)continue;
        Root root{};
        if(!scan_warp(-4*pi*pi,4*pi*pi-1e-8,samples,value,request,root,cached))continue;
        if(lane==0) {
            spacepdhcg_orbitweaver_lambert_result candidate{};
            if(solution(request,value,root,candidate)) {
                double dep2=0,arr2=0;
                for(int k=0;k<3;++k) {
                    const double dep=candidate.departure_velocity[k]-hop.departure_body_velocity[k];
                    const double arr=candidate.arrival_velocity[k]-hop.arrival_body_velocity[k];
                    dep2+=dep*dep;arr2+=arr*arr;
                }
                const double dep=fmax(sqrt(dep2)-hop.departure_allowance,0.0);
                const double arr=fmax(sqrt(arr2)-hop.arrival_allowance,0.0);
                if(isfinite(dep)&&isfinite(arr)&&dep+arr<output.departure_delta_v+output.arrival_delta_v) {
                    output.feasible=1;output.long_way=direction;
                    output.departure_delta_v=dep;output.arrival_delta_v=arr;
                    for(int k=0;k<3;++k){output.departure_velocity[k]=candidate.departure_velocity[k];output.arrival_velocity[k]=candidate.arrival_velocity[k];}
                }
            }
        }
    }
}
void launch_hops(const spacepdhcg_orbitweaver_hop_request* requests,size_t count,
    uint32_t samples,spacepdhcg_orbitweaver_hop_result* results,const double* cached,cudaStream_t stream) {
    if(count<=1024)hop_warp_kernel<<<static_cast<unsigned>((count+3)/4),128,0,stream>>>(requests,count,samples,results,cached);
    else hop_kernel<<<static_cast<unsigned>((count+63)/64),64,0,stream>>>(requests,count,samples,results,cached);
}
