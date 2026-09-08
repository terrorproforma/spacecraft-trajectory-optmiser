// Included in gtoc12_joint.cu: uses the retained workspace and shared evaluator.
namespace {
__global__ void generate_insertions(int count, int n, int camp, const Policy* policy,
    const int32_t* slots, const double* base_arr, const double* base_dep,
    double* arrivals, double* departures, uint8_t* enabled, int split_points=1) {
    const size_t row=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(row>=size_t(count))return;
    const int variants=4*split_points*split_points;
    const int i=slots[2*(row/variants)],k=slots[2*(row/variants)+1],seed=int(row%4);
    const double gap_d=base_arr[i+1]-base_dep[i],gap_c=base_arr[k+1]-base_dep[k];
    const double dwell=base_dep[camp]-base_arr[camp];
    const double slack=fmax(0.0,dwell-policy->minimum_stay-5.0);
    double lend_d=0.0,lend_c=0.0;
    if(seed) {
        const double share_d=seed==1?0.5:seed==2?1.0:0.0;
        const double share_c=seed==1?0.5:seed==2?0.0:1.0;
        lend_d=fmin(0.5*gap_d,share_d*slack);
        lend_c=fmin(0.5*gap_c,share_c*slack);
    }
    enabled[row]=!seed || lend_d+lend_c>1.0;
    // Grid offsets preserve the existing midpoint arithmetic exactly. Splits
    // move only the inserted visits; borrowed-time shifts remain unchanged.
    const int grid=int(row%variants)/4;
    const double fd=double(grid/split_points+1)/(split_points+1);
    const double fc=double(grid%split_points+1)/(split_points+1);
    const double td=base_dep[i]+0.5*(gap_d+lend_d)+(fd-0.5)*(gap_d+lend_d);
    const double tc=base_dep[k]+0.5*(gap_c-lend_c)+(fc-0.5)*(gap_c+lend_c);
    double* arr=arrivals+row*n;double* dep=departures+row*n;
    for(int j=0;j<n;++j) {
        if(j==i+1){arr[j]=dep[j]=td;continue;}
        if(j==k+2){arr[j]=dep[j]=tc;continue;}
        const int old=j-(j>i+1)-(j>k+2);
        arr[j]=base_arr[old];dep[j]=base_dep[old];
        // Scalar construction moves the collection side before deployment.
        if(old>=camp && old<=k) {
            if(old!=camp) arr[j]-=lend_c;
            dep[j]-=lend_c;
        }
        if(old>i && old<=camp) {
            arr[j]+=lend_d;
            if(old!=camp)dep[j]+=lend_d;
        }
    }
}
} // namespace

extern "C" int spacepdhcg_gtoc12_joint_insertions_host(void* opaque,
    int32_t layouts,int32_t camp,const Policy* policy,const Visit* visits,const Stage* stages,
    const int32_t* slots,const double* base_arr,const double* base_dep,
    const spacepdhcg_orbitweaver_hop_elements* elements,const CachedCost* records,
    int32_t record_count,const int32_t* offsets,Result* results,uint8_t* enabled,
    double* masses,double* inflations,double* proxies,double* collected,
    double* arrivals,double* departures,GeometryStats* stats) {
    auto* w=static_cast<Workspace*>(opaque);
    if(!correct_device(w)||layouts<0||layouts>w->capacity/4||!stats||record_count<0)return 1;
    std::unique_lock<std::mutex> lock(w->mutex,std::try_to_lock);
    if(!lock.owns_lock())return 3;
    if(!layouts){*stats={};return 0;}
    const int n=w->n,base_n=n-2,count=4*layouts;
    if(base_n<4||camp<2||camp>=base_n-1||!slots||!base_arr||!base_dep
        ||!elements||!offsets||!enabled||!visits||!stages||!results
        ||(record_count&&!records)||offsets[0]!=0||offsets[layouts]!=record_count)return 1;
    for(int j=0;j<base_n;++j)
        if(!std::isfinite(base_arr[j])||!std::isfinite(base_dep[j])
            ||std::abs(base_arr[j])>std::numeric_limits<double>::max()/16
            ||std::abs(base_dep[j])>std::numeric_limits<double>::max()/16)return 1;
    for(int l=0;l<layouts;++l) {
        if(slots[2*l]<1||slots[2*l]>=camp||slots[2*l+1]<camp||slots[2*l+1]>=base_n-1
            ||offsets[l]<0||offsets[l+1]<offsets[l]||offsets[l+1]>record_count)return 1;
        const int status=validate(n,0,policy,visits+size_t(l)*n,stages+size_t(l)*(n-1),
            base_arr,base_dep,nullptr,results,false);
        if(status)return status;
        for(int j=0;j<n-1;++j) {
            const auto& e=elements[size_t(l)*(n-1)+j];
            if(!valid_orbit(e.departure)||!valid_orbit(e.arrival)
                ||!std::isfinite(e.gravitational_parameter)||e.gravitational_parameter<=0
                ||!std::isfinite(e.departure_allowance)||e.departure_allowance<0
                ||!std::isfinite(e.arrival_allowance)||e.arrival_allowance<0)return 1;
        }
        for(int j=offsets[l];j<offsets[l+1];++j) {
            const auto& r=records[j];
            if(!flag(r.cached)||!flag(r.value.measured)||r.value.reserved)return 4;
            if(r.leg<0||r.leg>=n-1||!std::isfinite(r.departure)||!std::isfinite(r.arrival)
                ||(j>offsets[l]&&!key_before(records[j-1],r))
                ||(r.value.measured&&(!std::isfinite(r.value.measured_delta_v)
                    ||!std::isfinite(r.value.measured_mass))))return 1;
        }
    }
    const size_t cap=size_t(w->capacity),lc=cap/4,legs=size_t(count)*(n-1),epochs=size_t(count)*n;
    if(legs>UINT32_MAX)return 1;
    if((!w->insertion_visits&&!allocate(w->insertion_visits,lc*n))
        ||(!w->insertion_stages&&!allocate(w->insertion_stages,lc*(n-1)))
        ||(!w->insertion_elements&&!allocate(w->insertion_elements,lc*(n-1)))
        ||(!w->insertion_slots&&!allocate(w->insertion_slots,2*lc))
        ||(!w->insertion_offsets&&!allocate(w->insertion_offsets,lc+1))
        ||(!w->insertion_enabled&&!allocate(w->insertion_enabled,cap))
        ||(!w->base_arrivals&&!allocate(w->base_arrivals,size_t(n)))
        ||(!w->base_departures&&!allocate(w->base_departures,size_t(n)))
        ||(!w->hop_requests&&!allocate(w->hop_requests,cap*(n-1)))
        ||(!w->hop_results&&!allocate(w->hop_results,cap*(n-1)))
        ||(!w->geometry_stats&&!allocate(w->geometry_stats,1)))return 2;
    if(record_count>w->cached_capacity) {
        CachedCost* next=nullptr;
        if(!allocate(next,size_t(record_count)))return 2;
        if(w->cached_costs&&cudaFree(w->cached_costs)!=cudaSuccess){cudaFree(next);return 2;}
        w->cached_costs=next;w->cached_capacity=record_count;
    }
    const auto failed=[&](){cudaStreamSynchronize(w->stream);return 2;};
    if(!upload(w->policy,policy,1,w->stream)
        ||!upload(w->insertion_visits,visits,size_t(layouts)*n,w->stream)
        ||!upload(w->insertion_stages,stages,size_t(layouts)*(n-1),w->stream)
        ||!upload(w->insertion_elements,elements,size_t(layouts)*(n-1),w->stream)
        ||!upload(w->insertion_slots,slots,2*size_t(layouts),w->stream)
        ||!upload(w->insertion_offsets,offsets,size_t(layouts)+1,w->stream)
        ||!upload(w->base_arrivals,base_arr,size_t(base_n),w->stream)
        ||!upload(w->base_departures,base_dep,size_t(base_n),w->stream)
        ||(record_count&&!upload(w->cached_costs,records,size_t(record_count),w->stream))
        ||cudaMemsetAsync(w->geometry_stats,0,sizeof(GeometryStats),w->stream)!=cudaSuccess)return failed();
    const unsigned blocks=unsigned((size_t(count)+127)/128);
    generate_insertions<<<blocks,128,0,w->stream>>>(count,n,camp,w->policy,w->insertion_slots,
        w->base_arrivals,w->base_departures,w->arrivals,w->departures,w->insertion_enabled);
    if(cudaGetLastError()!=cudaSuccess)return failed();
    clear_geometry_costs<<<unsigned((legs+127)/128),128,0,w->stream>>>(legs,w->costs);
    if(cudaGetLastError()!=cudaSuccess)return failed();
    const auto evaluate=[&](){evaluate_candidates<<<blocks,128,0,w->stream>>>(count,n,w->policy,
        w->insertion_visits,w->insertion_stages,w->arrivals,w->departures,w->costs,w->results,
        w->masses,w->inflations,w->proxies,w->collected,4,w->insertion_enabled);
        return cudaGetLastError();};
    if(evaluate()!=cudaSuccess)return failed();
    if(spacepdhcg_joint_geometry_launch(count,n,w->insertion_elements,w->arrivals,w->departures,
        w->results,w->cached_costs,record_count,w->costs,w->hop_requests,w->hop_results,
        w->geometry_stats,w->stream,4,w->insertion_offsets)!=cudaSuccess)return failed();
    if(evaluate()!=cudaSuccess)return failed();
    if(!download(results,w->results,size_t(count),w->stream)
        ||!download(enabled,w->insertion_enabled,size_t(count),w->stream)
        ||!download(masses,w->masses,legs,w->stream)||!download(inflations,w->inflations,legs,w->stream)
        ||!download(proxies,w->proxies,legs,w->stream)||!download(collected,w->collected,epochs,w->stream)
        ||!download(arrivals,w->arrivals,epochs,w->stream)||!download(departures,w->departures,epochs,w->stream)
        ||!download(stats,w->geometry_stats,1,w->stream))return failed();
    return cudaStreamSynchronize(w->stream)==cudaSuccess?0:2;
}
