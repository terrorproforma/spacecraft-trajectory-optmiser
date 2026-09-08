// Compact immutable inputs for device construction of insertion layouts.
namespace {
struct PreparedInsertion {
    int candidates=0,camp=0,edges=0,record_count=0;
    double dwell_limit=0;
    Policy* policy=nullptr;
    Visit* visits=nullptr;
    double *arrivals=nullptr,*departures=nullptr,*weights=nullptr;
    Stage* stages=nullptr;
    spacepdhcg_orbitweaver_hop_elements* elements=nullptr;
    CachedCost* records=nullptr;
};
bool release_prepared_insertion(PreparedInsertion* s) {
    if(!s)return true;
    bool ok=true;
    const auto free=[&](void* p){if(p&&cudaFree(p)!=cudaSuccess)ok=false;};
    free(s->policy);free(s->visits);free(s->arrivals);free(s->departures);
    free(s->weights);free(s->stages);free(s->elements);free(s->records);
    delete s;return ok;
}
__global__ void generate_insertion_layouts(int layouts,int n,int camp,int64_t first,
    const Visit* original,const double* weights,double dwell,const Stage* edges,
    Visit* visits,Stage* stages,int32_t* slots,int32_t* edge_ids) {
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=size_t(layouts)*n)return;
    const int row=int(index/n),j=int(index%n),base=n-2,D=camp-1,K=base-1-camp;
    const int64_t layout=first+row;
    const int asteroid=int(layout/(int64_t(D)*K));
    const int i=1+int(layout/K%D),k=camp+int(layout%K);
    if(!j){slots[2*row]=i;slots[2*row+1]=k;}
    Visit v{};
    if(j==i+1 || j==k+2) {
        v.deploy=j==i+1;v.collect=j==k+2;v.donor=i+1;
        v.foreign_epoch=v.pinned_arrival=NAN;v.dwell_limit=dwell;v.weight=weights[asteroid];
    } else {
        const int old=j-(j>i+1)-(j>k+2);
        v=original[old];
        if(v.donor>=0)v.donor+=(v.donor>i)+(v.donor>k);
    }
    visits[index]=v;
    if(j<n-1) {
        const int first_edge=base-1+asteroid*2*(base-2);
        int edge;
        if(j==i)edge=first_edge+2*(i-1);
        else if(j==i+1)edge=first_edge+2*(i-1)+1;
        else if(j==k+1)edge=first_edge+2*D+2*(k-camp);
        else if(j==k+2)edge=first_edge+2*D+2*(k-camp)+1;
        else edge=j-(j>i+1)-(j>k+2);
        edge_ids[size_t(row)*(n-1)+j]=edge;
        stages[size_t(row)*(n-1)+j]=edges[edge];
    }
}
} // namespace

extern "C" int spacepdhcg_gtoc12_joint_prepare_insertions_host(void* opaque,
    int32_t candidates,int32_t camp,const Policy* policy,const Visit* visits,
    const double* arr,const double* dep,const double* weights,double dwell,
    const Stage* stages,const spacepdhcg_orbitweaver_hop_elements* elements,
    const CachedCost* records,int32_t record_count) {
    auto* w=static_cast<Workspace*>(opaque);
    if(!correct_device(w)||candidates<1||record_count<0)return 1;
    std::unique_lock<std::mutex> lock(w->mutex,std::try_to_lock);
    if(!lock.owns_lock())return 3;
    const int base=w->n-2;
    const int64_t edge_count=int64_t(base)-1+int64_t(candidates)*2*(int64_t(base)-2);
    if(base<4||camp<2||camp>=base-1||edge_count>INT32_MAX||!weights
        ||!elements||!nonnegative_bound(dwell)||(record_count&&!records))return 1;
    Result ignored{};
    const int status=validate(base,1,policy,visits,stages,arr,dep,nullptr,&ignored,false);
    if(status)return status;
    for(int j=0;j<base;++j)
        if(std::abs(arr[j])>std::numeric_limits<double>::max()/16
            ||std::abs(dep[j])>std::numeric_limits<double>::max()/16)return 1;
    for(int j=0;j<candidates;++j)if(!std::isfinite(weights[j]))return 1;
    for(int j=0;j<int(edge_count);++j) {
        const Stage& s=stages[j];const auto& e=elements[j];
        if(s.model<0||s.model>2||!flag(s.earth_out)||s.reserved0||s.reserved1)return 4;
        if(!std::isfinite(s.tof_min)||!nonnegative_bound(s.tof_max)||s.tof_max<s.tof_min
            ||!nonnegative_bound(s.ratio_limit)||!std::isfinite(s.flat)||!std::isfinite(s.floor)
            ||!std::isfinite(s.slope)||!std::isfinite(s.calibration)
            ||!valid_orbit(e.departure)||!valid_orbit(e.arrival)
            ||!std::isfinite(e.gravitational_parameter)||e.gravitational_parameter<=0
            ||!std::isfinite(e.departure_allowance)||e.departure_allowance<0
            ||!std::isfinite(e.arrival_allowance)||e.arrival_allowance<0)return 1;
    }
    for(int j=0;j<record_count;++j) {
        const auto& r=records[j];
        if(!flag(r.cached)||!flag(r.value.measured)||r.value.reserved)return 4;
        if(r.leg<0||r.leg>=edge_count||!std::isfinite(r.departure)||!std::isfinite(r.arrival)
            ||(j&&!key_before(records[j-1],r))
            ||(r.value.measured&&(!std::isfinite(r.value.measured_delta_v)
                ||!std::isfinite(r.value.measured_mass))))return 1;
    }
    auto* s=new(std::nothrow) PreparedInsertion;
    if(!s)return 2;
    s->candidates=candidates;s->camp=camp;s->edges=int(edge_count);
    s->record_count=record_count;s->dwell_limit=dwell;
    const auto failed=[&](){cudaStreamSynchronize(w->stream);release_prepared_insertion(s);return 2;};
    if(!allocate(s->policy,1)||!allocate(s->visits,size_t(base))
        ||!allocate(s->arrivals,size_t(base))||!allocate(s->departures,size_t(base))
        ||!allocate(s->weights,size_t(candidates))||!allocate(s->stages,size_t(edge_count))
        ||!allocate(s->elements,size_t(edge_count))
        ||(record_count&&!allocate(s->records,size_t(record_count))))return failed();
    if(!upload(s->policy,policy,1,w->stream)||!upload(s->visits,visits,size_t(base),w->stream)
        ||!upload(s->arrivals,arr,size_t(base),w->stream)||!upload(s->departures,dep,size_t(base),w->stream)
        ||!upload(s->weights,weights,size_t(candidates),w->stream)
        ||!upload(s->stages,stages,size_t(edge_count),w->stream)
        ||!upload(s->elements,elements,size_t(edge_count),w->stream)
        ||(record_count&&!upload(s->records,records,size_t(record_count),w->stream))
        ||cudaStreamSynchronize(w->stream)!=cudaSuccess)return failed();
    auto* previous=w->prepared_insertion;w->prepared_insertion=s;
    return release_prepared_insertion(previous)?0:2;
}

extern "C" int spacepdhcg_gtoc12_joint_prepared_insertion_grid_host(void* opaque,
    int64_t first,int32_t layouts,int32_t split_points,Result* results,uint8_t* enabled,
    double* masses,double* inflations,double* proxies,double* collected,
    double* arrivals,double* departures,GeometryStats* stats,Visit* generated_visits,int32_t* edge_ids) {
    auto* w=static_cast<Workspace*>(opaque);
    if(!correct_device(w)||!stats||first<0||layouts<0
        ||split_points<1||split_points>9||split_points%2!=1)return 1;
    const int variants=4*split_points*split_points;
    if(layouts>w->capacity/variants)return 1;
    std::unique_lock<std::mutex> lock(w->mutex,std::try_to_lock);
    if(!lock.owns_lock())return 3;
    const auto* s=w->prepared_insertion;if(!s)return 1;
    const int n=w->n,base=n-2,count=variants*layouts;
    const int64_t total=int64_t(s->candidates)*(s->camp-1)*(base-1-s->camp);
    if(first>total||layouts>total-first||(!results&&layouts)||(!enabled&&layouts))return 1;
    if(!layouts){*stats={};return 0;}
    const size_t cap=size_t(w->capacity),lc=cap/4,legs=size_t(count)*(n-1),epochs=size_t(count)*n;
    if(legs>UINT32_MAX)return 1;
    if((!w->insertion_visits&&!allocate(w->insertion_visits,lc*n))
        ||(!w->insertion_stages&&!allocate(w->insertion_stages,lc*(n-1)))
        ||(!w->insertion_slots&&!allocate(w->insertion_slots,2*lc))
        ||(!w->insertion_edge_ids&&!allocate(w->insertion_edge_ids,lc*(n-1)))
        ||(!w->insertion_enabled&&!allocate(w->insertion_enabled,cap))
        ||(!w->hop_requests&&!allocate(w->hop_requests,cap*(n-1)))
        ||(!w->hop_results&&!allocate(w->hop_results,cap*(n-1)))
        ||(!w->geometry_stats&&!allocate(w->geometry_stats,1)))return 2;
    const auto failed=[&](){cudaStreamSynchronize(w->stream);return 2;};
    if(cudaMemsetAsync(w->geometry_stats,0,sizeof(GeometryStats),w->stream)!=cudaSuccess)return failed();
    generate_insertion_layouts<<<unsigned((size_t(layouts)*n+127)/128),128,0,w->stream>>>(layouts,n,s->camp,first,
        s->visits,s->weights,s->dwell_limit,s->stages,w->insertion_visits,w->insertion_stages,
        w->insertion_slots,w->insertion_edge_ids);
    if(cudaGetLastError()!=cudaSuccess)return failed();
    const unsigned blocks=unsigned((size_t(count)+127)/128);
    generate_insertions<<<blocks,128,0,w->stream>>>(count,n,s->camp,s->policy,w->insertion_slots,
        s->arrivals,s->departures,w->arrivals,w->departures,w->insertion_enabled,split_points);
    if(cudaGetLastError()!=cudaSuccess)return failed();
    clear_geometry_costs<<<unsigned((legs+127)/128),128,0,w->stream>>>(legs,w->costs);
    if(cudaGetLastError()!=cudaSuccess)return failed();
    const auto evaluate=[&](){evaluate_candidates<<<blocks,128,0,w->stream>>>(count,n,s->policy,
        w->insertion_visits,w->insertion_stages,w->arrivals,w->departures,w->costs,w->results,
        w->masses,w->inflations,w->proxies,w->collected,variants,w->insertion_enabled);return cudaGetLastError();};
    if(evaluate()!=cudaSuccess)return failed();
    if(spacepdhcg_joint_geometry_launch(count,n,s->elements,w->arrivals,w->departures,w->results,
        s->records,s->record_count,w->costs,w->hop_requests,w->hop_results,w->geometry_stats,w->stream,
        variants,nullptr,w->insertion_edge_ids)!=cudaSuccess||evaluate()!=cudaSuccess)return failed();
    if(!download(results,w->results,size_t(count),w->stream)||!download(enabled,w->insertion_enabled,size_t(count),w->stream)
        ||!download(masses,w->masses,legs,w->stream)||!download(inflations,w->inflations,legs,w->stream)
        ||!download(proxies,w->proxies,legs,w->stream)||!download(collected,w->collected,epochs,w->stream)
        ||!download(arrivals,w->arrivals,epochs,w->stream)||!download(departures,w->departures,epochs,w->stream)
        ||!download(stats,w->geometry_stats,1,w->stream)
        ||!download(generated_visits,w->insertion_visits,size_t(layouts)*n,w->stream)
        ||!download(edge_ids,w->insertion_edge_ids,size_t(layouts)*(n-1),w->stream))return failed();
    return cudaStreamSynchronize(w->stream)==cudaSuccess?0:2;
}

extern "C" int spacepdhcg_gtoc12_joint_prepared_insertions_host(void* opaque,
    int64_t first,int32_t layouts,Result* results,uint8_t* enabled,
    double* masses,double* inflations,double* proxies,double* collected,
    double* arrivals,double* departures,GeometryStats* stats,Visit* generated_visits,int32_t* edge_ids) {
    return spacepdhcg_gtoc12_joint_prepared_insertion_grid_host(opaque,first,layouts,1,
        results,enabled,masses,inflations,proxies,collected,arrivals,departures,stats,
        generated_visits,edge_ids);
}
