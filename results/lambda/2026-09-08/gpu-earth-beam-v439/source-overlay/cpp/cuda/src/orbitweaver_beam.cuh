// Included by orbitweaver_gpu.cu: uses the same ephemeris and Lambert operators.
struct RankedBeamOption {
    spacepdhcg_orbitweaver_beam_option value;
    uint64_t index;
};
struct BeamLess {
    bool final_order;
    __host__ __device__ bool operator()(const RankedBeamOption& a,const RankedBeamOption& b) const {
        if(a.value.score!=b.value.score)return a.value.score>b.value.score;
        if(final_order){
            if(a.value.asteroid!=b.value.asteroid)return a.value.asteroid<b.value.asteroid;
            if(a.value.departure!=b.value.departure)return a.value.departure<b.value.departure;
            if(a.value.tof!=b.value.tof)return a.value.tof<b.value.tof;
        }
        return a.index<b.index;
    }
};
struct BeamScratch {
    spacepdhcg_orbitweaver_beam_target* targets{};
    double *epochs{},*tofs{};
    RankedBeamOption *rows{},*kept{};
    spacepdhcg_orbitweaver_beam_option* packed{};
    unsigned* selected{};
    void* temporary{};
    size_t targets_capacity{},epochs_capacity{},tofs_capacity{},rows_capacity{},kept_capacity{},limit_capacity{},temporary_bytes{};
    size_t bytes()const{return targets_capacity*sizeof(*targets)+(epochs_capacity+tofs_capacity)*sizeof(double)
        +(rows_capacity+kept_capacity)*sizeof(*rows)+limit_capacity*sizeof(*packed)+2*sizeof(unsigned)+temporary_bytes;}
    ~BeamScratch(){cudaFree(targets);cudaFree(epochs);cudaFree(tofs);cudaFree(rows);cudaFree(kept);cudaFree(packed);cudaFree(selected);cudaFree(temporary);}
};

__global__ void beam_requests(spacepdhcg_orbitweaver_beam_config config,
    const spacepdhcg_orbitweaver_beam_target* targets,const double* epochs,const double* tofs,
    size_t ne,size_t nt,size_t start,size_t count,spacepdhcg_orbitweaver_hop_request* hops) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
    const size_t row=start+i,body=row/(ne*nt),ei=(row/nt)%ne,ti=row%nt;
    auto& h=hops[i];h={};auto& q=h.lambert;
    q.deterministic_id=row;q.gravitational_parameter=config.mu;q.time_tolerance=1e-8;
    q.maximum_iterations=256;q.time_of_flight=tofs[ti]*86400.;h.departure_allowance=config.allowance;
    bool a=element_state(config.earth,epochs[ei],config.mu,q.departure_position,h.departure_body_velocity);
    bool b=element_state(targets[body].elements,epochs[ei]+tofs[ti],config.mu,q.arrival_position,h.arrival_body_velocity);
    if(!a||!b)q.time_of_flight=NAN;
}

__global__ void beam_scores(spacepdhcg_orbitweaver_beam_config c,
    const spacepdhcg_orbitweaver_beam_target* targets,const double* epochs,const double* tofs,
    size_t ne,size_t nt,size_t start,size_t count,const spacepdhcg_orbitweaver_hop_result* hops,
    RankedBeamOption* rows,unsigned* feasible) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
    const size_t row=start+i,body=row/(ne*nt),ei=(row/nt)%ne,ti=row%nt;
    const unsigned mask=__ballot_sync(__activemask(),hops[i].feasible!=0);
    if((threadIdx.x&31)==0)atomicAdd(feasible,unsigned(__popc(mask)));
    const auto& target=targets[body];const double tof=tofs[ti],dep=epochs[ei];
    // Explicit rounded operations preserve NumPy's unfused expression order.
    const double dv=__dadd_rn(hops[i].departure_delta_v,hops[i].arrival_delta_v);
    const double acceleration=__dmul_rn(__ddiv_rn(c.thrust,c.initial_mass),1e-3);
    const double authority=__dmul_rn(c.authority_ratio,__dmul_rn(__dmul_rn(acceleration,tof),86400.));
    const double prop=__dmul_rn(c.initial_mass,__dsub_rn(1.,exp(__ddiv_rn(-__dmul_rn(dv,c.inflation),c.exhaust_velocity))));
    const double mined=__ddiv_rn(__dmul_rn(c.mining_rate,fmax(__dsub_rn(c.horizon,__dadd_rn(dep,tof)),0.)),c.year_days);
    double score=__dsub_rn(__dmul_rn(target.weight,mined),__dmul_rn(c.propellant_weight,__dadd_rn(prop,c.miner_mass)));
    if(c.cluster_bonus>0)score=__dadd_rn(score,__dmul_rn(c.cluster_bonus,__ddiv_rn(fmin(target.density,c.density_cap),c.density_cap)));
    if(c.seed_bonus>0)score=__dadd_rn(score,__dmul_rn(c.seed_bonus,target.seeded));
    const bool valid=hops[i].feasible&&isfinite(dv)&&dv<=authority&&isfinite(score);
    rows[i]={{valid?score:-INFINITY,target.asteroid,dep,tof,dv,prop},valid?uint64_t(row):UINT64_MAX};
}

__global__ void beam_pack(const RankedBeamOption* rows,size_t count,
    spacepdhcg_orbitweaver_beam_option* packed,unsigned* selected) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count&&rows[i].index!=UINT64_MAX){packed[i]=rows[i].value;atomicAdd(selected,1U);}
}

template<class Workspace>
spacepdhcg_cuda_status earth_beam_host(Workspace* w,const spacepdhcg_orbitweaver_beam_config* c,
    const spacepdhcg_orbitweaver_beam_target* targets,size_t na,const double* epochs,size_t ne,
    const double* tofs,size_t nt,size_t block,size_t limit,
    spacepdhcg_orbitweaver_beam_option* options,size_t capacity,size_t* selected) {
    if(!w||!c||!targets||!epochs||!tofs||!options||!selected||!na||!ne||!nt||!block||!limit||capacity<limit
        ||na>INT_MAX||ne>INT_MAX||nt>INT_MAX||limit>INT_MAX||ne>SIZE_MAX/nt||na>UINT_MAX/(ne*nt)
        ||!valid_elements(c->earth))return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    const double scalars[]={c->mu,c->allowance,c->initial_mass,c->inflation,c->authority_ratio,c->thrust,
        c->exhaust_velocity,c->horizon,c->mining_rate,c->year_days,c->miner_mass,c->propellant_weight,c->cluster_bonus,c->density_cap,c->seed_bonus};
    for(double v:scalars)if(!std::isfinite(v))return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    if(c->mu<=0||c->initial_mass<=0||c->exhaust_velocity<=0||c->year_days<=0||c->density_cap<=0||c->allowance<0)
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    for(size_t i=0;i<na;++i)if(!valid_elements(targets[i].elements)||!std::isfinite(targets[i].weight)
        ||!std::isfinite(targets[i].density)||!std::isfinite(targets[i].seeded))return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    for(size_t i=0;i<ne;++i)if(!std::isfinite(epochs[i]))return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    for(size_t i=0;i<nt;++i)if(!std::isfinite(tofs[i])||tofs[i]<=0)return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    block=std::min(block,na);const size_t rows=block*ne*nt,blocks=(na-1)/block+1;
    const size_t per_block=std::min(limit,rows);
    if(rows>INT_MAX||blocks>size_t(INT_MAX)/per_block)return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    const size_t kept=blocks*per_block;
    *selected=0;std::unique_lock<std::mutex> lock(w->api_mutex,std::try_to_lock);
    if(!lock.owns_lock()||w->busy.load())return SPACEPDHCG_CUDA_BUSY;
    int device=-1;auto status=cudaGetDevice(&device);if(status!=cudaSuccess)return mapped(status);
    if(device!=int(w->config.device_id))return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    if(!w->beam_scratch||w->beam_scratch->targets_capacity<na||w->beam_scratch->epochs_capacity<ne
        ||w->beam_scratch->tofs_capacity<nt||w->beam_scratch->rows_capacity<rows
        ||w->beam_scratch->kept_capacity<kept||w->beam_scratch->limit_capacity<limit){
        std::unique_ptr<BeamScratch> s(new(std::nothrow) BeamScratch);if(!s)return SPACEPDHCG_CUDA_OUT_OF_MEMORY;
        s->targets_capacity=na;s->epochs_capacity=ne;s->tofs_capacity=nt;s->rows_capacity=rows;s->kept_capacity=kept;s->limit_capacity=limit;
#define BEAM_ALLOC(field,n) if(status==cudaSuccess)status=cudaMalloc(&s->field,(n)*sizeof(*s->field))
        BEAM_ALLOC(targets,na);BEAM_ALLOC(epochs,ne);BEAM_ALLOC(tofs,nt);BEAM_ALLOC(rows,rows);
        BEAM_ALLOC(kept,kept);BEAM_ALLOC(packed,limit);BEAM_ALLOC(selected,2);
#undef BEAM_ALLOC
        size_t a=0,b=0;
        if(status==cudaSuccess)status=cub::DeviceMergeSort::SortKeys(nullptr,a,s->rows,int(rows),BeamLess{false},w->stream);
        if(status==cudaSuccess)status=cub::DeviceMergeSort::SortKeys(nullptr,b,s->kept,int(kept),BeamLess{true},w->stream);
        s->temporary_bytes=std::max(a,b);if(status==cudaSuccess)status=cudaMalloc(&s->temporary,s->temporary_bytes);
        if(status!=cudaSuccess)return mapped(status);w->beam_scratch=std::move(s);
    }
    auto& s=*w->beam_scratch;
    status=cudaMemcpyAsync(s.targets,targets,na*sizeof(*targets),cudaMemcpyHostToDevice,w->stream);
    if(status==cudaSuccess)status=cudaMemcpyAsync(s.epochs,epochs,ne*sizeof(double),cudaMemcpyHostToDevice,w->stream);
    if(status==cudaSuccess)status=cudaMemcpyAsync(s.tofs,tofs,nt*sizeof(double),cudaMemcpyHostToDevice,w->stream);
    if(status==cudaSuccess)status=cudaMemsetAsync(s.selected,0,2*sizeof(unsigned),w->stream);
    size_t written=0,batches=0;
    for(size_t first=0;status==cudaSuccess&&first<na;first+=block){
        const size_t count=std::min(block,na-first)*ne*nt;
        for(size_t offset=0;status==cudaSuccess&&offset<count;){
            const size_t n=std::min(count-offset,w->config.maximum_batch_size),start=first*ne*nt+offset;
            beam_requests<<<(n+127)/128,128,0,w->stream>>>(*c,s.targets,s.epochs,s.tofs,ne,nt,start,n,w->hops);status=cudaGetLastError();
            if(status==cudaSuccess){launch_hops(w->hops,n,w->config.scan_samples_per_band,w->hop_results,w->scan_grid,w->stream);status=cudaGetLastError();}
            if(status==cudaSuccess){beam_scores<<<(n+127)/128,128,0,w->stream>>>(*c,s.targets,s.epochs,s.tofs,ne,nt,start,n,w->hop_results,s.rows+offset,s.selected+1);status=cudaGetLastError();}
            offset+=n;++batches;
        }
        auto bytes=s.temporary_bytes;
        if(status==cudaSuccess)status=cub::DeviceMergeSort::SortKeys(s.temporary,bytes,s.rows,int(count),BeamLess{false},w->stream);
        const size_t take=std::min(limit,count);
        if(status==cudaSuccess)status=cudaMemcpyAsync(s.kept+written,s.rows,take*sizeof(*s.rows),cudaMemcpyDeviceToDevice,w->stream);
        written+=take;
    }
    auto bytes=s.temporary_bytes;const size_t take=std::min(limit,written);
    if(status==cudaSuccess)status=cub::DeviceMergeSort::SortKeys(s.temporary,bytes,s.kept,int(written),BeamLess{true},w->stream);
    if(status==cudaSuccess)status=cudaMemsetAsync(s.selected,0,sizeof(unsigned),w->stream);
    if(status==cudaSuccess){beam_pack<<<(take+127)/128,128,0,w->stream>>>(s.kept,take,s.packed,s.selected);status=cudaGetLastError();}
    unsigned counts[2]={};if(status==cudaSuccess)status=cudaMemcpyAsync(counts,s.selected,sizeof(counts),cudaMemcpyDeviceToHost,w->stream);
    auto done=cudaStreamSynchronize(w->stream);if(status==cudaSuccess)status=done;
    if(status!=cudaSuccess)return mapped(status);
    const unsigned count=counts[0];if(count>take||counts[1]>na*ne*nt)return SPACEPDHCG_CUDA_RUNTIME_ERROR;
    if(count)status=cudaMemcpyAsync(options,s.packed,count*sizeof(*options),cudaMemcpyDeviceToHost,w->stream);
    done=cudaStreamSynchronize(w->stream);if(status==cudaSuccess)status=done;
    if(status==cudaSuccess){*selected=count;w->batches+=batches;w->request_count+=na*ne*nt;w->result_count+=count;
        w->feasible+=counts[1];w->failed+=na*ne*nt-counts[1];
        w->input_bytes+=na*sizeof(*targets)+(ne+nt)*sizeof(double)+sizeof(*c);w->output_bytes+=count*sizeof(*options)+sizeof(counts);}
    return mapped(status);
}
