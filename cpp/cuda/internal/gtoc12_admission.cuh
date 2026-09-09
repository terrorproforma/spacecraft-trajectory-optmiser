// Included inside the expansion operator's namespace after its row definitions.
using Admission=spacepdhcg_gtoc12_admission_policy;
struct ReturnView { const spacepdhcg_gtoc12_collection_option* rows; int count; };
static_assert(sizeof(Admission)==48);

__global__ void eligibility(Policy p,Admission a,const Parent* parents,const Deploy* deploys,
    const Ranked* rows,int n,const ReturnView* returns,int* eligible) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    eligible[i]=0;const Result r=rows[i].row;const Parent parent=parents[r.parent];
    if(r.mass<a.dry_mass+(a.reserve_fraction*r.hop_propellant+a.return_reserve))return;
    Sum mined;
    for(int j=0;j<parent.deploy_count;++j)
        mined.add(p.mining_rate*fmax(p.mining_horizon-deploys[parent.deploy_begin+j].epoch,0.0)/p.year_days,p.compensated_sum);
    mined.add(p.mining_rate*fmax(p.mining_horizon-r.arrival,0.0)/p.year_days,p.compensated_sum);
    const double cargo=mined.value();
    const double mass=fmin(r.mass+cargo,a.dry_mass+cargo+a.return_reserve);
    const ReturnView table=returns[r.parent];
    for(int j=0;j<table.count;++j) {
        const auto o=table.rows[j];
        const double authority=(p.thrust/mass*1e-3)*o.tof*p.day_seconds;
        if(o.delta_v<=a.return_authority_ratio*authority){eligible[i]=1;return;}
    }
}
__device__ int64_t body(const Result& r,const Parent* parents,const Deploy* deploys,int i) {
    const Parent p=parents[r.parent];return i==p.deploy_count?r.target:deploys[p.deploy_begin+i].body;
}
__device__ bool same_set(const Result& x,const Result& y,const Parent* parents,const Deploy* deploys) {
    const int n=parents[x.parent].deploy_count+1;if(n!=parents[y.parent].deploy_count+1)return false;
    // Exact multiset equality; hashes cannot merge distinct deployed sets.
    for(int i=0;i<n;++i) {
        const int64_t b=body(x,parents,deploys,i);int nx=0,ny=0;
        for(int j=0;j<n;++j){nx+=body(x,parents,deploys,j)==b;ny+=body(y,parents,deploys,j)==b;}
        if(nx!=ny)return false;
    }
    return true;
}
__global__ void admit_ordered(Admission a,const Parent* parents,const Deploy* deploys,
    const Ranked* rows,int n,const int* eligible,int* selected,int* count) {
    using Reduce=cub::BlockReduce<int,256>;
    __shared__ typename Reduce::TempStorage storage;
    __shared__ int used,chosen;
    if(threadIdx.x==0)used=0;__syncthreads();
    // Ordered greedy dependence: lanes check conflicts in parallel within each
    // ranked tile and cooperatively elect the earliest remaining survivor.
    for(int64_t base=0;base<n && used<a.limit;base+=256) {
        const int64_t slot=base+threadIdx.x;const bool exists=slot<n;
        const int i=exists?int(slot):0;
        bool active=exists&&eligible[i];int first_count=0,set_count=0;
        Result r{};if(active)r=rows[i].row;
        if(active)for(int j=0;j<used;++j) {
            const Result prior=rows[selected[j]].row;
            first_count+=body(r,parents,deploys,0)==body(prior,parents,deploys,0);
            if(first_count>=a.max_per_first){active=false;break;}
            set_count+=same_set(r,prior,parents,deploys);
            if(set_count>=a.max_per_set){active=false;break;}
        }
        active=active&&a.max_per_first>0&&a.max_per_set>0;
        while(used<a.limit) {
            const int best=Reduce(storage).Reduce(active?i:n,cub::Min());
            if(threadIdx.x==0)chosen=best;__syncthreads();
            if(chosen==n)break;
            const Result prior=rows[chosen].row;
            if(threadIdx.x==0)selected[used++]=chosen;
            if(active) {
                first_count+=body(r,parents,deploys,0)==body(prior,parents,deploys,0);
                set_count+=same_set(r,prior,parents,deploys);
                active=i!=chosen&&first_count<a.max_per_first&&set_count<a.max_per_set;
            }
            __syncthreads();
        }
        __syncthreads();
    }
    if(threadIdx.x==0)*count=used;
}
__global__ void unpack_admitted(const Ranked* rows,const int* selected,int offset,int n,Result* out) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=rows[selected[offset+i]].row;
}
