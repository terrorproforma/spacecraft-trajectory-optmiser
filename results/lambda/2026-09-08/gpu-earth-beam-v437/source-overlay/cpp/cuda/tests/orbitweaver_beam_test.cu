// Exercise the production scoring and both CUB ranking stages with exact ties.
#include "../src/orbitweaver_gpu.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define REQUIRE(x) do{if(!(x)){std::fprintf(stderr,"failure line %d: %s\n",__LINE__,#x);std::exit(1);}}while(0)
#define CUDA(x) REQUIRE((x)==cudaSuccess)
using Beam=spacepdhcg_orbitweaver_beam_option;
using Target=spacepdhcg_orbitweaver_beam_target;
using Hop=spacepdhcg_orbitweaver_hop_result;
static_assert(sizeof(Beam)==48&&sizeof(Target)==88);

void check(size_t bodies,size_t block,size_t limit,bool all_invalid) {
    constexpr size_t ne=5,nt=7;
    std::vector<double> epochs(ne),tofs(nt);
    for(size_t i=0;i<ne;++i)epochs[i]=64000+double(i%3);
    for(size_t i=0;i<nt;++i)tofs[i]=100+double(i%4);
    std::vector<Target> targets(bodies);
    for(size_t i=0;i<bodies;++i){targets[i].asteroid=bodies-i;targets[i].weight=1+i%3;targets[i].density=i%4;targets[i].seeded=i%2;}
    spacepdhcg_orbitweaver_beam_config c{};c.initial_mass=3000;c.inflation=1;c.authority_ratio=1;c.thrust=.6;
    c.exhaust_velocity=40;c.horizon=70000;c.mining_rate=1;c.year_days=1;c.miner_mass=40;c.propellant_weight=.5;
    c.cluster_bonus=2;c.density_cap=4;c.seed_bonus=3;
    std::vector<Hop> hops(bodies*ne*nt);
    for(size_t i=0;i<hops.size();++i)hops[i].feasible=!all_invalid&&i%3!=0;
    std::vector<Beam> expected;
    for(size_t first=0;first<bodies;first+=block){
        std::vector<Beam> local;
        for(size_t i=first*ne*nt;i<std::min(first+block,bodies)*ne*nt;++i){
            if(!hops[i].feasible)continue;
            const auto& t=targets[i/(ne*nt)];const double dep=epochs[(i/nt)%ne],tof=tofs[i%nt];
            const double score=t.weight*(c.horizon-dep-tof)-20+2*(t.density/4)+3*t.seeded;
            local.push_back({score,t.asteroid,dep,tof,0,0});
        }
        std::stable_sort(local.begin(),local.end(),[](const Beam&a,const Beam&b){return a.score>b.score;});
        local.resize(std::min(limit,local.size()));expected.insert(expected.end(),local.begin(),local.end());
    }
    std::stable_sort(expected.begin(),expected.end(),[](const Beam&a,const Beam&b){
        if(a.score!=b.score)return a.score>b.score;
        if(a.asteroid!=b.asteroid)return a.asteroid<b.asteroid;
        if(a.departure!=b.departure)return a.departure<b.departure;
        return a.tof<b.tof;
    });expected.resize(std::min(limit,expected.size()));
    BeamScratch s;Hop* dh{};
    CUDA(cudaMalloc(&dh,hops.size()*sizeof(Hop)));CUDA(cudaMemcpy(dh,hops.data(),hops.size()*sizeof(Hop),cudaMemcpyHostToDevice));
    CUDA(cudaMalloc(&s.targets,bodies*sizeof(Target)));CUDA(cudaMemcpy(s.targets,targets.data(),bodies*sizeof(Target),cudaMemcpyHostToDevice));
    CUDA(cudaMalloc(&s.epochs,ne*sizeof(double)));CUDA(cudaMemcpy(s.epochs,epochs.data(),ne*sizeof(double),cudaMemcpyHostToDevice));
    CUDA(cudaMalloc(&s.tofs,nt*sizeof(double)));CUDA(cudaMemcpy(s.tofs,tofs.data(),nt*sizeof(double),cudaMemcpyHostToDevice));
    const size_t rows=std::min(block,bodies)*ne*nt,kept=((bodies-1)/block+1)*std::min(limit,rows);
    CUDA(cudaMalloc(&s.rows,rows*sizeof(RankedBeamOption)));CUDA(cudaMalloc(&s.kept,kept*sizeof(RankedBeamOption)));
    CUDA(cudaMalloc(&s.packed,limit*sizeof(Beam)));CUDA(cudaMalloc(&s.selected,2*sizeof(unsigned)));CUDA(cudaMemset(s.selected,0,2*sizeof(unsigned)));
    size_t a=0,b=0;CUDA(cub::DeviceMergeSort::SortKeys(nullptr,a,s.rows,int(rows),BeamLess{false}));
    CUDA(cub::DeviceMergeSort::SortKeys(nullptr,b,s.kept,int(kept),BeamLess{true}));
    s.temporary_bytes=std::max(a,b);CUDA(cudaMalloc(&s.temporary,s.temporary_bytes));size_t written=0;
    for(size_t first=0;first<bodies;first+=block){
        const size_t count=std::min(block,bodies-first)*ne*nt;
        for(size_t offset=0;offset<count;offset+=31){
            const size_t n=std::min(size_t(31),count-offset),start=first*ne*nt+offset;
            beam_scores<<<(n+127)/128,128>>>(c,s.targets,s.epochs,s.tofs,ne,nt,start,n,dh+start,s.rows+offset,s.selected+1);
        }
        auto bytes=s.temporary_bytes;CUDA(cub::DeviceMergeSort::SortKeys(s.temporary,bytes,s.rows,int(count),BeamLess{false}));
        const size_t take=std::min(limit,count);CUDA(cudaMemcpy(s.kept+written,s.rows,take*sizeof(RankedBeamOption),cudaMemcpyDeviceToDevice));written+=take;
    }
    auto bytes=s.temporary_bytes;CUDA(cub::DeviceMergeSort::SortKeys(s.temporary,bytes,s.kept,int(written),BeamLess{true}));
    const size_t take=std::min(limit,written);beam_pack<<<(take+127)/128,128>>>(s.kept,take,s.packed,s.selected);CUDA(cudaGetLastError());
    unsigned counts[2]={};CUDA(cudaMemcpy(counts,s.selected,sizeof(counts),cudaMemcpyDeviceToHost));
    REQUIRE(counts[0]==expected.size());REQUIRE(counts[1]==(all_invalid?0:hops.size()-(hops.size()+2)/3));
    std::vector<Beam> actual(counts[0]);
    if(counts[0]){CUDA(cudaMemcpy(actual.data(),s.packed,counts[0]*sizeof(Beam),cudaMemcpyDeviceToHost));REQUIRE(std::memcmp(actual.data(),expected.data(),counts[0]*sizeof(Beam))==0);}
    CUDA(cudaFree(dh));
}
int main(){
    for(size_t bodies:{1,19,201})for(size_t block:{1,7,201})for(size_t limit:{1,37,1000})for(bool invalid:{false,true})check(bodies,block,limit,invalid);
    std::puts("PASS: 54 exact beam scoring/ranking cases, block top-k, final ties, invalids and feasibility counts");
}
