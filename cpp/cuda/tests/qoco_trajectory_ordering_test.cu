// Standalone graph/elimination test; no vendor factorization.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <set>
#include <vector>
extern "C" int qoco_gpu_set_trajectory(int,int,int,const int*,const int*,const int*,cudaStream_t);
extern "C" int qoco_gpu_trajectory_ordering(int,int,const int*,const int*,int*);
extern "C" int qoco_gpu_trajectory_ordering_with_tree(int,int,const int*,const int*,int*,int**,int*);
static void require(bool ok, const char* s) { if (!ok) { fprintf(stderr,"FAIL: %s\n",s); exit(1); } }
static void check(cudaError_t s) { require(s == cudaSuccess,cudaGetErrorString(s)); }
static int* upload(const std::vector<int>& v, cudaStream_t stream) {
    int* p{}; check(cudaMalloc(&p,v.size()*sizeof(int)));
    check(cudaMemcpyAsync(p,v.data(),v.size()*sizeof(int),cudaMemcpyHostToDevice,stream)); return p;
}
static int largest_front(std::vector<std::set<int>> graph, const std::vector<int>& order) {
    int maximum=0;
    for(int v:order) {
        auto neighbours=graph[v]; maximum=std::max(maximum,static_cast<int>(neighbours.size()));
        for(int a:neighbours) { graph[a].erase(v); for(int b:neighbours) if(a!=b) graph[a].insert(b); }
        graph[v].clear();
    }
    return maximum;
}
static void run(int N, bool nonlocal, bool reversed, bool with_tree=false) {
    const int n=4*N+1,total=n+3*N;
    auto id=[&](int v){return reversed ? n-1-v : v;};
    std::vector<int> states(N+1),controls(N),virtuals(N);
    std::vector<std::set<int>> graph(total);
    auto edge=[&](int a,int b){graph[a].insert(b);graph[b].insert(a);};
    for(int k=0;k<=N;++k) states[k]=id(k);
    for(int k=0;k<N;++k) {
        controls[k]=id(N+1+k); virtuals[k]=id(2*N+1+k);
        int y=n+k,z=n+N+2*k;
        edge(states[k],y); edge(states[k+1],y); edge(controls[k],y); edge(virtuals[k],y);
        edge(id(3*N+1+k),z); edge(virtuals[k],z+1); edge(z,z+1);
    }
    if(nonlocal) { edge(states[0],states[N]); if(N>2) edge(n+N,n+3*N-1); }
    std::vector<int> offsets(total+1),columns;
    for(int v=0;v<total;++v) {
        offsets[v]=columns.size(); columns.push_back(v);
        for(int c:graph[v]) if(c>v) columns.push_back(c);
    }
    offsets[total]=columns.size();
    cudaStream_t stream{}; check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    int *ds=upload(states,stream),*dc=upload(controls,stream),*dv=upload(virtuals,stream),
        *dr=upload(offsets,stream),*di=upload(columns,stream),*out{};
    check(cudaMalloc(&out,(total+2)*sizeof(int)));
    std::vector<int> previous;
    for(int repeat=0;repeat<3;++repeat) {
        std::vector<int> guarded(total+2,-19);
        check(cudaMemcpyAsync(out,guarded.data(),guarded.size()*sizeof(int),cudaMemcpyHostToDevice,stream));
        require(qoco_gpu_set_trajectory(N,1,1,ds,dc,dv,stream)==0,"layout submission");
        int* device_sizes{}; int levels{};
        const int ordered=with_tree
            ? qoco_gpu_trajectory_ordering_with_tree(total,n,dr,di,out+1,&device_sizes,&levels)
            : qoco_gpu_trajectory_ordering(total,n,dr,di,out+1);
        require(ordered==1,"ordering applied");
        check(cudaMemcpy(guarded.data(),out,guarded.size()*sizeof(int),cudaMemcpyDeviceToHost));
        require(guarded.front()==-19 && guarded.back()==-19,"output guards");
        std::vector<int> order(guarded.begin()+1,guarded.end()-1),sorted=order;
        std::sort(sorted.begin(),sorted.end()); for(int k=0;k<total;++k) require(sorted[k]==k,"bijection");
        if(with_tree) {
            require(levels>=2 && levels<=8,"bounded tree levels");
            std::vector<int> sizes((1<<levels)-1),vertex_node(total);
            check(cudaMemcpy(sizes.data(),device_sizes,sizes.size()*sizeof(int),cudaMemcpyDeviceToHost));
            int group=0,position=0;
            for(int d=levels-1;d>=0;--d) for(int leaf=0;leaf<(1<<d);++leaf,++group) {
                require(sizes[group]>=0 && sizes[group]<=total-position,"tree size bounds");
                for(int j=0;j<sizes[group];++j) vertex_node[order[position++]]=(1<<d)+leaf;
            }
            require(position==total,"tree counts cover permutation");
            for(int r=0;r<total;++r) for(int c:graph[r]) {
                int a=vertex_node[r],b=vertex_node[c];
                while(a!=b && a>0 && b>0) { if(a>b) a/=2; else b/=2; }
                // An edge may join ancestors, not siblings sharing an ancestor.
                const int original_a=vertex_node[r],original_b=vertex_node[c];
                require(a==original_a || a==original_b,"independent separator subgraphs");
            }
            check(cudaFree(device_sizes));
        }
        if(repeat) require(order==previous,"repeatability"); previous=order;
        if(!nonlocal) require(largest_front(graph,order)<=16,"bounded chain elimination front");
        require(qoco_gpu_trajectory_ordering(total,n,dr,di,out+1)==0,"metadata consumed exactly once");
    }
    auto invalid=states; invalid[0]=n;
    check(cudaMemcpyAsync(ds,invalid.data(),invalid.size()*sizeof(int),cudaMemcpyHostToDevice,stream));
    require(qoco_gpu_set_trajectory(N,1,1,ds,dc,dv,stream)==0,"invalid layout deferred");
    require(qoco_gpu_trajectory_ordering(total,n,dr,di,out+1)==-2,"index bounds rejected");
    invalid=states; invalid[0]=controls[0];
    check(cudaMemcpyAsync(ds,invalid.data(),invalid.size()*sizeof(int),cudaMemcpyHostToDevice,stream));
    require(qoco_gpu_set_trajectory(N,1,1,ds,dc,dv,stream)==0,"duplicate layout deferred");
    require(qoco_gpu_trajectory_ordering(total,n,dr,di,out+1)==-2,"duplicate indices rejected");
    invalid=states; invalid[0]=states[1];
    check(cudaMemcpyAsync(ds,invalid.data(),invalid.size()*sizeof(int),cudaMemcpyHostToDevice,stream));
    require(qoco_gpu_set_trajectory(N,1,1,ds,dc,dv,stream)==0,"concurrent duplicates deferred");
    require(qoco_gpu_trajectory_ordering(total,n,dr,di,out+1)==-2,"concurrent duplicates rejected");
    for(int* p:{ds,dc,dv,dr,di,out}) check(cudaFree(p)); check(cudaStreamDestroy(stream));
}
int main() {
    require(qoco_gpu_set_trajectory(-1,1,1,nullptr,nullptr,nullptr,nullptr)==-1,"invalid dimensions");
    for(int n:{1,2,7,24,65}) for(bool reversed:{false,true}) for(bool nonlocal:{false,true}) run(n,nonlocal,reversed);
    run(10000,false,true);
    for(int n:{1,2,7,65}) for(bool nonlocal:{false,true}) run(n,nonlocal,true,true);
    run(10000,false,true,true);
    puts("Trajectory GPU ordering: bijection, independent elimination, nonlocal edges, shuffled maps, guards, streams, invalid metadata PASS");
}
