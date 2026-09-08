#include "spacepdhcg/cuda/gtoc12_joint_c_api.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using Visit = spacepdhcg_gtoc12_joint_visit;
using Stage = spacepdhcg_gtoc12_joint_stage;
using Cost = spacepdhcg_gtoc12_joint_cost;
using Policy = spacepdhcg_gtoc12_joint_policy;
using Result = spacepdhcg_gtoc12_joint_result;
using Selection = spacepdhcg_gtoc12_joint_selection;

int main() {
    assert(cudaSetDevice(0) == cudaSuccess);
    constexpr int count = 257, n = 3, legs = 2;
    Policy policy{0, 2000, 3000, 500, 40, 365.25, 10, 365.25, .6, 30,
        60, 0, NAN, NAN, 0, 0, 0, 0};
    Visit visits[n]{{0, 0, -2, 0, 0, NAN, 0, 1},
        {1, 1, 1, 0, 0, NAN, 1200, .5}, {0, 0, -2, 0, 0, NAN, 0, 1}};
    Stage stages[legs]{{0, 1, 0, 0, 300, 600, 1, 1, 0, 0, 1},
        {0, 0, 0, 0, 100, 1200, 1, 1, 0, 0, 1}};
    std::vector<double> arr(count*n), dep(count*n), masses(count*legs),
        inflations(count*legs), proxies(count*legs), payload(count*n);
    std::vector<Cost> costs(count*legs);
    std::vector<Result> results(count);
    for (int i=0; i<count; ++i) {
        arr[i*n]=dep[i*n]=0;
        arr[i*n+1]=400; dep[i*n+1]=(i==129 || i==200)?1201:1200;
        arr[i*n+2]=dep[i*n+2]=1500;
        costs[i*legs]={0,0,2,0,0}; costs[i*legs+1]={0,0,1,0,0};
    }
    arr[n]=dep[n]=-1;
    void* workspace=nullptr;
    assert(spacepdhcg_gtoc12_joint_create(0,count,n,&workspace)==0);
    for (int size : {1,127,128,129,257,1}) {
        assert(spacepdhcg_gtoc12_joint_evaluate_host(workspace,size,&policy,visits,stages,
            arr.data(),dep.data(),costs.data(),results.data(),masses.data(),inflations.data(),
            proxies.data(),payload.data())==0);
        int winner=-1;
        for (int i=0; i<size; ++i)
            if (results[i].failure==0 && (winner<0 || results[i].objective>results[winner].objective)) winner=i;
        Selection selected{};
        std::vector<double> m(legs+2,777),f(legs+2,777),p(legs+2,777),c(n+2,777);
        const auto run=[&](double minimum) {
            return spacepdhcg_gtoc12_joint_best_host(workspace,size,&policy,visits,stages,
                arr.data(),dep.data(),costs.data(),minimum,&selected,m.data()+1,f.data()+1,
                p.data()+1,c.data()+1);
        };
        assert(run(-INFINITY)==0 && selected.index==winner && selected.invalid_stay==0);
        assert(std::memcmp(&selected.value,&results[winner],sizeof(Result))==0);
        for (int j=0; j<legs; ++j) {
            assert(m[j+1]==masses[winner*legs+j]);
            assert(f[j+1]==inflations[winner*legs+j]);
            assert(p[j+1]==proxies[winner*legs+j]);
        }
        for (int j=0; j<n; ++j) assert(c[j+1]==payload[winner*n+j]);
        assert(m.front()==777 && m.back()==777 && f.front()==777 && f.back()==777);
        assert(p.front()==777 && p.back()==777 && c.front()==777 && c.back()==777);
        for (double minimum : {results[winner].objective,double(INFINITY),double(NAN)})
            assert(run(minimum)==0 && selected.index==-1);
    }
    Selection selected{};
    selected.index=55; stages[0].model=99;
    assert(spacepdhcg_gtoc12_joint_best_host(workspace,1,&policy,visits,stages,arr.data(),
        dep.data(),costs.data(),0,&selected,nullptr,nullptr,nullptr,nullptr)==4);
    assert(selected.index==55);
    assert(spacepdhcg_gtoc12_joint_best_host(workspace,0,nullptr,nullptr,nullptr,nullptr,
        nullptr,nullptr,0,&selected,nullptr,nullptr,nullptr,nullptr)==0);
    assert(selected.index==-1 && selected.invalid_stay==0);
    assert(spacepdhcg_gtoc12_joint_destroy(&workspace)==0 && !workspace);
    std::puts("joint GPU selection passed: ties, thresholds, compact rows, reuse and guards");
}
