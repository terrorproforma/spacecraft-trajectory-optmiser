#include "qoco.h"
#include "../algebra/cuda/cuda_types.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
extern "C" int qoco_test_vector_arena_info(QOCOSolver*, void**, size_t*);
static void require(bool ok, const char* text) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", text); std::exit(1); }
}
static void check(cudaError_t code) { require(code == cudaSuccess, cudaGetErrorString(code)); }
static QOCOSolver* create(int n, int cone) {
    std::vector<int> offsets(n+1), rows(n);
    std::vector<double> values(n, 2.0), c(n, 0.0);
    for (int i=0;i<n;++i) offsets[i]=rows[i]=i;
    offsets[n]=n;
    QOCOCscMatrix P{};qoco_set_csc(&P,n,n,n,values.data(),offsets.data(),rows.data());
    // Empty equality rows and an orthant or one SOC exercise zero/nonzero views.
    const int p=cone ? 3 : 0, m=cone;
    std::vector<int> empty(n+1,0);
    std::vector<double> b(p,0.0),h(m,1.0);
    QOCOCscMatrix A{},G{};
    qoco_set_csc(&A,p,n,0,nullptr,empty.data(),nullptr);
    qoco_set_csc(&G,m,n,0,nullptr,empty.data(),nullptr);
    auto* solver=static_cast<QOCOSolver*>(std::calloc(1,sizeof(QOCOSolver)));
    QOCOSettings settings{};set_default_settings(&settings);
    require(qoco_setup(solver,n,m,p,&P,c.data(),p?&A:nullptr,b.data(),m?&G:nullptr,h.data(),
        cone==1?1:0,cone>1?1:0,&cone,&settings)==0,"arena solver setup");
    return solver;
}
static std::vector<QOCOVectorf*> vectors(QOCOSolver* s) {
    auto* w=s->work;
    return {w->x,w->x0,w->s,w->y,w->z,w->best_x,w->best_s,w->best_y,w->best_z,
        w->W,w->nt_scaling,w->Winv,w->WtW,w->lambda,w->sbar,w->zbar,w->xbuff,w->ybuff,
        w->ubuff1,w->ubuff2,w->ubuff3,w->Ds,w->rhs,w->kktres,w->xyz,w->xyzbuff1};
}
static void verify(QOCOSolver* s, int tag) {
    void* allocation{};size_t bytes{};
    require(qoco_test_vector_arena_info(s,&allocation,&bytes)==0,"explicit arena capability");
    const auto base=reinterpret_cast<uintptr_t>(allocation);
    auto views=vectors(s);std::vector<std::pair<uintptr_t,uintptr_t>> ranges;
    for (size_t i=0;i<views.size();++i) {
        auto* v=views[i];require(v->arena_owned==1,"arena owns every scratch view");
        if (!v->len) {require(!v->d_data,"empty view is null");continue;}
        const auto start=reinterpret_cast<uintptr_t>(v->d_data),end=start+v->len*sizeof(double);
        require(start%256==0 && start>=base && end<=base+bytes,"aligned in-bounds arena view");
        ranges.emplace_back(start,end);
        std::vector<double> data(v->len);
        check(cudaMemcpy(data.data(),v->d_data,data.size()*sizeof(double),cudaMemcpyDeviceToHost));
        for (int j=0;j<v->len;++j) require(data[j]==0.0 && v->data[j]==0.0,"initial device/host zeros");
        std::fill(data.begin(),data.end(),tag+static_cast<double>(i));
        check(cudaMemcpy(v->d_data,data.data(),data.size()*sizeof(double),cudaMemcpyHostToDevice));
    }
    std::sort(ranges.begin(),ranges.end());
    for (size_t i=1;i<ranges.size();++i) require(ranges[i-1].second<=ranges[i].first,"disjoint arena views");
    for (size_t i=0;i<views.size();++i) if (views[i]->len) {
        std::vector<double> data(views[i]->len);
        check(cudaMemcpy(data.data(),views[i]->d_data,data.size()*sizeof(double),cudaMemcpyDeviceToHost));
        for (double value:data) require(value==tag+static_cast<double>(i),"no cross-vector aliasing");
    }
    std::printf("Arena n=%d m=%d bytes=%zu PASS\n",s->work->data->n,s->work->data->m,bytes);
}
int main() {
    auto* a=create(17,0);auto* b=create(1031,257);auto* c=create(33,1);
    verify(a,100);verify(b,200);verify(c,300);
    qoco_cleanup(b); // Free a middle solver while its neighbours remain live.
    auto* d=create(65,3);verify(d,400);
    for (auto* s:{a,c}) {
        const auto v=vectors(s);std::vector<double> data(v[0]->len);
        check(cudaMemcpy(data.data(),v[0]->d_data,data.size()*sizeof(double),cudaMemcpyDeviceToHost));
        for(double value:data) require(value==(s==a?100.0:300.0),"other solver survives teardown/reallocation");
    }
    qoco_cleanup(a);qoco_cleanup(d);qoco_cleanup(c);
    check(cudaDeviceSynchronize());
}
