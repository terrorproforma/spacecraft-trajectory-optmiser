// Check GPU termination against original QP equations, not a second copy of
// the vendor scaling formulas. The vectors deliberately need not be optimal.
#include "qoco.h"
#include "kkt.h"
#include "../algebra/cuda/cuda_types.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <vector>

#define REQUIRE(x) do { if(!(x)){std::fprintf(stderr,"line %d: %s\n",__LINE__,#x);std::exit(1);} }while(0)
using Vec=std::vector<double>;
void upload(double* dst,const Vec& v) { if(!v.empty()) REQUIRE(cudaMemcpy(dst,v.data(),v.size()*8,cudaMemcpyHostToDevice)==cudaSuccess); }
Vec download(const double* src,int n) { Vec v(n);if(n) REQUIRE(cudaMemcpy(v.data(),src,n*8,cudaMemcpyDeviceToHost)==cudaSuccess);return v; }
long double dot(const Vec& a,const Vec& b) { long double s=0;for(size_t i=0;i<a.size();++i)s+=(long double)a[i]*b[i];return s; }
double norm(const Vec& a) { double n=0;for(double v:a)n=std::max(n,std::abs(v));return n; }
Vec product(const Vec& a,int rows,int cols,const Vec& x,bool transpose=false) {
    Vec result(transpose?cols:rows);
    for(int i=0;i<(transpose?cols:rows);++i) {
        long double v=0;
        for(int j=0;j<(transpose?rows:cols);++j)v+=(long double)(transpose?a[j*cols+i]:a[i*cols+j])*x[j];
        result[i]=(double)v;
    }
    return result;
}
struct Sparse {
    std::vector<int> offsets,indices;Vec values;QOCOCscMatrix view;
    Sparse(const Vec& a,int rows,int cols,bool upper=false) {
        for(int j=0;j<cols;++j) {offsets.push_back(values.size());for(int i=0;i<rows;++i)
            if((!upper||i<=j)&&a[i*cols+j]!=0){indices.push_back(i);values.push_back(a[i*cols+j]);}}
        offsets.push_back(values.size());view={rows,cols,(int)values.size(),indices.data(),offsets.data(),values.data()};
    }
};
void compare(double actual,double expected,int metric,int ruiz,int trial) {
    if(!std::isfinite(actual)||std::abs(actual-expected)>2e-10*std::max(1.,std::abs(expected))) {
        std::fprintf(stderr,"Ruiz=%d trial=%d metric=%d %.17g expected %.17g\n",ruiz,trial,metric,actual,expected);std::exit(1);
    }
}
int main() {
    using Metrics=void(*)(QOCOSolver*,double*);
    auto metrics=reinterpret_cast<Metrics>(dlsym(RTLD_DEFAULT,"qoco_gpu_iteration_metrics"));REQUIRE(metrics);
    for(int ruiz:{0,1,5,10}) for(bool absent:{false,true}) for(bool zero_p:{false,true}) {
        const int n=3,p=absent?0:2,m=absent?0:5;
        Vec P{4,.2,0,.2,30,.4,0,.4,.08},A{.02,3,0,4,0,.2},G{2,0,0,0,.03,0,0,0,4,.1,0,0,0,.1,0};
        if(zero_p)std::fill(P.begin(),P.end(),0);
        if(absent){A.clear();G.clear();}
        Vec c{20000,-.03,4},b=absent?Vec{}:Vec{.1,3},h=absent?Vec{}:Vec{1,2,4,.1,.2};
        Sparse sp(P,n,n,true),sa(A,p,n),sg(G,m,n);
        auto* solver=static_cast<QOCOSolver*>(std::calloc(1,sizeof(QOCOSolver)));
        QOCOSettings settings{};set_default_settings(&settings);settings.ruiz_iters=ruiz;settings.verbose=0;
        int cone=3;
        REQUIRE(qoco_setup(solver,n,m,p,zero_p?nullptr:&sp.view,c.data(),p?&sa.view:nullptr,b.data(),m?&sg.view:nullptr,h.data(),absent?0:2,absent?0:1,&cone,&settings)==0);
        auto* w=solver->work;auto* scale=w->scaling;
        const auto D=download(scale->Druiz->d_data,n),E=download(scale->Eruiz->d_data,p),F=download(scale->Fruiz->d_data,m);
        const double k=scale->k;
        for(int trial=0;trial<4;++trial) {
            Vec x{.4+.1*trial,-.7,.2},y(p,.2-.03*trial),z=absent?Vec{}:Vec{.4,.7,1.3,.2,-.1},s=absent?Vec{}:Vec{.8,.9,2,.1,.3};
            // Include zero complementarity with nonzero stationarity/objective
            // error: a complementarity-only stopping check cannot certify it.
            if(trial==3)std::fill(z.begin(),z.end(),0);
            auto scaled=x;for(int i=0;i<n;++i)scaled[i]/=D[i];upload(w->x->d_data,scaled);
            scaled=y;for(int i=0;i<p;++i)scaled[i]*=k/E[i];upload(w->y->d_data,scaled);
            scaled=z;for(int i=0;i<m;++i)scaled[i]*=k/F[i];upload(w->z->d_data,scaled);
            scaled=s;for(int i=0;i<m;++i)scaled[i]*=F[i];upload(w->s->d_data,scaled);
            compute_kkt_residual(w->data,w->x,w->y,w->s,w->z,w->kktres,settings.kkt_static_reg_P,w->xyzbuff1,w->xbuff,w->ubuff1);
            const auto px=product(P,n,n,x),aty=product(A,p,n,y,true),gtz=product(G,m,n,z,true),ax=product(A,p,n,x),gx=product(G,m,n,x);
            Vec eq=ax,ineq=gx,dual=px;
            for(int i=0;i<p;++i)eq[i]-=b[i];for(int i=0;i<m;++i)ineq[i]+=s[i]-h[i];
            for(int i=0;i<n;++i)dual[i]+=aty[i]+gtz[i]+c[i];
            const double primal=(double)(.5L*dot(x,px)+dot(c,x)),dobj=(double)(-.5L*dot(x,px)-dot(b,y)-dot(h,z));
            const double expected[]{std::max(norm(eq),norm(ineq)),norm(dual),std::max((double)std::abs(dot(s,z)),std::abs(primal-dobj)),
                std::max({norm(ax),norm(b),norm(gx),norm(h),norm(s)}),std::max({norm(px),norm(aty),norm(gtz),norm(c)}),
                std::max({1.,std::abs(primal),std::abs(dobj)}),primal,m?(double)(dot(s,z)*k/m):0};
            double got[8];metrics(solver,got);for(int i=0;i<8;++i)compare(got[i],expected[i],i,ruiz,trial);
        }
        qoco_cleanup(solver);
    }
    std::puts("Original-equation stopping metrics: 64 scaled/unscaled QP iterates, 8 metrics PASS");
}
