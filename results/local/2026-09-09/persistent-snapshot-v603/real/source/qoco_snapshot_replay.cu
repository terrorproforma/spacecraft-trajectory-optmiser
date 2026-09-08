// Diagnostic replay of an explicitly exported native QP. No trajectory code.
#include "qoco.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" int qoco_gpu_begin_reduction_scope();
extern "C" void qoco_gpu_end_reduction_scope();
extern "C" int qoco_gpu_set_device_io(QOCOSolver*,int);
extern "C" int qoco_gpu_primal_start(QOCOSolver*,int);
extern "C" int qoco_gpu_download_solution(QOCOSolver*);
extern "C" int qoco_gpu_create_numeric_update(QOCOSolver*,int,int,int,void**);
extern "C" int qoco_gpu_update_numeric(void*,const double*,cudaStream_t);
extern "C" void qoco_gpu_destroy_numeric_update(void*);
extern "C" int qoco_gpu_set_stopping_origin(QOCOSolver*,const double*,const double*);
#define CHECK(x) do { if ((x)!=0) throw std::runtime_error(#x); } while(0)
template<class T> std::vector<T> read_vector(std::istream& in) {
    size_t n{};in>>n;if(!in || n>100000000) throw std::runtime_error("invalid vector length");
    std::vector<T> v(n);for(auto& x:v) in>>x;
    if(!in) throw std::runtime_error("truncated snapshot");return v;
}
void number(double x) { if(std::isfinite(x)) std::cout<<x;else std::cout<<"null"; }
void vector_json(const double* x,int n) {
    std::cout<<'[';for(int i=0;i<n;++i) { if(i) std::cout<<',';number(x[i]); }std::cout<<']';
}
int main(int argc,char** argv) try {
    if(argc<2 || argc>3) throw std::runtime_error("usage: qoco_snapshot_replay snapshot [repeats]");
    const int repeats=argc==3 ? std::stoi(argv[2]) : 1;
    if(repeats<1 || repeats>10000) throw std::runtime_error("invalid repeats");
    std::ifstream in(argv[1]);std::string magic;std::getline(in,magic);
    if(magic!="SPACEPDHCG_QOCO_QP_V1") throw std::runtime_error("invalid snapshot version");
    int n,p,m,np,na,ng,l,ns,shift;in>>n>>p>>m>>np>>na>>ng>>l>>ns>>shift;
    if(n<1 || p<0 || m<0 || np<0 || na<0 || ng<0 || l<0 || ns<0 || (shift!=0 && shift!=1))
        throw std::runtime_error("invalid dimensions");
    QOCOSettings settings{};set_default_settings(&settings);int verbose;
    in>>settings.max_iters>>settings.ruiz_iters>>settings.max_ir_iters>>verbose;
    settings.verbose=verbose;
    in>>settings.ir_tol>>settings.kkt_static_reg_P>>settings.kkt_static_reg_A>>settings.kkt_static_reg_G
      >>settings.kkt_dynamic_reg>>settings.abstol>>settings.reltol>>settings.abstol_inacc>>settings.reltol_inacc;
    if(const char* tolerance=std::getenv("QOCO_REPLAY_TOLERANCE")) {
        const double value=std::stod(tolerance);
        if(!std::isfinite(value) || value<=0) throw std::runtime_error("invalid diagnostic tolerance");
        settings.abstol=settings.reltol=value;
    }
    if(const char* regularization=std::getenv("QOCO_REPLAY_REGULARIZATION")) {
        const double value=std::stod(regularization);
        if(!std::isfinite(value) || value<=0) throw std::runtime_error("invalid diagnostic regularization");
        settings.kkt_static_reg_P=settings.kkt_static_reg_A=settings.kkt_static_reg_G=value;
    }
    auto pp=read_vector<int>(in),pi=read_vector<int>(in),ap=read_vector<int>(in),ai=read_vector<int>(in),gp=read_vector<int>(in),gi=read_vector<int>(in),soc=read_vector<int>(in);
    auto original=read_vector<double>(in),translated=read_vector<double>(in),origin=read_vector<double>(in);
    double offset;in>>offset;
    const size_t count=size_t(np)+na+ng+n+p+m;
    if(!in || original.size()!=count || soc.size()!=size_t(ns)
       || translated.size()!=(shift ? count : 0) || origin.size()!=(shift ? size_t(n) : 0))
        throw std::runtime_error("inconsistent snapshot");
    const auto topology=[&](const auto& offsets,const auto& indices,int rows,int nnz) {
        if(offsets.size()!=size_t(n+1) || indices.size()!=size_t(nnz) || offsets.front()!=0 || offsets.back()!=nnz
           || !std::is_sorted(offsets.begin(),offsets.end())) throw std::runtime_error("invalid CSC topology");
        for(int r:indices) if(r<0 || r>=rows) throw std::runtime_error("invalid CSC row");
    };
    topology(pp,pi,n,np);topology(ap,ai,p,na);topology(gp,gi,m,ng);
    int cone_rows=l;for(int q:soc) { if(q<2) throw std::runtime_error("invalid SOC");cone_rows+=q; }
    if(cone_rows!=m) throw std::runtime_error("cone dimension mismatch");
    for(const auto* v:{&original,&translated,&origin}) for(double x:*v)
        if(!std::isfinite(x)) throw std::runtime_error("nonfinite input");
    CHECK(qoco_gpu_begin_reduction_scope());
    QOCOCscMatrix P{},A{},G{};
    auto host=original;
    qoco_set_csc(&P,n,n,np,host.data(),pp.data(),pi.data());
    qoco_set_csc(&A,p,n,na,host.data()+np,ap.data(),ai.data());
    qoco_set_csc(&G,m,n,ng,host.data()+np+na,gp.data(),gi.data());
    auto setup_settings=settings;setup_settings.ruiz_iters=0;
    auto* solver=static_cast<QOCOSolver*>(std::malloc(sizeof(QOCOSolver)));
    if(!solver) throw std::bad_alloc();
    double* c=host.data()+np+na+ng;
    CHECK(qoco_setup(solver,n,m,p,&P,c,p?&A:nullptr,p?c+n:nullptr,m?&G:nullptr,m?c+n+p:nullptr,l,ns,soc.data(),&setup_settings));
    CHECK(qoco_gpu_set_device_io(solver,1));
    void* update{};CHECK(qoco_gpu_create_numeric_update(solver,np,na,ng,&update));
    double *values{},*dorigin{},*doffset{};
    CHECK(cudaMalloc(&values,count*sizeof(double)));
    CHECK(cudaMemcpy(values,original.data(),count*sizeof(double),cudaMemcpyHostToDevice));
    CHECK(qoco_update_settings(solver,&settings));
    CHECK(qoco_gpu_update_numeric(update,values,nullptr));
    if(shift) {
        CHECK(cudaMalloc(&dorigin,n*sizeof(double)));CHECK(cudaMalloc(&doffset,sizeof(double)));
        CHECK(cudaMemcpy(dorigin,origin.data(),n*sizeof(double),cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(doffset,&offset,sizeof(double),cudaMemcpyHostToDevice));
        CHECK(qoco_gpu_set_stopping_origin(solver,dorigin,doffset));
        CHECK(cudaMemcpy(values,translated.data(),count*sizeof(double),cudaMemcpyHostToDevice));
    }
    std::cout<<std::setprecision(17);
    for(int repeat=0;repeat<repeats;++repeat) {
        CHECK(qoco_update_settings(solver,&settings));
        CHECK(qoco_gpu_update_numeric(update,values,nullptr));
        CHECK(qoco_gpu_primal_start(solver,0));
        const int status=qoco_solve(solver);
        CHECK(qoco_gpu_download_solution(solver));
        const auto* s=solver->sol;
        std::cout<<"QP_REPLAY {\"repeat\":"<<repeat<<",\"status\":"<<status<<",\"iterations\":"<<s->iters
                 <<",\"ir_iterations\":"<<s->ir_iters<<",\"primal\":";number(s->pres);
        std::cout<<",\"dual\":";number(s->dres);std::cout<<",\"gap\":";number(s->gap);
        std::cout<<",\"objective\":";number(s->obj);std::cout<<",\"x\":";vector_json(s->x,n);
        std::cout<<",\"y\":";vector_json(s->y,p);std::cout<<",\"z\":";vector_json(s->z,m);
        std::cout<<",\"s\":";vector_json(s->s,m);std::cout<<"}\n"<<std::flush;
    }
    qoco_gpu_destroy_numeric_update(update);qoco_cleanup(solver);
    CHECK(cudaFree(values));CHECK(cudaFree(dorigin));CHECK(cudaFree(doffset));
    qoco_gpu_end_reduction_scope();return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
