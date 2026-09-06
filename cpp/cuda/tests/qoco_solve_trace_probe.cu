// Diagnostic proxy only: inspect device inputs/results without changing numerics.
#include "qoco.h"
#include "../algebra/cuda/cuda_types.h"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
namespace {
using Hash=unsigned long long;
constexpr Hash seed=14695981039346656037ULL;
Hash bytes(Hash hash,const void* source,size_t size) {
    const auto* p=static_cast<const unsigned char*>(source);
    for(size_t i=0;i<size;++i) { hash^=p[i];hash*=1099511628211ULL; }
    return hash;
}
template<class T> Hash device(Hash hash,const T* source,size_t count) {
    std::vector<T> value(count);
    if(count && cudaMemcpy(value.data(),source,count*sizeof(T),cudaMemcpyDeviceToHost)!=cudaSuccess)
        std::abort();
    return bytes(hash,value.data(),count*sizeof(T));
}
Hash matrix(QOCOMatrix* matrix) {
    if(!matrix) return seed;
    const auto* c=matrix->d_csc_host;
    Hash hash=bytes(seed,&c->m,sizeof(c->m));hash=bytes(hash,&c->n,sizeof(c->n));
    hash=bytes(hash,&c->nnz,sizeof(c->nnz));
    if(c->p) hash=device(hash,c->p,c->n+1);
    hash=device(hash,c->i,c->nnz);return device(hash,c->x,c->nnz);
}
Hash vector(Hash hash,QOCOVectorf* value) {return device(hash,value->d_data,value->len);}
thread_local int ordinal{};
}
extern "C" QOCOInt qoco_solve(QOCOSolver* solver) {
    const auto actual=reinterpret_cast<QOCOInt(*)(QOCOSolver*)>(dlsym(RTLD_NEXT,"qoco_solve"));
    if(!actual) std::abort();
    auto* w=solver->work;auto* d=w->data;auto* sc=w->scaling;auto* settings=solver->settings;
    Hash vectors=seed,scales=seed;
    for(auto* v:{d->c,d->b,d->h}) vectors=vector(vectors,v);
    for(auto* v:{sc->Druiz,sc->Eruiz,sc->Fruiz,sc->Dinvruiz,sc->Einvruiz,sc->Finvruiz}) scales=vector(scales,v);
    scales=bytes(scales,&sc->k,sizeof(double));scales=bytes(scales,&sc->kinv,sizeof(double));
    const int id=ordinal++;
    std::fprintf(stderr,"{\"case\":\"qoco_trace_input\",\"ordinal\":%d,\"p\":\"%016llx\",\"a\":\"%016llx\",\"g\":\"%016llx\",\"at\":\"%016llx\",\"gt\":\"%016llx\",\"vectors\":\"%016llx\",\"scales\":\"%016llx\",\"warm\":%d,\"x0\":\"%016llx\",\"ir_tol\":%.17g,\"max_ir\":%d,\"dynamic_reg\":%.17g,\"abstol\":%.17g,\"reltol\":%.17g}\n",
        id,matrix(d->P),matrix(d->A),matrix(d->G),matrix(d->At),matrix(d->Gt),vectors,scales,
        w->use_x0,w->use_x0?vector(seed,w->x0):seed,settings->ir_tol,settings->max_ir_iters,
        settings->kkt_dynamic_reg,settings->abstol,settings->reltol);
    const int result=actual(solver);
    const auto* s=solver->sol;
    std::fprintf(stderr,"{\"case\":\"qoco_trace_output\",\"ordinal\":%d,\"status\":%d,\"iters\":%d,\"ir_iters\":%d,\"obj\":%.17g,\"pres\":%.17g,\"dres\":%.17g,\"gap\":%.17g,\"best_iter\":%d,\"best_metric\":%.17g,\"dynamic_reg\":%.17g,\"x\":\"%016llx\"}\n",
        id,result,s->iters,s->ir_iters,s->obj,s->pres,s->dres,s->gap,w->best_iter,w->best_metric,
        settings->kkt_dynamic_reg,vector(seed,w->x));
    return result;
}
