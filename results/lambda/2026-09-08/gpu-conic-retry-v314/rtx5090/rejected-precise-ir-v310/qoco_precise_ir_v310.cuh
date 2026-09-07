// Diagnostic: compensated unregularized KKT residual, entirely on device.
namespace qoco_precise_ir {
struct DD {
    double hi, lo;
    __device__ DD(double h=0.0, double l=0.0):hi(h),lo(l) {}
};
__device__ DD add(DD a, DD b) {
    const double s=__dadd_rn(a.hi,b.hi);
    const double v=__dadd_rn(s,-a.hi);
    double e=__dadd_rn(__dadd_rn(a.hi,-__dadd_rn(s,-v)),__dadd_rn(b.hi,-v));
    e=__dadd_rn(e,__dadd_rn(a.lo,b.lo));
    const double h=__dadd_rn(s,e);
    return {h,__dadd_rn(e,-__dadd_rn(h,-s))};
}
__device__ DD neg(DD a) { return {-a.hi,-a.lo}; }
__device__ DD mul(DD a, DD b) {
    const double h=__dmul_rn(a.hi,b.hi);
    double e=__fma_rn(a.hi,b.hi,-h);
    e=__dadd_rn(e,__dmul_rn(a.hi,b.lo));
    e=__dadd_rn(e,__dmul_rn(a.lo,b.hi));
    e=__dadd_rn(e,__dmul_rn(a.lo,b.lo));
    return add(DD(h),DD(e));
}
__device__ DD reciprocal(DD a) {
    const DD q(__ddiv_rn(1.0,a.hi));
    return add(q,mul(q,add(DD(1.0),neg(mul(a,q)))));
}
__device__ DD load(const double* hi,const double* lo,int i) { return {hi[i],lo?lo[i]:0.0}; }
__device__ void save(DD a,double* hi,double* lo,int i) { hi[i]=a.hi;lo[i]=a.lo; }
// One thread per cone follows the existing compact NT representation. Keeping
// both limbs between the two products avoids rounding W*x before W*(W*x).
__global__ void nt(const double* W,const int* ntidx,const int* socidx,const int* q,
                  int l,int ns,const double* xhi,const double* xlo,
                  double* yhi,double* ylo,bool accumulate) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=l+ns) return;
    if(i<l) {
        DD value=mul(DD(W[i]),load(xhi,xlo,i));
        if(accumulate) value=add(load(yhi,ylo,i),value);
        save(value,yhi,ylo,i);return;
    }
    const int c=i-l,start=socidx[c],offset=ntidx[c],count=q[c];
    const DD scale(W[offset]),w0(W[offset+1]),x0=load(xhi,xlo,start);
    DD zeta;
    for(int j=1;j<count;++j) zeta=add(zeta,mul(DD(W[offset+1+j]),load(xhi,xlo,start+j)));
    const DD coefficient=add(x0,mul(zeta,reciprocal(add(DD(1.0),w0))));
    DD value=mul(scale,add(mul(w0,x0),zeta));
    if(accumulate) value=add(load(yhi,ylo,start),value);
    save(value,yhi,ylo,start);
    for(int j=1;j<count;++j) {
        value=mul(scale,add(load(xhi,xlo,start+j),mul(coefficient,DD(W[offset+1+j]))));
        if(accumulate) value=add(load(yhi,ylo,start+j),value);
        save(value,yhi,ylo,start+j);
    }
}
using Matrix=qoco_fused_kkt::Matrix;
__device__ DD row(Matrix a,const double* x,int r,DD sum) {
    for(int k=a.offsets[r];k<a.offsets[r+1];++k)
        sum=add(sum,neg(mul(DD(a.csc->x[a.entries[k]]),DD(x[a.columns[k]]))));
    return sum;
}
__device__ DD transpose(Matrix a,const double* x,int c,DD sum,bool skip_diagonal=false) {
    for(int e=a.csc->p[c];e<a.csc->p[c+1];++e)
        if(!skip_diagonal || a.csc->i[e]!=c)
            sum=add(sum,neg(mul(DD(a.csc->x[e]),DD(x[a.csc->i[e]]))));
    return sum;
}
__global__ void sparse(Matrix P,Matrix A,Matrix G,int n,int p,int m,double reg,
                       const double* b,const double* x,double* hi,double* lo) {
    const int r=blockIdx.x*blockDim.x+threadIdx.x;
    if(r>=n+p+m) return;
    DD sum(b[r]);
    if(r<n) {
        if(P.csc) sum=transpose(P,x,r,row(P,x,r,sum),true);
        sum=add(sum,mul(DD(reg),DD(x[r])));
        if(p) sum=transpose(A,x+n,r,sum);
        if(m) sum=transpose(G,x+n+p,r,sum);
    } else if(r<n+p) sum=row(A,x,r-n,sum);
    else sum=row(G,x,r-n-p,sum);
    save(sum,hi,lo,r);
}
}
extern "C" void qoco_gpu_precise_ir(QOCOWorkspace* w,const double* b,const double* x,
                                     double* output,double* low,double reg) {
    using namespace qoco_precise_ir;
    auto* d=w->data;
    if(d->P) qoco_materialize_device_transpose(d->P);
    if(d->p) qoco_materialize_device_transpose(d->A);
    if(d->m) qoco_materialize_device_transpose(d->G);
    const int total=d->n+d->p+d->m,blocks=(d->l+d->nsoc+255)/256;
    auto* high=w->ubuff1->d_data;auto* tail=w->ubuff2->d_data;
    if(blocks) nt<<<blocks,256,0,qoco_metric_stream>>>(w->nt_scaling->d_data,
        w->nt_scaling_soc_idx->d_data,w->soc_idx->d_data,d->q->d_data,d->l,d->nsoc,
        x+d->n+d->p,nullptr,high,tail,false);
    sparse<<<(total+255)/256,256,0,qoco_metric_stream>>>(qoco_fused_kkt::matrix(d->P),
        qoco_fused_kkt::matrix(d->A),qoco_fused_kkt::matrix(d->G),d->n,d->p,d->m,reg,b,x,output,low);
    if(blocks) nt<<<blocks,256,0,qoco_metric_stream>>>(w->nt_scaling->d_data,
        w->nt_scaling_soc_idx->d_data,w->soc_idx->d_data,d->q->d_data,d->l,d->nsoc,
        high,tail,output+d->n+d->p,low+d->n+d->p,true);
    CUDA_CHECK(cudaGetLastError());
}
