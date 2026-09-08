// Diagnostic: compensated unregularized KKT residual, entirely on device.
namespace qoco_cone_arithmetic {
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
__device__ DD inner(const double* u,const double* v,int n) {
    DD sum=mul(DD(u[0]),DD(v[0]));
    for(int i=1;i<n;++i) sum=add(sum,neg(mul(DD(u[i]),DD(v[i]))));
    return sum;
}
__device__ double rounded(DD a) {return __dadd_rn(a.hi,a.lo);}
__device__ double determinant(const double* u,int n) {return rounded(inner(u,u,n));}
}
