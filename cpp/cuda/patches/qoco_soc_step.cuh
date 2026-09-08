// SPDX-License-Identifier: Apache-2.0
#pragma once
// Included where the prepared QOCO SOC step function was defined.
// Compensated Lorentz inner products for the SOC line-search polynomial.
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

__device__ QOCOFloat soc_step_length_dev(const QOCOFloat* x,
                                         const QOCOFloat* dx, QOCOInt n,
                                         QOCOFloat alpha_max)
{
  const QOCOFloat two = 2.0;
  const QOCOFloat four = 4.0;

  QOCOFloat alpha = alpha_max;

  // ----------------------------------
  // Scalar safeguard: x0 + alpha dx0 >= 0
  // ----------------------------------
  if (x[0] >= 0.0 && dx[0] < 0.0) {
    QOCOFloat a = -x[0] / dx[0];
    if (a < alpha)
      alpha = a;
  }

  // ----------------------------------
  // Compute quadratic coefficients
  // ----------------------------------

  using namespace qoco_cone_arithmetic;
  DD aa=inner(dx,dx,n),bb=mul(DD(2.0),inner(x,dx,n)),cc=inner(x,x,n);
  QOCOFloat a=rounded(aa),b=rounded(bb),c=rounded(cc);
  if (c < 0.0)
    { c = 0.0; cc = DD(0.0); } // retain existing boundary safeguard

  // ----------------------------------
  // Discriminant
  // ----------------------------------
  QOCOFloat d = rounded(add(mul(bb,bb),neg(mul(DD(four),mul(aa,cc)))));

  // ----------------------------------
  // Case analysis (same as Clarabel)
  // ----------------------------------

  // No positive root → no restriction
  if ((a > 0.0 && b > 0.0) || d < 0.0)
    return alpha;

  // Linear case
  if (a == 0.0)
    return b < 0.0 ? qoco_min(alpha, -c / b) : alpha;

  // Boundary case
  if (c == 0.0) {
    if (b < 0.0 || (b == 0.0 && a < 0.0)) return 0.0;
    return a < 0.0 ? qoco_min(alpha, -b / a) : alpha;
  }

  // ----------------------------------
  // Stable quadratic root computation
  // ----------------------------------
  QOCOFloat sqrt_d = qoco_sqrt(d);

  QOCOFloat t;
  if (b >= 0.0)
    t = -b - sqrt_d;
  else
    t = -b + sqrt_d;

  QOCOFloat r1 = (two * c) / t;
  QOCOFloat r2 = t / (two * a);

  // Keep only positive roots
  if (r1 < 0.0)
    r1 = QOCOFloat_MAX;
  if (r2 < 0.0)
    r2 = QOCOFloat_MAX;

  QOCOFloat r = qoco_min(r1, r2);

  if (r < alpha)
    alpha = r;

  return alpha;
}

