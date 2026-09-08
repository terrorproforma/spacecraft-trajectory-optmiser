// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cmath>

#if defined(__CUDACC__)
#define SPACEPDHCG_KKT_HD __host__ __device__
#else
#define SPACEPDHCG_KKT_HD
#endif

namespace spacepdhcg::cuda::common_kkt {
// Two-component FP64 arithmetic retains cancellation terms in sparse products
// and objective reductions. Overflow is deliberately propagated, never hidden.
struct Sum { double hi{}, lo{}; };
SPACEPDHCG_KKT_HD inline Sum add(Sum a, Sum b) {
    const double s=a.hi+b.hi, v=s-a.hi;
    const double e=(a.hi-(s-v))+(b.hi-v)+a.lo+b.lo;
    const double h=s+e;
    return {h,e-(h-s)};
}
SPACEPDHCG_KKT_HD inline Sum negate(Sum a) { return {-a.hi,-a.lo}; }
SPACEPDHCG_KKT_HD inline Sum product(double a,double b) {
    const double p=a*b;return {p,::fma(a,b,-p)};
}
SPACEPDHCG_KKT_HD inline Sum multiply(Sum a,double b) {
    return add(product(a.hi,b),product(a.lo,b));
}
SPACEPDHCG_KKT_HD inline double value(Sum a) {return a.hi+a.lo;}
SPACEPDHCG_KKT_HD inline double absolute(Sum a) {return ::fabs(value(a));}
SPACEPDHCG_KKT_HD inline bool finite(Sum a) {
    return std::isfinite(a.hi) && std::isfinite(a.lo);
}
SPACEPDHCG_KKT_HD inline Sum square_root(Sum a) {
    const double s=::sqrt(a.hi);
    if(s==0.0)return {s,0.0};
    const Sum remainder=add(a,negate(product(s,s)));
    return add({s,0.0},{value(remainder)/(2.0*s),0.0});
}
} // namespace spacepdhcg::cuda::common_kkt
#undef SPACEPDHCG_KKT_HD
