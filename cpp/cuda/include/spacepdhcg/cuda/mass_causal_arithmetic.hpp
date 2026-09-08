// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cmath>
#include <limits>
#ifdef __CUDACC__
#define SPACEPDHCG_MASS_HD __host__ __device__
#else
#define SPACEPDHCG_MASS_HD
#endif
namespace spacepdhcg::cuda::mass {
// The represented binary64 constant is below the exact rational 19/20.
inline constexpr double theta = 0.95;
// Positive sums/products only. CPU fallback deliberately rounds outward by a
// full ULP; the CUDA path uses directed instructions. Zero is kept exact.
SPACEPDHCG_MASS_HD inline double add_up(double a,double b) {
#ifdef __CUDA_ARCH__
    return __dadd_ru(a,b);
#else
    if(a==0.0)return b;if(b==0.0)return a;
    return std::nextafter(a+b,std::numeric_limits<double>::infinity());
#endif
}
SPACEPDHCG_MASS_HD inline double multiply_up(double a,double b) {
#ifdef __CUDA_ARCH__
    return __dmul_ru(a,b);
#else
    if(a==0.0||b==0.0)return 0.0;
    return std::nextafter(a*b,std::numeric_limits<double>::infinity());
#endif
}
SPACEPDHCG_MASS_HD inline double step_down(double denominator) {
    if(denominator==0.0)return theta; // dummy denominator one; constraint retained
#ifdef __CUDA_ARCH__
    return __ddiv_rd(theta,denominator);
#else
    return std::nextafter(theta/denominator,0.0);
#endif
}
}
#undef SPACEPDHCG_MASS_HD
