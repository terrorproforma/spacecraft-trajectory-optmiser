// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cmath>
#ifdef __CUDACC__
#define SPACEPDHCG_L1_HD __host__ __device__
#else
#define SPACEPDHCG_L1_HD
#endif
namespace spacepdhcg::cuda::l1 {
// Caller checks that argument and threshold are finite and threshold>0.
SPACEPDHCG_L1_HD inline double soft_threshold(double argument,double threshold) {
    return argument>threshold ? argument-threshold
        : argument < -threshold ? argument+threshold : 0.0;
}
struct DualPair { double positive,negative; };
// Exact-zero means the represented FP64 value, with no activity tolerance.
// Completion is for the exported original QP point, not the working dual map.
SPACEPDHCG_L1_HD inline DualPair complete_dual(double value,double retained_gradient,double lambda) {
    if(value>0.0)return{lambda,0.0};
    if(value<0.0)return{0.0,lambda};
    double delta=-retained_gradient;
    if(delta>lambda)delta=lambda;
    if(delta < -lambda)delta=-lambda;
    // Avoid lambda+abs(delta) overflow. Each subtraction is between nonnegative
    // quantities no larger than lambda, and the two outputs remain nonnegative.
    if(delta>=0.0){const double negative=0.5*(lambda-delta);return{lambda-negative,negative};}
    const double positive=0.5*(lambda+delta);return{positive,lambda-positive};
}
}
#undef SPACEPDHCG_L1_HD
