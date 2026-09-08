// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cmath>
#include <cstdint>
#ifdef __CUDACC__
#define SPACEPDHCG_HALPERN_HD __host__ __device__
#else
#define SPACEPDHCG_HALPERN_HD
#endif
namespace spacepdhcg::cuda::halpern {
// Scalar pieces shared with independent CPU counter/sign regressions.
SPACEPDHCG_HALPERN_HD inline double blend(double anchor, double reflected, std::uint64_t inner) {
    const double alpha = 1.0 / (static_cast<double>(inner) + 2.0);
    return alpha * anchor + (1.0 - alpha) * reflected;
}
SPACEPDHCG_HALPERN_HD inline double metric_squared(
    double primal_squared, double dual_squared, double cross, double eta, double omega) {
    return omega * primal_squared + dual_squared / omega - 2.0 * eta * cross;
}
SPACEPDHCG_HALPERN_HD inline bool restart(
    std::uint64_t total, std::uint64_t inner, double error, double initial, double previous) {
    return total == 200 || (total > 200 &&
        (error <= 0.2 * initial || (error <= 0.8 * initial && error > previous)
         || static_cast<double>(inner) >= 0.36 * static_cast<double>(total)));
}
}
#undef SPACEPDHCG_HALPERN_HD
