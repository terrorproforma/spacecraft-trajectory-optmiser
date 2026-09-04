#include "cuda_test_support.hpp"

#include <cstdio>

namespace test = spacepdhcg::cuda::test;

int main() {
    const spacepdhcg_cuda_cone_kind unsupported[] = {
        SPACEPDHCG_CUDA_CONE_EXPONENTIAL,
        SPACEPDHCG_CUDA_CONE_POWER,
        SPACEPDHCG_CUDA_CONE_POSITIVE_SEMIDEFINITE,
    };
    for (const auto kind : unsupported) {
        auto problem = test::make_box_problem(false, true);
        const spacepdhcg_cuda_cone_descriptor descriptor{kind, 0, 1, 0.5};
        problem.structure.variable_cones = &descriptor;
        problem.structure.variable_cone_count = 1U;
        auto options = test::create_options();
        spacepdhcg_cuda_workspace* workspace{nullptr};
        const auto status = spacepdhcg_cuda_workspace_create(
            &problem.structure,
            &problem.exchange,
            &options,
            &workspace
        );
        test::require(
            status == SPACEPDHCG_CUDA_UNSUPPORTED && workspace == nullptr,
            "non-G2 cone was not classified as unsupported"
        );
    }
    std::printf(
        "{\"case\":\"cone_inventory\",\"upstream_supports\":"
        "[\"exponential\",\"power\",\"psd\"],"
        "\"persistent_g2_boundary\":\"explicit_unsupported\","
        "\"numerical_failure\":false}\n"
    );
    return 0;
}
