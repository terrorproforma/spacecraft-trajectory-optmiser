#include "cuda_test_support.hpp"

#include <cstdio>

namespace test = spacepdhcg::cuda::test;

namespace {

struct LifetimeCounter {
    int retains{0};
    int releases{0};
};

void retain(void* context) {
    ++static_cast<LifetimeCounter*>(context)->retains;
}

void release(void* context) {
    ++static_cast<LifetimeCounter*>(context)->releases;
}

}  // namespace

int main() {
    auto problem = test::make_box_problem(false, false);
    LifetimeCounter lifetime{};
    auto options = test::create_options();
    options.external_lifetime_context = &lifetime;
    options.retain_external = retain;
    options.release_external = release;
    auto* workspace = test::create_workspace(problem, options);
    test::require(lifetime.retains == 1 && lifetime.releases == 0, "borrow not retained");

    auto numeric = problem.numeric_views();
    test::require(
        spacepdhcg_cuda_workspace_update_async(
            workspace,
            problem.fingerprint ^ 1U,
            &numeric,
            problem.exchange.consumer_stream
        ) == SPACEPDHCG_CUDA_TOPOLOGY_MISMATCH,
        "topology mutation was not rejected"
    );
    auto invalid_stride = numeric;
    invalid_stride.linear_objective.element_stride = 2;
    test::require(
        spacepdhcg_cuda_workspace_update_async(
            workspace,
            problem.fingerprint,
            &invalid_stride,
            problem.exchange.consumer_stream
        ) == SPACEPDHCG_CUDA_POINTER_CONTRACT,
        "strided value view was not rejected"
    );
    auto invalid_dtype = numeric;
    invalid_dtype.linear_objective.scalar_type = SPACEPDHCG_SCALAR_FLOAT32;
    test::require(
        spacepdhcg_cuda_workspace_update_async(
            workspace,
            problem.fingerprint,
            &invalid_dtype,
            problem.exchange.consumer_stream
        ) == SPACEPDHCG_CUDA_POINTER_CONTRACT,
        "wrong dtype was not rejected"
    );
    auto invalid_device = numeric;
    invalid_device.linear_objective.device.id = 1;
    test::require(
        spacepdhcg_cuda_workspace_update_async(
            workspace,
            problem.fingerprint,
            &invalid_device,
            problem.exchange.consumer_stream
        ) == SPACEPDHCG_CUDA_POINTER_CONTRACT,
        "cross-device value view was not rejected"
    );

    const auto diagnostics = test::solve_and_wait(workspace, problem);
    test::require(
        diagnostics.used_declared_stream == 1 && diagnostics.hidden_cpu_fallback == 0,
        "default stream contract was not recorded"
    );
    test::destroy_workspace(workspace);
    test::require(lifetime.retains == 1 && lifetime.releases == 1, "borrow release count wrong");

    auto alias_problem = test::make_box_problem(false, true);
    alias_problem.exchange.numeric.linear_objective =
        alias_problem.exchange.numeric.quadratic;
    spacepdhcg_cuda_workspace* alias_workspace{nullptr};
    const auto alias_status = spacepdhcg_cuda_workspace_create(
        &alias_problem.structure,
        &alias_problem.exchange,
        &options,
        &alias_workspace
    );
    test::require(
        alias_status == SPACEPDHCG_CUDA_POINTER_CONTRACT && alias_workspace == nullptr,
        "writable alias was not rejected"
    );

    auto topology_problem = test::make_box_problem(false, true);
    topology_problem.exchange.topology_fingerprint ^= 2U;
    spacepdhcg_cuda_workspace* topology_workspace{nullptr};
    const auto topology_status = spacepdhcg_cuda_workspace_create(
        &topology_problem.structure,
        &topology_problem.exchange,
        &options,
        &topology_workspace
    );
    test::require(
        topology_status == SPACEPDHCG_CUDA_TOPOLOGY_MISMATCH
            && topology_workspace == nullptr,
        "create accepted mismatched topology fingerprint"
    );

    std::printf(
        "{\"case\":\"pointer_contract\",\"default_stream\":true,"
        "\"retains\":%d,\"releases\":%d,\"topology_rejected\":true,"
        "\"stride_rejected\":true,\"dtype_rejected\":true,"
        "\"alias_rejected\":true}\n",
        lifetime.retains,
        lifetime.releases
    );
    return 0;
}
