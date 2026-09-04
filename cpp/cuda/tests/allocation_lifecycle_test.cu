#include "cuda_test_support.hpp"

#include <cstdio>
#include <cstring>

namespace test = spacepdhcg::cuda::test;

int main() {
    auto problem = test::make_box_problem(false, true);
    auto* workspace = test::create_workspace(problem);
    auto diagnostics = test::solve_and_wait(workspace, problem);
    const auto allocation_count = diagnostics.allocation_count;
    const auto topology_allocations = diagnostics.topology_allocation_count;
    const auto topology_copies = diagnostics.topology_index_copy_count;
    const auto active_bytes = diagnostics.active_bytes;

    spacepdhcg_cuda_pointer_snapshot initial{};
    test::status_require(
        spacepdhcg_cuda_workspace_pointer_snapshot(workspace, &initial),
        "pointer snapshot"
    );
    for (int update = 0; update < 10; ++update) {
        problem.h_c[0] -= 0.001;
        problem.upload_numeric();
        auto values = problem.numeric_views();
        test::status_require(
            spacepdhcg_cuda_workspace_update_async(
                workspace,
                problem.fingerprint,
                &values,
                problem.exchange.consumer_stream
            ),
            "lifecycle update"
        );
        diagnostics = test::solve_and_wait(workspace, problem);
        test::require(
            diagnostics.allocation_count == allocation_count
                && diagnostics.active_bytes == active_bytes
                && diagnostics.topology_allocation_count == topology_allocations
                && diagnostics.topology_index_copy_count == topology_copies,
            "allocation lifecycle changed during hot loop"
        );
    }

    std::size_t checkpoint_bytes{0U};
    test::status_require(
        spacepdhcg_cuda_workspace_checkpoint_bytes(workspace, &checkpoint_bytes),
        "checkpoint bytes"
    );
    test::CudaBuffer<double> checkpoint(checkpoint_bytes / sizeof(double), false);
    auto checkpoint_view = test::view(
        checkpoint.get(),
        checkpoint.size(),
        false,
        SPACEPDHCG_SCALAR_FLOAT64,
        SPACEPDHCG_ACCESS_READ_WRITE
    );
    test::status_require(
        spacepdhcg_cuda_workspace_checkpoint_async(
            workspace,
            checkpoint_view,
            problem.exchange.consumer_stream
        ),
        "checkpoint async"
    );
    test::status_require(spacepdhcg_cuda_workspace_wait(workspace), "checkpoint wait");
    test::require(
        spacepdhcg_cuda_workspace_restore_async(
            workspace,
            problem.fingerprint ^ 0x100U,
            checkpoint_view,
            problem.exchange.consumer_stream
        ) == SPACEPDHCG_CUDA_TOPOLOGY_MISMATCH,
        "incompatible checkpoint was not rejected"
    );
    test::status_require(
        spacepdhcg_cuda_workspace_restore_async(
            workspace,
            problem.fingerprint,
            checkpoint_view,
            problem.exchange.consumer_stream
        ),
        "checkpoint restore"
    );
    diagnostics = test::solve_and_wait(workspace, problem);
    test::require(
        diagnostics.allocation_count == allocation_count
            && diagnostics.topology_allocation_count == topology_allocations,
        "checkpoint lifecycle allocated workspace memory"
    );
    spacepdhcg_cuda_pointer_snapshot final{};
    test::status_require(
        spacepdhcg_cuda_workspace_pointer_snapshot(workspace, &final),
        "final pointer snapshot"
    );
    test::require(
        std::memcmp(&initial, &final, sizeof(initial)) == 0,
        "checkpoint lifecycle changed pointers"
    );
    const auto hot_update_epoch = diagnostics.update_epoch;
    test::destroy_workspace(workspace);

    workspace = test::create_workspace(problem);
    test::status_require(
        spacepdhcg_cuda_workspace_restore_async(
            workspace,
            problem.fingerprint,
            checkpoint_view,
            problem.exchange.consumer_stream
        ),
        "restore into recreated workspace"
    );
    diagnostics = test::solve_and_wait(workspace, problem);
    test::require(
        diagnostics.termination == SPACEPDHCG_CUDA_TERMINATION_OPTIMAL,
        "recreated checkpoint solve did not converge"
    );

    std::printf(
        "{\"case\":\"allocation_lifecycle\",\"updates\":%llu,"
        "\"allocations\":%llu,\"active_allocations\":%llu,"
        "\"active_bytes\":%llu,\"peak_bytes\":%llu,"
        "\"topology_allocations\":%llu,\"topology_index_copies\":%llu,"
        "\"post_create_allocation_delta\":%llu,"
        "\"checkpoint_recreated\":true}\n",
        static_cast<unsigned long long>(hot_update_epoch),
        static_cast<unsigned long long>(diagnostics.allocation_count),
        static_cast<unsigned long long>(diagnostics.active_allocation_count),
        static_cast<unsigned long long>(diagnostics.active_bytes),
        static_cast<unsigned long long>(diagnostics.peak_active_bytes),
        static_cast<unsigned long long>(diagnostics.topology_allocation_count),
        static_cast<unsigned long long>(diagnostics.topology_index_copy_count),
        static_cast<unsigned long long>(diagnostics.allocation_delta_last_update)
    );
    test::destroy_workspace(workspace);
    return 0;
}
