#include "cuda_test_support.hpp"

#include <chrono>
#include <cstdio>

namespace test = spacepdhcg::cuda::test;

int main() {
    auto problem = test::make_box_problem(false, true);
    auto* workspace = test::create_workspace(problem);

    const auto default_stream = spacepdhcg_accelerator_stream{
        {SPACEPDHCG_DEVICE_CUDA, 0},
        0U,
    };
    test::require(
        spacepdhcg_cuda_workspace_reset_async(
            workspace,
            SPACEPDHCG_CUDA_RESET_ITERATES,
            default_stream
        ) == SPACEPDHCG_CUDA_POINTER_CONTRACT,
        "operation accepted a stream other than the declared consumer"
    );

    auto long_solve = test::solve_options(1.0e-30, 100'000'000'000ULL);
    const auto started = std::chrono::steady_clock::now();
    test::status_require(
        spacepdhcg_cuda_workspace_solve_async(
            workspace,
            &long_solve,
            problem.exchange.consumer_stream
        ),
        "long solve launch"
    );
    const auto launch_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started
    ).count();
    test::require(launch_seconds < 1.0, "solve_async blocked on GPU completion");

    auto values = problem.numeric_views();
    test::require(
        spacepdhcg_cuda_workspace_update_async(
            workspace,
            problem.fingerprint,
            &values,
            problem.exchange.consumer_stream
        ) == SPACEPDHCG_CUDA_BUSY,
        "update was accepted while solve was in flight"
    );
    test::status_require(spacepdhcg_cuda_workspace_cancel(workspace), "cancel");
    test::status_require(spacepdhcg_cuda_workspace_wait(workspace), "cancel wait");
    spacepdhcg_cuda_diagnostics diagnostics{};
    test::status_require(
        spacepdhcg_cuda_workspace_diagnostics(workspace, &diagnostics),
        "cancel diagnostics"
    );
    test::require(
        diagnostics.state == SPACEPDHCG_CUDA_CANCELLED
            && diagnostics.termination == SPACEPDHCG_CUDA_TERMINATION_CANCELLED,
        "cooperative cancellation did not reach cancelled state"
    );
    test::require(
        spacepdhcg_cuda_workspace_cancel(workspace) == SPACEPDHCG_CUDA_INVALID_STATE,
        "cancel outside solving state was accepted"
    );
    test::destroy_workspace(workspace);

    auto destruction_problem = test::make_box_problem(false, true);
    auto* destruction_workspace = test::create_workspace(destruction_problem);
    test::status_require(
        spacepdhcg_cuda_workspace_solve_async(
            destruction_workspace,
            &long_solve,
            destruction_problem.exchange.consumer_stream
        ),
        "destruction solve launch"
    );
    const auto destroy_started = std::chrono::steady_clock::now();
    test::destroy_workspace(destruction_workspace);
    const auto destroy_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - destroy_started
    ).count();
    test::require(destroy_seconds < 5.0, "destroy did not safely cancel in-flight work");

    std::printf(
        "{\"case\":\"stream_lifetime\",\"async_launch_seconds\":%.9g,"
        "\"cancel_iterations\":%llu,\"destroy_seconds\":%.9g,"
        "\"wrong_stream_rejected\":true,\"busy_update_rejected\":true}\n",
        launch_seconds,
        static_cast<unsigned long long>(diagnostics.iterations),
        destroy_seconds
    );
    return 0;
}
