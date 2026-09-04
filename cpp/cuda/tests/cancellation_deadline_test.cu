// Cross-thread cancellation must bound every persistent inner solve within a small grace,
// whichever kernel phase the cancel lands in: the PDHG loop, the recovery projected-gradient
// loop, or the recovery KKT/CGLS refinement.
//
// Background (integration/single-gpu-v1, campaign g4-claim-core-4db5047, ordinal 73): an
// adaptive P1-E N=100 censoring twin (600 s deadline, 1,000,000 inner cap) reached the
// recovery kernel on every attempt. Five attempts were cancelled at 600 s; the sixth ran for
// more than 47 minutes and pushed the group past its safety boundary. The recovery kernel read
// the mapped cancellation flag independently in every thread, so a flag that flipped between
// two warps' reads split the block around __syncthreads(), and several recovery phases never
// polled the flag at all. This test cancels from a watchdog thread at offsets spread across the
// PDHG and recovery phases, requires SPACEPDHCG_CUDA_TERMINATION_CANCELLED, and bounds the
// owner's wait latency. A hard watchdog converts a kernel hang into a failure instead of a
// ctest timeout.
//
// Usage: cancellation_deadline_test [--repeats N] [--grace SECONDS] [--sanitizer]
#include "cuda_test_support.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace test = spacepdhcg::cuda::test;

namespace {

// One variable, two contradictory equality rows: PDHG can never satisfy the tolerance, so a
// 350,000-iteration tight solve runs its 300,000 PDHG iterations and then enters the recovery
// kernel (projected gradient, KKT reconstruction), exactly as the campaign twins do.
void initialise_inconsistent_problem(test::ProblemStorage& problem) {
    problem.fingerprint = 0x8e2e5df339787de1ULL;
    problem.variables = 1;
    problem.scalar_rows = 2;
    problem.affine_rows = 0;
    problem.h_q_offsets = {0, 1};
    problem.h_q_indices = {0};
    problem.h_a_offsets = {0, 2};
    problem.h_a_indices = {0, 1};
    problem.h_q = {1.0};
    problem.h_a = {1.0, 1.0};
    problem.h_c = {-0.25};
    problem.h_scalar_lower = {0.0, 1.0};
    problem.h_scalar_upper = {0.0, 1.0};
    problem.h_variable_lower = {
        -std::numeric_limits<double>::infinity(),
    };
    problem.h_variable_upper = {
        std::numeric_limits<double>::infinity(),
    };
    problem.materialise();
}

struct Calibration {
    double total_seconds{0.0};
    double pdhg_seconds{0.0};
    double recovery_seconds{0.0};
    std::uint64_t pdhg_iterations{0U};
    spacepdhcg_cuda_termination termination{SPACEPDHCG_CUDA_TERMINATION_UNSPECIFIED};
};

Calibration calibrate(
    test::ProblemStorage& problem,
    spacepdhcg_cuda_workspace* workspace,
    const spacepdhcg_cuda_solve_options& options
) {
    const auto started = std::chrono::steady_clock::now();
    test::status_require(
        spacepdhcg_cuda_workspace_solve_async(
            workspace, &options, problem.exchange.consumer_stream
        ),
        "calibration launch"
    );
    test::status_require(
        spacepdhcg_cuda_workspace_wait(workspace), "calibration wait"
    );
    Calibration calibration{};
    calibration.total_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started
    ).count();
    spacepdhcg_cuda_diagnostics diagnostics{};
    test::status_require(
        spacepdhcg_cuda_workspace_diagnostics(workspace, &diagnostics),
        "calibration diagnostics"
    );
    calibration.recovery_seconds = diagnostics.recovery_seconds;
    calibration.pdhg_seconds =
        std::max(0.0, diagnostics.solve_seconds - diagnostics.recovery_seconds);
    calibration.pdhg_iterations = diagnostics.iterations;
    calibration.termination = diagnostics.termination;
    test::require(
        diagnostics.recovery_trigger_reason
            == SPACEPDHCG_CUDA_RECOVERY_TIGHT_ITERATION_LIMIT,
        "calibration solve did not enter the recovery kernel"
    );
    test::require(
        diagnostics.termination != SPACEPDHCG_CUDA_TERMINATION_CANCELLED,
        "calibration solve must not be cancelled"
    );
    return calibration;
}

struct StressOutcome {
    int cancelled{0};
    int completed_before_cancel{0};
    int late_returns{0};
    int full_budget_reports{0};
    double maximum_latency_seconds{0.0};
    double maximum_overrun_seconds{0.0};
};

// Launches `repeats` solves and cancels each one from a watchdog thread `offset` seconds after
// the launch. The owner blocks in spacepdhcg_cuda_workspace_wait exactly as the G4 driver
// does. `latency` is the time from the cancel store to the owner's return; `overrun` is the
// solve wall beyond the requested offset. Both must stay within `grace`.
StressOutcome stress(
    const char* phase,
    test::ProblemStorage& problem,
    spacepdhcg_cuda_workspace* workspace,
    const spacepdhcg_cuda_solve_options& options,
    const std::vector<double>& offsets,
    const double grace_seconds,
    const double hang_limit_seconds
) {
    StressOutcome outcome{};
    for (std::size_t repeat = 0; repeat < offsets.size(); ++repeat) {
        const double offset = offsets[repeat];
        std::atomic<bool> returned{false};
        std::atomic<int> cancel_status{-1};
        const auto launched = std::chrono::steady_clock::now();
        test::status_require(
            spacepdhcg_cuda_workspace_solve_async(
                workspace, &options, problem.exchange.consumer_stream
            ),
            "stress launch"
        );
        std::chrono::steady_clock::time_point cancel_at{};
        std::thread watchdog([&]() {
            std::this_thread::sleep_until(
                launched
                + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(offset)
                )
            );
            cancel_at = std::chrono::steady_clock::now();
            cancel_status.store(
                static_cast<int>(spacepdhcg_cuda_workspace_cancel(workspace))
            );
            // Re-assert once per second like the executor's deadline thread, and abort the
            // process if the owner never returns: a hung kernel must fail, not time out.
            const auto hang_deadline = cancel_at
                + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(hang_limit_seconds)
                );
            while (!returned.load(std::memory_order_acquire)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                if (std::chrono::steady_clock::now() >= hang_deadline) {
                    std::fprintf(
                        stderr,
                        "{\"case\":\"cancellation_deadline\",\"phase\":\"%s\",\"repeat\":%zu,"
                        "\"offset_seconds\":%.6f,\"hang_limit_seconds\":%.3f,"
                        "\"failure\":\"owner wait did not return after cancellation\"}\n",
                        phase,
                        repeat,
                        offset,
                        hang_limit_seconds
                    );
                    std::fflush(stderr);
                    std::_Exit(3);
                }
                static_cast<void>(spacepdhcg_cuda_workspace_cancel(workspace));
            }
        });
        test::status_require(
            spacepdhcg_cuda_workspace_wait(workspace), "stress wait"
        );
        const auto finished = std::chrono::steady_clock::now();
        returned.store(true, std::memory_order_release);
        watchdog.join();
        spacepdhcg_cuda_diagnostics diagnostics{};
        test::status_require(
            spacepdhcg_cuda_workspace_diagnostics(workspace, &diagnostics),
            "stress diagnostics"
        );
        const double wall = std::chrono::duration<double>(finished - launched).count();
        if (diagnostics.termination == SPACEPDHCG_CUDA_TERMINATION_CANCELLED) {
            ++outcome.cancelled;
            test::require(
                cancel_status.load() == static_cast<int>(SPACEPDHCG_CUDA_SUCCESS),
                "cancel did not reach a solve that later reported CANCELLED"
            );
            const double latency =
                std::chrono::duration<double>(finished - cancel_at).count();
            const double overrun = wall - offset;
            outcome.maximum_latency_seconds =
                std::max(outcome.maximum_latency_seconds, latency);
            outcome.maximum_overrun_seconds =
                std::max(outcome.maximum_overrun_seconds, overrun);
            // The honest spent-work report: a cancelled solve never claims the full
            // iteration budget, and its residual is finite. Violations are counted and
            // reported per repeat so one run characterises the whole phase; the caller
            // fails the test if any occurred.
            const bool full_budget = diagnostics.iterations >= options.iteration_limit;
            if (full_budget) {
                ++outcome.full_budget_reports;
            }
            if (latency > grace_seconds) {
                ++outcome.late_returns;
            }
            if (latency > grace_seconds || full_budget) {
                std::fprintf(
                    stderr,
                    "{\"case\":\"cancellation_deadline\",\"phase\":\"%s\",\"repeat\":%zu,"
                    "\"offset_seconds\":%.6f,\"latency_seconds\":%.6f,"
                    "\"grace_seconds\":%.3f,\"iterations\":%llu,"
                    "\"iteration_limit\":%llu,\"recovery_iterations\":%llu,"
                    "\"recovery_outcome\":%d,\"late\":%s,\"full_budget_report\":%s}\n",
                    phase,
                    repeat,
                    offset,
                    latency,
                    grace_seconds,
                    static_cast<unsigned long long>(diagnostics.iterations),
                    static_cast<unsigned long long>(options.iteration_limit),
                    static_cast<unsigned long long>(diagnostics.recovery_iterations),
                    static_cast<int>(diagnostics.recovery_outcome_reason),
                    latency > grace_seconds ? "true" : "false",
                    full_budget ? "true" : "false"
                );
                std::fflush(stderr);
            }
            test::require(
                std::isfinite(diagnostics.natural_residual_inf),
                "cancelled solve reported a non-finite residual"
            );
        } else {
            // The solve legitimately finished before the cancel landed; that is only
            // acceptable when the cancel arrived after the solve's natural end.
            ++outcome.completed_before_cancel;
            test::require(
                cancel_status.load() != static_cast<int>(SPACEPDHCG_CUDA_SUCCESS)
                    || wall <= offset + grace_seconds,
                "a solve that accepted a cancel neither cancelled nor returned promptly"
            );
        }
    }
    return outcome;
}

std::vector<double> spread(
    std::mt19937_64& generator,
    const int count,
    const double begin,
    const double end
) {
    std::uniform_real_distribution<double> uniform(begin, end);
    std::vector<double> offsets;
    offsets.reserve(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        offsets.push_back(uniform(generator));
    }
    std::sort(offsets.begin(), offsets.end());
    return offsets;
}

}  // namespace

int main(const int argc, char** argv) {
    int repeats = 24;
    bool repeats_requested = false;
    double grace_seconds = 0.5;
    bool sanitizer = false;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--repeats" && index + 1 < argc) {
            repeats = std::atoi(argv[++index]);
            repeats_requested = true;
        } else if (argument == "--grace" && index + 1 < argc) {
            grace_seconds = std::atof(argv[++index]);
        } else if (argument == "--sanitizer") {
            sanitizer = true;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[index]);
            return 2;
        }
    }
    if (sanitizer) {
        // Compute Sanitizer slows every kernel by one to two orders of magnitude; keep the
        // instrumented run short and let the grace follow the calibrated phase lengths.
        repeats = std::min(repeats, 4);
    }
    test::require(repeats > 0 && grace_seconds > 0.0, "invalid stress parameters");

    test::ProblemStorage problem(false, true);
    initialise_inconsistent_problem(problem);
    auto* workspace = test::create_workspace(problem);
    const auto options = test::solve_options(1.0e-20, 350'000U);
    // The first solve pays module loading and stream warm-up inside its event window; time
    // the phases on the second, warm solve. Under Compute Sanitizer one solve costs minutes,
    // so the instrumented run calibrates once and widens the grace accordingly.
    if (!sanitizer) {
        static_cast<void>(calibrate(problem, workspace, options));
    }
    const Calibration calibration = calibrate(problem, workspace, options);
    test::require(
        calibration.recovery_seconds > 0.0,
        "calibration recorded no recovery time"
    );
    if (sanitizer) {
        grace_seconds = std::max(grace_seconds, 0.25 * calibration.total_seconds);
    }
    if (!repeats_requested) {
        // Keep the default run inside the ctest budget whatever the build type: Debug device
        // code runs the 350,000-iteration solve several times slower than Release.
        const double budget_solves = 60.0 / std::max(0.05, calibration.total_seconds);
        repeats = std::clamp(static_cast<int>(budget_solves / 1.75), 4, repeats);
    }
    const double hang_limit_seconds = std::max(30.0, 4.0 * calibration.total_seconds);
    std::printf(
        "{\"case\":\"cancellation_deadline_calibration\",\"total_seconds\":%.6f,"
        "\"pdhg_seconds\":%.6f,\"recovery_seconds\":%.6f,\"pdhg_iterations\":%llu,"
        "\"termination\":%d,\"grace_seconds\":%.3f,\"repeats\":%d}\n",
        calibration.total_seconds,
        calibration.pdhg_seconds,
        calibration.recovery_seconds,
        static_cast<unsigned long long>(calibration.pdhg_iterations),
        static_cast<int>(calibration.termination),
        grace_seconds,
        repeats
    );
    std::fflush(stdout);

    std::mt19937_64 generator(0x5eed5eedULL);
    // Wall offsets measured from the launch. The PDHG phase spans roughly [0, pdhg); the
    // recovery kernel follows. Recovery offsets stop short of the natural end so the cancel
    // lands inside the kernel rather than after it.
    const double pdhg_end = calibration.pdhg_seconds;
    const double recovery_end = calibration.total_seconds;
    const auto pdhg_offsets = spread(
        generator, std::max(1, repeats / 4), 0.05 * pdhg_end, 0.90 * pdhg_end
    );
    const auto recovery_early = spread(
        generator,
        repeats,
        pdhg_end + 0.02 * calibration.recovery_seconds,
        pdhg_end + 0.35 * calibration.recovery_seconds
    );
    const auto recovery_late = spread(
        generator,
        std::max(1, repeats / 2),
        pdhg_end + 0.35 * calibration.recovery_seconds,
        std::max(pdhg_end + 0.36 * calibration.recovery_seconds, recovery_end - 2.0 * grace_seconds)
    );
    const auto pdhg = stress(
        "pdhg", problem, workspace, options, pdhg_offsets, grace_seconds, hang_limit_seconds
    );
    const auto early = stress(
        "recovery_early", problem, workspace, options, recovery_early, grace_seconds,
        hang_limit_seconds
    );
    const auto late = stress(
        "recovery_late", problem, workspace, options, recovery_late, grace_seconds,
        hang_limit_seconds
    );
    test::require(
        pdhg.cancelled == static_cast<int>(pdhg_offsets.size()),
        "every PDHG-phase cancel must terminate as CANCELLED"
    );
    test::require(
        early.cancelled == static_cast<int>(recovery_early.size()),
        "every early-recovery cancel must terminate as CANCELLED"
    );
    test::require(
        late.cancelled >= static_cast<int>(recovery_late.size()) / 2,
        "late-recovery cancels mostly completed before the cancel landed; the phase was not "
        "exercised"
    );
    test::destroy_workspace(workspace);
    const auto print_phase = [](const char* name, const StressOutcome& outcome) {
        std::printf(
            "\"%s\":{\"cancelled\":%d,\"completed\":%d,\"late_returns\":%d,"
            "\"full_budget_reports\":%d,\"max_latency_seconds\":%.6f,"
            "\"max_overrun_seconds\":%.6f}",
            name,
            outcome.cancelled,
            outcome.completed_before_cancel,
            outcome.late_returns,
            outcome.full_budget_reports,
            outcome.maximum_latency_seconds,
            outcome.maximum_overrun_seconds
        );
    };
    const int late_returns = pdhg.late_returns + early.late_returns + late.late_returns;
    const int full_budget_reports = pdhg.full_budget_reports + early.full_budget_reports
        + late.full_budget_reports;
    std::printf("{\"case\":\"cancellation_deadline\",");
    print_phase("pdhg", pdhg);
    std::printf(",");
    print_phase("recovery_early", early);
    std::printf(",");
    print_phase("recovery_late", late);
    std::printf(
        ",\"grace_seconds\":%.3f,\"late_returns\":%d,\"full_budget_reports\":%d,"
        "\"sanitizer\":%s}\n",
        grace_seconds,
        late_returns,
        full_budget_reports,
        sanitizer ? "true" : "false"
    );
    std::fflush(stdout);
    test::require(
        late_returns == 0,
        "a cancelled solve returned later than the grace allows"
    );
    test::require(
        full_budget_reports == 0,
        "a cancelled solve reported the full iteration budget as spent"
    );
    return 0;
}
