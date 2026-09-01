#include "spacepdhcg/cuda/orbitweaver_gpu_c_api.h"
#include "spacepdhcg/orbitweaver/lambert_family.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <thread>
#include <vector>

namespace ow = spacepdhcg::orbitweaver;

#ifdef NDEBUG
#  undef assert
#  define assert(condition) ((condition) ? static_cast<void>(0) : std::abort())
#endif

int main() {
    spacepdhcg_orbitweaver_lambert_config config{
        SPACEPDHCG_ORBITWEAVER_GPU_ABI_VERSION,
        0U,
        2U,
        1U,
        512U,
    };
    spacepdhcg_accelerator_stream stream{{SPACEPDHCG_DEVICE_CUDA, 0}, 0U};
    spacepdhcg_orbitweaver_lambert_workspace* workspace = nullptr;
    assert(
        spacepdhcg_orbitweaver_lambert_workspace_create(
            &config, stream, &workspace
        )
        == SPACEPDHCG_CUDA_SUCCESS
    );
    spacepdhcg_orbitweaver_lambert_request requests[2]{};
    requests[0].deterministic_id = 17U;
    requests[0].departure_position[0] = 7.0e6;
    requests[0].arrival_position[1] = 8.0e6;
    requests[0].time_of_flight = 3'600.0;
    requests[0].gravitational_parameter = 3.986004418e14;
    requests[0].time_tolerance = 1.0e-8;
    requests[0].maximum_iterations = 256U;
    requests[0].maximum_revolutions = 1U;
    requests[0].include_short_way = 1;
    requests[1] = requests[0];
    requests[1].deterministic_id = 18U;
    requests[1].arrival_position[0] = 8.0e6;
    requests[1].arrival_position[1] = 0.0;

    const auto stride = spacepdhcg_orbitweaver_lambert_result_stride(1U);
    std::vector<spacepdhcg_orbitweaver_lambert_result> results(2U * stride);
    assert(
        spacepdhcg_orbitweaver_lambert_evaluate_async(
            workspace, requests, 2U, results.data(), results.size(), stream
        )
        == SPACEPDHCG_CUDA_SUCCESS
    );
    spacepdhcg_orbitweaver_batch_telemetry telemetry{};
    telemetry.abi_version = SPACEPDHCG_ORBITWEAVER_GPU_ABI_VERSION;
    auto status =
        spacepdhcg_orbitweaver_lambert_workspace_telemetry(workspace, &telemetry);
    for (std::size_t attempt = 0U;
         status == SPACEPDHCG_CUDA_BUSY && attempt < 10'000U;
         ++attempt) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        status =
            spacepdhcg_orbitweaver_lambert_workspace_telemetry(workspace, &telemetry);
    }
    assert(status == SPACEPDHCG_CUDA_SUCCESS);
    assert(telemetry.requests_submitted == 2U);
    assert(results[0].status == SPACEPDHCG_ORBITWEAVER_ARC_FEASIBLE);
    for (std::size_t slot = stride; slot < 2U * stride; ++slot) {
        assert(results[slot].status == SPACEPDHCG_ORBITWEAVER_ARC_INVALID_INPUT);
    }
    const auto cpu = ow::enumerate_lambert_families(
        {7.0e6, 0.0, 0.0},
        {0.0, 8.0e6, 0.0},
        3'600.0,
        3.986004418e14,
        1U,
        true,
        false,
        1.0e-8,
        256U,
        512U
    );
    assert(
        std::abs(
            results[0].universal_parameter
            - cpu.front().solution.universal_parameter
        )
        < 1.0e-10
    );
    for (std::size_t component = 0U; component < 3U; ++component) {
        assert(
            std::abs(
                results[0].departure_velocity[component]
                - cpu.front().solution.departure_velocity[component]
            )
            < 1.0e-7
        );
    }
    assert(
        spacepdhcg_orbitweaver_lambert_workspace_destroy(&workspace)
        == SPACEPDHCG_CUDA_SUCCESS
    );
    return 0;
}
