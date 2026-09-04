#include "spacepdhcg/distributed/runtime.hpp"
#include "spacepdhcg/distributed/workspace.hpp"

#include "cuda_test_support.hpp"

#include <cuda_runtime.h>
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace g5 = spacepdhcg::distributed::g5;
namespace cuda_test = spacepdhcg::cuda::test;

namespace {

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename T>
class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_ > 0) {
            cuda_check(cudaMalloc(&pointer_, count_ * sizeof(T)), "cudaMalloc");
        }
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() noexcept { return pointer_; }
    const T* get() const noexcept { return pointer_; }
    std::size_t size() const noexcept { return count_; }

    void upload(std::span<const T> source, cudaStream_t stream) {
        require(source.size() == count_, "device upload size mismatch");
        cuda_check(
            cudaMemcpyAsync(
                pointer_,
                source.data(),
                count_ * sizeof(T),
                cudaMemcpyHostToDevice,
                stream
            ),
            "cudaMemcpyAsync upload"
        );
    }

    [[nodiscard]] std::vector<T> download(cudaStream_t stream) const {
        std::vector<T> result(count_);
        cuda_check(
            cudaMemcpyAsync(
                result.data(),
                pointer_,
                count_ * sizeof(T),
                cudaMemcpyDeviceToHost,
                stream
            ),
            "cudaMemcpyAsync download"
        );
        cuda_check(cudaStreamSynchronize(stream), "download synchronization");
        return result;
    }

  private:
    T* pointer_{nullptr};
    std::size_t count_{0};
};

void require_vector(
    std::span<const double> actual,
    std::span<const double> expected,
    const std::string& message
) {
    require(actual.size() == expected.size(), message + " size");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (std::abs(actual[index] - expected[index]) > 1.0e-12) {
            throw std::runtime_error(message + " at index " + std::to_string(index));
        }
    }
}

const char* collective_name(g5::CollectiveKind kind) {
    switch (kind) {
        case g5::CollectiveKind::shared_arrowhead_sum:
            return "shared_arrowhead_sum";
        case g5::CollectiveKind::residual_sum:
            return "residual_sum";
        case g5::CollectiveKind::residual_max:
            return "residual_max";
        case g5::CollectiveKind::expected_risk_sum:
            return "expected_risk_sum";
        case g5::CollectiveKind::worst_risk_max:
            return "worst_risk_max";
        case g5::CollectiveKind::cvar_epigraph_sum:
            return "cvar_epigraph_sum";
        case g5::CollectiveKind::status_max:
            return "status_max";
    }
    return "unknown";
}

void run_one_rank_correctness() {
    g5::MpiNcclRuntime runtime(g5::RuntimeOptions{
        MPI_COMM_WORLD,
        true,
        false,
        0xabcddcba,
        0x12344321,
    });
    require(runtime.rank() == 0, "one-rank test has a nonzero global rank");
    require(runtime.world_size() == 1, "one-rank test was launched with multiple ranks");
    require(runtime.local_rank() == 0 && runtime.device() == 0, "rank-device mapping is not deterministic");

    const std::array<int, 4> offsets{0, 2, 3, 5};
    const std::array<int, 5> indices{0, 1, 1, 0, 1};
    const std::array<double, 5> values{1.0, 2.0, 3.0, 4.0, 5.0};
    const std::array<double, 3> x{2.0, 3.0, 4.0};
    DeviceBuffer<int> device_offsets(offsets.size());
    DeviceBuffer<int> device_indices(indices.size());
    DeviceBuffer<double> device_values(values.size());
    DeviceBuffer<double> device_x(x.size());
    DeviceBuffer<double> device_y(2);
    device_offsets.upload(offsets, runtime.compute_stream());
    device_indices.upload(indices, runtime.compute_stream());
    device_values.upload(values, runtime.compute_stream());
    device_x.upload(x, runtime.compute_stream());
    cuda_check(
        g5::csc_forward_async(
            2,
            3,
            device_offsets.get(),
            device_indices.get(),
            device_values.get(),
            device_x.get(),
            device_y.get(),
            runtime.compute_stream()
        ),
        "Q/A/F forward product"
    );
    const auto y = device_y.download(runtime.compute_stream());
    const std::array<double, 2> expected_y{18.0, 33.0};
    require_vector(y, expected_y, "deterministic CSC forward product");

    const std::array<double, 2> transpose_input{7.0, 11.0};
    DeviceBuffer<double> device_transpose_input(transpose_input.size());
    DeviceBuffer<double> device_transpose_output(3);
    device_transpose_input.upload(transpose_input, runtime.compute_stream());
    cuda_check(
        g5::csc_transpose_async(
            2,
            3,
            device_offsets.get(),
            device_indices.get(),
            device_values.get(),
            device_transpose_input.get(),
            device_transpose_output.get(),
            runtime.compute_stream()
        ),
        "Q/A/F transpose product"
    );
    const auto transpose_output = device_transpose_output.download(runtime.compute_stream());
    const std::array<double, 3> expected_transpose{29.0, 33.0, 83.0};
    require_vector(
        transpose_output,
        expected_transpose,
        "deterministic CSC transpose product"
    );

    const std::array<double, 3> cone_values{1.0, 3.0, 4.0};
    const std::array<int, 1> cone_starts{0};
    const std::array<int, 1> cone_dimensions{3};
    DeviceBuffer<double> device_cone(cone_values.size());
    DeviceBuffer<int> device_cone_starts(1);
    DeviceBuffer<int> device_cone_dimensions(1);
    device_cone.upload(cone_values, runtime.compute_stream());
    device_cone_starts.upload(cone_starts, runtime.compute_stream());
    device_cone_dimensions.upload(cone_dimensions, runtime.compute_stream());
    cuda_check(
        g5::project_soc_blocks_async(
            device_cone.get(),
            device_cone_starts.get(),
            device_cone_dimensions.get(),
            1,
            runtime.compute_stream()
        ),
        "SOC projection"
    );
    const auto projected = device_cone.download(runtime.compute_stream());
    const std::array<double, 3> expected_projected{3.0, 1.8, 2.4};
    require_vector(projected, expected_projected, "SOC projection");

    const std::array<double, 2> arrowhead{2.0, -3.0};
    DeviceBuffer<double> device_arrowhead(arrowhead.size());
    device_arrowhead.upload(arrowhead, runtime.compute_stream());
    runtime.mark_values_updated();
    runtime.mark_warm_started();
    runtime.begin_solve();
    runtime.allreduce_sum(
        device_arrowhead.get(),
        device_arrowhead.size(),
        g5::CollectiveKind::shared_arrowhead_sum,
        1,
        "non-anticipativity shared gradient"
    );
    runtime.allreduce_max(
        device_arrowhead.get(),
        1,
        g5::CollectiveKind::residual_max,
        5,
        "global cone/non-anticipativity/risk residual"
    );
    runtime.synchronize();
    const auto reduced = device_arrowhead.download(runtime.compute_stream());
    require_vector(reduced, arrowhead, "one-rank NCCL device reduction");
    runtime.finish_solve();
    require(runtime.synchronize_status() == g5::RankStatus::healthy, "rank status is unhealthy");

    const auto& telemetry = runtime.telemetry();
    require(telemetry.collectives.size() == 3, "collective telemetry lost a collective class");
    require(
        telemetry.collectives[0].call_count == 1
            && telemetry.collectives[0].element_count == arrowhead.size()
            && telemetry.collectives[0].payload_bytes == arrowhead.size() * sizeof(double)
            && telemetry.collectives[0].wire_bytes_estimate == 0
            && telemetry.collectives[0].frequency == 1
            && !telemetry.collectives[0].purpose.empty(),
        "shared-arrowhead collective telemetry is incomplete"
    );
    require(
        telemetry.collectives[0].collective_seconds >= 0.0
            && telemetry.exposed_communication_seconds >= 0.0,
        "collective timing telemetry is invalid"
    );

    const std::array<std::byte, 4> local_state{
        std::byte{0xaa},
        std::byte{0xbb},
        std::byte{0xcc},
        std::byte{0xdd},
    };
    g5::RankCheckpointHeader checkpoint{};
    checkpoint.topology_fingerprint = 0xabcddcba;
    checkpoint.partition_fingerprint = 0x12344321;
    checkpoint.local_workspace_bytes = local_state.size();
    checkpoint.local_scenario_count = 2;
    checkpoint.primal_elements = 3;
    checkpoint.dual_elements = 2;
    checkpoint.scaling_elements = 3;
    checkpoint.world_size = runtime.world_size();
    checkpoint.rank = runtime.rank();
    checkpoint.device = runtime.device();
    checkpoint.warm_ownership = g5::WarmOwnership::full_state;
    const auto packed = g5::pack_rank_checkpoint(checkpoint, local_state);
    const auto restored = g5::validate_rank_checkpoint(
        packed,
        checkpoint.topology_fingerprint,
        checkpoint.partition_fingerprint,
        runtime.world_size(),
        runtime.rank(),
        runtime.device()
    );
    require(
        restored.warm_ownership == g5::WarmOwnership::full_state,
        "one-rank checkpoint lost full-state warm ownership"
    );

    std::cout << "one-rank MPI/NCCL/CUDA algebra and ordering passed\n";
}

void run_persistent_workspace_composition() {
    auto problem = cuda_test::make_box_problem(false, true);
    const std::vector<g5::ScenarioWork> scenario_work{
        g5::ScenarioWork{2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1},
    };
    auto partition = g5::partition_scenarios(
        scenario_work,
        1,
        g5::PartitionKind::scenario_aware
    );
    g5::ArrowheadMetadata arrowhead{
        2,
        {0, 1},
        0,
        std::nullopt,
        {},
        0xbedabb1e,
    };
    auto create_options = cuda_test::create_options();
    const std::array<g5::LocalScenarioCreate, 1> local_scenarios{{
        g5::LocalScenarioCreate{
            0,
            &problem.structure,
            &problem.exchange,
            &create_options,
        },
    }};
    g5::DistributedWorkspace workspace(
        g5::RuntimeOptions{
            MPI_COMM_WORLD,
            true,
            false,
            problem.fingerprint,
            partition.fingerprint,
        },
        partition,
        arrowhead,
        local_scenarios
    );
    require(workspace.local_scenario_count() == 1, "rank-local workspace ownership is incomplete");

    auto solve_options = cuda_test::solve_options();
    workspace.solve_all_async(solve_options);
    workspace.wait_all();
    workspace.residuals_all_async();
    workspace.wait_all();
    auto diagnostics = workspace.diagnostics();
    require(
        diagnostics.size() == 1
            && diagnostics[0].diagnostics.termination
                   == SPACEPDHCG_CUDA_TERMINATION_OPTIMAL,
        "rank-local persistent solve did not reach the CPU-truth-qualified status"
    );

    const auto checkpoint = workspace.checkpoint();
    workspace.restore(checkpoint);
    auto numeric = problem.numeric_views();
    bool topology_mutation_rejected = false;
    try {
        workspace.update_local_async(0, numeric, problem.fingerprint ^ 1U);
    } catch (const std::invalid_argument&) {
        topology_mutation_rejected = true;
    }
    require(topology_mutation_rejected, "rank-local topology mutation was accepted");
    workspace.update_local_async(0, numeric, problem.fingerprint);
    workspace.warm_start_local_async(
        0,
        SPACEPDHCG_CUDA_WARM_START_FULL_RETAINED,
        nullptr
    );
    workspace.refresh_scaling_all_async();
    workspace.solve_all_async(solve_options);
    workspace.wait_all();
    diagnostics = workspace.diagnostics();
    require(
        diagnostics[0].diagnostics.termination == SPACEPDHCG_CUDA_TERMINATION_OPTIMAL
            && diagnostics[0].diagnostics.warm_start_mode
                   == SPACEPDHCG_CUDA_WARM_START_FULL_RETAINED,
        "checkpoint/restart did not preserve full-state warm ownership"
    );

    const std::array<double, 2> reductions{1.25, -0.75};
    DeviceBuffer<double> device_reductions(reductions.size());
    device_reductions.upload(reductions, workspace.runtime().compute_stream());
    workspace.reduce_shared_arrowhead(device_reductions.get(), reductions.size());
    workspace.reduce_global_residual_sums(device_reductions.get(), 1, 25);
    workspace.reduce_global_residual_maxima(device_reductions.get(), 1, 25);
    workspace.reduce_expected_risk(device_reductions.get(), 1, 1);
    workspace.reduce_worst_risk(device_reductions.get(), 1, 1);
    workspace.reduce_cvar_epigraph(device_reductions.get(), 1, 1);
    workspace.runtime().synchronize();
    require(
        workspace.runtime().synchronize_status() == g5::RankStatus::healthy,
        "persistent workspace status reduction failed"
    );
    const auto reduced = device_reductions.download(workspace.runtime().compute_stream());
    require_vector(reduced, reductions, "persistent one-rank reduction path");
    require(
        workspace.runtime().telemetry().collectives.size() == 7,
        "persistent workspace omitted a collective telemetry class"
    );

    const auto& telemetry = workspace.runtime().telemetry();
    std::cout
        << "{\"schema_version\":\"1.0.0\",\"gate\":\"G5-implementation\","
        << "\"verification\":\"one-rank-mpi-nccl-cuda\",\"rank\":0,\"world_size\":1,"
        << "\"device\":0,\"deterministic\":true,\"overlap\":false,"
        << "\"rank_status\":\"healthy\",\"partition\":{\"kind\":\"scenario_aware\","
        << "\"fingerprint\":\"" << std::hex << std::setw(16) << std::setfill('0')
        << partition.fingerprint << std::dec << "\",\"scenario_owner\":[0],"
        << "\"predicted_rank_load\":[" << partition.predicted_rank_load[0]
        << "],\"measured_rank_load\":null},\"collectives\":[";
    for (std::size_t index = 0; index < telemetry.collectives.size(); ++index) {
        const auto& collective = telemetry.collectives[index];
        if (index > 0) {
            std::cout << ',';
        }
        std::cout
            << "{\"kind\":\"" << collective_name(collective.kind)
            << "\",\"count\":" << collective.call_count
            << ",\"elements\":" << collective.element_count
            << ",\"payload_bytes\":" << collective.payload_bytes
            << ",\"wire_bytes_estimate\":" << collective.wire_bytes_estimate
            << ",\"frequency\":" << collective.frequency
            << ",\"purpose\":\"" << collective.purpose
            << "\",\"collective_seconds\":" << collective.collective_seconds
            << ",\"exposed_seconds\":" << collective.exposed_seconds
            << ",\"overlapped_seconds\":" << collective.overlapped_seconds << '}';
    }
    std::cout
        << "],\"multi_gpu_scaling_verified\":false,"
        << "\"physical_rank_counts_deferred\":[2,4,8]}\n";
    workspace.cancel();
    require(
        workspace.runtime().status() == g5::RankStatus::cancelled,
        "rank-local cancellation did not reach the terminal runtime state"
    );
}

}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    const int initialise = MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (initialise != MPI_SUCCESS) {
        std::cerr << "MPI_Init_thread failed\n";
        return 1;
    }
    int result = 0;
    try {
        require(provided >= MPI_THREAD_SERIALIZED, "MPI thread support is below serialized");
        run_one_rank_correctness();
        run_persistent_workspace_composition();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        result = 1;
    }
    MPI_Finalize();
    return result;
}
