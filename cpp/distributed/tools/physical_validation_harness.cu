#include "spacepdhcg/distributed/runtime.hpp"

#include <cuda_runtime.h>
#include <mpi.h>

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace g5 = spacepdhcg::distributed::g5;

namespace {

struct Arguments {
    std::map<std::string, std::string> values{};
    bool test_mode{false};
};

Arguments parse_arguments(int argc, char** argv) {
    Arguments result{};
    for (int index = 1; index < argc; ++index) {
        const std::string key(argv[index]);
        if (key == "--test-mode") {
            result.test_mode = true;
            continue;
        }
        if (!key.starts_with("--") || index + 1 >= argc) {
            throw std::invalid_argument("every harness option must be --key value");
        }
        result.values.emplace(key, argv[++index]);
    }
    for (const auto* required : {
             "--campaign-mode",
             "--scaling",
             "--partition",
             "--scenarios",
             "--nodes",
             "--risk",
             "--seed",
             "--warmups",
             "--repeats",
             "--evidence-directory",
             "--topology-fingerprint",
             "--partition-fingerprint",
         }) {
        if (!result.values.contains(required)) {
            throw std::invalid_argument(std::string("missing required option ") + required);
        }
    }
    return result;
}

void cuda_require(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status)
        );
    }
}

std::string rank_status_name(g5::RankStatus status) {
    switch (status) {
        case g5::RankStatus::healthy:
            return "healthy";
        case g5::RankStatus::cancelled:
            return "cancelled";
        case g5::RankStatus::failed:
            return "failed";
        case g5::RankStatus::rank_lost:
            return "rank_lost";
    }
    return "unknown";
}

std::string collective_name(g5::CollectiveKind kind) {
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

void write_rank_record(
    const std::filesystem::path& directory,
    const g5::RuntimeTelemetry& telemetry,
    const Arguments& arguments,
    std::string_view outcome,
    bool complete,
    double reduced_value
) {
    std::filesystem::create_directories(directory);
    const auto destination = directory / ("rank-" + std::to_string(telemetry.rank) + ".json");
    std::ofstream stream(destination);
    if (!stream) {
        throw std::runtime_error("could not create rank evidence record");
    }
    stream << std::setprecision(17);
    stream << "{\n";
    stream << "  \"schema_version\": \"1.0.0\",\n";
    stream << "  \"record_type\": \"g5-physical-rank-telemetry\",\n";
    stream << "  \"validation_scope\": \"launch-and-collective-harness-only\",\n";
    stream << "  \"qualification_claim\": false,\n";
    stream << "  \"rank\": " << telemetry.rank << ",\n";
    stream << "  \"world_size\": " << telemetry.world_size << ",\n";
    stream << "  \"local_rank\": " << telemetry.local_rank << ",\n";
    stream << "  \"device\": " << telemetry.device << ",\n";
    stream << "  \"complete\": " << (complete ? "true" : "false") << ",\n";
    stream << "  \"outcome\": " << std::quoted(std::string(outcome)) << ",\n";
    stream << "  \"rank_status\": " << std::quoted(rank_status_name(telemetry.rank_status))
           << ",\n";
    stream << "  \"scaling\": " << std::quoted(arguments.values.at("--scaling")) << ",\n";
    stream << "  \"partition\": " << std::quoted(arguments.values.at("--partition")) << ",\n";
    stream << "  \"risk\": " << std::quoted(arguments.values.at("--risk")) << ",\n";
    stream << "  \"reduced_probe_value\": " << reduced_value << ",\n";
    stream << "  \"local_compute_seconds\": " << telemetry.local_compute_seconds << ",\n";
    stream << "  \"communication_exposed_seconds\": "
           << telemetry.exposed_communication_seconds << ",\n";
    stream << "  \"communication_overlapped_seconds\": "
           << telemetry.overlapped_communication_seconds << ",\n";
    stream << "  \"collectives\": [";
    for (std::size_t index = 0; index < telemetry.collectives.size(); ++index) {
        const auto& item = telemetry.collectives[index];
        if (index != 0) {
            stream << ",";
        }
        stream << "\n    {"
               << "\"kind\": " << std::quoted(collective_name(item.kind))
               << ", \"count\": " << item.call_count
               << ", \"elements\": " << item.element_count
               << ", \"payload_bytes\": " << item.payload_bytes
               << ", \"wire_bytes_estimate\": " << item.wire_bytes_estimate
               << ", \"frequency\": " << item.frequency
               << ", \"purpose\": " << std::quoted(item.purpose)
               << ", \"collective_seconds\": " << item.collective_seconds
               << ", \"exposed_seconds\": " << item.exposed_seconds
               << ", \"overlapped_seconds\": " << item.overlapped_seconds << "}";
    }
    if (!telemetry.collectives.empty()) {
        stream << "\n  ";
    }
    stream << "],\n";
    stream << "  \"memory\": {\"peak_device_bytes\": null, \"free_device_bytes\": null},\n";
    stream << "  \"energy_joules\": null,\n";
    stream << "  \"quality\": {"
           << "\"canonical_primal_residual\": null, "
           << "\"canonical_dual_residual\": null, "
           << "\"canonical_cone_residual\": null, "
           << "\"nonanticipativity_residual\": null, "
           << "\"risk_epigraph_residual\": null, "
           << "\"nonlinear_quality\": null}\n";
    stream << "}\n";
}

void run_checkpoint_injection(
    const g5::MpiNcclRuntime& runtime,
    std::string_view mode,
    const std::filesystem::path& directory
) {
    const std::vector<std::byte> payload(32, std::byte{0x5a});
    g5::RankCheckpointHeader header{};
    header.topology_fingerprint = runtime.topology_fingerprint();
    header.partition_fingerprint = runtime.partition_fingerprint();
    header.local_workspace_bytes = payload.size();
    header.local_scenario_count = 1;
    header.world_size = runtime.world_size();
    header.rank = runtime.rank();
    header.device = runtime.device();
    header.warm_ownership = g5::WarmOwnership::full_state;
    const auto checkpoint = g5::pack_rank_checkpoint(header, payload);
    std::filesystem::create_directories(directory);
    const auto checkpoint_path =
        directory / ("checkpoint-rank-" + std::to_string(runtime.rank()) + ".bin");
    const auto temporary_path = checkpoint_path.string() + ".tmp";
    {
        std::ofstream stream(temporary_path, std::ios::binary);
        stream.write(
            reinterpret_cast<const char*>(checkpoint.data()),
            static_cast<std::streamsize>(checkpoint.size())
        );
        if (!stream) {
            throw std::runtime_error("could not write injected rank checkpoint");
        }
    }
    std::filesystem::rename(temporary_path, checkpoint_path);
    std::ifstream input(checkpoint_path, std::ios::binary | std::ios::ate);
    const auto input_bytes = input.tellg();
    if (input_bytes < 0) {
        throw std::runtime_error("could not size injected rank checkpoint");
    }
    std::vector<std::byte> restored(static_cast<std::size_t>(input_bytes));
    input.seekg(0);
    input.read(
        reinterpret_cast<char*>(restored.data()),
        static_cast<std::streamsize>(restored.size())
    );
    if (!input) {
        throw std::runtime_error("could not read injected rank checkpoint");
    }
    if (mode == "checkpoint_restart") {
        static_cast<void>(g5::validate_rank_checkpoint(
            restored,
            runtime.topology_fingerprint(),
            runtime.partition_fingerprint(),
            runtime.world_size(),
            runtime.rank(),
            runtime.device()
        ));
        return;
    }
    bool rejected = false;
    try {
        static_cast<void>(g5::validate_rank_checkpoint(
            restored,
            mode == "topology_mismatch"
                ? runtime.topology_fingerprint() + 1
                : runtime.topology_fingerprint(),
            runtime.partition_fingerprint(),
            runtime.world_size(),
            runtime.rank(),
            mode == "device_mismatch" ? runtime.device() + 1 : runtime.device()
        ));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected) {
        throw std::runtime_error("injected checkpoint ownership mismatch was accepted");
    }
}

std::string run_injection(
    g5::MpiNcclRuntime& runtime,
    const Arguments& arguments,
    const std::filesystem::path& directory
) {
    const auto mode = arguments.values.at("--inject");
    if (!arguments.test_mode || std::getenv("SPACEPDHCG_G5_FAILURE_TEST") == nullptr) {
        throw std::runtime_error(
            "failure injection requires --test-mode and SPACEPDHCG_G5_FAILURE_TEST"
        );
    }
    if (mode == "rank_failure") {
        if (runtime.world_size() < 2) {
            throw std::runtime_error("rank_failure injection requires at least two ranks");
        }
        if (runtime.rank() == runtime.world_size() - 1) {
            write_rank_record(
                directory,
                runtime.telemetry(),
                arguments,
                "injected-rank-failure",
                false,
                0.0
            );
            std::_Exit(86);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        return "unexpected-rank-failure-survivor";
    }
    if (mode == "communicator_error") {
        MPI_Comm probe = MPI_COMM_NULL;
        MPI_Comm_dup(MPI_COMM_WORLD, &probe);
        MPI_Comm_set_errhandler(probe, MPI_ERRORS_RETURN);
        int value = runtime.rank();
        const int injected_status = MPI_Bcast(
            &value,
            1,
            MPI_INT,
            runtime.world_size(),
            probe
        );
        MPI_Comm_free(&probe);
        if (injected_status == MPI_SUCCESS) {
            throw std::runtime_error("injected MPI communicator error was not reported");
        }
        runtime.fail("injected communicator error");
        static_cast<void>(runtime.synchronize_status());
        return "injected-communicator-error-propagated";
    }
    if (mode == "collective_order") {
        g5::CollectiveOrdering ordering{};
        bool rejected = false;
        try {
            ordering.enqueue();
        } catch (const std::logic_error&) {
            rejected = true;
        }
        if (!rejected) {
            throw std::runtime_error("mismatched collective ordering was accepted");
        }
        return "injected-collective-order-rejected";
    }
    if (mode == "cancellation") {
        runtime.cancel();
        static_cast<void>(runtime.synchronize_status());
        return "injected-cancellation-propagated";
    }
    if (
        mode == "checkpoint_restart" || mode == "topology_mismatch"
        || mode == "device_mismatch"
    ) {
        run_checkpoint_injection(runtime, mode, directory);
        return "injected-" + mode + "-handled";
    }
    if (mode == "timeout") {
        std::this_thread::sleep_for(std::chrono::hours(1));
        return "unexpected-timeout-survivor";
    }
    throw std::invalid_argument("unsupported failure-injection mode");
}

int run_harness(const Arguments& arguments) {
    const auto topology_fingerprint =
        std::stoull(arguments.values.at("--topology-fingerprint"), nullptr, 16);
    const auto partition_fingerprint =
        std::stoull(arguments.values.at("--partition-fingerprint"), nullptr, 16);
    g5::MpiNcclRuntime runtime(g5::RuntimeOptions{
        MPI_COMM_WORLD,
        true,
        true,
        topology_fingerprint,
        partition_fingerprint,
    });
    const std::filesystem::path evidence_directory(
        arguments.values.at("--evidence-directory")
    );
    double reduced_value = static_cast<double>(runtime.rank() + 1);
    double* device_value = nullptr;
    cuda_require(cudaMalloc(&device_value, sizeof(double)), "cudaMalloc launch probe");
    try {
        cuda_require(
            cudaMemcpyAsync(
                device_value,
                &reduced_value,
                sizeof(double),
                cudaMemcpyHostToDevice,
                runtime.compute_stream()
            ),
            "cudaMemcpyAsync launch probe"
        );
        runtime.allreduce_sum(
            device_value,
            1,
            g5::CollectiveKind::shared_arrowhead_sum,
            1,
            "physical launch and rank-device mapping probe"
        );
        runtime.synchronize();
        cuda_require(
            cudaMemcpy(
                &reduced_value,
                device_value,
                sizeof(double),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy launch probe result"
        );
        const double expected =
            static_cast<double>(runtime.world_size() * (runtime.world_size() + 1)) / 2.0;
        if (reduced_value != expected) {
            throw std::runtime_error("NCCL launch-probe allreduce returned an invalid value");
        }
        std::string outcome = "launch-probe-complete";
        if (arguments.values.contains("--inject")) {
            outcome = run_injection(runtime, arguments, evidence_directory);
        }
        write_rank_record(
            evidence_directory,
            runtime.telemetry(),
            arguments,
            outcome,
            true,
            reduced_value
        );
    } catch (...) {
        cudaFree(device_value);
        throw;
    }
    cuda_require(cudaFree(device_value), "cudaFree launch probe");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int result = 0;
    try {
        const auto arguments = parse_arguments(argc, argv);
        result = run_harness(arguments);
    } catch (const std::exception& error) {
        int rank = -1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::cerr << "G5 physical harness rank " << rank << " failed: " << error.what()
                  << '\n';
        result = 2;
    }
    MPI_Finalize();
    return result;
}
