/*
 * Persistent composition of G2 workspaces under one rank-local G5 runtime.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "spacepdhcg/distributed/workspace.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace spacepdhcg::distributed::g5 {
namespace {

struct LocalCheckpointEntry {
    std::uint64_t scenario{0};
    std::uint64_t topology_fingerprint{0};
    std::uint64_t offset{0};
    std::uint64_t bytes{0};
};

void cuda_require(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

WarmOwnership warm_ownership(spacepdhcg_cuda_warm_start_mode mode) {
    switch (mode) {
        case SPACEPDHCG_CUDA_WARM_START_NONE:
            return WarmOwnership::none;
        case SPACEPDHCG_CUDA_WARM_START_PRIMAL:
            return WarmOwnership::primal;
        case SPACEPDHCG_CUDA_WARM_START_PRIMAL_DUAL:
            return WarmOwnership::primal_dual;
        case SPACEPDHCG_CUDA_WARM_START_FULL_RETAINED:
            return WarmOwnership::full_state;
    }
    throw std::invalid_argument("unknown local warm-start mode");
}

spacepdhcg_accelerator_buffer_view checkpoint_view(
    void* pointer,
    std::size_t bytes,
    int device
) {
    if (bytes % sizeof(double) != 0) {
        throw std::logic_error("G2 checkpoint size is not float64 aligned");
    }
    return spacepdhcg_accelerator_buffer_view{
        pointer,
        spacepdhcg_accelerator_device{SPACEPDHCG_DEVICE_CUDA, device},
        SPACEPDHCG_SCALAR_FLOAT64,
        bytes / sizeof(double),
        0,
        1,
        SPACEPDHCG_ACCESS_READ_WRITE,
    };
}

}  // namespace

DistributedWorkspace::DistributedWorkspace(
    RuntimeOptions runtime_options,
    PartitionPlan partition,
    ArrowheadMetadata arrowhead,
    std::span<const LocalScenarioCreate> local_scenarios
)
    : runtime_(runtime_options),
      partition_(std::move(partition)),
      arrowhead_(std::move(arrowhead)) {
    partition_.validate(partition_.scenario_owner.size(), runtime_.world_size());
    arrowhead_.validate();
    if (runtime_options.partition_fingerprint != partition_.fingerprint) {
        throw std::invalid_argument("runtime and scenario partition fingerprints differ");
    }
    const auto& expected =
        partition_.rank_scenarios[static_cast<std::size_t>(runtime_.rank())];
    if (local_scenarios.size() != expected.size()) {
        throw std::invalid_argument("rank did not receive exactly its owned whole scenarios");
    }

    std::vector<std::size_t> supplied{};
    supplied.reserve(local_scenarios.size());
    for (const auto& input : local_scenarios) {
        supplied.push_back(input.scenario);
    }
    std::sort(supplied.begin(), supplied.end());
    if (supplied != expected) {
        throw std::invalid_argument("local scenario inputs do not match deterministic ownership");
    }

    try {
        local_.reserve(local_scenarios.size());
        for (const auto& input : local_scenarios) {
            if (input.structure == nullptr || input.exchange == nullptr || input.options == nullptr) {
                throw std::invalid_argument("local scenario workspace input is incomplete");
            }
            if (input.exchange->consumer_stream.device.type != SPACEPDHCG_DEVICE_CUDA
                || input.exchange->consumer_stream.device.id != runtime_.device()) {
                throw std::invalid_argument("local scenario exchange is not owned by the rank GPU");
            }
            OwnedScenario local{
                input.scenario,
                input.structure->topology_fingerprint,
                static_cast<std::size_t>(input.structure->variables),
                static_cast<std::size_t>(
                    input.structure->scalar_rows + input.structure->affine_rows
                ),
                nullptr,
            };
            auto local_exchange = *input.exchange;
            local_exchange.consumer_stream = stream();
            require_success(
                spacepdhcg_cuda_workspace_create(
                    input.structure,
                    &local_exchange,
                    input.options,
                    &local.workspace
                ),
                "create local persistent scenario workspace"
            );
            local_.push_back(local);
        }
    } catch (...) {
        for (auto& local : local_) {
            spacepdhcg_cuda_workspace_destroy(&local.workspace);
        }
        runtime_.fail("rank-local persistent workspace creation failed");
        throw;
    }
}

DistributedWorkspace::~DistributedWorkspace() {
    for (auto& local : local_) {
        if (local.workspace != nullptr) {
            spacepdhcg_cuda_workspace_destroy(&local.workspace);
        }
    }
}

spacepdhcg_accelerator_stream DistributedWorkspace::stream() const noexcept {
    return spacepdhcg_accelerator_stream{
        spacepdhcg_accelerator_device{SPACEPDHCG_DEVICE_CUDA, runtime_.device()},
        reinterpret_cast<std::uintptr_t>(runtime_.compute_stream()),
    };
}

DistributedWorkspace::OwnedScenario& DistributedWorkspace::owned(std::size_t scenario) {
    const auto iterator = std::find_if(local_.begin(), local_.end(), [scenario](const auto& local) {
        return local.scenario == scenario;
    });
    if (iterator == local_.end()) {
        throw std::invalid_argument("scenario is not owned by this rank");
    }
    return *iterator;
}

const DistributedWorkspace::OwnedScenario& DistributedWorkspace::owned(
    std::size_t scenario
) const {
    const auto iterator = std::find_if(local_.begin(), local_.end(), [scenario](const auto& local) {
        return local.scenario == scenario;
    });
    if (iterator == local_.end()) {
        throw std::invalid_argument("scenario is not owned by this rank");
    }
    return *iterator;
}

void DistributedWorkspace::require_success(
    spacepdhcg_cuda_status status,
    const char* operation
) {
    if (status != SPACEPDHCG_CUDA_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with status " + std::to_string(status)
        );
    }
}

void DistributedWorkspace::update_local_async(
    std::size_t scenario,
    const spacepdhcg_cqp_numeric_accelerator_views& values,
    std::uint64_t topology_fingerprint
) {
    auto& local = owned(scenario);
    if (topology_fingerprint != local.topology_fingerprint) {
        throw std::invalid_argument("local scenario topology mutated");
    }
    require_success(
        spacepdhcg_cuda_workspace_update_async(
            local.workspace,
            topology_fingerprint,
            &values,
            stream()
        ),
        "update local scenario values"
    );
    runtime_.mark_values_updated();
}

void DistributedWorkspace::warm_start_local_async(
    std::size_t scenario,
    spacepdhcg_cuda_warm_start_mode mode,
    const spacepdhcg_cqp_iterate_accelerator_views* iterates
) {
    auto& local = owned(scenario);
    require_success(
        spacepdhcg_cuda_workspace_warm_start_async(
            local.workspace,
            mode,
            iterates,
            stream()
        ),
        "apply local scenario warm start"
    );
    warm_ownership_ = std::max(warm_ownership_, warm_ownership(mode));
    runtime_.mark_warm_started();
}

void DistributedWorkspace::refresh_scaling_all_async() {
    for (auto& local : local_) {
        require_success(
            spacepdhcg_cuda_workspace_refresh_scaling_async(local.workspace, stream()),
            "refresh local scenario scaling"
        );
    }
}

void DistributedWorkspace::solve_all_async(const spacepdhcg_cuda_solve_options& options) {
    runtime_.begin_solve();
    try {
        for (auto& local : local_) {
            require_success(
                spacepdhcg_cuda_workspace_solve_async(local.workspace, &options, stream()),
                "solve local scenario"
            );
        }
    } catch (const std::exception& error) {
        runtime_.fail(error.what());
        throw;
    }
}

void DistributedWorkspace::residuals_all_async() {
    for (auto& local : local_) {
        require_success(
            spacepdhcg_cuda_workspace_residuals_async(local.workspace, stream()),
            "compute local scenario residuals"
        );
    }
}

void DistributedWorkspace::wait_all() {
    for (auto& local : local_) {
        require_success(spacepdhcg_cuda_workspace_wait(local.workspace), "wait local scenario");
    }
    runtime_.synchronize();
    if (runtime_.state() == RuntimeState::solving) {
        runtime_.finish_solve();
    }
}

void DistributedWorkspace::reduce_shared_arrowhead(
    double* device_values,
    std::size_t count,
    std::uint64_t frequency
) {
    if (count != arrowhead_.shared_primal_indices.size()) {
        throw std::invalid_argument("shared-arrowhead payload dimension is inconsistent");
    }
    runtime_.allreduce_sum(
        device_values,
        count,
        CollectiveKind::shared_arrowhead_sum,
        frequency,
        "non-anticipativity shared primal/gradient"
    );
}

void DistributedWorkspace::reduce_global_residual_sums(
    double* device_values,
    std::size_t count,
    std::uint64_t frequency
) {
    runtime_.allreduce_sum(
        device_values,
        count,
        CollectiveKind::residual_sum,
        frequency,
        "global squared primal/dual/gap residuals"
    );
}

void DistributedWorkspace::reduce_global_residual_maxima(
    double* device_values,
    std::size_t count,
    std::uint64_t frequency
) {
    runtime_.allreduce_max(
        device_values,
        count,
        CollectiveKind::residual_max,
        frequency,
        "global cone/non-anticipativity/risk residuals"
    );
}

void DistributedWorkspace::reduce_expected_risk(
    double* device_values,
    std::size_t count,
    std::uint64_t frequency
) {
    runtime_.allreduce_sum(
        device_values,
        count,
        CollectiveKind::expected_risk_sum,
        frequency,
        "probability-weighted scenario objective/risk"
    );
}

void DistributedWorkspace::reduce_worst_risk(
    double* device_values,
    std::size_t count,
    std::uint64_t frequency
) {
    runtime_.allreduce_max(
        device_values,
        count,
        CollectiveKind::worst_risk_max,
        frequency,
        "worst-case epigraph loss and violation"
    );
}

void DistributedWorkspace::reduce_cvar_epigraph(
    double* device_values,
    std::size_t count,
    std::uint64_t frequency
) {
    runtime_.allreduce_sum(
        device_values,
        count,
        CollectiveKind::cvar_epigraph_sum,
        frequency,
        "CVaR weighted excess and threshold dual"
    );
}

std::vector<LocalScenarioDiagnostics> DistributedWorkspace::diagnostics() const {
    std::vector<LocalScenarioDiagnostics> result{};
    result.reserve(local_.size());
    for (const auto& local : local_) {
        spacepdhcg_cuda_diagnostics diagnostics{};
        diagnostics.abi_version = SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION;
        require_success(
            spacepdhcg_cuda_workspace_diagnostics(local.workspace, &diagnostics),
            "read local scenario diagnostics"
        );
        result.push_back(LocalScenarioDiagnostics{local.scenario, diagnostics});
    }
    return result;
}

std::vector<std::byte> DistributedWorkspace::checkpoint() {
    wait_all();
    std::vector<LocalCheckpointEntry> entries{};
    std::vector<std::byte> payload{};
    entries.reserve(local_.size());
    for (auto& local : local_) {
        std::size_t bytes = 0;
        require_success(
            spacepdhcg_cuda_workspace_checkpoint_bytes(local.workspace, &bytes),
            "query local checkpoint bytes"
        );
        void* device_checkpoint = nullptr;
        cuda_require(cudaMalloc(&device_checkpoint, bytes), "allocate local checkpoint staging");
        try {
            auto view = checkpoint_view(device_checkpoint, bytes, runtime_.device());
            require_success(
                spacepdhcg_cuda_workspace_checkpoint_async(
                    local.workspace,
                    view,
                    stream()
                ),
                "checkpoint local scenario"
            );
            require_success(
                spacepdhcg_cuda_workspace_wait(local.workspace),
                "wait local scenario checkpoint"
            );
            const auto offset = payload.size();
            payload.resize(offset + bytes);
            cuda_require(
                cudaMemcpyAsync(
                    payload.data() + offset,
                    device_checkpoint,
                    bytes,
                    cudaMemcpyDeviceToHost,
                    runtime_.compute_stream()
                ),
                "copy local checkpoint to host"
            );
            cuda_require(
                cudaStreamSynchronize(runtime_.compute_stream()),
                "synchronize local checkpoint copy"
            );
            entries.push_back(LocalCheckpointEntry{
                local.scenario,
                local.topology_fingerprint,
                offset,
                bytes,
            });
        } catch (...) {
            cudaFree(device_checkpoint);
            throw;
        }
        cuda_require(cudaFree(device_checkpoint), "free local checkpoint staging");
    }

    std::vector<std::byte> body(
        sizeof(std::uint64_t) + entries.size() * sizeof(LocalCheckpointEntry) + payload.size()
    );
    const auto entry_count = static_cast<std::uint64_t>(entries.size());
    std::memcpy(body.data(), &entry_count, sizeof(entry_count));
    if (!entries.empty()) {
        std::memcpy(
            body.data() + sizeof(entry_count),
            entries.data(),
            entries.size() * sizeof(LocalCheckpointEntry)
        );
    }
    if (!payload.empty()) {
        std::memcpy(
            body.data() + sizeof(entry_count) + entries.size() * sizeof(LocalCheckpointEntry),
            payload.data(),
            payload.size()
        );
    }

    RankCheckpointHeader header{};
    header.topology_fingerprint = runtime_.topology_fingerprint();
    header.partition_fingerprint = partition_.fingerprint;
    header.local_workspace_bytes = body.size();
    header.local_scenario_count = local_.size();
    for (const auto& local : local_) {
        header.primal_elements += local.primal_elements;
        header.dual_elements += local.dual_elements;
        header.scaling_elements += local.primal_elements;
    }
    header.world_size = runtime_.world_size();
    header.rank = runtime_.rank();
    header.device = runtime_.device();
    header.warm_ownership = warm_ownership_;
    return pack_rank_checkpoint(header, body);
}

void DistributedWorkspace::restore(std::span<const std::byte> checkpoint_bytes) {
    const auto header = validate_rank_checkpoint(
        checkpoint_bytes,
        runtime_.topology_fingerprint(),
        partition_.fingerprint,
        runtime_.world_size(),
        runtime_.rank(),
        runtime_.device()
    );
    const auto body = checkpoint_bytes.subspan(sizeof(RankCheckpointHeader));
    if (body.size() < sizeof(std::uint64_t)) {
        throw std::invalid_argument("distributed checkpoint table is truncated");
    }
    std::uint64_t entry_count = 0;
    std::memcpy(&entry_count, body.data(), sizeof(entry_count));
    if (entry_count != local_.size()
        || body.size() < sizeof(entry_count) + entry_count * sizeof(LocalCheckpointEntry)) {
        throw std::invalid_argument("distributed checkpoint local ownership changed");
    }
    std::vector<LocalCheckpointEntry> entries(entry_count);
    std::memcpy(
        entries.data(),
        body.data() + sizeof(entry_count),
        entries.size() * sizeof(LocalCheckpointEntry)
    );
    const auto payload = body.subspan(
        sizeof(entry_count) + entries.size() * sizeof(LocalCheckpointEntry)
    );
    for (const auto& entry : entries) {
        auto& local = owned(entry.scenario);
        if (entry.topology_fingerprint != local.topology_fingerprint
            || entry.offset > payload.size() || entry.bytes > payload.size() - entry.offset) {
            throw std::invalid_argument("local checkpoint topology or bounds are incompatible");
        }
        void* device_checkpoint = nullptr;
        cuda_require(
            cudaMalloc(&device_checkpoint, static_cast<std::size_t>(entry.bytes)),
            "allocate local restore staging"
        );
        try {
            cuda_require(
                cudaMemcpyAsync(
                    device_checkpoint,
                    payload.data() + entry.offset,
                    static_cast<std::size_t>(entry.bytes),
                    cudaMemcpyHostToDevice,
                    runtime_.compute_stream()
                ),
                "copy local checkpoint to device"
            );
            auto view = checkpoint_view(
                device_checkpoint,
                static_cast<std::size_t>(entry.bytes),
                runtime_.device()
            );
            require_success(
                spacepdhcg_cuda_workspace_restore_async(
                    local.workspace,
                    local.topology_fingerprint,
                    view,
                    stream()
                ),
                "restore local scenario"
            );
            require_success(
                spacepdhcg_cuda_workspace_wait(local.workspace),
                "wait local scenario restore"
            );
        } catch (...) {
            cudaFree(device_checkpoint);
            throw;
        }
        cuda_require(cudaFree(device_checkpoint), "free local restore staging");
    }
    warm_ownership_ = header.warm_ownership;
    runtime_.mark_warm_started();
}

void DistributedWorkspace::cancel() noexcept {
    for (auto& local : local_) {
        spacepdhcg_cuda_workspace_cancel(local.workspace);
    }
    runtime_.cancel();
}

}  // namespace spacepdhcg::distributed::g5
