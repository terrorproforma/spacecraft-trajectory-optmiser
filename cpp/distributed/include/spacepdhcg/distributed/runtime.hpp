/*
 * Scenario-aware MPI/NCCL runtime contracts for Gate G5.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda_runtime_api.h>
#include <mpi.h>
#include <nccl.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace spacepdhcg::distributed::g5 {

enum class PartitionKind : std::uint8_t {
    scenario_aware,
    nonzero_balanced,
};

enum class ConeKind : std::uint8_t {
    second_order,
    rotated_second_order,
    exponential,
    power,
    positive_semidefinite,
};

struct ScenarioWork {
    std::uint64_t q_nonzeros{0};
    std::uint64_t a_nonzeros{0};
    std::uint64_t f_nonzeros{0};
    std::uint64_t second_order_slots{0};
    std::uint64_t rotated_second_order_slots{0};
    std::uint64_t exponential_slots{0};
    std::uint64_t power_slots{0};
    std::uint64_t semidefinite_slots{0};
    std::uint64_t time_nodes{0};
    std::uint64_t replay_work{0};
    std::uint64_t risk_work{0};
    std::uint64_t update_work{0};
};

struct ScenarioCostModel {
    double q_nonzero{1.0};
    double a_nonzero{1.0};
    double f_nonzero{1.25};
    double second_order_slot{2.0};
    double rotated_second_order_slot{2.5};
    double exponential_slot{8.0};
    double power_slot{8.0};
    double semidefinite_slot{16.0};
    double time_node{4.0};
    double replay_work{1.0};
    double risk_work{2.0};
    double update_work{1.0};

    void validate() const {
        const std::array values{
            q_nonzero,
            a_nonzero,
            f_nonzero,
            second_order_slot,
            rotated_second_order_slot,
            exponential_slot,
            power_slot,
            semidefinite_slot,
            time_node,
            replay_work,
            risk_work,
            update_work,
        };
        if (std::any_of(values.begin(), values.end(), [](double value) {
                return !std::isfinite(value) || value < 0.0;
            })) {
            throw std::invalid_argument("scenario cost coefficients must be finite and non-negative");
        }
    }
};

[[nodiscard]] inline double scenario_cost(
    const ScenarioWork& work,
    const ScenarioCostModel& model,
    PartitionKind kind
) {
    model.validate();
    const auto nonzeros = model.q_nonzero * static_cast<double>(work.q_nonzeros)
                          + model.a_nonzero * static_cast<double>(work.a_nonzeros)
                          + model.f_nonzero * static_cast<double>(work.f_nonzeros);
    if (kind == PartitionKind::nonzero_balanced) {
        return nonzeros;
    }
    return nonzeros
           + model.second_order_slot * static_cast<double>(work.second_order_slots)
           + model.rotated_second_order_slot
                 * static_cast<double>(work.rotated_second_order_slots)
           + model.exponential_slot * static_cast<double>(work.exponential_slots)
           + model.power_slot * static_cast<double>(work.power_slots)
           + model.semidefinite_slot * static_cast<double>(work.semidefinite_slots)
           + model.time_node * static_cast<double>(work.time_nodes)
           + model.replay_work * static_cast<double>(work.replay_work)
           + model.risk_work * static_cast<double>(work.risk_work)
           + model.update_work * static_cast<double>(work.update_work);
}

struct MeasuredRankLoad {
    std::size_t rank{0};
    double local_compute_seconds{0.0};
    double exposed_communication_seconds{0.0};
    double overlapped_communication_seconds{0.0};
    std::uint64_t work_units{0};

    void validate(std::size_t world_size) const {
        if (rank >= world_size) {
            throw std::invalid_argument("measured load rank is outside the partition");
        }
        const std::array times{
            local_compute_seconds,
            exposed_communication_seconds,
            overlapped_communication_seconds,
        };
        if (std::any_of(times.begin(), times.end(), [](double value) {
                return !std::isfinite(value) || value < 0.0;
            })) {
            throw std::invalid_argument("measured load timings must be finite and non-negative");
        }
    }
};

struct PartitionPlan {
    PartitionKind kind{PartitionKind::scenario_aware};
    std::vector<std::vector<std::size_t>> rank_scenarios{};
    std::vector<std::size_t> scenario_owner{};
    std::vector<double> predicted_rank_load{};
    std::vector<MeasuredRankLoad> measured_rank_load{};
    std::uint64_t fingerprint{0};

    [[nodiscard]] double predicted_imbalance() const noexcept {
        if (predicted_rank_load.empty()) {
            return 1.0;
        }
        const auto mean = std::accumulate(
                              predicted_rank_load.begin(),
                              predicted_rank_load.end(),
                              0.0
                          )
                          / static_cast<double>(predicted_rank_load.size());
        return mean > 0.0
                   ? *std::max_element(
                         predicted_rank_load.begin(),
                         predicted_rank_load.end()
                     )
                         / mean
                   : 1.0;
    }

    void validate(std::size_t scenario_count, std::size_t world_size) const {
        if (world_size == 0 || rank_scenarios.size() != world_size
            || predicted_rank_load.size() != world_size
            || scenario_owner.size() != scenario_count) {
            throw std::invalid_argument("partition dimensions are inconsistent");
        }
        std::vector<std::uint8_t> seen(scenario_count, 0);
        for (std::size_t rank = 0; rank < world_size; ++rank) {
            if (!std::is_sorted(rank_scenarios[rank].begin(), rank_scenarios[rank].end())) {
                throw std::invalid_argument("rank scenario ownership must be sorted");
            }
            for (const auto scenario : rank_scenarios[rank]) {
                if (scenario >= scenario_count || seen[scenario] != 0
                    || scenario_owner[scenario] != rank) {
                    throw std::invalid_argument("each whole scenario must have exactly one owner");
                }
                seen[scenario] = 1;
            }
        }
        if (std::any_of(seen.begin(), seen.end(), [](std::uint8_t value) {
                return value != 1;
            })) {
            throw std::invalid_argument("partition omits a scenario");
        }
        for (const auto& load : measured_rank_load) {
            load.validate(world_size);
        }
    }
};

namespace detail {

inline void hash_u64(std::uint64_t& hash, std::uint64_t value) noexcept {
    constexpr std::uint64_t prime = 1099511628211ULL;
    for (unsigned int byte = 0; byte < 8; ++byte) {
        hash ^= (value >> (byte * 8U)) & 0xffU;
        hash *= prime;
    }
}

}  // namespace detail

[[nodiscard]] inline PartitionPlan partition_scenarios(
    std::span<const ScenarioWork> work,
    std::size_t world_size,
    PartitionKind kind,
    const ScenarioCostModel& model = {}
) {
    if (work.empty() || world_size == 0) {
        throw std::invalid_argument("partitioning requires scenarios and ranks");
    }
    std::vector<double> costs;
    costs.reserve(work.size());
    for (const auto& scenario : work) {
        costs.push_back(scenario_cost(scenario, model, kind));
    }
    std::vector<std::size_t> order(work.size());
    std::iota(order.begin(), order.end(), 0U);
    std::stable_sort(order.begin(), order.end(), [&costs](std::size_t left, std::size_t right) {
        return costs[left] != costs[right] ? costs[left] > costs[right] : left < right;
    });

    PartitionPlan plan{
        kind,
        std::vector<std::vector<std::size_t>>(world_size),
        std::vector<std::size_t>(work.size(), 0),
        std::vector<double>(world_size, 0.0),
        {},
        1469598103934665603ULL,
    };
    for (const auto scenario : order) {
        const auto owner = static_cast<std::size_t>(std::distance(
            plan.predicted_rank_load.begin(),
            std::min_element(
                plan.predicted_rank_load.begin(),
                plan.predicted_rank_load.end()
            )
        ));
        plan.rank_scenarios[owner].push_back(scenario);
        plan.scenario_owner[scenario] = owner;
        plan.predicted_rank_load[owner] += costs[scenario];
    }
    detail::hash_u64(plan.fingerprint, static_cast<std::uint64_t>(kind));
    detail::hash_u64(plan.fingerprint, world_size);
    for (std::size_t rank = 0; rank < world_size; ++rank) {
        std::sort(plan.rank_scenarios[rank].begin(), plan.rank_scenarios[rank].end());
        detail::hash_u64(plan.fingerprint, rank);
        for (const auto scenario : plan.rank_scenarios[rank]) {
            detail::hash_u64(plan.fingerprint, scenario);
            detail::hash_u64(plan.fingerprint, std::bit_cast<std::uint64_t>(costs[scenario]));
        }
    }
    plan.validate(work.size(), world_size);
    return plan;
}

inline void record_measured_loads(
    PartitionPlan& plan,
    std::vector<MeasuredRankLoad> loads
) {
    if (loads.size() != plan.rank_scenarios.size()) {
        throw std::invalid_argument("one measured load is required per rank");
    }
    std::sort(loads.begin(), loads.end(), [](const auto& left, const auto& right) {
        return left.rank < right.rank;
    });
    for (std::size_t rank = 0; rank < loads.size(); ++rank) {
        loads[rank].validate(loads.size());
        if (loads[rank].rank != rank) {
            throw std::invalid_argument("measured ranks must be unique and complete");
        }
    }
    plan.measured_rank_load = std::move(loads);
}

enum class RuntimeState : std::uint8_t {
    uninitialised,
    created,
    values_updated,
    warm_started,
    solving,
    solved,
    failed,
    cancelled,
    destroyed,
};

enum class RankStatus : std::int32_t {
    healthy = 0,
    cancelled = 1,
    failed = 2,
    rank_lost = 3,
};

[[nodiscard]] inline RankStatus aggregate_rank_status(
    std::span<const RankStatus> statuses
) {
    if (statuses.empty()) {
        throw std::invalid_argument("rank status aggregation requires at least one rank");
    }
    return *std::max_element(statuses.begin(), statuses.end());
}

[[nodiscard]] inline std::vector<double> reduce_shared_arrowhead(
    std::span<const std::vector<double>> contributions
) {
    if (contributions.empty()) {
        throw std::invalid_argument("arrowhead reduction requires at least one rank");
    }
    std::vector<double> result(contributions.front().size(), 0.0);
    for (const auto& contribution : contributions) {
        if (contribution.size() != result.size()) {
            throw std::invalid_argument("arrowhead contributions must have equal dimensions");
        }
        for (std::size_t index = 0; index < result.size(); ++index) {
            result[index] += contribution[index];
        }
    }
    return result;
}

struct ResidualPartial {
    double squared_primal{0.0};
    double squared_dual{0.0};
    double squared_gap{0.0};
    double maximum_cone_distance{0.0};
    double maximum_nonanticipativity{0.0};
    double maximum_risk_epigraph{0.0};
};

[[nodiscard]] inline ResidualPartial reduce_residuals(
    std::span<const ResidualPartial> partials
) {
    if (partials.empty()) {
        throw std::invalid_argument("residual reduction requires at least one rank");
    }
    ResidualPartial result{};
    for (const auto& partial : partials) {
        const std::array values{
            partial.squared_primal,
            partial.squared_dual,
            partial.squared_gap,
            partial.maximum_cone_distance,
            partial.maximum_nonanticipativity,
            partial.maximum_risk_epigraph,
        };
        if (std::any_of(values.begin(), values.end(), [](double value) {
                return !std::isfinite(value) || value < 0.0;
            })) {
            throw std::invalid_argument("residual partials must be finite and non-negative");
        }
        result.squared_primal += partial.squared_primal;
        result.squared_dual += partial.squared_dual;
        result.squared_gap += partial.squared_gap;
        result.maximum_cone_distance =
            std::max(result.maximum_cone_distance, partial.maximum_cone_distance);
        result.maximum_nonanticipativity = std::max(
            result.maximum_nonanticipativity,
            partial.maximum_nonanticipativity
        );
        result.maximum_risk_epigraph =
            std::max(result.maximum_risk_epigraph, partial.maximum_risk_epigraph);
    }
    return result;
}

struct ExpectedRiskPartial {
    double weighted_loss{0.0};
    double probability{0.0};
};

[[nodiscard]] inline double reduce_expected_risk(
    std::span<const ExpectedRiskPartial> partials
) {
    double probability = 0.0;
    double expected = 0.0;
    for (const auto& partial : partials) {
        if (!std::isfinite(partial.weighted_loss) || !std::isfinite(partial.probability)
            || partial.probability < 0.0) {
            throw std::invalid_argument("expected-risk partial is invalid");
        }
        expected += partial.weighted_loss;
        probability += partial.probability;
    }
    if (std::abs(probability - 1.0) > 1.0e-12) {
        throw std::invalid_argument("expected-risk rank probabilities must sum to one");
    }
    return expected;
}

struct WorstRiskPartial {
    double loss{-std::numeric_limits<double>::infinity()};
    std::size_t scenario{std::numeric_limits<std::size_t>::max()};
};

[[nodiscard]] inline WorstRiskPartial reduce_worst_risk(
    std::span<const WorstRiskPartial> partials
) {
    if (partials.empty()) {
        throw std::invalid_argument("worst-risk reduction requires at least one rank");
    }
    WorstRiskPartial result{};
    for (const auto& partial : partials) {
        if (!std::isfinite(partial.loss)
            || partial.scenario == std::numeric_limits<std::size_t>::max()) {
            throw std::invalid_argument("worst-risk partial is invalid");
        }
        if (partial.loss > result.loss
            || (partial.loss == result.loss && partial.scenario < result.scenario)) {
            result = partial;
        }
    }
    return result;
}

struct CvarEpigraphPartial {
    double weighted_excess{0.0};
    double maximum_epigraph_violation{0.0};
    double threshold_dual_sum{0.0};
};

[[nodiscard]] inline CvarEpigraphPartial reduce_cvar_epigraph(
    std::span<const CvarEpigraphPartial> partials
) {
    CvarEpigraphPartial result{};
    for (const auto& partial : partials) {
        if (!std::isfinite(partial.weighted_excess)
            || !std::isfinite(partial.maximum_epigraph_violation)
            || !std::isfinite(partial.threshold_dual_sum)
            || partial.maximum_epigraph_violation < 0.0) {
            throw std::invalid_argument("CVaR epigraph partial is invalid");
        }
        result.weighted_excess += partial.weighted_excess;
        result.maximum_epigraph_violation = std::max(
            result.maximum_epigraph_violation,
            partial.maximum_epigraph_violation
        );
        result.threshold_dual_sum += partial.threshold_dual_sum;
    }
    return result;
}

enum class CollectiveKind : std::uint8_t {
    shared_arrowhead_sum,
    residual_sum,
    residual_max,
    expected_risk_sum,
    worst_risk_max,
    cvar_epigraph_sum,
    status_max,
};

enum class OrderingState : std::uint8_t {
    idle,
    local_ready_recorded,
    collective_waiting,
    collective_enqueued,
    collective_complete_recorded,
    compute_waiting,
    complete,
    failed,
    cancelled,
};

class CollectiveOrdering {
  public:
    [[nodiscard]] OrderingState state() const noexcept { return state_; }
    [[nodiscard]] std::uint64_t epoch() const noexcept { return epoch_; }

    void begin() {
        require(OrderingState::idle, OrderingState::complete);
        ++epoch_;
        state_ = OrderingState::local_ready_recorded;
    }
    void collective_wait() {
        require(OrderingState::local_ready_recorded);
        state_ = OrderingState::collective_waiting;
    }
    void enqueue() {
        require(OrderingState::collective_waiting);
        state_ = OrderingState::collective_enqueued;
    }
    void collective_complete() {
        require(OrderingState::collective_enqueued);
        state_ = OrderingState::collective_complete_recorded;
    }
    void compute_wait() {
        require(OrderingState::collective_complete_recorded);
        state_ = OrderingState::compute_waiting;
    }
    void finish() {
        require(OrderingState::compute_waiting);
        state_ = OrderingState::complete;
    }
    void fail() noexcept { state_ = OrderingState::failed; }
    void cancel() noexcept { state_ = OrderingState::cancelled; }

  private:
    OrderingState state_{OrderingState::idle};
    std::uint64_t epoch_{0};

    template <typename... States>
    void require(States... allowed) const {
        if (!((state_ == allowed) || ...)) {
            throw std::logic_error("invalid NCCL stream-event ordering transition");
        }
    }
};

struct CollectiveTelemetry {
    CollectiveKind kind{CollectiveKind::shared_arrowhead_sum};
    std::uint64_t call_count{0};
    std::uint64_t element_count{0};
    std::uint64_t payload_bytes{0};
    std::uint64_t wire_bytes_estimate{0};
    std::uint64_t frequency{0};
    std::string purpose{};
    double collective_seconds{0.0};
    double exposed_seconds{0.0};
    double overlapped_seconds{0.0};
};

struct RuntimeTelemetry {
    int rank{-1};
    int world_size{0};
    int local_rank{-1};
    int device{-1};
    bool deterministic{true};
    bool overlap_enabled{false};
    RankStatus rank_status{RankStatus::healthy};
    double local_compute_seconds{0.0};
    double exposed_communication_seconds{0.0};
    double overlapped_communication_seconds{0.0};
    std::vector<CollectiveTelemetry> collectives{};
};

struct ArrowheadMetadata {
    std::size_t global_variables{0};
    std::vector<std::size_t> shared_primal_indices{};
    std::size_t nonanticipativity_rows{0};
    std::optional<std::size_t> risk_threshold_index{};
    std::vector<std::size_t> risk_excess_indices{};
    std::uint64_t fingerprint{0};

    void validate() const {
        auto indices = shared_primal_indices;
        if (risk_threshold_index.has_value()) {
            indices.push_back(*risk_threshold_index);
        }
        indices.insert(indices.end(), risk_excess_indices.begin(), risk_excess_indices.end());
        if (std::any_of(indices.begin(), indices.end(), [this](std::size_t index) {
                return index >= global_variables;
            })) {
            throw std::invalid_argument("shared arrowhead index is outside the global primal");
        }
        std::sort(indices.begin(), indices.end());
        if (std::adjacent_find(indices.begin(), indices.end()) != indices.end()) {
            throw std::invalid_argument("shared arrowhead indices must be unique");
        }
    }
};

enum class WarmOwnership : std::uint8_t {
    none,
    primal,
    primal_dual,
    full_state,
};

struct RankCheckpointHeader {
    static constexpr std::array<char, 8> magic{{'S', 'P', 'G', '5', 'C', 'K', 'P', 'T'}};
    static constexpr std::uint32_t version = 1;

    std::array<char, 8> file_magic{magic};
    std::uint32_t schema_version{version};
    std::uint32_t header_bytes{0};
    std::uint64_t topology_fingerprint{0};
    std::uint64_t partition_fingerprint{0};
    std::uint64_t local_workspace_bytes{0};
    std::uint64_t local_scenario_count{0};
    std::uint64_t primal_elements{0};
    std::uint64_t dual_elements{0};
    std::uint64_t scaling_elements{0};
    std::int32_t world_size{0};
    std::int32_t rank{-1};
    std::int32_t device{-1};
    WarmOwnership warm_ownership{WarmOwnership::none};
    std::array<std::uint8_t, 3> reserved{};
};

[[nodiscard]] inline std::vector<std::byte> pack_rank_checkpoint(
    const RankCheckpointHeader& header,
    std::span<const std::byte> local_workspace
) {
    if (header.file_magic != RankCheckpointHeader::magic
        || header.schema_version != RankCheckpointHeader::version
        || (header.header_bytes != 0 && header.header_bytes != sizeof(RankCheckpointHeader))
        || header.local_workspace_bytes != local_workspace.size()) {
        throw std::invalid_argument("rank checkpoint header is inconsistent");
    }
    auto normalized = header;
    normalized.header_bytes = sizeof(RankCheckpointHeader);
    std::vector<std::byte> bytes(sizeof(header) + local_workspace.size());
    std::memcpy(bytes.data(), &normalized, sizeof(normalized));
    std::memcpy(bytes.data() + sizeof(header), local_workspace.data(), local_workspace.size());
    return bytes;
}

[[nodiscard]] inline RankCheckpointHeader validate_rank_checkpoint(
    std::span<const std::byte> bytes,
    std::uint64_t topology_fingerprint,
    std::uint64_t partition_fingerprint,
    int world_size,
    int rank,
    int device
) {
    if (bytes.size() < sizeof(RankCheckpointHeader)) {
        throw std::invalid_argument("rank checkpoint is truncated");
    }
    RankCheckpointHeader header{};
    std::memcpy(&header, bytes.data(), sizeof(header));
    if (header.file_magic != RankCheckpointHeader::magic
        || header.schema_version != RankCheckpointHeader::version
        || header.header_bytes != sizeof(RankCheckpointHeader)
        || header.local_workspace_bytes != bytes.size() - sizeof(RankCheckpointHeader)) {
        throw std::invalid_argument("rank checkpoint schema is incompatible");
    }
    if (header.topology_fingerprint != topology_fingerprint
        || header.partition_fingerprint != partition_fingerprint
        || header.world_size != world_size || header.rank != rank || header.device != device) {
        throw std::invalid_argument("rank checkpoint ownership or topology is incompatible");
    }
    return header;
}

struct RuntimeOptions {
    MPI_Comm communicator{MPI_COMM_WORLD};
    bool deterministic{true};
    bool enable_overlap{false};
    std::uint64_t topology_fingerprint{0};
    std::uint64_t partition_fingerprint{0};
};

class MpiNcclRuntime {
  public:
    explicit MpiNcclRuntime(RuntimeOptions options);
    ~MpiNcclRuntime();

    MpiNcclRuntime(const MpiNcclRuntime&) = delete;
    MpiNcclRuntime& operator=(const MpiNcclRuntime&) = delete;

    [[nodiscard]] int rank() const noexcept { return telemetry_.rank; }
    [[nodiscard]] int world_size() const noexcept { return telemetry_.world_size; }
    [[nodiscard]] int local_rank() const noexcept { return telemetry_.local_rank; }
    [[nodiscard]] int device() const noexcept { return telemetry_.device; }
    [[nodiscard]] cudaStream_t compute_stream() const noexcept { return compute_stream_; }
    [[nodiscard]] cudaStream_t collective_stream() const noexcept {
        return options_.enable_overlap ? collective_stream_ : compute_stream_;
    }
    [[nodiscard]] RuntimeState state() const noexcept { return state_; }
    [[nodiscard]] RankStatus status() const noexcept { return telemetry_.rank_status; }
    [[nodiscard]] std::uint64_t topology_fingerprint() const noexcept {
        return options_.topology_fingerprint;
    }
    [[nodiscard]] std::uint64_t partition_fingerprint() const noexcept {
        return options_.partition_fingerprint;
    }
    [[nodiscard]] const RuntimeTelemetry& telemetry() const noexcept { return telemetry_; }

    void allreduce_sum(
        double* device_values,
        std::size_t count,
        CollectiveKind kind,
        std::uint64_t frequency,
        std::string_view purpose
    );
    void allreduce_max(
        double* device_values,
        std::size_t count,
        CollectiveKind kind,
        std::uint64_t frequency,
        std::string_view purpose
    );
    void synchronize();
    void record_local_compute(double seconds);
    void record_overlap(double seconds);
    void mark_values_updated();
    void mark_warm_started();
    void begin_solve();
    void finish_solve();
    void cancel() noexcept;
    void fail(std::string message) noexcept;
    [[nodiscard]] RankStatus synchronize_status();
    [[nodiscard]] const std::string& last_error() const noexcept { return last_error_; }

  private:
    RuntimeOptions options_{};
    MPI_Comm communicator_{MPI_COMM_NULL};
    MPI_Comm local_communicator_{MPI_COMM_NULL};
    ncclComm_t nccl_communicator_{nullptr};
    cudaStream_t compute_stream_{nullptr};
    cudaStream_t collective_stream_{nullptr};
    cudaEvent_t local_ready_{nullptr};
    cudaEvent_t collective_complete_{nullptr};
    RuntimeState state_{RuntimeState::uninitialised};
    CollectiveOrdering ordering_{};
    RuntimeTelemetry telemetry_{};
    std::string last_error_{};
    struct PendingTiming {
        std::size_t telemetry_index{0};
        cudaEvent_t start{nullptr};
        cudaEvent_t stop{nullptr};
    };
    std::vector<PendingTiming> pending_timings_{};

    void allreduce(
        double* device_values,
        std::size_t count,
        ncclRedOp_t operation,
        CollectiveKind kind,
        std::uint64_t frequency,
        std::string_view purpose
    );
};

cudaError_t csc_forward_async(
    int rows,
    int columns,
    const int* offsets,
    const int* indices,
    const double* values,
    const double* input,
    double* output,
    cudaStream_t stream
) noexcept;

cudaError_t csc_transpose_async(
    int rows,
    int columns,
    const int* offsets,
    const int* indices,
    const double* values,
    const double* input,
    double* output,
    cudaStream_t stream
) noexcept;

cudaError_t project_soc_blocks_async(
    double* values,
    const int* starts,
    const int* dimensions,
    int cone_count,
    cudaStream_t stream
) noexcept;

}  // namespace spacepdhcg::distributed::g5
