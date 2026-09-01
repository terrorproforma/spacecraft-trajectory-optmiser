#pragma once

#include "spacepdhcg/orbitweaver/trajectory_oracle.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace spacepdhcg::orbitweaver::g7 {

enum class ArcExecutionStatus : std::uint8_t {
    feasible,
    infeasible,
    unsupported,
    invalid_input,
    numerical_failure,
    backend_failure,
    cancelled,
    certification_rejected,
};

struct TopologyFidelityKey {
    std::uint64_t topology_fingerprint{0U};
    ArcFidelity fidelity{ArcFidelity::analytical_screening};
    std::size_t intervals{0U};
    std::size_t scenario_count{1U};

    [[nodiscard]] auto tie() const noexcept {
        return std::tie(topology_fingerprint, fidelity, intervals, scenario_count);
    }
    [[nodiscard]] bool operator<(const TopologyFidelityKey& other) const noexcept {
        return tie() < other.tie();
    }
};

struct ScheduledArc {
    std::uint64_t deterministic_id{0U};
    ArcRequest request{};
    TopologyFidelityKey group{};
    double inherited_lower_bound{0.0};
    std::size_t route_index{0U};
    std::size_t trajectory_arc_index{0U};
    std::size_t scenario_index{0U};
    std::size_t time_node_index{0U};

    void validate() const {
        request.validate();
        if (group.topology_fingerprint == 0U || group.intervals < 2U
            || group.scenario_count != request.scenario_count
            || group.fidelity != request.fidelity
            || !std::isfinite(inherited_lower_bound)
            || inherited_lower_bound < 0.0) {
            throw std::invalid_argument("scheduled G7 arc metadata is invalid");
        }
    }
};

struct ArcExecution {
    std::uint64_t deterministic_id{0U};
    ArcExecutionStatus status{ArcExecutionStatus::backend_failure};
    ArcSolution solution{};
    std::size_t owner_rank{0U};
    std::size_t owner_device{0U};
    std::size_t batch_sequence{0U};
    std::string diagnostic{};

    [[nodiscard]] bool feasible() const noexcept {
        return status == ArcExecutionStatus::feasible && solution.feasible;
    }
};

struct Ownership {
    std::size_t rank{0U};
    std::size_t device{0U};
    [[nodiscard]] bool operator<(const Ownership& other) const noexcept {
        return std::tie(rank, device) < std::tie(other.rank, other.device);
    }
};

class OwnershipPolicy {
  public:
    virtual ~OwnershipPolicy() = default;
    [[nodiscard]] virtual Ownership owner(
        const ScheduledArc& arc,
        std::size_t batch
    ) const = 0;
};

class SingleDeviceOwnership final : public OwnershipPolicy {
  public:
    explicit SingleDeviceOwnership(std::size_t device = 0U) : device_(device) {}
    [[nodiscard]] Ownership owner(const ScheduledArc&, std::size_t) const override {
        return {0U, device_};
    }

  private:
    std::size_t device_{0U};
};

/// Contract-only rank mock. It performs no communication and is never scaling evidence.
class LogicalRankOwnership final : public OwnershipPolicy {
  public:
    explicit LogicalRankOwnership(std::vector<std::size_t> devices)
        : devices_(std::move(devices)) {
        if (devices_.empty()) {
            throw std::invalid_argument("logical ownership requires devices");
        }
    }
    [[nodiscard]] Ownership owner(
        const ScheduledArc& arc,
        const std::size_t batch
    ) const override {
        const auto rank =
            static_cast<std::size_t>((arc.deterministic_id + batch) % devices_.size());
        return {rank, devices_[rank]};
    }

  private:
    std::vector<std::size_t> devices_{};
};

class ArcBatchBackend {
  public:
    virtual ~ArcBatchBackend() = default;
    [[nodiscard]] virtual std::vector<ArcExecution> evaluate(
        const TopologyFidelityKey& group,
        const std::vector<ScheduledArc>& batch,
        Ownership owner,
        const std::atomic<bool>& cancelled
    ) = 0;
};

/// The callback owns public G3 persistent drivers today. G5 later supplies rank-local
/// drivers through this same seam; G7 never imports G5 implementation details.
class PersistentArcCallbackBackend final : public ArcBatchBackend {
  public:
    using Callback = std::function<std::vector<ArcExecution>(
        const TopologyFidelityKey&,
        const std::vector<ScheduledArc>&,
        Ownership,
        const std::atomic<bool>&
    )>;
    explicit PersistentArcCallbackBackend(Callback callback)
        : callback_(std::move(callback)) {
        if (!callback_) {
            throw std::invalid_argument("persistent callback is empty");
        }
    }
    [[nodiscard]] std::vector<ArcExecution> evaluate(
        const TopologyFidelityKey& group,
        const std::vector<ScheduledArc>& batch,
        const Ownership owner,
        const std::atomic<bool>& cancelled
    ) override {
        return callback_(group, batch, owner, cancelled);
    }

  private:
    Callback callback_{};
};

struct SchedulerConfig {
    std::size_t maximum_batch_size{128U};
    std::size_t maximum_buffered_arcs{1'024U};
    std::size_t bytes_per_arc_budget{1U << 20U};
    std::size_t maximum_workspace_bytes{1U << 30U};
    void validate() const {
        if (maximum_batch_size == 0U || maximum_buffered_arcs == 0U
            || bytes_per_arc_budget == 0U || maximum_workspace_bytes == 0U
            || maximum_batch_size > maximum_buffered_arcs
            || maximum_batch_size
                   > maximum_workspace_bytes / bytes_per_arc_budget) {
            throw std::invalid_argument("G7 scheduler memory limits are invalid");
        }
    }
};

struct SchedulerTelemetry {
    std::size_t submitted{0U};
    std::size_t completed{0U};
    std::size_t feasible{0U};
    std::size_t failed{0U};
    std::size_t cancelled{0U};
    std::size_t batches{0U};
    std::size_t maximum_observed_batch{0U};
    std::size_t estimated_peak_buffer_bytes{0U};
    std::map<TopologyFidelityKey, std::size_t> group_batches{};
    std::map<Ownership, std::size_t> ownership_batches{};
};

class BoundedArcScheduler {
  public:
    BoundedArcScheduler(
        std::shared_ptr<ArcBatchBackend> backend,
        std::shared_ptr<OwnershipPolicy> ownership =
            std::make_shared<SingleDeviceOwnership>(),
        SchedulerConfig config = {}
    )
        : backend_(std::move(backend)),
          ownership_(std::move(ownership)),
          config_(config) {
        if (!backend_ || !ownership_) {
            throw std::invalid_argument("G7 scheduler dependencies are missing");
        }
        config_.validate();
    }

    [[nodiscard]] std::vector<ArcExecution> run(std::vector<ScheduledArc> arcs) {
        if (arcs.size() > config_.maximum_buffered_arcs) {
            throw std::invalid_argument("G7 scheduler backpressure limit exceeded");
        }
        telemetry_ = {};
        telemetry_.submitted = arcs.size();
        for (const auto& arc : arcs) {
            arc.validate();
        }
        std::stable_sort(arcs.begin(), arcs.end(), [](const auto& left, const auto& right) {
            if (left.group < right.group) {
                return true;
            }
            if (right.group < left.group) {
                return false;
            }
            return left.deterministic_id < right.deterministic_id;
        });
        std::vector<ArcExecution> output{};
        for (std::size_t cursor = 0U, sequence = 0U; cursor < arcs.size(); ++sequence) {
            const auto group = arcs[cursor].group;
            auto group_end = cursor;
            while (group_end < arcs.size() && same_group(group, arcs[group_end].group)) {
                ++group_end;
            }
            const auto count = std::min(config_.maximum_batch_size, group_end - cursor);
            std::vector<ScheduledArc> batch(
                arcs.begin() + static_cast<std::ptrdiff_t>(cursor),
                arcs.begin() + static_cast<std::ptrdiff_t>(cursor + count)
            );
            const auto owner = ownership_->owner(batch.front(), sequence);
            auto evaluated = invoke(group, batch, owner);
            for (std::size_t index = 0U; index < batch.size(); ++index) {
                auto& result = evaluated[index];
                result.owner_rank = owner.rank;
                result.owner_device = owner.device;
                result.batch_sequence = sequence;
                if (result.deterministic_id != batch[index].deterministic_id) {
                    result = failure(batch[index], "backend changed deterministic ordering");
                }
                ++telemetry_.completed;
                if (result.feasible()) {
                    ++telemetry_.feasible;
                } else if (result.status == ArcExecutionStatus::cancelled) {
                    ++telemetry_.cancelled;
                } else {
                    ++telemetry_.failed;
                }
                output.push_back(std::move(result));
            }
            ++telemetry_.batches;
            ++telemetry_.group_batches[group];
            ++telemetry_.ownership_batches[owner];
            telemetry_.maximum_observed_batch =
                std::max(telemetry_.maximum_observed_batch, count);
            telemetry_.estimated_peak_buffer_bytes = std::max(
                telemetry_.estimated_peak_buffer_bytes,
                count * config_.bytes_per_arc_budget
            );
            cursor += count;
        }
        std::stable_sort(output.begin(), output.end(), [](const auto& left, const auto& right) {
            return left.deterministic_id < right.deterministic_id;
        });
        return output;
    }

    void cancel() noexcept { cancelled_.store(true); }
    void reset_cancellation() noexcept { cancelled_.store(false); }
    [[nodiscard]] const SchedulerTelemetry& telemetry() const noexcept {
        return telemetry_;
    }

  private:
    std::shared_ptr<ArcBatchBackend> backend_{};
    std::shared_ptr<OwnershipPolicy> ownership_{};
    SchedulerConfig config_{};
    std::atomic<bool> cancelled_{false};
    SchedulerTelemetry telemetry_{};

    static bool same_group(
        const TopologyFidelityKey& left,
        const TopologyFidelityKey& right
    ) {
        return !(left < right) && !(right < left);
    }
    static ArcExecution failure(const ScheduledArc& arc, std::string diagnostic) {
        return {
            arc.deterministic_id,
            ArcExecutionStatus::backend_failure,
            {},
            0U,
            0U,
            0U,
            std::move(diagnostic),
        };
    }
    std::vector<ArcExecution> invoke(
        const TopologyFidelityKey& group,
        const std::vector<ScheduledArc>& batch,
        const Ownership owner
    ) {
        if (cancelled_.load()) {
            std::vector<ArcExecution> result{};
            for (const auto& arc : batch) {
                auto item = failure(arc, "G7 execution cancelled");
                item.status = ArcExecutionStatus::cancelled;
                result.push_back(std::move(item));
            }
            return result;
        }
        try {
            auto result = backend_->evaluate(group, batch, owner, cancelled_);
            if (result.size() != batch.size()) {
                throw std::runtime_error("backend returned mismatched batch length");
            }
            return result;
        } catch (const std::exception& error) {
            std::vector<ArcExecution> result{};
            for (const auto& arc : batch) {
                result.push_back(failure(arc, error.what()));
            }
            return result;
        }
    }
};

[[nodiscard]] inline std::vector<ArcExecution> deterministic_top_k(
    std::vector<ArcExecution> candidates,
    const std::size_t count,
    const bool retain_failures = true
) {
    if (count == 0U) {
        throw std::invalid_argument("G7 top-K count must be positive");
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
        return std::tuple{
                   !left.feasible(),
                   left.solution.cost,
                   left.solution.lower_bound,
                   left.deterministic_id}
               < std::tuple{
                   !right.feasible(),
                   right.solution.cost,
                   right.solution.lower_bound,
                   right.deterministic_id};
    });
    if (!retain_failures) {
        candidates.erase(
            std::remove_if(candidates.begin(), candidates.end(), [](const auto& item) {
                return !item.feasible();
            }),
            candidates.end()
        );
    }
    if (candidates.size() > count) {
        candidates.resize(count);
    }
    return candidates;
}

enum class RiskMeasure : std::uint8_t { expected, worst_case, cvar };

struct ScenarioOutcome {
    std::size_t scenario{0U};
    double probability{0.0};
    double cost{0.0};
    double lower_bound{0.0};
    std::vector<double> nonanticipative_controls{};
    ArcExecutionStatus status{ArcExecutionStatus::feasible};
};

struct RiskResult {
    bool feasible{false};
    double objective{std::numeric_limits<double>::infinity()};
    double lower_bound{std::numeric_limits<double>::infinity()};
    double nonanticipativity_violation{std::numeric_limits<double>::infinity()};
    double cvar_threshold{std::numeric_limits<double>::quiet_NaN()};
};

[[nodiscard]] inline RiskResult aggregate_risk(
    std::vector<ScenarioOutcome> scenarios,
    const RiskMeasure measure,
    const double alpha = 0.9,
    const double nonanticipativity_tolerance = 1.0e-10
) {
    if (scenarios.empty() || (measure == RiskMeasure::cvar && !(alpha > 0.0 && alpha < 1.0))) {
        throw std::invalid_argument("G7 risk configuration is invalid");
    }
    std::stable_sort(scenarios.begin(), scenarios.end(), [](const auto& left, const auto& right) {
        return left.scenario < right.scenario;
    });
    double probability = 0.0;
    double expected = 0.0;
    double expected_bound = 0.0;
    double violation = 0.0;
    const auto& reference = scenarios.front().nonanticipative_controls;
    for (const auto& scenario : scenarios) {
        if (scenario.status != ArcExecutionStatus::feasible
            || !std::isfinite(scenario.probability) || scenario.probability < 0.0
            || !std::isfinite(scenario.cost) || !std::isfinite(scenario.lower_bound)
            || scenario.lower_bound > scenario.cost
            || scenario.nonanticipative_controls.size() != reference.size()) {
            return {};
        }
        probability += scenario.probability;
        expected += scenario.probability * scenario.cost;
        expected_bound += scenario.probability * scenario.lower_bound;
        for (std::size_t index = 0U; index < reference.size(); ++index) {
            violation = std::max(
                violation,
                std::abs(scenario.nonanticipative_controls[index] - reference[index])
            );
        }
    }
    if (std::abs(probability - 1.0) > 1.0e-12) {
        throw std::invalid_argument("G7 scenario probabilities must sum to one");
    }
    if (violation > nonanticipativity_tolerance) {
        return {false, std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity(), violation};
    }
    if (measure == RiskMeasure::expected) {
        return {true, expected, expected_bound, violation};
    }
    if (measure == RiskMeasure::worst_case) {
        double cost = 0.0;
        double bound = 0.0;
        for (const auto& scenario : scenarios) {
            cost = std::max(cost, scenario.cost);
            bound = std::max(bound, scenario.lower_bound);
        }
        return {true, cost, bound, violation};
    }
    std::stable_sort(scenarios.begin(), scenarios.end(), [](const auto& left, const auto& right) {
        return std::tie(left.cost, left.scenario) < std::tie(right.cost, right.scenario);
    });
    double cumulative = 0.0;
    double threshold = scenarios.back().cost;
    for (const auto& scenario : scenarios) {
        cumulative += scenario.probability;
        if (cumulative >= alpha) {
            threshold = scenario.cost;
            break;
        }
    }
    double cvar = threshold;
    for (const auto& scenario : scenarios) {
        cvar += scenario.probability * std::max(0.0, scenario.cost - threshold)
                / (1.0 - alpha);
    }
    return {true, cvar, std::min(cvar, expected_bound), violation, threshold};
}

struct CertificationChecks {
    double dynamics_defect{std::numeric_limits<double>::infinity()};
    double path_violation{std::numeric_limits<double>::infinity()};
    double terminal_error{std::numeric_limits<double>::infinity()};
    double uncertainty_violation{std::numeric_limits<double>::infinity()};
    double integration_error{std::numeric_limits<double>::infinity()};
    [[nodiscard]] double maximum() const noexcept {
        return std::max({
            dynamics_defect,
            path_violation,
            terminal_error,
            uncertainty_violation,
            integration_error,
        });
    }
};

struct CertificationRecord {
    bool accepted{false};
    CertificationChecks checks{};
    std::string backend_identifier{};
    std::string diagnostic{};
};

class IndependentCertifier {
  public:
    using Callback = std::function<CertificationChecks(const ArcExecution&)>;
    IndependentCertifier(Callback callback, std::string backend, const double tolerance)
        : callback_(std::move(callback)),
          backend_(std::move(backend)),
          tolerance_(tolerance) {
        if (!callback_ || backend_.empty() || !(tolerance_ > 0.0)) {
            throw std::invalid_argument("independent certifier is invalid");
        }
    }
    [[nodiscard]] CertificationRecord certify(const ArcExecution& incumbent) const {
        if (!incumbent.feasible()) {
            return {false, {}, backend_, "optimizer status is not certification"};
        }
        const auto checks = callback_(incumbent);
        const auto accepted = std::isfinite(checks.maximum())
                              && checks.maximum() <= tolerance_;
        return {
            accepted,
            checks,
            backend_,
            accepted ? "independent certification accepted"
                     : "independent certification rejected incumbent",
        };
    }

  private:
    Callback callback_{};
    std::string backend_{};
    double tolerance_{0.0};
};

struct Checkpoint {
    std::uint32_t schema_version{1U};
    std::uint64_t seed{0U};
    std::size_t completed_batches{0U};
    double incumbent{std::numeric_limits<double>::infinity()};
    double lower_bound{-std::numeric_limits<double>::infinity()};
    std::vector<std::uint64_t> completed_arc_ids{};
    std::vector<std::uint64_t> warm_tokens{};

    [[nodiscard]] std::string encode() const {
        std::ostringstream stream{};
        stream << schema_version << ' ' << seed << ' ' << completed_batches << ' '
               << std::setprecision(17) << incumbent << ' ' << lower_bound << '\n';
        write(stream, completed_arc_ids);
        write(stream, warm_tokens);
        return stream.str();
    }
    [[nodiscard]] static Checkpoint decode(const std::string& value) {
        std::istringstream stream{value};
        Checkpoint result{};
        if (!(stream >> result.schema_version >> result.seed >> result.completed_batches
              >> result.incumbent >> result.lower_bound)) {
            throw std::invalid_argument("G7 checkpoint header is malformed");
        }
        read(stream, result.completed_arc_ids);
        read(stream, result.warm_tokens);
        if (result.schema_version != 1U
            || !std::is_sorted(result.completed_arc_ids.begin(), result.completed_arc_ids.end())) {
            throw std::invalid_argument("G7 checkpoint is incompatible");
        }
        return result;
    }

  private:
    static void write(std::ostringstream& stream, const std::vector<std::uint64_t>& values) {
        stream << values.size();
        for (const auto value : values) {
            stream << ' ' << value;
        }
        stream << '\n';
    }
    static void read(std::istringstream& stream, std::vector<std::uint64_t>& values) {
        std::size_t size = 0U;
        if (!(stream >> size) || size > 10'000'000U) {
            throw std::invalid_argument("G7 checkpoint vector is malformed");
        }
        values.resize(size);
        for (auto& value : values) {
            if (!(stream >> value)) {
                throw std::invalid_argument("G7 checkpoint is truncated");
            }
        }
    }
};

}  // namespace spacepdhcg::orbitweaver::g7
