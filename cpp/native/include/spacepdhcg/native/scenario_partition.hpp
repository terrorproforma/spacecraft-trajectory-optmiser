#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace spacepdhcg::native {

struct ScenarioPartition {
    std::vector<std::vector<std::size_t>> assignments{};
    std::vector<double> loads{};

    [[nodiscard]] std::size_t device_count() const noexcept { return assignments.size(); }

    [[nodiscard]] double maximum_load() const noexcept {
        return loads.empty() ? 0.0 : *std::max_element(loads.begin(), loads.end());
    }

    [[nodiscard]] double mean_load() const noexcept {
        if (loads.empty()) {
            return 0.0;
        }
        return std::accumulate(loads.begin(), loads.end(), 0.0) /
               static_cast<double>(loads.size());
    }

    [[nodiscard]] double imbalance() const noexcept {
        const double mean = mean_load();
        return mean > 0.0 ? maximum_load() / mean : 1.0;
    }

    [[nodiscard]] std::size_t owner(std::size_t scenario) const {
        for (std::size_t device = 0; device < assignments.size(); ++device) {
            if (std::ranges::find(assignments[device], scenario) != assignments[device].end()) {
                return device;
            }
        }
        throw std::out_of_range("scenario has no assigned device");
    }
};

[[nodiscard]] inline ScenarioPartition partition_scenarios(
    std::span<const double> weights,
    std::size_t device_count
) {
    if (weights.empty()) {
        throw std::invalid_argument("scenario weights may not be empty");
    }
    if (device_count == 0) {
        throw std::invalid_argument("scenario partition requires at least one device");
    }
    for (double weight : weights) {
        if (!std::isfinite(weight) || weight < 0.0) {
            throw std::invalid_argument("scenario weights must be finite and non-negative");
        }
    }

    std::vector<std::size_t> order(weights.size());
    std::iota(order.begin(), order.end(), 0U);
    std::sort(order.begin(), order.end(), [weights](std::size_t left, std::size_t right) {
        if (weights[left] != weights[right]) {
            return weights[left] > weights[right];
        }
        return left < right;
    });

    ScenarioPartition result{
        std::vector<std::vector<std::size_t>>(device_count),
        std::vector<double>(device_count, 0.0),
    };
    for (std::size_t scenario : order) {
        const auto device = static_cast<std::size_t>(std::distance(
            result.loads.begin(),
            std::min_element(result.loads.begin(), result.loads.end())
        ));
        result.assignments[device].push_back(scenario);
        result.loads[device] += weights[scenario];
    }
    for (auto& assignment : result.assignments) {
        std::sort(assignment.begin(), assignment.end());
    }
    return result;
}

struct LogicalGpuGrid {
    std::size_t scenario_partitions{1};
    std::size_t time_partitions{1};

    void validate() const {
        if (scenario_partitions == 0 || time_partitions == 0) {
            throw std::invalid_argument("logical GPU grid dimensions must be positive");
        }
    }

    [[nodiscard]] std::size_t device_count() const {
        validate();
        return scenario_partitions * time_partitions;
    }

    [[nodiscard]] std::size_t rank(
        std::size_t scenario_partition,
        std::size_t time_partition
    ) const {
        validate();
        if (scenario_partition >= scenario_partitions || time_partition >= time_partitions) {
            throw std::out_of_range("logical GPU coordinate lies outside the grid");
        }
        return scenario_partition * time_partitions + time_partition;
    }
};

struct CommunicationProfile {
    std::size_t device_count{0};
    std::size_t shared_dimension{0};
    std::size_t payload_bytes{0};
    double bytes_per_device{0.0};
    double aggregate_bytes{0.0};
    std::size_t collective_count{0};
};

[[nodiscard]] inline CommunicationProfile ring_allreduce_profile(
    std::size_t shared_dimension,
    std::size_t device_count,
    std::size_t scalar_bytes = sizeof(double),
    std::size_t collective_count = 1
) {
    if (device_count == 0 || scalar_bytes == 0) {
        throw std::invalid_argument("communication dimensions must be positive");
    }
    if (shared_dimension > std::numeric_limits<std::size_t>::max() / scalar_bytes) {
        throw std::overflow_error("communication payload size overflows size_t");
    }
    const std::size_t payload = shared_dimension * scalar_bytes;
    double per_device = 0.0;
    double aggregate = 0.0;
    if (device_count > 1 && payload > 0 && collective_count > 0) {
        const double one_collective =
            2.0 * static_cast<double>(device_count - 1) /
            static_cast<double>(device_count) * static_cast<double>(payload);
        per_device = static_cast<double>(collective_count) * one_collective;
        aggregate = static_cast<double>(device_count) * per_device;
    }
    return CommunicationProfile{
        device_count,
        shared_dimension,
        payload,
        per_device,
        aggregate,
        collective_count,
    };
}

}  // namespace spacepdhcg::native
