#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace spacepdhcg::orbitweaver {

struct ScheduledArcEstimate {
    bool feasible{false};
    double nominal_cost{std::numeric_limits<double>::infinity()};
    double lower_bound{std::numeric_limits<double>::infinity()};
    double delta_v{0.0};
};

struct TimeExpandedNode {
    std::size_t target{0U};
    std::size_t epoch_index{0U};
    double epoch{0.0};
};

struct TimeExpandedArc {
    std::size_t from{0U};
    std::size_t to{0U};
    double nominal_cost{0.0};
    double lower_bound{0.0};
    double delta_v{0.0};

    [[nodiscard]] double duration(const std::vector<TimeExpandedNode>& nodes) const {
        return nodes.at(to).epoch - nodes.at(from).epoch;
    }
};

class TimeExpandedGraph {
  public:
    TimeExpandedGraph(std::size_t target_count, std::vector<double> epochs)
        : target_count_(target_count), epochs_(std::move(epochs)) {
        if (target_count_ == 0U) {
            throw std::invalid_argument("time-expanded graph requires at least one target");
        }
        if (epochs_.size() < 2U) {
            throw std::invalid_argument("time-expanded graph requires at least two epochs");
        }
        for (std::size_t index = 0; index < epochs_.size(); ++index) {
            if (!std::isfinite(epochs_[index])) {
                throw std::invalid_argument("time-expanded epochs must be finite");
            }
            if (index > 0U && epochs_[index] <= epochs_[index - 1U]) {
                throw std::invalid_argument("time-expanded epochs must be strictly increasing");
            }
        }
        nodes_.reserve(target_count_ * epochs_.size());
        for (std::size_t epoch_index = 0; epoch_index < epochs_.size(); ++epoch_index) {
            for (std::size_t target = 0; target < target_count_; ++target) {
                nodes_.push_back(TimeExpandedNode{target, epoch_index, epochs_[epoch_index]});
            }
        }
        outgoing_.resize(nodes_.size());
    }

    [[nodiscard]] std::size_t target_count() const noexcept { return target_count_; }
    [[nodiscard]] std::size_t epoch_count() const noexcept { return epochs_.size(); }
    [[nodiscard]] const std::vector<double>& epochs() const noexcept { return epochs_; }
    [[nodiscard]] const std::vector<TimeExpandedNode>& nodes() const noexcept { return nodes_; }
    [[nodiscard]] const std::vector<TimeExpandedArc>& arcs() const noexcept { return arcs_; }

    [[nodiscard]] std::size_t node_id(std::size_t target, std::size_t epoch_index) const {
        if (target >= target_count_ || epoch_index >= epochs_.size()) {
            throw std::out_of_range("time-expanded node coordinate is outside the graph");
        }
        return epoch_index * target_count_ + target;
    }

    [[nodiscard]] const std::vector<std::size_t>& outgoing(std::size_t node) const {
        return outgoing_.at(node);
    }

    void add_arc(TimeExpandedArc arc) {
        if (arc.from >= nodes_.size() || arc.to >= nodes_.size()) {
            throw std::invalid_argument("time-expanded arc endpoint is outside the graph");
        }
        const auto& from = nodes_[arc.from];
        const auto& to = nodes_[arc.to];
        if (to.epoch_index <= from.epoch_index) {
            throw std::invalid_argument("time-expanded arcs must move forward in time");
        }
        if (from.target == to.target) {
            throw std::invalid_argument("inter-target graph arcs must change targets");
        }
        for (const auto value : {arc.nominal_cost, arc.lower_bound, arc.delta_v}) {
            if (!std::isfinite(value) || value < 0.0) {
                throw std::invalid_argument("time-expanded arc metrics must be finite and non-negative");
            }
        }
        if (arc.lower_bound > arc.nominal_cost) {
            throw std::invalid_argument("arc lower bound may not exceed nominal cost");
        }
        const auto arc_index = arcs_.size();
        arcs_.push_back(arc);
        outgoing_[arc.from].push_back(arc_index);
    }

    void sort_outgoing() {
        for (auto& adjacency : outgoing_) {
            std::sort(
                adjacency.begin(),
                adjacency.end(),
                [this](std::size_t left_index, std::size_t right_index) {
                    const auto& left = arcs_[left_index];
                    const auto& right = arcs_[right_index];
                    const auto& left_node = nodes_[left.to];
                    const auto& right_node = nodes_[right.to];
                    return std::tie(
                               left.lower_bound,
                               left.nominal_cost,
                               left_node.epoch_index,
                               left_node.target,
                               left.to
                           )
                           < std::tie(
                               right.lower_bound,
                               right.nominal_cost,
                               right_node.epoch_index,
                               right_node.target,
                               right.to
                           );
                }
            );
        }
    }

  private:
    std::size_t target_count_{0U};
    std::vector<double> epochs_{};
    std::vector<TimeExpandedNode> nodes_{};
    std::vector<TimeExpandedArc> arcs_{};
    std::vector<std::vector<std::size_t>> outgoing_{};
};

/// Build every feasible scheduled arc returned by `oracle`.
///
/// The oracle signature is `(from_target, to_target, departure_epoch, arrival_epoch)` and returns
/// `ScheduledArcEstimate`. This exhaustive builder is a correctness baseline. Production search
/// should add arcs lazily or use dynamic discretisation discovery.
template <typename ScheduledOracle>
TimeExpandedGraph build_time_expanded_graph(
    std::size_t target_count,
    const std::vector<double>& epochs,
    ScheduledOracle&& oracle
) {
    TimeExpandedGraph graph(target_count, epochs);
    for (std::size_t departure_index = 0; departure_index + 1U < epochs.size();
         ++departure_index) {
        for (std::size_t arrival_index = departure_index + 1U;
             arrival_index < epochs.size();
             ++arrival_index) {
            for (std::size_t from = 0; from < target_count; ++from) {
                for (std::size_t to = 0; to < target_count; ++to) {
                    if (from == to) {
                        continue;
                    }
                    const auto estimate = oracle(
                        from,
                        to,
                        epochs[departure_index],
                        epochs[arrival_index]
                    );
                    if (!estimate.feasible) {
                        continue;
                    }
                    graph.add_arc(
                        TimeExpandedArc{
                            graph.node_id(from, departure_index),
                            graph.node_id(to, arrival_index),
                            estimate.nominal_cost,
                            estimate.lower_bound,
                            estimate.delta_v,
                        }
                    );
                }
            }
        }
    }
    graph.sort_outgoing();
    return graph;
}

struct ScheduledRoute {
    std::vector<std::size_t> node_sequence{};
    std::vector<std::size_t> arc_sequence{};
    double nominal_cost{0.0};
    double lower_bound{0.0};
    double delta_v{0.0};
};

namespace time_graph_detail {

struct LabelKey {
    std::size_t node{0U};
    std::uint64_t visited_targets{0U};
    std::size_t legs{0U};

    [[nodiscard]] auto tie() const noexcept {
        return std::tie(node, visited_targets, legs);
    }
};

inline bool operator<(const LabelKey& left, const LabelKey& right) noexcept {
    return left.tie() < right.tie();
}

struct Label {
    LabelKey key{};
    double nominal_cost{0.0};
    double lower_bound{0.0};
    double delta_v{0.0};
    std::vector<std::size_t> nodes{};
    std::vector<std::size_t> arcs{};
};

inline bool better_label(const Label& left, const Label& right) {
    return std::tie(left.nominal_cost, left.lower_bound, left.delta_v, left.nodes)
           < std::tie(right.nominal_cost, right.lower_bound, right.delta_v, right.nodes);
}

}  // namespace time_graph_detail

/// Exact elementary label search over a finite time-expanded graph.
///
/// This is exponential in target count and is intended as a small-instance truth model for beam
/// search and future column-generation pricing. It supports at most 64 targets.
inline std::optional<ScheduledRoute> minimum_cost_elementary_route(
    const TimeExpandedGraph& graph,
    std::size_t start_target,
    std::size_t start_epoch_index,
    std::size_t required_legs
) {
    if (graph.target_count() > 64U) {
        throw std::invalid_argument("exact elementary labels support at most 64 targets");
    }
    if (start_target >= graph.target_count() || start_epoch_index >= graph.epoch_count()) {
        throw std::invalid_argument("elementary-route start is outside the graph");
    }
    if (required_legs >= graph.target_count()) {
        throw std::invalid_argument("an elementary route cannot use that many legs");
    }
    using time_graph_detail::Label;
    using time_graph_detail::LabelKey;
    std::map<LabelKey, Label> current{};
    const auto start_node = graph.node_id(start_target, start_epoch_index);
    const auto start_mask = std::uint64_t{1U} << start_target;
    const Label start{
        LabelKey{start_node, start_mask, 0U},
        0.0,
        0.0,
        0.0,
        {start_node},
        {},
    };
    current.emplace(start.key, start);

    for (std::size_t leg = 0; leg < required_legs; ++leg) {
        std::map<LabelKey, Label> next{};
        for (const auto& [key, label] : current) {
            static_cast<void>(key);
            for (const auto arc_index : graph.outgoing(label.key.node)) {
                const auto& arc = graph.arcs()[arc_index];
                const auto target = graph.nodes()[arc.to].target;
                const auto bit = std::uint64_t{1U} << target;
                if ((label.key.visited_targets & bit) != 0U) {
                    continue;
                }
                auto candidate = label;
                candidate.key = LabelKey{
                    arc.to,
                    label.key.visited_targets | bit,
                    leg + 1U,
                };
                candidate.nominal_cost += arc.nominal_cost;
                candidate.lower_bound += arc.lower_bound;
                candidate.delta_v += arc.delta_v;
                candidate.nodes.push_back(arc.to);
                candidate.arcs.push_back(arc_index);
                const auto iterator = next.find(candidate.key);
                if (iterator == next.end()
                    || time_graph_detail::better_label(candidate, iterator->second)) {
                    next[candidate.key] = std::move(candidate);
                }
            }
        }
        if (next.empty()) {
            return std::nullopt;
        }
        current = std::move(next);
    }

    const auto best = std::min_element(
        current.begin(),
        current.end(),
        [](const auto& left, const auto& right) {
            return time_graph_detail::better_label(left.second, right.second);
        }
    );
    if (best == current.end()) {
        return std::nullopt;
    }
    return ScheduledRoute{
        best->second.nodes,
        best->second.arcs,
        best->second.nominal_cost,
        best->second.lower_bound,
        best->second.delta_v,
    };
}

/// Optimistic non-elementary dynamic-programming bound for `required_legs` remaining legs.
inline double optimistic_route_lower_bound(
    const TimeExpandedGraph& graph,
    std::size_t start_node,
    std::size_t required_legs
) {
    if (start_node >= graph.nodes().size()) {
        throw std::invalid_argument("lower-bound start node is outside the graph");
    }
    std::vector<double> current(
        graph.nodes().size(),
        std::numeric_limits<double>::infinity()
    );
    current[start_node] = 0.0;
    for (std::size_t leg = 0; leg < required_legs; ++leg) {
        std::vector<double> next(
            graph.nodes().size(),
            std::numeric_limits<double>::infinity()
        );
        for (std::size_t node = 0; node < graph.nodes().size(); ++node) {
            if (!std::isfinite(current[node])) {
                continue;
            }
            for (const auto arc_index : graph.outgoing(node)) {
                const auto& arc = graph.arcs()[arc_index];
                next[arc.to] = std::min(next[arc.to], current[node] + arc.lower_bound);
            }
        }
        current = std::move(next);
    }
    return *std::min_element(current.begin(), current.end());
}

}  // namespace spacepdhcg::orbitweaver
