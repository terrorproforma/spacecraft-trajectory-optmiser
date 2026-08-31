#include "spacepdhcg/core/checkpoint.hpp"
#include "spacepdhcg/core/persistent_session.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

using spacepdhcg::DeviceConstSpan;
using spacepdhcg::Index;
using spacepdhcg::NumericValuesView;
using spacepdhcg::PersistentCQP;
using spacepdhcg::RescalePolicy;
using spacepdhcg::ResidualView;
using spacepdhcg::SolutionView;
using spacepdhcg::SolveOptions;
using spacepdhcg::SolveReport;
using spacepdhcg::SolveStatus;
using spacepdhcg::StreamHandle;
using spacepdhcg::WarmStartView;
using spacepdhcg::WorkspaceState;
using spacepdhcg::core::NumericValues;

NumericValuesView view(const NumericValues& values) {
    return NumericValuesView{
        DeviceConstSpan<double>{values.quadratic.data(), values.quadratic.size()},
        DeviceConstSpan<double>{
            values.scalar_constraint.data(),
            values.scalar_constraint.size(),
        },
        DeviceConstSpan<double>{values.affine_cone.data(), values.affine_cone.size()},
        DeviceConstSpan<double>{
            values.linear_objective.data(),
            values.linear_objective.size(),
        },
        DeviceConstSpan<double>{values.scalar_lower.data(), values.scalar_lower.size()},
        DeviceConstSpan<double>{values.scalar_upper.data(), values.scalar_upper.size()},
        DeviceConstSpan<double>{values.affine_offset.data(), values.affine_offset.size()},
        DeviceConstSpan<double>{values.variable_lower.data(), values.variable_lower.size()},
        DeviceConstSpan<double>{values.variable_upper.data(), values.variable_upper.size()},
    };
}

class FakePersistentCqp final : public PersistentCQP {
  public:
    FakePersistentCqp(Index variables, Index scalar_rows, Index affine_rows)
        : variables_(variables),
          scalar_rows_(scalar_rows),
          affine_rows_(affine_rows),
          primal_(static_cast<std::size_t>(variables), 0.0),
          dual_(static_cast<std::size_t>(scalar_rows + affine_rows), 0.0),
          primal_residual_(2U, 0.0),
          dual_residual_(2U, 0.0) {}

    [[nodiscard]] WorkspaceState state() const noexcept override { return state_; }
    [[nodiscard]] Index variables() const noexcept override { return variables_; }
    [[nodiscard]] Index scalar_constraints() const noexcept override { return scalar_rows_; }
    [[nodiscard]] Index affine_cone_rows() const noexcept override { return affine_rows_; }

    void update_values(
        const NumericValuesView& values,
        RescalePolicy rescale_policy,
        StreamHandle stream
    ) override {
        if (state_ == WorkspaceState::solving) {
            throw std::logic_error("fake workspace is already solving");
        }
        if (stream.native != nullptr) {
            last_stream_ = stream.native;
        }
        last_policy_ = rescale_policy;
        objective_.assign(
            values.linear_objective.data,
            values.linear_objective.data + values.linear_objective.size
        );
        state_ = WorkspaceState::update_pending;
        ++updates_;
    }

    void set_warm_start(const WarmStartView& warm_start, StreamHandle) override {
        if (!warm_start.primal.empty()) {
            primal_.assign(
                warm_start.primal.data,
                warm_start.primal.data + warm_start.primal.size
            );
        }
        if (!warm_start.dual.empty()) {
            dual_.assign(warm_start.dual.data, warm_start.dual.data + warm_start.dual.size);
        }
        ++warm_starts_;
    }

    void reset_iterates(StreamHandle) override {
        std::fill(primal_.begin(), primal_.end(), 0.0);
        std::fill(dual_.begin(), dual_.end(), 0.0);
        state_ = WorkspaceState::ready;
    }

    void solve_async(const SolveOptions& options, StreamHandle) override {
        if (state_ != WorkspaceState::update_pending && state_ != WorkspaceState::ready
            && state_ != WorkspaceState::solved) {
            throw std::logic_error("fake workspace is not ready to solve");
        }
        last_options_ = options;
        state_ = WorkspaceState::solving;
    }

    [[nodiscard]] SolveReport synchronize() override {
        if (state_ != WorkspaceState::solving) {
            throw std::logic_error("fake workspace has no solve in flight");
        }
        state_ = WorkspaceState::solved;
        return SolveReport{
            SolveStatus::optimal,
            4,
            12,
            objective_.empty() ? 0.0 : objective_.front(),
            1.0e-7,
            2.0e-7,
            last_policy_ == RescalePolicy::force_refresh,
            {},
        };
    }

    [[nodiscard]] SolutionView solution() const noexcept override {
        return SolutionView{
            DeviceConstSpan<double>{primal_.data(), primal_.size()},
            DeviceConstSpan<double>{dual_.data(), dual_.size()},
        };
    }

    [[nodiscard]] ResidualView residuals() const noexcept override {
        return ResidualView{
            DeviceConstSpan<double>{primal_residual_.data(), primal_residual_.size()},
            DeviceConstSpan<double>{dual_residual_.data(), dual_residual_.size()},
        };
    }

    void request_cancel() noexcept override { cancelled_ = true; }

    [[nodiscard]] RescalePolicy last_policy() const noexcept { return last_policy_; }
    [[nodiscard]] std::size_t updates() const noexcept { return updates_; }
    [[nodiscard]] std::size_t warm_starts() const noexcept { return warm_starts_; }
    [[nodiscard]] bool cancelled() const noexcept { return cancelled_; }

  private:
    Index variables_{0};
    Index scalar_rows_{0};
    Index affine_rows_{0};
    WorkspaceState state_{WorkspaceState::ready};
    RescalePolicy last_policy_{RescalePolicy::reuse};
    SolveOptions last_options_{};
    void* last_stream_{nullptr};
    std::vector<double> objective_{};
    std::vector<double> primal_{};
    std::vector<double> dual_{};
    std::vector<double> primal_residual_{};
    std::vector<double> dual_residual_{};
    std::size_t updates_{0U};
    std::size_t warm_starts_{0U};
    bool cancelled_{false};
};

}  // namespace

int main() {
    using spacepdhcg::ConeBlockDescriptor;
    using spacepdhcg::ConeKind;
    using spacepdhcg::ScalingThresholds;
    using spacepdhcg::core::CscPattern;
    using spacepdhcg::core::FixedStructure;
    using spacepdhcg::core::PersistentSolveSession;
    using spacepdhcg::core::decode_checkpoint;
    using spacepdhcg::core::encode_checkpoint;

    FixedStructure structure{
        CscPattern{2, 2, {0, 1, 2}, {0, 1}},
        CscPattern{1, 2, {0, 1, 2}, {0, 0}},
        CscPattern{3, 2, {0, 1, 2}, {0, 1}},
        {ConeBlockDescriptor{ConeKind::second_order, 0, 1, 0.0}},
        {},
    };
    NumericValues initial{
        {1.0, 2.0},
        {1.0, -1.0},
        {1.0, 1.0},
        {0.0, 0.0},
        {-1.0},
        {1.0},
        {0.0, 0.0, 2.0},
        {-10.0, -10.0},
        {10.0, 10.0},
    };
    auto fake = std::make_unique<FakePersistentCqp>(2, 1, 3);
    auto* fake_pointer = fake.get();
    PersistentSolveSession session(
        std::move(fake),
        structure,
        initial,
        ScalingThresholds{0.25, 0.50, 20}
    );

    auto first = initial;
    first.linear_objective[0] = 0.1;
    const std::vector<double> primal{0.5, -0.5};
    const std::vector<double> dual{0.0, 0.0, 0.0, 0.0};
    const WarmStartView warm{
        DeviceConstSpan<double>{primal.data(), primal.size()},
        DeviceConstSpan<double>{dual.data(), dual.size()},
    };
    const auto first_record = session.submit(
        first,
        view(first),
        SolveOptions{1.0e-5, 1.0e-5, 1'000, RescalePolicy::refresh_if_needed},
        StreamHandle{},
        &warm
    );
    if (first_record.report.status != SolveStatus::optimal
        || first_record.rescale_policy != RescalePolicy::reuse
        || !first_record.warm_start_applied || fake_pointer->updates() != 1U
        || fake_pointer->warm_starts() != 1U) {
        return 1;
    }

    auto second = first;
    second.linear_objective[0] = 10.0;
    const auto second_record = session.submit(
        second,
        view(second),
        SolveOptions{},
        StreamHandle{}
    );
    if (second_record.rescale_policy != RescalePolicy::force_refresh
        || !second_record.report.scaling_refreshed || session.update_count() != 2U) {
        return 2;
    }

    const auto checkpoint = encode_checkpoint(structure, session.update_count(), second, primal, dual);
    const auto restored = decode_checkpoint(checkpoint, structure);
    if (restored.numerical_update != 2U || restored.primal != primal || restored.dual != dual
        || restored.values.linear_objective != second.linear_objective) {
        return 3;
    }

    auto wrong_structure = structure;
    wrong_structure.quadratic.indices[0] = 1;
    bool rejected{false};
    try {
        static_cast<void>(decode_checkpoint(checkpoint, wrong_structure));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected) {
        return 4;
    }

    session.request_cancel();
    if (!fake_pointer->cancelled()) {
        return 5;
    }
    return 0;
}
