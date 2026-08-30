#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

namespace spacepdhcg {

using Index = std::int32_t;

/// Opaque CUDA stream handle. The implementation interprets `native` as cudaStream_t.
struct StreamHandle {
    void* native{nullptr};
};

enum class SparseFormat : std::uint8_t {
    csc,
    csr,
};

enum class ConeKind : std::uint8_t {
    second_order,
    rotated_second_order,
    exponential,
    power,
    positive_semidefinite,
};

enum class RescalePolicy : std::uint8_t {
    /// Retain the existing scaling and preconditioner unconditionally.
    reuse,
    /// Refresh only when device-side change diagnostics cross configured thresholds.
    refresh_if_needed,
    /// Recompute all numerical scaling before the next solve.
    force_refresh,
};

enum class WorkspaceState : std::uint8_t {
    ready,
    update_pending,
    solving,
    solved,
    failed,
};

enum class SolveStatus : std::uint8_t {
    optimal,
    primal_infeasible,
    dual_infeasible,
    iteration_limit,
    interrupted,
    numerical_failure,
    internal_error,
};

template <typename T>
struct HostConstSpan {
    const T* data{nullptr};
    std::size_t size{0};

    [[nodiscard]] constexpr bool empty() const noexcept { return size == 0; }
};

template <typename T>
struct DeviceConstSpan {
    const T* data{nullptr};
    std::size_t size{0};

    [[nodiscard]] constexpr bool empty() const noexcept { return size == 0; }
};

template <typename T>
struct DeviceSpan {
    T* data{nullptr};
    std::size_t size{0};

    [[nodiscard]] constexpr bool empty() const noexcept { return size == 0; }
};

/// Fixed sparse index structure. Numerical values are supplied separately on device.
struct SparsePatternView {
    SparseFormat format{SparseFormat::csc};
    Index rows{0};
    Index columns{0};
    HostConstSpan<Index> offsets{};
    HostConstSpan<Index> indices{};

    [[nodiscard]] constexpr std::size_t nonzeros() const noexcept { return indices.size; }

    [[nodiscard]] constexpr bool well_formed() const noexcept {
        if (rows < 0 || columns < 0) {
            return false;
        }
        const auto major = format == SparseFormat::csc ? columns : rows;
        return offsets.size == static_cast<std::size_t>(major) + 1;
    }
};

/// Native PDHCG cone metadata. `vector_dimension` follows upstream `v_dim` semantics.
struct ConeBlockDescriptor {
    ConeKind kind{ConeKind::second_order};
    Index start{0};
    Index vector_dimension{0};
    double power_alpha{0.0};
};

/// Immutable symbolic structure copied during workspace creation.
struct StructureDescriptor {
    Index variables{0};
    SparsePatternView quadratic{};
    SparsePatternView scalar_constraint{};
    SparsePatternView affine_cone{};
    HostConstSpan<ConeBlockDescriptor> affine_cones{};
    HostConstSpan<ConeBlockDescriptor> variable_cones{};
};

/// Device-resident numerical values matching one immutable structure.
struct NumericValuesView {
    DeviceConstSpan<double> quadratic_values{};
    DeviceConstSpan<double> scalar_constraint_values{};
    DeviceConstSpan<double> affine_cone_values{};
    DeviceConstSpan<double> linear_objective{};
    DeviceConstSpan<double> scalar_lower{};
    DeviceConstSpan<double> scalar_upper{};
    DeviceConstSpan<double> affine_offset{};
    DeviceConstSpan<double> variable_lower{};
    DeviceConstSpan<double> variable_upper{};
};

/// Device-resident primal and dual starts. Dual ordering is [dual_A, dual_F].
struct WarmStartView {
    DeviceConstSpan<double> primal{};
    DeviceConstSpan<double> dual{};
};

struct SolutionView {
    DeviceConstSpan<double> primal{};
    DeviceConstSpan<double> dual{};
};

struct ResidualView {
    DeviceConstSpan<double> primal_components{};
    DeviceConstSpan<double> dual_components{};
};

struct ScalingThresholds {
    double maximum_relative_matrix_change{0.25};
    double maximum_relative_vector_change{0.50};
    Index maximum_reuse_updates{20};
};

struct CreateOptions {
    Index device_ordinal{0};
    bool retain_unscaled_values{true};
    bool enable_cuda_graph_capture{false};
    ScalingThresholds scaling_thresholds{};
};

struct SolveOptions {
    double optimality_tolerance{1.0e-4};
    double feasibility_tolerance{1.0e-4};
    Index iteration_limit{1'000'000};
    RescalePolicy rescale_policy{RescalePolicy::refresh_if_needed};
};

struct TimingBreakdown {
    double update_seconds{0.0};
    double rescale_seconds{0.0};
    double solve_seconds{0.0};
    double residual_seconds{0.0};
    double total_seconds{0.0};
};

struct SolveReport {
    SolveStatus status{SolveStatus::internal_error};
    Index outer_iterations{0};
    Index inner_iterations{0};
    double objective{0.0};
    double relative_primal_residual{0.0};
    double relative_dual_residual{0.0};
    bool scaling_refreshed{false};
    TimingBreakdown timing{};
};

using DiagnosticCallback = void (*)(std::string_view message, void* user_data);

/// Abstract persistent solver workspace implemented by the CUDA/PDHCG integration.
class PersistentCQP {
  public:
    PersistentCQP(const PersistentCQP&) = delete;
    PersistentCQP& operator=(const PersistentCQP&) = delete;
    PersistentCQP(PersistentCQP&&) = delete;
    PersistentCQP& operator=(PersistentCQP&&) = delete;
    virtual ~PersistentCQP() = default;

    [[nodiscard]] virtual WorkspaceState state() const noexcept = 0;
    [[nodiscard]] virtual Index variables() const noexcept = 0;
    [[nodiscard]] virtual Index scalar_constraints() const noexcept = 0;
    [[nodiscard]] virtual Index affine_cone_rows() const noexcept = 0;

    /// Enqueue numerical updates on `stream`; all pointers are borrowed until completion.
    virtual void update_values(
        const NumericValuesView& values,
        RescalePolicy rescale_policy,
        StreamHandle stream
    ) = 0;

    /// Enqueue a primal-dual warm-start copy into workspace-owned device buffers.
    virtual void set_warm_start(const WarmStartView& warm_start, StreamHandle stream) = 0;

    /// Reset iterates while retaining structure, allocations, descriptors and scaling.
    virtual void reset_iterates(StreamHandle stream) = 0;

    /// Enqueue PDHCG iterations and residual evaluation on `stream`.
    virtual void solve_async(const SolveOptions& options, StreamHandle stream) = 0;

    /// Synchronise the most recently submitted solve and return compact host diagnostics.
    [[nodiscard]] virtual SolveReport synchronize() = 0;

    /// Borrow workspace-owned device solution and residual buffers.
    [[nodiscard]] virtual SolutionView solution() const noexcept = 0;
    [[nodiscard]] virtual ResidualView residuals() const noexcept = 0;

    /// Request cooperative cancellation without freeing workspace state.
    virtual void request_cancel() noexcept = 0;

  protected:
    PersistentCQP() = default;
};

/// Create one workspace and copy immutable sparse/cone metadata exactly once.
[[nodiscard]] std::unique_ptr<PersistentCQP> create_persistent_cqp(
    const StructureDescriptor& structure,
    const NumericValuesView& initial_values,
    const CreateOptions& options,
    StreamHandle stream,
    DiagnosticCallback callback = nullptr,
    void* callback_user_data = nullptr
);

}  // namespace spacepdhcg
