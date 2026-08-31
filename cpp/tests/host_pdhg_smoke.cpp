#include "spacepdhcg/core/host_pdhg.hpp"
#include "spacepdhcg/core/sparse_builder.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace {

[[nodiscard]] spacepdhcg::core::CQPValues unbounded_values(
    const std::size_t variables,
    const std::size_t scalar_rows,
    const std::size_t affine_rows
) {
    using spacepdhcg::core::CQPValues;
    return CQPValues{
        {},
        {},
        {},
        std::vector<double>(variables, 0.0),
        std::vector<double>(scalar_rows, 0.0),
        std::vector<double>(scalar_rows, 0.0),
        std::vector<double>(affine_rows, 0.0),
        std::vector<double>(variables, -std::numeric_limits<double>::infinity()),
        std::vector<double>(variables, std::numeric_limits<double>::infinity()),
    };
}

[[nodiscard]] int equality_qp() {
    using namespace spacepdhcg;
    using namespace spacepdhcg::core;

    CscBuilder quadratic_builder{2, 2};
    quadratic_builder.add(0, 0, 1.0);
    quadratic_builder.add(1, 1, 1.0);
    auto quadratic = quadratic_builder.build();
    CscBuilder constraint_builder{1, 2};
    constraint_builder.add(0, 0, 1.0);
    constraint_builder.add(0, 1, 1.0);
    auto constraint = constraint_builder.build();
    CQPStructure structure{
        std::move(quadratic.structure),
        std::move(constraint.structure),
    };
    auto values = unbounded_values(2U, 1U, 0U);
    values.quadratic = std::move(quadratic.values);
    values.scalar_constraint = std::move(constraint.values);
    values.scalar_lower[0] = 1.0;
    values.scalar_upper[0] = 1.0;

    PersistentHostPDHG workspace{structure, values};
    HostPDHGOptions options;
    options.tolerance = 2.0e-7;
    options.iteration_limit = 50'000U;
    options.check_interval = 10U;
    const auto first = workspace.solve(options);
    if (!first.solved() || first.primal_residual > options.tolerance ||
        first.dual_residual > options.tolerance ||
        std::abs(first.primal[0] - 0.5) > 2.0e-6 ||
        std::abs(first.primal[1] - 0.5) > 2.0e-6) {
        return 1;
    }

    auto updated = values;
    updated.scalar_lower[0] = 2.0;
    updated.scalar_upper[0] = 2.0;
    workspace.update_values(updated);
    workspace.warm_start(first.primal, first.dual);
    const auto second = workspace.solve(options);
    if (!second.solved() || std::abs(second.primal[0] - 1.0) > 2.0e-6 ||
        std::abs(second.primal[1] - 1.0) > 2.0e-6 || workspace.update_count() != 1U ||
        workspace.warm_start_count() != 1U || workspace.solve_count() != 2U) {
        return 2;
    }
    return 0;
}

[[nodiscard]] int soc_projection_qp() {
    using namespace spacepdhcg;
    using namespace spacepdhcg::core;

    CscBuilder quadratic_builder{3, 3};
    for (Index index = 0; index < 3; ++index) {
        quadratic_builder.add(index, index, 1.0);
    }
    auto quadratic = quadratic_builder.build();
    CscBuilder scalar_builder{0, 3};
    auto scalar = scalar_builder.build();
    CscBuilder affine_builder{3, 3};
    for (Index index = 0; index < 3; ++index) {
        affine_builder.add(index, index, 1.0);
    }
    auto affine = affine_builder.build();
    CQPStructure structure{
        std::move(quadratic.structure),
        std::move(scalar.structure),
        std::move(affine.structure),
        {ConeBlock{ConeKind::second_order, 0, 1}},
    };
    auto values = unbounded_values(3U, 0U, 3U);
    values.quadratic = std::move(quadratic.values);
    values.scalar_constraint = std::move(scalar.values);
    values.affine_cone = std::move(affine.values);
    values.linear_objective = {-2.0, 0.0, -1.0};

    PersistentHostPDHG workspace{structure, values};
    HostPDHGOptions options;
    options.tolerance = 5.0e-7;
    options.iteration_limit = 100'000U;
    options.check_interval = 20U;
    const auto solution = workspace.solve(options);
    if (!solution.solved() || solution.primal_residual > options.tolerance ||
        solution.dual_residual > options.tolerance ||
        std::abs(solution.primal[0] - 1.5) > 5.0e-5 ||
        std::abs(solution.primal[1]) > 5.0e-5 ||
        std::abs(solution.primal[2] - 1.5) > 5.0e-5) {
        return 3;
    }
    return 0;
}

}  // namespace

int main() {
    const int equality = equality_qp();
    if (equality != 0) {
        return equality;
    }
    return soc_projection_qp();
}
