#include "spacepdhcg/core/fixed_cqp.hpp"

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

int main() {
    using spacepdhcg::ConeBlockDescriptor;
    using spacepdhcg::ConeKind;
    using spacepdhcg::core::CscPattern;
    using spacepdhcg::core::FixedCQP;
    using spacepdhcg::core::FixedStructure;
    using spacepdhcg::core::NumericValues;

    FixedStructure structure{
        CscPattern{2, 2, {0, 1, 2}, {0, 1}},
        CscPattern{1, 2, {0, 1, 2}, {0, 0}},
        CscPattern{3, 2, {0, 1, 2}, {0, 1}},
        {ConeBlockDescriptor{ConeKind::second_order, 0, 1, 0.0}},
        {},
    };
    NumericValues values{
        {1.0, 2.0},
        {1.0, -1.0},
        {1.0, 1.0},
        {0.0, 0.0},
        {-1.0},
        {1.0},
        {0.0, 0.0, 2.0},
        {-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()},
        {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()},
    };

    FixedCQP problem(std::move(structure), values);
    const auto fingerprint = problem.topology_fingerprint();
    if (fingerprint == 0U || problem.structure().descriptor().variables != 2) {
        return 1;
    }
    values.linear_objective[0] = 0.25;
    problem.update_values(values);
    if (problem.update_count() != 1U || problem.topology_fingerprint() != fingerprint) {
        return 2;
    }

    auto invalid = values;
    invalid.scalar_lower.clear();
    bool rejected{false};
    try {
        problem.update_values(std::move(invalid));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    return rejected ? 0 : 3;
}
