#pragma once

#include "spacepdhcg/orbitweaver/column_generation.hpp"
#include "spacepdhcg/orbitweaver/route_master.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace spacepdhcg::orbitweaver {

struct RestrictedMasterLPConfig {
    double feasibility_tolerance{1.0e-9};
    double pivot_tolerance{1.0e-12};

    void validate() const {
        if (!std::isfinite(feasibility_tolerance) || feasibility_tolerance <= 0.0
            || !std::isfinite(pivot_tolerance) || pivot_tolerance <= 0.0) {
            throw std::invalid_argument(
                "restricted-master LP tolerances must be finite and positive"
            );
        }
    }
};

namespace restricted_master_lp_detail {

/// Dense two-phase simplex for max c^T x subject to A x <= b, x >= 0.
/// The implementation follows the deterministic tableau convention used by standard
/// competitive-programming/reference simplex solvers, augmented with basis-dual recovery.
class DenseSimplex {
  public:
    DenseSimplex(
        std::vector<std::vector<double>> constraints,
        std::vector<double> right_hand_side,
        std::vector<double> objective,
        RestrictedMasterLPConfig config
    )
        : rows_(right_hand_side.size()),
          columns_(objective.size()),
          original_constraints_(constraints),
          objective_(std::move(objective)),
          config_(config),
          basis_(rows_),
          nonbasis_(columns_ + 1U),
          tableau_(rows_ + 2U, std::vector<double>(columns_ + 2U, 0.0)) {
        config_.validate();
        if (constraints.size() != rows_) {
            throw std::invalid_argument("simplex constraint and RHS row counts differ");
        }
        for (const auto& row : constraints) {
            if (row.size() != columns_) {
                throw std::invalid_argument("simplex constraint row has the wrong width");
            }
        }
        if (!finite_matrix(constraints) || !finite_vector(right_hand_side)
            || !finite_vector(objective_)) {
            throw std::invalid_argument("simplex data must be finite");
        }
        for (std::size_t row = 0; row < rows_; ++row) {
            for (std::size_t column = 0; column < columns_; ++column) {
                tableau_[row][column] = constraints[row][column];
            }
            basis_[row] = columns_ + row;
            tableau_[row][columns_] = -1.0;
            tableau_[row][columns_ + 1U] = right_hand_side[row];
        }
        for (std::size_t column = 0; column < columns_; ++column) {
            nonbasis_[column] = column;
            tableau_[rows_][column] = -objective_[column];
        }
        nonbasis_[columns_] = artificial_index();
        tableau_[rows_ + 1U][columns_] = 1.0;
    }

    enum class Status {
        optimal,
        infeasible,
        unbounded,
    };

    struct Result {
        Status status{Status::infeasible};
        double objective{0.0};
        std::vector<double> primal{};
        std::vector<double> dual{};
        std::size_t pivots{0U};
    };

    [[nodiscard]] Result solve() {
        if (rows_ == 0U) {
            for (const auto coefficient : objective_) {
                if (coefficient > config_.pivot_tolerance) {
                    return Result{Status::unbounded, 0.0, {}, {}, pivots_};
                }
            }
            return Result{
                Status::optimal,
                0.0,
                std::vector<double>(columns_, 0.0),
                {},
                pivots_,
            };
        }
        const auto minimum_row = static_cast<std::size_t>(std::min_element(
            tableau_.begin(),
            tableau_.begin() + static_cast<std::ptrdiff_t>(rows_),
            [this](const auto& left, const auto& right) {
                return left[columns_ + 1U] < right[columns_ + 1U];
            }
        ) - tableau_.begin());
        if (tableau_[minimum_row][columns_ + 1U] < -config_.feasibility_tolerance) {
            pivot(minimum_row, columns_);
            if (!simplex(1U)
                || tableau_[rows_ + 1U][columns_ + 1U]
                       < -config_.feasibility_tolerance) {
                return Result{Status::infeasible, 0.0, {}, {}, pivots_};
            }
            if (std::abs(tableau_[rows_ + 1U][columns_ + 1U])
                > config_.feasibility_tolerance) {
                return Result{Status::infeasible, 0.0, {}, {}, pivots_};
            }
            const auto artificial = std::find(
                basis_.begin(),
                basis_.end(),
                artificial_index()
            );
            if (artificial != basis_.end()) {
                const auto row = static_cast<std::size_t>(artificial - basis_.begin());
                std::optional<std::size_t> entering{};
                for (std::size_t column = 0; column <= columns_; ++column) {
                    if (nonbasis_[column] == artificial_index()) {
                        continue;
                    }
                    if (!entering.has_value()
                        || tableau_[row][column]
                               < tableau_[row][*entering] - config_.pivot_tolerance
                        || (std::abs(
                                tableau_[row][column] - tableau_[row][*entering]
                            ) <= config_.pivot_tolerance
                            && nonbasis_[column] < nonbasis_[*entering])) {
                        entering = column;
                    }
                }
                if (entering.has_value()
                    && std::abs(tableau_[row][*entering]) > config_.pivot_tolerance) {
                    pivot(row, *entering);
                }
            }
        }
        if (!simplex(2U)) {
            return Result{Status::unbounded, 0.0, {}, {}, pivots_};
        }
        std::vector<double> primal(columns_, 0.0);
        for (std::size_t row = 0; row < rows_; ++row) {
            if (basis_[row] < columns_) {
                primal[basis_[row]] = tableau_[row][columns_ + 1U];
            }
        }
        return Result{
            Status::optimal,
            tableau_[rows_][columns_ + 1U],
            std::move(primal),
            recover_dual(),
            pivots_,
        };
    }

  private:
    std::size_t rows_{0U};
    std::size_t columns_{0U};
    std::vector<std::vector<double>> original_constraints_{};
    std::vector<double> objective_{};
    RestrictedMasterLPConfig config_{};
    std::vector<std::size_t> basis_{};
    std::vector<std::size_t> nonbasis_{};
    std::vector<std::vector<double>> tableau_{};
    std::size_t pivots_{0U};

    [[nodiscard]] std::size_t artificial_index() const noexcept {
        return std::numeric_limits<std::size_t>::max();
    }

    static bool finite_vector(const std::vector<double>& values) {
        return std::all_of(values.begin(), values.end(), [](const double value) {
            return std::isfinite(value);
        });
    }

    static bool finite_matrix(const std::vector<std::vector<double>>& values) {
        return std::all_of(values.begin(), values.end(), [](const auto& row) {
            return finite_vector(row);
        });
    }

    void pivot(const std::size_t leaving_row, const std::size_t entering_column) {
        const auto inverse = 1.0 / tableau_[leaving_row][entering_column];
        for (std::size_t row = 0; row < rows_ + 2U; ++row) {
            if (row == leaving_row) {
                continue;
            }
            for (std::size_t column = 0; column < columns_ + 2U; ++column) {
                if (column == entering_column) {
                    continue;
                }
                tableau_[row][column] -= tableau_[leaving_row][column]
                                          * tableau_[row][entering_column] * inverse;
            }
        }
        for (std::size_t column = 0; column < columns_ + 2U; ++column) {
            if (column != entering_column) {
                tableau_[leaving_row][column] *= inverse;
            }
        }
        for (std::size_t row = 0; row < rows_ + 2U; ++row) {
            if (row != leaving_row) {
                tableau_[row][entering_column] *= -inverse;
            }
        }
        tableau_[leaving_row][entering_column] = inverse;
        std::swap(basis_[leaving_row], nonbasis_[entering_column]);
        ++pivots_;
    }

    [[nodiscard]] bool simplex(const std::size_t phase) {
        const auto objective_row = phase == 1U ? rows_ + 1U : rows_;
        while (true) {
            std::optional<std::size_t> entering{};
            for (std::size_t column = 0; column <= columns_; ++column) {
                if (phase == 2U && nonbasis_[column] == artificial_index()) {
                    continue;
                }
                if (!entering.has_value()
                    || tableau_[objective_row][column]
                           < tableau_[objective_row][*entering]
                                 - config_.pivot_tolerance
                    || (std::abs(
                            tableau_[objective_row][column]
                            - tableau_[objective_row][*entering]
                        ) <= config_.pivot_tolerance
                        && nonbasis_[column] < nonbasis_[*entering])) {
                    entering = column;
                }
            }
            if (!entering.has_value()
                || tableau_[objective_row][*entering]
                       >= -config_.pivot_tolerance) {
                return true;
            }
            std::optional<std::size_t> leaving{};
            for (std::size_t row = 0; row < rows_; ++row) {
                if (tableau_[row][*entering] <= config_.pivot_tolerance) {
                    continue;
                }
                if (!leaving.has_value()) {
                    leaving = row;
                    continue;
                }
                const auto ratio = tableau_[row][columns_ + 1U]
                                   / tableau_[row][*entering];
                const auto incumbent_ratio = tableau_[*leaving][columns_ + 1U]
                                             / tableau_[*leaving][*entering];
                if (ratio < incumbent_ratio - config_.feasibility_tolerance
                    || (std::abs(ratio - incumbent_ratio)
                            <= config_.feasibility_tolerance
                        && basis_[row] < basis_[*leaving])) {
                    leaving = row;
                }
            }
            if (!leaving.has_value()) {
                return false;
            }
            pivot(*leaving, *entering);
        }
    }

    [[nodiscard]] std::vector<double> recover_dual() const {
        std::vector<std::vector<double>> transposed_basis(
            rows_,
            std::vector<double>(rows_, 0.0)
        );
        std::vector<double> basis_objective(rows_, 0.0);
        for (std::size_t basis_column = 0; basis_column < rows_; ++basis_column) {
            const auto variable = basis_[basis_column];
            if (variable < columns_) {
                basis_objective[basis_column] = objective_[variable];
                for (std::size_t row = 0; row < rows_; ++row) {
                    transposed_basis[basis_column][row] =
                        original_constraints_[row][variable];
                }
            } else if (variable != artificial_index()) {
                const auto slack_row = variable - columns_;
                if (slack_row >= rows_) {
                    throw std::logic_error("simplex basis contains an invalid slack variable");
                }
                transposed_basis[basis_column][slack_row] = 1.0;
            } else {
                throw std::logic_error("artificial variable remained in optimal simplex basis");
            }
        }
        return solve_square_system(
            std::move(transposed_basis),
            std::move(basis_objective),
            config_.pivot_tolerance
        );
    }

    static std::vector<double> solve_square_system(
        std::vector<std::vector<double>> matrix,
        std::vector<double> right_hand_side,
        const double tolerance
    ) {
        const auto dimension = right_hand_side.size();
        for (std::size_t pivot_column = 0; pivot_column < dimension; ++pivot_column) {
            auto pivot_row = pivot_column;
            for (std::size_t row = pivot_column + 1U; row < dimension; ++row) {
                if (std::abs(matrix[row][pivot_column])
                    > std::abs(matrix[pivot_row][pivot_column])) {
                    pivot_row = row;
                }
            }
            if (std::abs(matrix[pivot_row][pivot_column]) <= tolerance) {
                throw std::runtime_error("simplex basis is singular during dual recovery");
            }
            std::swap(matrix[pivot_column], matrix[pivot_row]);
            std::swap(right_hand_side[pivot_column], right_hand_side[pivot_row]);
            const auto inverse = 1.0 / matrix[pivot_column][pivot_column];
            for (std::size_t column = pivot_column; column < dimension; ++column) {
                matrix[pivot_column][column] *= inverse;
            }
            right_hand_side[pivot_column] *= inverse;
            for (std::size_t row = 0; row < dimension; ++row) {
                if (row == pivot_column) {
                    continue;
                }
                const auto factor = matrix[row][pivot_column];
                if (std::abs(factor) <= tolerance) {
                    continue;
                }
                for (std::size_t column = pivot_column; column < dimension; ++column) {
                    matrix[row][column] -= factor * matrix[pivot_column][column];
                }
                right_hand_side[row] -= factor * right_hand_side[pivot_column];
            }
        }
        return right_hand_side;
    }
};

}  // namespace restricted_master_lp_detail

/// Dependency-free restricted-master LP reference for native route column generation.
///
/// Required targets are covered exactly once; each spacecraft contributes at most one route.
/// The returned dual prices use the same convention as `route_reduced_cost`:
///
///     c_r - sum(target_price) - spacecraft_price.
class DenseRestrictedMasterLP {
  public:
    DenseRestrictedMasterLP(
        const std::size_t spacecraft_count,
        const std::size_t target_count,
        std::vector<std::size_t> required_targets = {},
        RestrictedMasterLPConfig config = {}
    )
        : spacecraft_count_(spacecraft_count),
          target_count_(target_count),
          required_targets_(std::move(required_targets)),
          config_(config) {
        if (spacecraft_count_ == 0U || target_count_ == 0U) {
            throw std::invalid_argument("restricted-master dimensions must be positive");
        }
        config_.validate();
        if (required_targets_.empty()) {
            required_targets_.resize(target_count_);
            for (std::size_t target = 0; target < target_count_; ++target) {
                required_targets_[target] = target;
            }
        }
        std::sort(required_targets_.begin(), required_targets_.end());
        if (std::adjacent_find(required_targets_.begin(), required_targets_.end())
            != required_targets_.end()) {
            throw std::invalid_argument("restricted-master required targets must be unique");
        }
        for (const auto target : required_targets_) {
            if (target >= target_count_) {
                throw std::invalid_argument(
                    "restricted-master required target is outside the problem"
                );
            }
        }
    }

    [[nodiscard]] RestrictedMasterResult operator()(
        const std::vector<RouteColumn>& columns
    ) const {
        const auto start = std::chrono::steady_clock::now();
        if (columns.empty()) {
            return {};
        }
        for (const auto& column : columns) {
            column.validate(spacecraft_count_, target_count_);
        }

        const auto target_rows = required_targets_.size();
        const auto row_count = 2U * target_rows + spacecraft_count_;
        std::vector<std::vector<double>> matrix(
            row_count,
            std::vector<double>(columns.size(), 0.0)
        );
        std::vector<double> right_hand_side(row_count, 1.0);
        std::vector<double> objective(columns.size(), 0.0);
        for (std::size_t route = 0; route < columns.size(); ++route) {
            const auto& column = columns[route];
            objective[route] = -column.cost;
            for (std::size_t target_row = 0; target_row < target_rows; ++target_row) {
                const auto target = required_targets_[target_row];
                const auto covers = std::binary_search(
                    column.targets.begin(),
                    column.targets.end(),
                    target
                ) ? 1.0 : 0.0;
                matrix[target_row][route] = covers;
                matrix[target_rows + target_row][route] = -covers;
            }
            matrix[2U * target_rows + column.spacecraft][route] = 1.0;
        }
        for (std::size_t target_row = 0; target_row < target_rows; ++target_row) {
            right_hand_side[target_rows + target_row] = -1.0;
        }

        restricted_master_lp_detail::DenseSimplex simplex{
            std::move(matrix),
            std::move(right_hand_side),
            std::move(objective),
            config_,
        };
        const auto solution = simplex.solve();
        const auto stop = std::chrono::steady_clock::now();
        if (solution.status
            != restricted_master_lp_detail::DenseSimplex::Status::optimal) {
            return {};
        }
        if (solution.dual.size() != row_count) {
            throw std::logic_error("restricted-master dual vector has the wrong size");
        }

        RouteMasterDualPrices prices{};
        prices.target.assign(target_count_, 0.0);
        prices.spacecraft.assign(spacecraft_count_, 0.0);
        for (std::size_t target_row = 0; target_row < target_rows; ++target_row) {
            prices.target[required_targets_[target_row]] =
                solution.dual[target_rows + target_row]
                - solution.dual[target_row];
        }
        for (std::size_t spacecraft = 0; spacecraft < spacecraft_count_; ++spacecraft) {
            prices.spacecraft[spacecraft] =
                -solution.dual[2U * target_rows + spacecraft];
        }
        prices.validate(target_count_, spacecraft_count_);

        const auto minimum_objective = std::max(0.0, -solution.objective);
        RestrictedMasterResult result{
            true,
            minimum_objective,
            minimum_objective,
            std::move(prices),
            solution.pivots,
            std::chrono::duration<double>(stop - start).count(),
        };
        result.validate(target_count_, spacecraft_count_);
        return result;
    }

    [[nodiscard]] RestrictedMasterSolver callback() const {
        return [*this](const std::vector<RouteColumn>& columns) {
            return (*this)(columns);
        };
    }

  private:
    std::size_t spacecraft_count_{0U};
    std::size_t target_count_{0U};
    std::vector<std::size_t> required_targets_{};
    RestrictedMasterLPConfig config_{};
};

}  // namespace spacepdhcg::orbitweaver
