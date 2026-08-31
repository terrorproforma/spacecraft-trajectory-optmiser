#pragma once

#include "spacepdhcg/core/cqp.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace spacepdhcg::core {

struct CscMatrixData {
    CscStructure structure;
    std::vector<double> values;
};

class CscBuilder {
  public:
    CscBuilder(const Index rows, const Index columns) : rows_(rows), columns_(columns) {
        if (rows < 0 || columns < 0) {
            throw std::invalid_argument("sparse-builder dimensions must be non-negative");
        }
    }

    void add(const Index row, const Index column, const double value = 0.0) {
        if (row < 0 || row >= rows_ || column < 0 || column >= columns_) {
            throw std::out_of_range("sparse-builder entry lies outside matrix dimensions");
        }
        if (!std::isfinite(value)) {
            throw std::invalid_argument("sparse-builder numerical values must be finite");
        }
        entries_.push_back(Entry{row, column, value});
    }

    [[nodiscard]] CscMatrixData build() const {
        auto entries = entries_;
        std::sort(entries.begin(), entries.end(), [](const Entry& left, const Entry& right) {
            return std::tie(left.column, left.row) < std::tie(right.column, right.row);
        });

        std::vector<Index> offsets(static_cast<std::size_t>(columns_) + 1U, 0);
        std::vector<Index> indices;
        std::vector<double> values;
        indices.reserve(entries.size());
        values.reserve(entries.size());
        std::size_t cursor = 0U;
        for (Index column = 0; column < columns_; ++column) {
            offsets[static_cast<std::size_t>(column)] = static_cast<Index>(indices.size());
            while (cursor < entries.size() && entries[cursor].column == column) {
                const Index row = entries[cursor].row;
                double value = 0.0;
                while (cursor < entries.size() && entries[cursor].column == column &&
                       entries[cursor].row == row) {
                    value += entries[cursor].value;
                    ++cursor;
                }
                indices.push_back(row);
                values.push_back(value);
            }
        }
        offsets.back() = static_cast<Index>(indices.size());
        return CscMatrixData{
            CscStructure{rows_, columns_, std::move(offsets), std::move(indices)},
            std::move(values),
        };
    }

  private:
    struct Entry {
        Index row{0};
        Index column{0};
        double value{0.0};
    };

    Index rows_{0};
    Index columns_{0};
    std::vector<Entry> entries_;
};

}  // namespace spacepdhcg::core
