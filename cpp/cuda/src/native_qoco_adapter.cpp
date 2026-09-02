#include "native_qoco_adapter.h"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct CscAbi {
    int m;
    int n;
    int nnz;
    int* i;
    int* p;
    double* x;
};

struct SettingsAbi {
    int max_iters;
    int ruiz_iters;
    int max_ir_iters;
    double ir_tol;
    double kkt_static_reg_p;
    double kkt_static_reg_a;
    double kkt_static_reg_g;
    double kkt_dynamic_reg;
    double abstol;
    double reltol;
    double abstol_inaccurate;
    double reltol_inaccurate;
    unsigned char verbose;
};

struct SolutionAbi {
    double* x;
    double* s;
    double* y;
    double* z;
    int iters;
    int ir_iters;
    double setup_time_sec;
    double solve_time_sec;
    double analysis_time_sec;
    double obj;
    double pres;
    double dres;
    double gap;
    int status;
};

struct SolverAbi {
    void* settings;
    void* work;
    void* linsys;
    void* linsys_data;
    SolutionAbi* sol;
};

struct Csc {
    int rows{};
    int columns{};
    std::vector<int> indices{};
    std::vector<int> offsets{};
    std::vector<double> values{};

    CscAbi abi() {
        return {
            rows,
            columns,
            static_cast<int>(values.size()),
            indices.data(),
            offsets.data(),
            values.data(),
        };
    }
};

struct Triplet {
    int row;
    int column;
    double value;
};

enum class Source { scalar, variable, affine, variable_cone };

struct RowMap {
    Source source;
    int index;
    int side;
    int cone_start;
    int cone_size;
    int transformed_row;
    bool rotated;
};

struct Formulation {
    Csc p{};
    Csc a{};
    Csc g{};
    std::vector<double> c{};
    std::vector<double> b{};
    std::vector<double> h{};
    std::vector<int> soc{};
    std::vector<RowMap> equality_map{};
    std::vector<RowMap> conic_map{};
    int nonnegative{};
};

using SetupFn = int (*)(
    SolverAbi*, int, int, int, CscAbi*, double*, CscAbi*, double*,
    CscAbi*, double*, int, int, int*, SettingsAbi*
);
using UpdateSettingsFn = int (*)(SolverAbi*, SettingsAbi*);
using UpdateVectorFn = void (*)(SolverAbi*, double*, double*, double*);
using UpdateMatrixFn = void (*)(SolverAbi*, double*, double*, double*);
using SetX0Fn = void (*)(SolverAbi*, double*);
using SolveFn = int (*)(SolverAbi*);
using CleanupFn = int (*)(SolverAbi*);

template <typename T>
spacepdhcg_cuda_status download(
    const spacepdhcg_accelerator_buffer_view& view,
    const std::size_t count,
    const cudaStream_t stream,
    std::vector<T>* output
) {
    output->assign(count, T{});
    if (count == 0U) {
        return SPACEPDHCG_CUDA_SUCCESS;
    }
    if (view.data == nullptr || view.elements != count
        || view.element_stride != 1) {
        return SPACEPDHCG_CUDA_POINTER_CONTRACT;
    }
    const auto* source = static_cast<const unsigned char*>(view.data)
        + view.byte_offset;
    const auto status = cudaMemcpyAsync(
        output->data(),
        source,
        count * sizeof(T),
        cudaMemcpyDeviceToHost,
        stream
    );
    if (status != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess) {
        return status == cudaErrorMemoryAllocation
            ? SPACEPDHCG_CUDA_OUT_OF_MEMORY
            : SPACEPDHCG_CUDA_RUNTIME_ERROR;
    }
    return SPACEPDHCG_CUDA_SUCCESS;
}

Csc make_csc(int rows, int columns, std::vector<Triplet> entries) {
    std::sort(
        entries.begin(),
        entries.end(),
        [](const Triplet& left, const Triplet& right) {
            return std::tie(left.column, left.row)
                < std::tie(right.column, right.row);
        }
    );
    Csc result{};
    result.rows = rows;
    result.columns = columns;
    result.offsets.assign(static_cast<std::size_t>(columns) + 1U, 0);
    for (std::size_t cursor = 0U; cursor < entries.size();) {
        const int row = entries[cursor].row;
        const int column = entries[cursor].column;
        double value = 0.0;
        do {
            value += entries[cursor].value;
            ++cursor;
        } while (cursor < entries.size()
                 && entries[cursor].row == row
                 && entries[cursor].column == column);
        result.indices.push_back(row);
        result.values.push_back(value);
        ++result.offsets[static_cast<std::size_t>(column) + 1U];
    }
    std::partial_sum(
        result.offsets.begin(),
        result.offsets.end(),
        result.offsets.begin()
    );
    return result;
}

using SparseRows = std::vector<std::vector<std::pair<int, double>>>;

SparseRows rows_from_csc(
    int rows,
    int columns,
    const std::vector<int>& offsets,
    const std::vector<int>& indices,
    const std::vector<double>& values
) {
    SparseRows result(static_cast<std::size_t>(rows));
    for (int column = 0; column < columns; ++column) {
        for (int cursor = offsets[column]; cursor < offsets[column + 1]; ++cursor) {
            result[indices[cursor]].emplace_back(column, values[cursor]);
        }
    }
    return result;
}

void append_row(
    const SparseRows& rows,
    int source,
    int target,
    double scale,
    std::vector<Triplet>* output
) {
    for (const auto& [column, value] : rows[source]) {
        output->push_back({target, column, scale * value});
    }
}

int cone_size(const spacepdhcg_cuda_cone_descriptor& cone) {
    return cone.vector_dimension + 2;
}

void append_cone_row(
    const SparseRows& rows,
    const spacepdhcg_cuda_cone_descriptor& cone,
    int output_row,
    int target,
    std::vector<Triplet>* entries
) {
    if (cone.kind == SPACEPDHCG_CUDA_CONE_SECOND_ORDER) {
        append_row(
            rows,
            output_row == 0
                ? cone.start + cone.vector_dimension + 1
                : cone.start + output_row - 1,
            target,
            -1.0,
            entries
        );
        return;
    }
    const double scale = 1.0 / std::sqrt(2.0);
    if (output_row == 0 || output_row == cone.vector_dimension + 1) {
        append_row(
            rows,
            cone.start + cone.vector_dimension,
            target,
            -scale,
            entries
        );
        append_row(
            rows,
            cone.start + cone.vector_dimension + 1,
            target,
            output_row == 0 ? -scale : scale,
            entries
        );
    } else {
        append_row(rows, cone.start + output_row - 1, target, -1.0, entries);
    }
}

double cone_offset(
    const std::vector<double>& offset,
    const spacepdhcg_cuda_cone_descriptor& cone,
    int output_row
) {
    if (cone.kind == SPACEPDHCG_CUDA_CONE_SECOND_ORDER) {
        return offset[
            output_row == 0
                ? cone.start + cone.vector_dimension + 1
                : cone.start + output_row - 1
        ];
    }
    const double scale = 1.0 / std::sqrt(2.0);
    if (output_row == 0) {
        return scale * (
            offset[cone.start + cone.vector_dimension]
            + offset[cone.start + cone.vector_dimension + 1]
        );
    }
    if (output_row == cone.vector_dimension + 1) {
        return scale * (
            offset[cone.start + cone.vector_dimension]
            - offset[cone.start + cone.vector_dimension + 1]
        );
    }
    return offset[cone.start + output_row - 1];
}

bool same_pattern(const Csc& left, const Csc& right) {
    return left.rows == right.rows && left.columns == right.columns
        && left.indices == right.indices && left.offsets == right.offsets;
}

template <typename Function>
bool symbol(void* library, const char* name, Function* output) {
    *output = reinterpret_cast<Function>(dlsym(library, name));
    return *output != nullptr;
}

void matvec(const Csc& matrix, const std::vector<double>& x, std::vector<double>* y) {
    y->assign(static_cast<std::size_t>(matrix.rows), 0.0);
    for (int column = 0; column < matrix.columns; ++column) {
        for (int cursor = matrix.offsets[column];
             cursor < matrix.offsets[column + 1];
             ++cursor) {
            (*y)[matrix.indices[cursor]] += matrix.values[cursor] * x[column];
        }
    }
}

void transpose_accumulate(const Csc& matrix, const double* x, std::vector<double>* y) {
    for (int column = 0; column < matrix.columns; ++column) {
        for (int cursor = matrix.offsets[column];
             cursor < matrix.offsets[column + 1];
             ++cursor) {
            (*y)[column] += matrix.values[cursor] * x[matrix.indices[cursor]];
        }
    }
}

double soc_violation(const double* value, int size) {
    double norm_squared = 0.0;
    for (int index = 1; index < size; ++index) {
        norm_squared += value[index] * value[index];
    }
    return std::max(0.0, std::sqrt(norm_squared) - value[0]);
}

}  // namespace

struct spacepdhcg_native_qoco {
    void* library{};
    SolverAbi* solver{};
    SetupFn setup{};
    UpdateSettingsFn update_settings{};
    UpdateVectorFn update_vector{};
    UpdateMatrixFn update_matrix{};
    SetX0Fn set_x0{};
    SolveFn solve{};
    CleanupFn cleanup{};
    Formulation formulation{};
    std::vector<double> primal{};
    std::vector<double> accepted_primal{};
    std::vector<double> dual{};
    bool has_accepted{};
    spacepdhcg_native_qoco_report report{};

    ~spacepdhcg_native_qoco() {
        if (solver != nullptr && cleanup != nullptr) {
            static_cast<void>(cleanup(solver));
            solver = nullptr;
        }
        if (library != nullptr) {
            dlclose(library);
            library = nullptr;
        }
    }
};

namespace {

spacepdhcg_cuda_status convert(
    const spacepdhcg_cuda_scvx_problem& problem,
    cudaStream_t stream,
    Formulation* output,
    std::uint64_t* copy_count,
    std::uint64_t* copy_bytes
) {
    const auto& structure = problem.canonical_structure;
    if (structure.abi_version != SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION
        || structure.topology_fingerprint != problem.topology_fingerprint
        || structure.variables <= 0) {
        return SPACEPDHCG_CUDA_TOPOLOGY_MISMATCH;
    }
    const int n = structure.variables;
    std::vector<int> q_offsets{};
    std::vector<int> q_indices{};
    std::vector<int> a_offsets{};
    std::vector<int> a_indices{};
    std::vector<int> f_offsets{};
    std::vector<int> f_indices{};
    std::vector<double> q_values{};
    std::vector<double> a_values{};
    std::vector<double> f_values{};
    std::vector<double> lower{};
    std::vector<double> upper{};
    std::vector<double> affine_offset{};
    std::vector<double> variable_lower{};
    std::vector<double> variable_upper{};

    const auto load = [&](const auto& view, std::size_t count, auto* values) {
        const auto status = download(view, count, stream, values);
        if (status == SPACEPDHCG_CUDA_SUCCESS && count != 0U) {
            ++*copy_count;
            *copy_bytes += count * sizeof(typename std::decay_t<decltype(*values)>::value_type);
        }
        return status;
    };
    spacepdhcg_cuda_status status = load(
        problem.canonical_topology.quadratic_offsets,
        static_cast<std::size_t>(n) + 1U,
        &q_offsets
    );
#define SPACEPDHCG_QOCO_LOAD(view, count, target) \
    if (status == SPACEPDHCG_CUDA_SUCCESS) { \
        status = load((view), (count), &(target)); \
    }
    SPACEPDHCG_QOCO_LOAD(
        problem.canonical_topology.quadratic_indices,
        structure.quadratic_nonzeros,
        q_indices
    )
    SPACEPDHCG_QOCO_LOAD(
        problem.canonical_topology.scalar_offsets,
        static_cast<std::size_t>(n) + 1U,
        a_offsets
    )
    SPACEPDHCG_QOCO_LOAD(
        problem.canonical_topology.scalar_indices,
        structure.scalar_nonzeros,
        a_indices
    )
    SPACEPDHCG_QOCO_LOAD(
        problem.canonical_topology.affine_offsets,
        structure.affine_rows == 0 ? 0U : static_cast<std::size_t>(n) + 1U,
        f_offsets
    )
    SPACEPDHCG_QOCO_LOAD(
        problem.canonical_topology.affine_indices,
        structure.affine_nonzeros,
        f_indices
    )
    SPACEPDHCG_QOCO_LOAD(
        problem.numeric.quadratic,
        structure.quadratic_nonzeros,
        q_values
    )
    SPACEPDHCG_QOCO_LOAD(
        problem.numeric.scalar_constraint,
        structure.scalar_nonzeros,
        a_values
    )
    SPACEPDHCG_QOCO_LOAD(
        problem.numeric.affine_cone,
        structure.affine_nonzeros,
        f_values
    )
    SPACEPDHCG_QOCO_LOAD(
        problem.numeric.linear_objective,
        static_cast<std::size_t>(n),
        output->c
    )
    SPACEPDHCG_QOCO_LOAD(
        problem.numeric.scalar_lower,
        static_cast<std::size_t>(structure.scalar_rows),
        lower
    )
    SPACEPDHCG_QOCO_LOAD(
        problem.numeric.scalar_upper,
        static_cast<std::size_t>(structure.scalar_rows),
        upper
    )
    SPACEPDHCG_QOCO_LOAD(
        problem.numeric.affine_offset,
        static_cast<std::size_t>(structure.affine_rows),
        affine_offset
    )
    SPACEPDHCG_QOCO_LOAD(
        problem.numeric.variable_lower,
        static_cast<std::size_t>(n),
        variable_lower
    )
    SPACEPDHCG_QOCO_LOAD(
        problem.numeric.variable_upper,
        static_cast<std::size_t>(n),
        variable_upper
    )
#undef SPACEPDHCG_QOCO_LOAD
    if (status != SPACEPDHCG_CUDA_SUCCESS) {
        return status;
    }

    std::vector<Triplet> p_entries{};
    std::unordered_map<std::uint64_t, double> q_entries{};
    for (int column = 0; column < n; ++column) {
        for (int cursor = q_offsets[column]; cursor < q_offsets[column + 1]; ++cursor) {
            q_entries[
                (static_cast<std::uint64_t>(
                    static_cast<std::uint32_t>(q_indices[cursor])
                ) << 32U)
                | static_cast<std::uint32_t>(column)
            ] = q_values[cursor];
            if (q_indices[cursor] <= column) {
                p_entries.push_back({q_indices[cursor], column, q_values[cursor]});
            }
        }
    }
    for (const auto& [key, value] : q_entries) {
        const auto row = static_cast<std::uint32_t>(key >> 32U);
        const auto column = static_cast<std::uint32_t>(key);
        const auto reverse = q_entries.find(
            (static_cast<std::uint64_t>(column) << 32U) | row
        );
        if (reverse == q_entries.end()
            || std::abs(reverse->second - value)
                > 1.0e-12 * std::max({
                    1.0,
                    std::abs(value),
                    std::abs(reverse->second),
                })) {
            return SPACEPDHCG_CUDA_UNSUPPORTED;
        }
    }
    output->p = make_csc(n, n, std::move(p_entries));
    const auto scalar_rows = rows_from_csc(
        structure.scalar_rows, n, a_offsets, a_indices, a_values
    );
    const auto affine_rows = structure.affine_rows == 0
        ? SparseRows{}
        : rows_from_csc(
            structure.affine_rows, n, f_offsets, f_indices, f_values
        );
    std::vector<Triplet> equality_entries{};
    std::vector<Triplet> conic_entries{};
    output->b.clear();
    output->h.clear();
    output->equality_map.clear();
    output->conic_map.clear();
    const auto append_bound = [&](
        Source source,
        int index,
        const SparseRows* rows,
        double lo,
        double hi
    ) {
        const auto add = [&](std::vector<Triplet>* destination, int target, double scale) {
            if (rows == nullptr) {
                destination->push_back({target, index, scale});
            } else {
                append_row(*rows, index, target, scale, destination);
            }
        };
        if (std::isfinite(lo) && std::isfinite(hi) && lo == hi) {
            add(&equality_entries, static_cast<int>(output->b.size()), 1.0);
            output->b.push_back(lo);
            output->equality_map.push_back({source, index, 0, -1, 0, 0, false});
        } else {
            if (std::isfinite(hi)) {
                add(&conic_entries, static_cast<int>(output->h.size()), 1.0);
                output->h.push_back(hi);
                output->conic_map.push_back({source, index, 1, -1, 0, 0, false});
            }
            if (std::isfinite(lo)) {
                add(&conic_entries, static_cast<int>(output->h.size()), -1.0);
                output->h.push_back(-lo);
                output->conic_map.push_back({source, index, -1, -1, 0, 0, false});
            }
        }
    };
    for (int row = 0; row < structure.scalar_rows; ++row) {
        append_bound(Source::scalar, row, &scalar_rows, lower[row], upper[row]);
    }
    for (int variable = 0; variable < n; ++variable) {
        append_bound(
            Source::variable,
            variable,
            nullptr,
            variable_lower[variable],
            variable_upper[variable]
        );
    }
    output->nonnegative = static_cast<int>(output->h.size());
    output->soc.clear();
    const auto append_cones = [&](
        const spacepdhcg_cuda_cone_descriptor* cones,
        std::size_t count,
        bool affine_source
    ) {
        for (std::size_t index = 0U; index < count; ++index) {
            const auto& cone = cones[index];
            if (cone.kind != SPACEPDHCG_CUDA_CONE_SECOND_ORDER
                && cone.kind != SPACEPDHCG_CUDA_CONE_ROTATED_SECOND_ORDER) {
                return false;
            }
            const int size = cone_size(cone);
            output->soc.push_back(size);
            SparseRows identity{};
            if (!affine_source) {
                identity.resize(static_cast<std::size_t>(n));
                for (int slot = cone.start; slot < cone.start + size; ++slot) {
                    identity[slot].emplace_back(slot, 1.0);
                }
            }
            const auto& source_rows = affine_source ? affine_rows : identity;
            for (int row = 0; row < size; ++row) {
                append_cone_row(
                    source_rows,
                    cone,
                    row,
                    static_cast<int>(output->h.size()),
                    &conic_entries
                );
                output->h.push_back(
                    affine_source ? cone_offset(affine_offset, cone, row) : 0.0
                );
                output->conic_map.push_back({
                    affine_source ? Source::affine : Source::variable_cone,
                    cone.start,
                    0,
                    cone.start,
                    size,
                    row,
                    cone.kind == SPACEPDHCG_CUDA_CONE_ROTATED_SECOND_ORDER,
                });
            }
        }
        return true;
    };
    if (!append_cones(
            structure.affine_cones,
            structure.affine_cone_count,
            true
        )
        || !append_cones(
            structure.variable_cones,
            structure.variable_cone_count,
            false
        )) {
        return SPACEPDHCG_CUDA_UNSUPPORTED;
    }
    output->a = make_csc(
        static_cast<int>(output->b.size()), n, std::move(equality_entries)
    );
    output->g = make_csc(
        static_cast<int>(output->h.size()), n, std::move(conic_entries)
    );
    return SPACEPDHCG_CUDA_SUCCESS;
}

void residuals(spacepdhcg_native_qoco* workspace) {
    const auto& formulation = workspace->formulation;
    std::vector<double> ax{};
    std::vector<double> gx{};
    matvec(formulation.a, workspace->primal, &ax);
    matvec(formulation.g, workspace->primal, &gx);
    double primal = 0.0;
    for (std::size_t index = 0U; index < ax.size(); ++index) {
        primal = std::max(primal, std::abs(ax[index] - formulation.b[index]));
    }
    std::vector<double> slack(formulation.h.size(), 0.0);
    for (std::size_t index = 0U; index < slack.size(); ++index) {
        slack[index] = formulation.h[index] - gx[index];
    }
    for (int index = 0; index < formulation.nonnegative; ++index) {
        primal = std::max(primal, std::max(0.0, -slack[index]));
    }
    int cursor = formulation.nonnegative;
    for (int size : formulation.soc) {
        primal = std::max(primal, soc_violation(slack.data() + cursor, size));
        cursor += size;
    }
    std::vector<double> stationarity = formulation.c;
    for (int column = 0; column < formulation.p.columns; ++column) {
        for (int entry = formulation.p.offsets[column];
             entry < formulation.p.offsets[column + 1];
             ++entry) {
            const int row = formulation.p.indices[entry];
            stationarity[row] += formulation.p.values[entry] * workspace->primal[column];
            if (row != column) {
                stationarity[column] +=
                    formulation.p.values[entry] * workspace->primal[row];
            }
        }
    }
    transpose_accumulate(formulation.a, workspace->solver->sol->y, &stationarity);
    transpose_accumulate(formulation.g, workspace->solver->sol->z, &stationarity);
    double dual = 0.0;
    for (double value : stationarity) {
        dual = std::max(dual, std::abs(value));
    }
    workspace->report.primal_residual = primal;
    workspace->report.dual_residual = dual;
}

void map_dual(
    spacepdhcg_native_qoco* workspace,
    const spacepdhcg_cuda_structure& structure
) {
    std::fill(workspace->dual.begin(), workspace->dual.end(), 0.0);
    for (std::size_t index = 0U;
         index < workspace->formulation.equality_map.size();
         ++index) {
        const auto& row = workspace->formulation.equality_map[index];
        if (row.source == Source::scalar) {
            workspace->dual[row.index] += workspace->solver->sol->y[index];
        }
    }
    for (std::size_t index = 0U;
         index < workspace->formulation.conic_map.size();
         ++index) {
        const auto& row = workspace->formulation.conic_map[index];
        const double value = workspace->solver->sol->z[index];
        if (row.source == Source::scalar) {
            workspace->dual[row.index] += row.side * value;
        } else if (row.source == Source::affine) {
            const int base = structure.scalar_rows + row.cone_start;
            if (!row.rotated) {
                workspace->dual[
                    base + (row.transformed_row == 0
                        ? row.cone_size - 1
                        : row.transformed_row - 1)
                ] += value;
            } else if (row.transformed_row == 0) {
                const double scaled = value / std::sqrt(2.0);
                workspace->dual[base + row.cone_size - 2] += scaled;
                workspace->dual[base + row.cone_size - 1] += scaled;
            } else if (row.transformed_row == row.cone_size - 1) {
                const double scaled = value / std::sqrt(2.0);
                workspace->dual[base + row.cone_size - 2] += scaled;
                workspace->dual[base + row.cone_size - 1] -= scaled;
            } else {
                workspace->dual[base + row.transformed_row - 1] += value;
            }
        }
    }
}

}  // namespace

spacepdhcg_cuda_status native_qoco_create_impl(
    const spacepdhcg_cuda_scvx_problem* problem,
    cudaStream_t stream,
    spacepdhcg_native_qoco** workspace
) {
    if (problem == nullptr || workspace == nullptr) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    *workspace = nullptr;
    const char* path = std::getenv("SPACEPDHCG_QOCO_LIBRARY");
    if (path == nullptr || path[0] == '\0') {
        return SPACEPDHCG_CUDA_UNSUPPORTED;
    }
    std::unique_ptr<spacepdhcg_native_qoco> result{
        new (std::nothrow) spacepdhcg_native_qoco{}
    };
    if (result == nullptr) {
        return SPACEPDHCG_CUDA_OUT_OF_MEMORY;
    }
    result->library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (result->library == nullptr
        || !symbol(result->library, "qoco_setup", &result->setup)
        || !symbol(result->library, "qoco_update_settings", &result->update_settings)
        || !symbol(result->library, "qoco_update_vector_data", &result->update_vector)
        || !symbol(result->library, "qoco_update_matrix_data", &result->update_matrix)
        || !symbol(result->library, "qoco_set_x0", &result->set_x0)
        || !symbol(result->library, "qoco_solve", &result->solve)
        || !symbol(result->library, "qoco_cleanup", &result->cleanup)) {
        return SPACEPDHCG_CUDA_UNSUPPORTED;
    }
    const auto conversion_start = std::chrono::steady_clock::now();
    auto status = convert(
        *problem,
        stream,
        &result->formulation,
        &result->report.d2h_copy_count,
        &result->report.d2h_bytes
    );
    result->report.conversion_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - conversion_start
    ).count();
    if (status != SPACEPDHCG_CUDA_SUCCESS) {
        return status;
    }
    const std::size_t variables =
        static_cast<std::size_t>(problem->canonical_structure.variables);
    const std::size_t duals =
        static_cast<std::size_t>(
            problem->canonical_structure.scalar_rows
            + problem->canonical_structure.affine_rows
        );
    result->primal.assign(variables, 0.0);
    result->accepted_primal.assign(variables, 0.0);
    result->dual.assign(duals, 0.0);
    result->solver = static_cast<SolverAbi*>(std::calloc(1U, sizeof(SolverAbi)));
    if (result->solver == nullptr) {
        return SPACEPDHCG_CUDA_OUT_OF_MEMORY;
    }
    auto p = result->formulation.p.abi();
    auto a = result->formulation.a.abi();
    auto g = result->formulation.g.abi();
    const bool low_thrust =
        problem->dynamics.model == SPACEPDHCG_CUDA_DYNAMICS_LOW_THRUST;
    // Low-thrust mass-flow equalities are much smaller than the virtual
    // penalty scale, so they require less KKT bias and tighter refinement.
    SettingsAbi settings{
        200, 0, low_thrust ? 20 : 5, low_thrust ? 1.0e-12 : 1.0e-6,
        1.0e-13, low_thrust ? 1.0e-13 : 1.0e-8, 1.0e-13,
        low_thrust ? 1.0e-13 : 1.0e-11,
        1.0e-8, 1.0e-8, 1.0e-5, 1.0e-5, 0,
    };
    const auto setup_start = std::chrono::steady_clock::now();
    const int code = result->setup(
        result->solver,
        problem->canonical_structure.variables,
        static_cast<int>(result->formulation.h.size()),
        static_cast<int>(result->formulation.b.size()),
        &p,
        result->formulation.c.data(),
        result->formulation.b.empty() ? nullptr : &a,
        result->formulation.b.empty() ? nullptr : result->formulation.b.data(),
        result->formulation.h.empty() ? nullptr : &g,
        result->formulation.h.empty() ? nullptr : result->formulation.h.data(),
        result->formulation.nonnegative,
        static_cast<int>(result->formulation.soc.size()),
        result->formulation.soc.empty() ? nullptr : result->formulation.soc.data(),
        &settings
    );
    result->report.setup_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - setup_start
    ).count();
    if (code != 0) {
        std::free(result->solver);
        result->solver = nullptr;
        return code == 5 ? SPACEPDHCG_CUDA_OUT_OF_MEMORY
                         : SPACEPDHCG_CUDA_NUMERICAL_FAILURE;
    }
    result->report.workspace_creations = 1U;
    *workspace = result.release();
    return SPACEPDHCG_CUDA_SUCCESS;
}

spacepdhcg_cuda_status native_qoco_update_solve_impl(
    spacepdhcg_native_qoco* workspace,
    const spacepdhcg_cuda_scvx_problem* problem,
    cudaStream_t stream,
    spacepdhcg_cuda_warm_start_mode requested_warm,
    double* device_primal,
    double* device_dual,
    spacepdhcg_native_qoco_report* report
) {
    if (workspace == nullptr || problem == nullptr || device_primal == nullptr
        || device_dual == nullptr || report == nullptr) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    const auto finish = [&](spacepdhcg_cuda_status status) {
        *report = workspace->report;
        return status;
    };
    if (workspace->report.solves != 0U) {
        Formulation updated{};
        const auto update_start = std::chrono::steady_clock::now();
        auto status = convert(
            *problem,
            stream,
            &updated,
            &workspace->report.d2h_copy_count,
            &workspace->report.d2h_bytes
        );
        if (status != SPACEPDHCG_CUDA_SUCCESS) {
            workspace->report.failure =
                status == SPACEPDHCG_CUDA_UNSUPPORTED
                ? SPACEPDHCG_CUDA_QOCO_FAILURE_UNSUPPORTED
                : SPACEPDHCG_CUDA_QOCO_FAILURE_ABI;
            return finish(status);
        }
        if (!same_pattern(workspace->formulation.p, updated.p)
            || !same_pattern(workspace->formulation.a, updated.a)
            || !same_pattern(workspace->formulation.g, updated.g)
            || workspace->formulation.soc != updated.soc
            || workspace->formulation.nonnegative != updated.nonnegative) {
            workspace->report.failure = SPACEPDHCG_CUDA_QOCO_FAILURE_UNSUPPORTED;
            return finish(SPACEPDHCG_CUDA_TOPOLOGY_MISMATCH);
        }
        workspace->formulation = std::move(updated);
        workspace->update_matrix(
            workspace->solver,
            workspace->formulation.p.values.data(),
            workspace->formulation.a.values.empty()
                ? nullptr : workspace->formulation.a.values.data(),
            workspace->formulation.g.values.empty()
                ? nullptr : workspace->formulation.g.values.data()
        );
        workspace->update_vector(
            workspace->solver,
            workspace->formulation.c.data(),
            workspace->formulation.b.empty()
                ? nullptr : workspace->formulation.b.data(),
            workspace->formulation.h.empty()
                ? nullptr : workspace->formulation.h.data()
        );
        workspace->report.update_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - update_start
        ).count();
        ++workspace->report.numeric_updates;
    }
    const bool warm = workspace->has_accepted
        && requested_warm != SPACEPDHCG_CUDA_WARM_START_NONE;
    workspace->set_x0(
        workspace->solver,
        warm ? workspace->accepted_primal.data() : nullptr
    );
    workspace->report.warm_primal_accepted = warm ? 1 : 0;
    workspace->report.dual_discarded =
        warm
        && (requested_warm == SPACEPDHCG_CUDA_WARM_START_PRIMAL_DUAL
            || requested_warm == SPACEPDHCG_CUDA_WARM_START_FULL_RETAINED)
        ? 1 : 0;
    const auto solve_start = std::chrono::steady_clock::now();
    const int status_code = workspace->solve(workspace->solver);
    workspace->report.solve_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - solve_start
    ).count();
    ++workspace->report.solves;
    if (workspace->solver->sol == nullptr
        || workspace->solver->sol->status != status_code) {
        workspace->report.failure = SPACEPDHCG_CUDA_QOCO_FAILURE_ABI;
        return finish(SPACEPDHCG_CUDA_INTERNAL_ERROR);
    }
    workspace->report.iterations = workspace->solver->sol->iters;
    if (status_code != 1 && status_code != 2) {
        workspace->report.failure = status_code == 4
            ? SPACEPDHCG_CUDA_QOCO_FAILURE_MAX_ITERATIONS
            : SPACEPDHCG_CUDA_QOCO_FAILURE_NUMERICAL;
        return finish(SPACEPDHCG_CUDA_NUMERICAL_FAILURE);
    }
    std::copy_n(
        workspace->solver->sol->x,
        workspace->primal.size(),
        workspace->primal.begin()
    );
    residuals(workspace);
    map_dual(workspace, problem->canonical_structure);
    auto cuda_status = cudaMemcpyAsync(
        device_primal,
        workspace->primal.data(),
        workspace->primal.size() * sizeof(double),
        cudaMemcpyHostToDevice,
        stream
    );
    if (cuda_status == cudaSuccess) {
        cuda_status = cudaMemcpyAsync(
            device_dual,
            workspace->dual.data(),
            workspace->dual.size() * sizeof(double),
            cudaMemcpyHostToDevice,
            stream
        );
    }
    if (cuda_status != cudaSuccess) {
        workspace->report.failure =
            cuda_status == cudaErrorMemoryAllocation
            ? SPACEPDHCG_CUDA_QOCO_FAILURE_OUT_OF_MEMORY
            : SPACEPDHCG_CUDA_QOCO_FAILURE_ABI;
        return finish(
            cuda_status == cudaErrorMemoryAllocation
                ? SPACEPDHCG_CUDA_OUT_OF_MEMORY
                : SPACEPDHCG_CUDA_RUNTIME_ERROR
        );
    }
    workspace->report.failure = SPACEPDHCG_CUDA_QOCO_FAILURE_NONE;
    return finish(SPACEPDHCG_CUDA_SUCCESS);
}

spacepdhcg_cuda_status spacepdhcg_native_qoco_create(
    const spacepdhcg_cuda_scvx_problem* problem,
    cudaStream_t stream,
    spacepdhcg_native_qoco** workspace
) {
    try {
        return native_qoco_create_impl(problem, stream, workspace);
    } catch (const std::bad_alloc&) {
        return SPACEPDHCG_CUDA_OUT_OF_MEMORY;
    } catch (...) {
        return SPACEPDHCG_CUDA_INTERNAL_ERROR;
    }
}

spacepdhcg_cuda_status spacepdhcg_native_qoco_update_solve(
    spacepdhcg_native_qoco* workspace,
    const spacepdhcg_cuda_scvx_problem* problem,
    cudaStream_t stream,
    spacepdhcg_cuda_warm_start_mode requested_warm,
    double* device_primal,
    double* device_dual,
    spacepdhcg_native_qoco_report* report
) {
    try {
        return native_qoco_update_solve_impl(
            workspace,
            problem,
            stream,
            requested_warm,
            device_primal,
            device_dual,
            report
        );
    } catch (const std::bad_alloc&) {
        if (report != nullptr) {
            report->failure = SPACEPDHCG_CUDA_QOCO_FAILURE_OUT_OF_MEMORY;
        }
        return SPACEPDHCG_CUDA_OUT_OF_MEMORY;
    } catch (...) {
        if (report != nullptr) {
            report->failure = SPACEPDHCG_CUDA_QOCO_FAILURE_ABI;
        }
        return SPACEPDHCG_CUDA_INTERNAL_ERROR;
    }
}

void spacepdhcg_native_qoco_accept(spacepdhcg_native_qoco* workspace) {
    if (workspace != nullptr) {
        workspace->accepted_primal = workspace->primal;
        workspace->has_accepted = true;
    }
}

void spacepdhcg_native_qoco_reset_warm_state(
    spacepdhcg_native_qoco* workspace,
    const bool retain_primal
) {
    if (workspace == nullptr) {
        return;
    }
    if (!retain_primal) {
        std::fill(
            workspace->accepted_primal.begin(),
            workspace->accepted_primal.end(),
            0.0
        );
        workspace->has_accepted = false;
    }
    workspace->report.warm_primal_accepted = 0;
    workspace->report.dual_discarded = 0;
    if (workspace->solver != nullptr && workspace->set_x0 != nullptr) {
        workspace->set_x0(
            workspace->solver,
            retain_primal && workspace->has_accepted
                ? workspace->accepted_primal.data()
                : nullptr
        );
    }
}

void spacepdhcg_native_qoco_destroy(spacepdhcg_native_qoco* workspace) {
    delete workspace;
}
