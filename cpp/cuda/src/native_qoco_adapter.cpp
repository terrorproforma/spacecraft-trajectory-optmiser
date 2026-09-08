#include "native_qoco_adapter.h"
#include "native_qoco_gpu.h"
#include "graph_append.h"

#include <thread>
#include <atomic>
#include <unistd.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

// Optional prepared-QOCO replay ABI v1; kept independent of vendor headers.
struct CompletionAbi {
    int abi_version, status, iterations, ir_iterations, step_ir_iterations, restored;
    double primal_residual, dual_residual, gap, objective, dynamic_reg;
};
struct ReplayOutputAbi {
    const CompletionAbi* completion;
    const double *x, *y, *s, *z;
    int n, p, m;
};
static_assert(sizeof(CompletionAbi)==64 && sizeof(ReplayOutputAbi)==56);
using ReplayFn = int (*)(SolverAbi*, void*, ReplayOutputAbi*);
using ReplayUpdatedFn = int (*)(SolverAbi*, void*, const double*, ReplayOutputAbi*);
using FinishReplayFn = int (*)(SolverAbi*);
using EmitGraphFn = int (*)(SolverAbi*,cudaGraph_t,const cudaGraphNode_t*,size_t,
    const double*,ReplayOutputAbi*,cudaGraphNode_t*);

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
    bool operator==(const RowMap&) const = default;
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

struct TopologyCache {
    std::vector<int> arrays[6];
    std::uint64_t fingerprint{};
    QocoGpuTopology* device{};
    ~TopologyCache() { qoco_gpu_topology_destroy(device); }
};

struct ConversionCache {
    QocoGpuConversion* device{};
    int input_counts[9]{};
    std::vector<double> values;
    std::vector<spacepdhcg_cuda_cone_descriptor> affine_cones, variable_cones;
    ~ConversionCache() { qoco_gpu_conversion_destroy(device); }
};

// Complete queued downloads before their destination vectors can be destroyed,
// including allocation/validation failures while assembling a batch.
struct DownloadBatch {
    cudaStream_t stream;
    bool pending{true};
    cudaError_t finish() { pending = false; return cudaStreamSynchronize(stream); }
    ~DownloadBatch() { if (pending) cudaStreamSynchronize(stream); }
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
using BeginReductionScopeFn = int (*)();
using EndReductionScopeFn = void (*)();
using DeviceSolutionFn = int (*)(SolverAbi*, int, int, int, const double**, const double**, const double**);
using DeviceIoFn = int (*)(SolverAbi*, int);
using StoppingOriginFn = int (*)(SolverAbi*, const double*, const double*);
using DownloadSolutionFn = int (*)(SolverAbi*);
using CreateNumericUpdateFn = int (*)(SolverAbi*, int, int, int, void**);
using DeviceNumericUpdateFn = int (*)(void*, const double*, cudaStream_t);
using QueuedNumericUpdateFn = int (*)(void*, const double*, cudaStream_t, const double**);
using FinishNumericUpdateFn = int (*)(void*, int);
using NumericPreservesObjectiveFn = int (*)(const void*);
using DestroyNumericUpdateFn = void (*)(void*);
using SetTrajectoryFn = int (*)(int, int, int, const int*, const int*, const int*, cudaStream_t);

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
    if (status != cudaSuccess) {
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
    BeginReductionScopeFn begin_reduction_scope{};
    EndReductionScopeFn end_reduction_scope{};
    ReplayFn replay{};
    ReplayUpdatedFn replay_updated{};
    FinishReplayFn finish_replay{};
    EmitGraphFn emit_graph{};
    cudaEvent_t replay_events[3]{};
    bool deferred_active{};
    bool deferred_ready{};
    bool numeric_scale_primed{};
    bool retained_replay_ready{};
    bool origin_mode{}, origin_applied{};
    cudaStream_t deferred_stream{};
    const QocoReplayStatus* deferred_status{};
    int deferred_device{};
    std::thread::id deferred_owner{};
    DeviceSolutionFn device_solution{};
    DeviceIoFn set_device_io{};
    StoppingOriginFn set_stopping_origin{};
    DeviceIoFn primal_start{};
    DownloadSolutionFn download_solution{};
    CreateNumericUpdateFn create_numeric_update{};
    DeviceNumericUpdateFn device_numeric_update{};
    QueuedNumericUpdateFn queued_numeric_update{};
    QueuedNumericUpdateFn capture_numeric_update{};
    FinishNumericUpdateFn finish_numeric_update{};
    NumericPreservesObjectiveFn numeric_preserves_objective{};
    const double* queued_numeric_result{};
    bool numeric_update_invalid{};
    bool queue_validation_allowed{}, validation_pending{};
    const int* device_validation{};
    int validation_flags{};
    DestroyNumericUpdateFn destroy_numeric_update{};
    SetTrajectoryFn set_trajectory{};
    int trajectory_intervals{}, trajectory_nx{}, trajectory_nu{};
    const int *trajectory_states{}, *trajectory_controls{}, *trajectory_virtual{};
    int* trajectory_indices{};
    std::size_t trajectory_bytes{};
    cudaStream_t trajectory_stream{};
    void* numeric_update_context{};
    QocoGpuAudit* gpu_audit{};
    TopologyCache topology{};
    ConversionCache conversion{};
    Csc dual_transfer{};
    Formulation formulation{};
    // The settings handed to qoco_setup. QOCO's stall handler mutates
    // solver->settings->kkt_dynamic_reg in place (x10 per stalled step) and never
    // restores it, so after a "numerical error" exit every later solve on the
    // persistent workspace would start above the 1e-6 ceiling and abort at
    // iteration 1. Every re-solve restores these first.
    SettingsAbi configured_settings{};
    int variables{};
    // The last solve ended in "numerical error" / "maximum iterations". QOCO
    // keeps its best-iterate tracker (best_valid/best_metric) and the escalated
    // regularisation across solves, so the next solve on this workspace would
    // not be an independent attempt. The next update_solve tears the solver
    // down and sets it up again (counted in report.workspace_creations).
    bool needs_fresh_solver{};
    bool refresh_new_leg{};
    std::vector<double> primal{};
    std::vector<double> accepted_primal{};
    std::vector<double> dual{};
    bool has_accepted{};
    bool graph_active{}, destroy_after_graph{};
    cudaStream_t graph_stream{};
    int creation_device=-1;
    std::thread::id creation_owner=std::this_thread::get_id();
    QocoGraphProgress* graph_progress{};
    spacepdhcg_native_qoco_report report{};

    ~spacepdhcg_native_qoco() {
        // Consumers borrow audit/conversion storage. Drain before freeing any
        // of it, including when a submitted solve is abandoned by the caller.
        if (deferred_active) {
            cudaStreamSynchronize(deferred_stream);
            finish_replay(solver);
            finish_numeric_update(numeric_update_context,0);
        }
        for (auto event : replay_events) if (event) cudaEventDestroy(event);
        if (graph_progress) cudaFree(graph_progress);
        if (numeric_update_context) destroy_numeric_update(numeric_update_context);
        if (solver != nullptr && cleanup != nullptr) {
            static_cast<void>(cleanup(solver));
            solver = nullptr;
        }
        // Vendor stopping graphs borrow the audit's origin and offset buffers.
        qoco_gpu_audit_destroy(gpu_audit);
        if (trajectory_indices) cudaFree(trajectory_indices);
        if (library != nullptr) {
            dlclose(library);
            library = nullptr;
        }
    }
};

namespace {

QocoAuditCsc audit_view(const Csc& matrix) {
    return {matrix.rows, matrix.columns, static_cast<int>(matrix.values.size()),
        matrix.offsets.data(), matrix.indices.data(), matrix.values.data()};
}
QocoAuditInput audit_input(const spacepdhcg_native_qoco* w) {
    const auto& f = w->formulation;
    return {audit_view(f.p), audit_view(f.a), audit_view(f.g), audit_view(w->dual_transfer),
        f.c.data(), f.b.data(), f.h.data(), f.nonnegative, static_cast<int>(f.soc.size()), f.soc.data()};
}

Csc make_dual_transfer(const Formulation& f, const spacepdhcg_cuda_structure& structure) {
    std::vector<Triplet> entries;
    for (int i = 0; i < static_cast<int>(f.equality_map.size()); ++i) {
        const auto& row = f.equality_map[i];
        if (row.source == Source::scalar) entries.push_back({row.index, i, 1.0});
    }
    for (int i = 0; i < static_cast<int>(f.conic_map.size()); ++i) {
        const auto& row = f.conic_map[i];
        const int col = static_cast<int>(f.b.size()) + i;
        if (row.source == Source::scalar) entries.push_back({row.index, col, static_cast<double>(row.side)});
        else if (row.source == Source::affine) {
            const int base = structure.scalar_rows + row.cone_start;
            if (!row.rotated) entries.push_back({base + (row.transformed_row == 0 ? row.cone_size - 1 : row.transformed_row - 1), col, 1.0});
            else if (row.transformed_row == 0 || row.transformed_row == row.cone_size - 1) {
                entries.push_back({base + row.cone_size - 2, col, 1.0 / std::sqrt(2.0)});
                entries.push_back({base + row.cone_size - 1, col, (row.transformed_row == 0 ? 1.0 : -1.0) / std::sqrt(2.0)});
            } else entries.push_back({base + row.transformed_row - 1, col, 1.0});
        }
    }
    return make_csc(structure.scalar_rows + structure.affine_rows,
        static_cast<int>(f.b.size() + f.h.size()), std::move(entries));
}

bool solve_with_reduction_scope(spacepdhcg_native_qoco* workspace, int* status) {
    if (workspace->begin_reduction_scope != nullptr
        && workspace->begin_reduction_scope() != 0) return false;
    struct Scope {
        EndReductionScopeFn end;
        ~Scope() { if (end != nullptr) end(); }
    } scope{workspace->end_reduction_scope};
    *status = workspace->solve(workspace->solver);
    return true;
}

spacepdhcg_cuda_status canonical_validation_status(int flags) {
    if (flags & 16) return SPACEPDHCG_CUDA_NUMERICAL_FAILURE;
    if (flags & 9) return SPACEPDHCG_CUDA_TOPOLOGY_MISMATCH;
    if (flags & 2) return SPACEPDHCG_CUDA_NUMERICAL_FAILURE;
    if (flags & 4) return SPACEPDHCG_CUDA_UNSUPPORTED;
    return SPACEPDHCG_CUDA_SUCCESS;
}
bool collect_validation(spacepdhcg_native_qoco* w,cudaStream_t stream) {
    if (!w->device_validation) return true;
    const auto copied=qoco_gpu_conversion_download_flags_async(w->conversion.device,stream,&w->validation_flags);
    const auto waited=cudaStreamSynchronize(stream);
    w->validation_pending=false; w->device_validation=nullptr;
    return copied==cudaSuccess && waited==cudaSuccess;
}
bool collect_device_replay(spacepdhcg_native_qoco* w, cudaStream_t stream,
    const QocoReplayStatus* device_status, int* status, QocoAuditResult* audit) {
    QocoReplayStatus completion{};
    struct DrainDownloads {
        cudaStream_t stream;
        bool finished=false;
        ~DrainDownloads() { if (!finished) cudaStreamSynchronize(stream); }
    } downloads{stream};
    const bool updated=w->queued_numeric_result!=nullptr;
    if (qoco_gpu_audit_download_async(w->gpu_audit,stream,audit)!=cudaSuccess ||
        cudaMemcpyAsync(&completion,device_status,sizeof(completion),cudaMemcpyDeviceToHost,stream)!=cudaSuccess)
        return false;
    ++w->report.d2h_copy_count; w->report.d2h_bytes+=sizeof(completion);
    if (w->device_validation && qoco_gpu_conversion_download_flags_async(
        w->conversion.device,stream,&w->validation_flags)!=cudaSuccess) return false;
    if (w->finish_replay(w->solver)!=0) return false;
    downloads.finished=true;

    const bool validated=w->device_validation!=nullptr;
    w->validation_pending=false; w->device_validation=nullptr;
    if (w->queued_numeric_result) {
        if (w->finish_numeric_update(w->numeric_update_context,0)!=0) return false;
        w->queued_numeric_result=nullptr;
    }
    if (completion.status<0) return false;
    float solve_ms{},audit_ms{};
    if (cudaEventElapsedTime(&solve_ms,w->replay_events[0],w->replay_events[1])!=cudaSuccess ||
        cudaEventElapsedTime(&audit_ms,w->replay_events[1],w->replay_events[2])!=cudaSuccess) return false;
    w->report.solve_seconds+=solve_ms*.001;
    w->report.residual_seconds+=audit_ms*.001;
    // Materialise only legacy fields consumed by the adapter's status handling
    // and explicit accepted-primal API. Vectors themselves remain on the GPU.
    auto* sol=w->solver->sol;
    sol->status=completion.status; sol->iters=completion.iterations;
    *status=completion.status;
    if (std::getenv("SPACEPDHCG_TEST_QOCO_NATIVE_REPLAY_TRACE"))
        std::fprintf(stderr,"NATIVE_REPLAY status=%d ipm=%d audit_primal=%.17g audit_dual=%.17g\n",
            *status,sol->iters,audit->primal,audit->dual);
    if (updated && std::getenv("SPACEPDHCG_TEST_QOCO_NATIVE_NUMERIC_REPLAY_TRACE"))
        std::fprintf(stderr,"NATIVE_NUMERIC_REPLAY status=%d ipm=%d\n",*status,sol->iters);
    if (validated && std::getenv("SPACEPDHCG_TEST_QOCO_DEVICE_VALIDATION_TRACE"))
        std::fprintf(stderr,"DEVICE_VALIDATION flags=%d status=%d\n",w->validation_flags,*status);
    return true;
}

bool solve_with_device_audit(spacepdhcg_native_qoco* w, cudaStream_t stream,
    double* primal, double* dual, int* status, QocoAuditResult* audit, bool* replayed,
    spacepdhcg_native_qoco_consumer consumer, void* context, bool defer=false) {
    *replayed=false;
    const auto prime=[&]() {
        if (defer) return false;
        if (!collect_validation(w,stream) || w->validation_flags) return false;
        if (w->queued_numeric_result) {
            const int code=w->finish_numeric_update(w->numeric_update_context,1);
            w->queued_numeric_result=nullptr;
            if (code) { w->numeric_update_invalid=code==3; return false; }
        }
        return solve_with_reduction_scope(w,status);
    };
    const char* enabled=std::getenv("SPACEPDHCG_TEST_QOCO_NATIVE_REPLAY");
    if (!enabled || enabled[0]!='1') return prime();
    if (!w->replay || !w->finish_replay || !w->set_device_io) return false;
    // Per-leg counters reset when a compatible workspace is retained. Its
    // already-built vendor graph can consume the newly queued numeric packet.
    // A genuinely cold/rebuilt workspace still needs synchronous priming.
    const bool retained=w->retained_replay_ready;
    w->retained_replay_ready=false;
    if (!w->report.solves && !retained) return prime();
    for (auto& event : w->replay_events)
        if (!event && cudaEventCreate(&event)!=cudaSuccess) return false;
    if (cudaEventRecord(w->replay_events[0],stream)!=cudaSuccess) return false;
    ReplayOutputAbi output{};
    const bool updated=w->queued_numeric_result!=nullptr;
    const double* numeric=w->queued_numeric_result;
    if (updated && w->device_validation && qoco_gpu_conversion_guard_numeric(
        w->conversion.device,numeric,stream,&numeric)!=cudaSuccess) return false;
    const int submitted=updated
        ? w->replay_updated(w->solver,stream,numeric,&output)
        : w->replay(w->solver,stream,&output);
    if(retained && std::getenv("SPACEPDHCG_TEST_QOCO_NATIVE_REPLAY_TRACE"))
        std::fprintf(stderr,"RETAINED_REPLAY submitted=%d updated=%d\n",submitted,int(updated));
    if (submitted==2) return prime();
    struct Pending {
        spacepdhcg_native_qoco* w;
        bool finished=false;
        ~Pending() { if (!finished) w->finish_replay(w->solver); }
    } pending{w};
    if (submitted || !output.completion || output.n!=w->variables ||
        output.p!=static_cast<int>(w->formulation.b.size()) ||
        output.m!=static_cast<int>(w->formulation.h.size())) return false;
    if (cudaEventRecord(w->replay_events[1],stream)!=cudaSuccess) return false;
    const QocoAuditResult* device_audit{};
    const QocoReplayStatus* device_status{};
    if (w->origin_mode && qoco_gpu_audit_reconstruct(w->gpu_audit,output.x,primal,stream)!=cudaSuccess) return false;
    if (qoco_gpu_audit_run_device(w->gpu_audit,w->origin_mode ? primal : output.x,output.y,output.z,dual,stream,&device_audit)!=cudaSuccess ||
        qoco_gpu_audit_replay_status(w->gpu_audit,reinterpret_cast<const int*>(output.completion),stream,&device_status)!=cudaSuccess ||
        (!w->origin_mode && cudaMemcpyAsync(primal,output.x,w->variables*sizeof(double),cudaMemcpyDeviceToDevice,stream)!=cudaSuccess) ||
        cudaEventRecord(w->replay_events[2],stream)!=cudaSuccess) return false;
    if (!w->origin_mode) { ++w->report.d2d_copy_count; w->report.d2d_bytes+=w->variables*sizeof(double); }
    if (consumer && consumer(context,device_status,device_audit,stream)!=cudaSuccess) return false;
    if (defer) {
        w->deferred_active=true; w->deferred_stream=stream;
        w->deferred_status=device_status;
        cudaGetDevice(&w->deferred_device);
        w->deferred_owner=std::this_thread::get_id();
        pending.finished=true;
        return true;
    }
    const bool collected=collect_device_replay(w,stream,device_status,status,audit);
    if (collected) { pending.finished=true; *replayed=true; }
    const auto* early=std::getenv("SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH");
    // A successful synchronous scale-packet prime already satisfies the queued
    // updater's scale_valid prerequisite. Once a subsequent device IPM replay
    // has completed, another whole host-dispatched SCvx solve is unnecessary.
    // Retain the old readiness proof unless this measured path is selected.
    if (collected && (updated || (early && early[0]=='1' && w->numeric_scale_primed)))
        w->deferred_ready=true;
    return collected;
}

spacepdhcg_cuda_status validate_cached_topology(const spacepdhcg_cuda_scvx_problem& problem,
    cudaStream_t stream, const TopologyCache& topology, const int** device_mismatch=nullptr) {
    const auto& s = problem.canonical_structure;
    if (s.abi_version != SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION
        || s.topology_fingerprint != problem.topology_fingerprint || s.variables <= 0
        || topology.fingerprint != problem.topology_fingerprint)
        return SPACEPDHCG_CUDA_TOPOLOGY_MISMATCH;
    const auto& t = problem.canonical_topology;
    const spacepdhcg_accelerator_buffer_view views[]{t.quadratic_offsets,
        t.quadratic_indices, t.scalar_offsets, t.scalar_indices, t.affine_offsets, t.affine_indices};
    const std::size_t counts[]{static_cast<std::size_t>(s.variables) + 1, s.quadratic_nonzeros,
        static_cast<std::size_t>(s.variables) + 1, s.scalar_nonzeros,
        s.affine_rows == 0 ? 0 : static_cast<std::size_t>(s.variables) + 1, s.affine_nonzeros};
    QocoTopologyInput input{};
    for (int i = 0; i < 6; ++i) {
        if (counts[i] != topology.arrays[i].size()) return SPACEPDHCG_CUDA_TOPOLOGY_MISMATCH;
        if (counts[i] && (!views[i].data || views[i].elements != counts[i] || views[i].element_stride != 1))
            return SPACEPDHCG_CUDA_POINTER_CONTRACT;
        input.counts[i] = static_cast<int>(counts[i]);
        input.arrays[i] = counts[i] ? reinterpret_cast<const int*>(
            static_cast<const unsigned char*>(views[i].data) + views[i].byte_offset) : nullptr;
    }
    if (device_mismatch) return qoco_gpu_topology_validate_device(topology.device,input,stream,device_mismatch)==cudaSuccess
        ? SPACEPDHCG_CUDA_SUCCESS : SPACEPDHCG_CUDA_RUNTIME_ERROR;
    bool matches{};
    const auto status = qoco_gpu_topology_validate(topology.device, input, stream, &matches);
    if (status != cudaSuccess) return SPACEPDHCG_CUDA_RUNTIME_ERROR;
    return matches ? SPACEPDHCG_CUDA_SUCCESS : SPACEPDHCG_CUDA_TOPOLOGY_MISMATCH;
}

spacepdhcg_cuda_status convert(
    const spacepdhcg_cuda_scvx_problem& problem,
    cudaStream_t stream,
    Formulation* output,
    std::uint64_t* copy_count,
    std::uint64_t* copy_bytes,
    TopologyCache* topology
) {
    const auto& structure = problem.canonical_structure;
    if (structure.abi_version != SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION
        || structure.topology_fingerprint != problem.topology_fingerprint
        || structure.variables <= 0) {
        return SPACEPDHCG_CUDA_TOPOLOGY_MISMATCH;
    }
    const int n = structure.variables;
    auto& q_offsets = topology->arrays[0];
    auto& q_indices = topology->arrays[1];
    auto& a_offsets = topology->arrays[2];
    auto& a_indices = topology->arrays[3];
    auto& f_offsets = topology->arrays[4];
    auto& f_indices = topology->arrays[5];
    std::vector<double> q_values{};
    std::vector<double> a_values{};
    std::vector<double> f_values{};
    std::vector<double> lower{};
    std::vector<double> upper{};
    std::vector<double> affine_offset{};
    std::vector<double> variable_lower{};
    std::vector<double> variable_upper{};

    if (topology->device) {
        const auto status = validate_cached_topology(problem, stream, *topology);
        if (status != SPACEPDHCG_CUDA_SUCCESS) return status;
    }

    DownloadBatch downloads{stream};
    bool loading_topology = true;

    const auto load = [&](const auto& view, std::size_t count, auto* values) {
        if (loading_topology && topology->device) return SPACEPDHCG_CUDA_SUCCESS;
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
    loading_topology = false;
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
    const auto downloaded = downloads.finish();
    if (status != SPACEPDHCG_CUDA_SUCCESS) {
        return status;
    }
    if (downloaded != cudaSuccess) return SPACEPDHCG_CUDA_RUNTIME_ERROR;
    if (!topology->device) {
        QocoTopologyInput input{};
        for (int i = 0; i < 6; ++i) {
            if (topology->arrays[i].size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                return SPACEPDHCG_CUDA_UNSUPPORTED;
            input.counts[i] = static_cast<int>(topology->arrays[i].size());
            input.arrays[i] = topology->arrays[i].data();
        }
        const auto created = qoco_gpu_topology_create(input, stream, &topology->device);
        if (created != cudaSuccess) return created == cudaErrorMemoryAllocation
            ? SPACEPDHCG_CUDA_OUT_OF_MEMORY : SPACEPDHCG_CUDA_RUNTIME_ERROR;
        topology->fingerprint = problem.topology_fingerprint;
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

spacepdhcg_cuda_status compile_conversion(spacepdhcg_native_qoco* w,
    const spacepdhcg_cuda_scvx_problem& problem, cudaStream_t stream) {
    const auto& s = problem.canonical_structure;
    const auto& f = w->formulation;
    auto& cache = w->conversion;
    const std::size_t counts[]{s.quadratic_nonzeros, s.scalar_nonzeros, s.affine_nonzeros,
        static_cast<std::size_t>(s.variables), static_cast<std::size_t>(s.scalar_rows),
        static_cast<std::size_t>(s.scalar_rows), static_cast<std::size_t>(s.affine_rows),
        static_cast<std::size_t>(s.variables), static_cast<std::size_t>(s.variables)};
    QocoConversionPlan plan{};
    for (int i = 0; i < 9; ++i) {
        if (counts[i] > static_cast<std::size_t>(std::numeric_limits<int>::max())) return SPACEPDHCG_CUDA_UNSUPPORTED;
        cache.input_counts[i] = plan.input_counts[i] = static_cast<int>(counts[i]);
    }
    const auto save_cones = [](auto& saved, const auto* cones, std::size_t count) {
        if (count) saved.assign(cones, cones + count);
    };
    save_cones(cache.affine_cones, s.affine_cones, s.affine_cone_count);
    save_cones(cache.variable_cones, s.variable_cones, s.variable_cone_count);
    using References = std::vector<std::vector<std::pair<int, int>>>;
    const auto rows = [&](int array, int count) {
        References result(count);
        const auto& offsets = w->topology.arrays[array * 2];
        const auto& indices = w->topology.arrays[array * 2 + 1];
        if (!offsets.empty()) for (int col = 0; col < s.variables; ++col)
            for (int k = offsets[col]; k < offsets[col + 1]; ++k) result[indices[k]].emplace_back(col, k);
        return result;
    };
    const auto scalar = rows(1, s.scalar_rows), affine = rows(2, s.affine_rows);
    struct Entry { int row, column; QocoConversionTerm term; };
    std::vector<QocoConversionTerm> terms;
    std::vector<int> offsets{0}, bound_types(static_cast<std::size_t>(s.scalar_rows) + s.variables, 0);
    const auto term = [](int array, int index, double scale = 1.0, int other = -1, double other_scale = 0.0) {
        return QocoConversionTerm{array, index, other, scale, other_scale};
    };
    const auto finish_matrix = [&](const Csc& matrix, std::vector<Entry> entries) {
        std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
            return std::tie(a.column, a.row) < std::tie(b.column, b.row);
        });
        std::size_t cursor = 0;
        for (int col = 0; col < matrix.columns; ++col)
            for (int k = matrix.offsets[col]; k < matrix.offsets[col + 1]; ++k) {
                const auto before = cursor;
                while (cursor < entries.size() && entries[cursor].column == col && entries[cursor].row == matrix.indices[k])
                    terms.push_back(entries[cursor++].term);
                if (cursor == before || terms.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) return false;
                offsets.push_back(static_cast<int>(terms.size()));
            }
        return cursor == entries.size();
    };
    std::vector<Entry> p;
    std::unordered_map<std::uint64_t, int> q_entries;
    for (int col = 0; col < s.variables; ++col)
        for (int k = w->topology.arrays[0][col]; k < w->topology.arrays[0][col + 1]; ++k) {
            const int row = w->topology.arrays[1][k];
            q_entries[(static_cast<std::uint64_t>(row) << 32U) | static_cast<std::uint32_t>(col)] = k;
            if (row <= col) p.push_back({row, col, term(0, k)});
        }
    std::vector<QocoConversionPair> symmetry;
    for (const auto& [key, index] : q_entries) {
        const auto reverse = q_entries.find((key << 32U) | (key >> 32U));
        if (reverse == q_entries.end()) return SPACEPDHCG_CUDA_UNSUPPORTED;
        symmetry.push_back({index, reverse->second});
    }
    if (!finish_matrix(f.p, std::move(p))) return SPACEPDHCG_CUDA_INTERNAL_ERROR;
    const auto append_rows = [&](const std::vector<RowMap>& maps, const Csc& matrix) {
        std::vector<Entry> entries;
        for (int row = 0; row < static_cast<int>(maps.size()); ++row) {
            const auto& map = maps[row];
            const auto append = [&](int source, double scale) {
                if (map.source == Source::scalar || map.source == Source::affine) {
                    const auto& references = map.source == Source::scalar ? scalar : affine;
                    for (const auto& [col, k] : references[source])
                        entries.push_back({row, col, term(map.source == Source::scalar ? 1 : 2, k, scale)});
                } else entries.push_back({row, source, term(-1, 0, scale)});
            };
            if (map.source == Source::scalar || map.source == Source::variable) {
                append(map.index, map.side < 0 ? -1.0 : 1.0);
                auto& kind = bound_types[(map.source == Source::variable ? s.scalar_rows : 0) + map.index];
                kind = map.side == 0 ? 4 : kind | (map.side < 0 ? 2 : 1);
            } else if (!map.rotated) {
                append(map.cone_start + (map.transformed_row == 0 ? map.cone_size - 1 : map.transformed_row - 1), -1.0);
            } else if (map.transformed_row == 0 || map.transformed_row == map.cone_size - 1) {
                const double scale = 1.0 / std::sqrt(2.0);
                append(map.cone_start + map.cone_size - 2, -scale);
                append(map.cone_start + map.cone_size - 1, map.transformed_row == 0 ? -scale : scale);
            } else append(map.cone_start + map.transformed_row - 1, -1.0);
        }
        return finish_matrix(matrix, std::move(entries));
    };
    if (!append_rows(f.equality_map, f.a) || !append_rows(f.conic_map, f.g)) return SPACEPDHCG_CUDA_INTERNAL_ERROR;
    const auto append_value = [&](QocoConversionTerm entry) { terms.push_back(entry); offsets.push_back(static_cast<int>(terms.size())); };
    for (int i = 0; i < s.variables; ++i) append_value(term(3, i));
    const auto append_rhs = [&](const std::vector<RowMap>& maps) {
        for (const auto& map : maps) {
            if (map.source == Source::scalar || map.source == Source::variable) {
                const int lo = map.source == Source::scalar ? 4 : 7;
                append_value(term(map.side > 0 ? lo + 1 : lo, map.index, map.side < 0 ? -1.0 : 1.0));
            } else if (map.source == Source::variable_cone) append_value(term(-1, 0, 0.0));
            else if (!map.rotated) append_value(term(6,
                map.cone_start + (map.transformed_row == 0 ? map.cone_size - 1 : map.transformed_row - 1)));
            else if (map.transformed_row == 0 || map.transformed_row == map.cone_size - 1)
                append_value(term(6, map.cone_start + map.cone_size - 2, 1.0 / std::sqrt(2.0),
                    map.cone_start + map.cone_size - 1, map.transformed_row == 0 ? 1.0 : -1.0));
            else append_value(term(6, map.cone_start + map.transformed_row - 1));
        }
    };
    append_rhs(f.equality_map); append_rhs(f.conic_map);
    if (offsets.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())
        || terms.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) return SPACEPDHCG_CUDA_UNSUPPORTED;
    plan.outputs = static_cast<int>(offsets.size()) - 1; plan.terms = static_cast<int>(terms.size());
    plan.offsets = offsets.data(); plan.entries = terms.data(); plan.bound_types = bound_types.data();
    plan.symmetry_pairs = static_cast<int>(symmetry.size()); plan.symmetry = symmetry.data();
    cache.values.resize(plan.outputs);
    const auto status = qoco_gpu_conversion_create(plan, stream, &cache.device);
    return status == cudaSuccess ? SPACEPDHCG_CUDA_SUCCESS : status == cudaErrorMemoryAllocation
        ? SPACEPDHCG_CUDA_OUT_OF_MEMORY : SPACEPDHCG_CUDA_INTERNAL_ERROR;
}

bool conversion_needs_host(const spacepdhcg_native_qoco* w) {
    const auto enabled=[](const char* name) { const auto* value=std::getenv(name); return value && value[0]=='1'; };
    return !w->numeric_update_context || w->needs_fresh_solver || w->configured_settings.verbose
        || enabled("SPACEPDHCG_TEST_QOCO_GPU_CONVERSION_COMPARE") || enabled("SPACEPDHCG_TEST_QOCO_GPU_AUDIT_COMPARE");
}
spacepdhcg_cuda_status refresh_conversion(spacepdhcg_native_qoco* w,
    const spacepdhcg_cuda_scvx_problem& problem, cudaStream_t stream, const int* producer_invalid=nullptr) {
    auto& cache = w->conversion;
    const auto& s = problem.canonical_structure;
    const auto& n = problem.numeric;
    const bool host_values = conversion_needs_host(w);
    const bool deferred=w->queue_validation_allowed && !host_values;
    const int* device_topology{};
    if (deferred) w->validation_pending=true;
    const auto topology = validate_cached_topology(problem, stream, w->topology,deferred ? &device_topology : nullptr);
    if (topology != SPACEPDHCG_CUDA_SUCCESS) return topology;
    const auto same_cones = [](const auto& saved, const auto* cones, std::size_t count) {
        if (saved.size() != count || (count && !cones)) return false;
        for (std::size_t i = 0; i < count; ++i)
            if (saved[i].kind != cones[i].kind || saved[i].start != cones[i].start
                || saved[i].vector_dimension != cones[i].vector_dimension) return false;
        return true;
    };
    if (!same_cones(cache.affine_cones, s.affine_cones, s.affine_cone_count)
        || !same_cones(cache.variable_cones, s.variable_cones, s.variable_cone_count)
        || s.scalar_rows != cache.input_counts[4] || s.affine_rows != cache.input_counts[6])
        return SPACEPDHCG_CUDA_TOPOLOGY_MISMATCH;
    const spacepdhcg_accelerator_buffer_view views[]{n.quadratic, n.scalar_constraint, n.affine_cone,
        n.linear_objective, n.scalar_lower, n.scalar_upper, n.affine_offset, n.variable_lower, n.variable_upper};
    QocoConversionInputs inputs{};
    for (int i = 0; i < 9; ++i) if (cache.input_counts[i]) {
        if (!views[i].data || views[i].elements != static_cast<std::size_t>(cache.input_counts[i]) || views[i].element_stride != 1)
            return SPACEPDHCG_CUDA_POINTER_CONTRACT;
        inputs.arrays[i] = reinterpret_cast<const double*>(static_cast<const unsigned char*>(views[i].data) + views[i].byte_offset);
    }
    int invalid{};
    if (deferred) {
        if (qoco_gpu_conversion_run_device(cache.device,inputs,device_topology,stream,
            &w->device_validation)!=cudaSuccess) return SPACEPDHCG_CUDA_RUNTIME_ERROR;
        if (producer_invalid) {
            if (qoco_gpu_conversion_include_producer(cache.device,producer_invalid,stream)!=cudaSuccess)
                return SPACEPDHCG_CUDA_RUNTIME_ERROR;
            w->report.producer_validation_queued=1;
        }
        return SPACEPDHCG_CUDA_SUCCESS;
    }
    const auto status = qoco_gpu_conversion_run(cache.device, inputs,
        host_values ? cache.values.data() : nullptr, &invalid, stream);
    if (status != cudaSuccess) return SPACEPDHCG_CUDA_RUNTIME_ERROR;
    if (invalid & 1) return SPACEPDHCG_CUDA_TOPOLOGY_MISMATCH;
    if (invalid & 2) return SPACEPDHCG_CUDA_NUMERICAL_FAILURE;
    if (invalid & 4) return SPACEPDHCG_CUDA_UNSUPPORTED;
    if (!host_values) return SPACEPDHCG_CUDA_SUCCESS;
    if (const char* compare = std::getenv("SPACEPDHCG_TEST_QOCO_GPU_CONVERSION_COMPARE"); compare && compare[0] == '1') {
        Formulation reference;
        auto checked = convert(problem, stream, &reference, &w->report.d2h_copy_count, &w->report.d2h_bytes, &w->topology);
        if (checked != SPACEPDHCG_CUDA_SUCCESS) return checked;
        if (!same_pattern(w->formulation.p, reference.p) || !same_pattern(w->formulation.a, reference.a)
            || !same_pattern(w->formulation.g, reference.g) || w->formulation.equality_map != reference.equality_map
            || w->formulation.conic_map != reference.conic_map) return SPACEPDHCG_CUDA_INTERNAL_ERROR;
        std::size_t cursor = 0;
        for (const auto* values : {&reference.p.values, &reference.a.values, &reference.g.values,
                                  &reference.c, &reference.b, &reference.h})
            for (double value : *values) {
                const double actual = cache.values[cursor++];
                if (std::abs(value - actual) > 8 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(value))) {
                    std::fprintf(stderr, "QOCO GPU conversion mismatch at %zu: %.17g vs %.17g\n", cursor - 1, actual, value);
                    return SPACEPDHCG_CUDA_NUMERICAL_FAILURE;
                }
            }
    }
    std::size_t cursor = 0;
    for (auto* values : {&w->formulation.p.values, &w->formulation.a.values, &w->formulation.g.values,
                         &w->formulation.c, &w->formulation.b, &w->formulation.h}) {
        std::copy_n(cache.values.data() + cursor, values->size(), values->data());
        cursor += values->size();
    }
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
    // Stationarity c + P x + A^T y + G^T z, with the constraint-transposed term kept separately
    // so the dual scale sees the same |A^T y + G^T z| magnitude the CPU-reference audit uses.
    std::vector<double> constraint_dual(formulation.c.size(), 0.0);
    transpose_accumulate(formulation.a, workspace->solver->sol->y, &constraint_dual);
    transpose_accumulate(formulation.g, workspace->solver->sol->z, &constraint_dual);
    double dual = 0.0;
    double max_px = 0.0;
    double max_c = 0.0;
    double max_constraint_dual = 0.0;
    double primal_objective = 0.0;
    for (std::size_t index = 0U; index < stationarity.size(); ++index) {
        const double px = stationarity[index] - formulation.c[index];
        max_px = std::max(max_px, std::abs(px));
        max_c = std::max(max_c, std::abs(formulation.c[index]));
        max_constraint_dual = std::max(max_constraint_dual, std::abs(constraint_dual[index]));
        primal_objective +=
            (0.5 * px + formulation.c[index]) * workspace->primal[index];
        dual = std::max(dual, std::abs(stationarity[index] + constraint_dual[index]));
    }

    // Dual cone feasibility (z in K*) and per-cone complementarity |s . z|, which QOCO
    // drives to its own tolerances but which the planner certificate also audits.
    const double* z = workspace->solver->sol->z;
    double dual_cone = 0.0;
    double complementarity = 0.0;
    for (int index = 0; index < formulation.nonnegative; ++index) {
        dual_cone = std::max(dual_cone, std::max(0.0, -z[index]));
        complementarity = std::max(complementarity, std::abs(slack[index] * z[index]));
    }
    cursor = formulation.nonnegative;
    for (int size : formulation.soc) {
        dual_cone = std::max(dual_cone, soc_violation(z + cursor, size));
        double inner = 0.0;
        for (int offset = 0; offset < size; ++offset) {
            inner += slack[cursor + offset] * z[cursor + offset];
        }
        complementarity = std::max(complementarity, std::abs(inner));
        cursor += size;
    }

    // Relative KKT normalisation matching `spacepdhcg.cqp.quality.canonical_residual_audit`
    // (the definition the planner's `canonical_residual` certificate gate and the CPU
    // reference report).  An absolute residual is not comparable across families: the
    // pd3 canonical rows carry O(1e3) thrust/mass magnitudes, so QOCO's converged solution
    // showed an absolute 2.5e-3 against the 1e-6 gate while the relative residual was 1e-8.
    double max_rhs = 0.0;
    for (double value : formulation.b) {
        max_rhs = std::max(max_rhs, std::abs(value));
    }
    for (double value : formulation.h) {
        max_rhs = std::max(max_rhs, std::abs(value));
    }
    double max_ax = 0.0;
    for (double value : ax) {
        max_ax = std::max(max_ax, std::abs(value));
    }
    for (double value : gx) {
        max_ax = std::max(max_ax, std::abs(value));
    }
    double dual_objective = 0.0;
    for (std::size_t index = 0U; index < formulation.b.size(); ++index) {
        dual_objective += formulation.b[index] * workspace->solver->sol->y[index];
    }
    for (std::size_t index = 0U; index < formulation.h.size(); ++index) {
        dual_objective += formulation.h[index] * z[index];
    }
    const double primal_scale = 1.0 + max_rhs + max_ax;
    const double dual_scale = 1.0 + max_c + max_px + max_constraint_dual;
    const double gap_scale = 1.0 + std::abs(primal_objective) + std::abs(dual_objective);

    auto& report = workspace->report;
    report.absolute_primal_residual = primal;
    report.absolute_dual_residual = dual;
    report.dual_cone_residual = dual_cone / primal_scale;
    report.complementarity_residual = complementarity / gap_scale;
    report.primal_residual = std::max(primal, dual_cone) / primal_scale;
    report.dual_residual = std::max(dual / dual_scale, report.complementarity_residual);
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

// qoco_setup on the workspace's current formulation with its configured
// settings. Returns QOCO's setup code (0 on success) and accumulates the setup
// time; on failure the solver struct is released and nulled.
int setup_solver(spacepdhcg_native_qoco* workspace) {
    workspace->origin_applied=false;
    auto p = workspace->formulation.p.abi();
    auto a = workspace->formulation.a.abi();
    auto g = workspace->formulation.g.abi();
    SettingsAbi settings = workspace->configured_settings;
    // Establish storage and symbolic KKT structure without CPU equilibration.
    // The device update performs the requested initial Ruiz passes below.
    if (workspace->create_numeric_update) settings.ruiz_iters = 0;
    const auto setup_start = std::chrono::steady_clock::now();
    if (workspace->set_trajectory && workspace->set_trajectory(
            workspace->trajectory_intervals, workspace->trajectory_nx, workspace->trajectory_nu,
            workspace->trajectory_states, workspace->trajectory_controls,
            workspace->trajectory_virtual, workspace->trajectory_stream) != 0) {
        std::free(workspace->solver);
        workspace->solver = nullptr;
        return -1;
    }
    int code = workspace->setup(
        workspace->solver,
        workspace->variables,
        static_cast<int>(workspace->formulation.h.size()),
        static_cast<int>(workspace->formulation.b.size()),
        &p,
        workspace->formulation.c.data(),
        workspace->formulation.b.empty() ? nullptr : &a,
        workspace->formulation.b.empty() ? nullptr : workspace->formulation.b.data(),
        workspace->formulation.h.empty() ? nullptr : &g,
        workspace->formulation.h.empty() ? nullptr : workspace->formulation.h.data(),
        workspace->formulation.nonnegative,
        static_cast<int>(workspace->formulation.soc.size()),
        workspace->formulation.soc.empty() ? nullptr : workspace->formulation.soc.data(),
        &settings
    );
    if (workspace->set_trajectory)
        static_cast<void>(workspace->set_trajectory(0, 0, 0, nullptr, nullptr, nullptr, nullptr));
    if (code == 0 && workspace->set_device_io
        && workspace->set_device_io(workspace->solver, 1) != 0) {
        workspace->cleanup(workspace->solver);
        workspace->solver = nullptr;
        code = -1;
    }
    if (code == 0 && workspace->create_numeric_update) {
        int created = workspace->create_numeric_update(workspace->solver, p.nnz, a.nnz, g.nnz,
            &workspace->numeric_update_context);
        if (created == 0 && workspace->configured_settings.ruiz_iters > 0) {
            created = workspace->update_settings(workspace->solver, &workspace->configured_settings);
            if (created == 0) created = workspace->device_numeric_update(workspace->numeric_update_context,
                qoco_gpu_conversion_values(workspace->conversion.device), nullptr);
            if (created == 0) ++workspace->report.device_numeric_updates;
        }
        if (created != 0) {
            if (workspace->numeric_update_context) {
                workspace->destroy_numeric_update(workspace->numeric_update_context);
                workspace->numeric_update_context = nullptr;
            }
            workspace->cleanup(workspace->solver);
            workspace->solver = nullptr;
            code = -created;
        }
    }
    if (code == 0 && workspace->origin_mode && workspace->set_stopping_origin
        && workspace->set_stopping_origin(workspace->solver,
            qoco_gpu_audit_origin(workspace->gpu_audit),
            qoco_gpu_audit_origin_offset(workspace->gpu_audit)) != 0) {
        if (workspace->numeric_update_context) {
            workspace->destroy_numeric_update(workspace->numeric_update_context);
            workspace->numeric_update_context=nullptr;
        }
        workspace->cleanup(workspace->solver); workspace->solver=nullptr; code=-1;
    }
    workspace->report.setup_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - setup_start
    ).count();
    if (code != 0) {
        std::free(workspace->solver);
        workspace->solver = nullptr;
    }
    return code;
}

}  // namespace

spacepdhcg_cuda_status native_qoco_create_impl(
    const spacepdhcg_cuda_scvx_problem* problem,
    cudaStream_t stream,
    int ruiz_iterations,
    spacepdhcg_native_qoco** workspace,
    double tolerance,
    bool require_device_extensions
) {
    if (problem == nullptr || workspace == nullptr) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    *workspace = nullptr;
    if (!std::isfinite(tolerance) || tolerance <= 0.0)
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
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
    // Optional paired extension; unpatched pinned libraries retain their original
    // execution. A scope is destroyed before returning, including failed solves.
    symbol(result->library, "qoco_gpu_begin_reduction_scope", &result->begin_reduction_scope);
    symbol(result->library, "qoco_gpu_end_reduction_scope", &result->end_reduction_scope);
    symbol(result->library, "qoco_gpu_ipm_replay_device", &result->replay);
    symbol(result->library, "qoco_gpu_ipm_replay_updated_device", &result->replay_updated);
    symbol(result->library, "qoco_gpu_ipm_emit_graph", &result->emit_graph);
    if (cudaGetDevice(&result->creation_device)!=cudaSuccess) return SPACEPDHCG_CUDA_RUNTIME_ERROR;
    symbol(result->library, "qoco_gpu_ipm_finish_device", &result->finish_replay);
    symbol(result->library, "qoco_gpu_get_solution", &result->device_solution);
    symbol(result->library, "qoco_gpu_set_device_io", &result->set_device_io);
    symbol(result->library, "qoco_gpu_set_stopping_origin", &result->set_stopping_origin);
    symbol(result->library, "qoco_gpu_primal_start", &result->primal_start);
    symbol(result->library, "qoco_gpu_download_solution", &result->download_solution);
    if ((result->set_device_io || result->primal_start || result->download_solution)
        && !(result->set_device_io && result->primal_start && result->download_solution
             && result->device_solution)) return SPACEPDHCG_CUDA_UNSUPPORTED;
    if (const char* required = std::getenv("SPACEPDHCG_TEST_QOCO_DEVICE_IO_REQUIRED");
        required && required[0] == '1' && !result->set_device_io)
        return SPACEPDHCG_CUDA_UNSUPPORTED;
    symbol(result->library, "qoco_gpu_create_numeric_update", &result->create_numeric_update);
    symbol(result->library, "qoco_gpu_update_numeric", &result->device_numeric_update);
    symbol(result->library, "qoco_gpu_update_numeric_device", &result->queued_numeric_update);
    symbol(result->library, "qoco_gpu_capture_numeric_update", &result->capture_numeric_update);
    symbol(result->library, "qoco_gpu_finish_numeric_update", &result->finish_numeric_update);
    symbol(result->library, "qoco_gpu_numeric_preserves_objective", &result->numeric_preserves_objective);
    symbol(result->library, "qoco_gpu_destroy_numeric_update", &result->destroy_numeric_update);
    symbol(result->library, "qoco_gpu_set_trajectory", &result->set_trajectory);
    if (result->set_trajectory && problem->intervals > 0
        && problem->intervals < static_cast<std::size_t>(std::numeric_limits<int>::max())
        && problem->state_dimension > 0 && problem->state_dimension <= 64
        && problem->control_dimension > 0 && problem->control_dimension <= 64) {
        auto indices = [](const spacepdhcg_accelerator_buffer_view& view, std::size_t count) {
            return view.data && view.elements == count && view.element_stride == 1
                && view.scalar_type == SPACEPDHCG_SCALAR_INT32
                && (view.device.type == SPACEPDHCG_DEVICE_CUDA
                    || view.device.type == SPACEPDHCG_DEVICE_CUDA_MANAGED)
                ? reinterpret_cast<const int*>(static_cast<const unsigned char*>(view.data)
                    + view.byte_offset) : nullptr;
        };
        result->trajectory_intervals = static_cast<int>(problem->intervals);
        result->trajectory_nx = static_cast<int>(problem->state_dimension);
        result->trajectory_nu = static_cast<int>(problem->control_dimension);
        result->trajectory_states = indices(problem->state_variable_indices,
            (problem->intervals + 1) * problem->state_dimension);
        result->trajectory_controls = indices(problem->control_variable_indices,
            problem->intervals * problem->control_dimension);
        result->trajectory_virtual = indices(problem->virtual_variable_indices,
            problem->intervals * problem->state_dimension);
        result->trajectory_stream = stream;
        if (!result->trajectory_states || !result->trajectory_controls || !result->trajectory_virtual)
            return SPACEPDHCG_CUDA_POINTER_CONTRACT;
        const std::size_t counts[]{(problem->intervals + 1) * problem->state_dimension,
            problem->intervals * problem->control_dimension,
            problem->intervals * problem->state_dimension};
        result->trajectory_bytes = (counts[0] + counts[1] + counts[2]) * sizeof(int);
        if (cudaMalloc(&result->trajectory_indices, result->trajectory_bytes) != cudaSuccess)
            return SPACEPDHCG_CUDA_OUT_OF_MEMORY;
        const int* sources[]{result->trajectory_states, result->trajectory_controls,
            result->trajectory_virtual};
        std::size_t offset = 0;
        for (int part = 0; part < 3; ++part) {
            if (cudaMemcpyAsync(result->trajectory_indices + offset, sources[part],
                    counts[part] * sizeof(int), cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
                return SPACEPDHCG_CUDA_RUNTIME_ERROR;
            offset += counts[part];
        }
        result->report.d2d_copy_count += 3;
        result->report.d2d_bytes += result->trajectory_bytes;
        result->trajectory_states = result->trajectory_indices;
        result->trajectory_controls = result->trajectory_indices + counts[0];
        result->trajectory_virtual = result->trajectory_controls + counts[1];
    }
    const int update_symbols = (result->create_numeric_update != nullptr)
        + (result->device_numeric_update != nullptr) + (result->destroy_numeric_update != nullptr);
    if (update_symbols != 0 && update_symbols != 3) return SPACEPDHCG_CUDA_UNSUPPORTED;
    if (require_device_extensions && (!result->set_device_io || !result->device_solution
        || !result->primal_start || update_symbols != 3 || !result->begin_reduction_scope
        || !result->end_reduction_scope)) return SPACEPDHCG_CUDA_UNSUPPORTED;
    if ((result->begin_reduction_scope == nullptr) != (result->end_reduction_scope == nullptr)) {
        return SPACEPDHCG_CUDA_UNSUPPORTED;
    }
    const auto conversion_start = std::chrono::steady_clock::now();
    auto status = convert(
        *problem,
        stream,
        &result->formulation,
        &result->report.d2h_copy_count,
        &result->report.d2h_bytes,
        &result->topology
    );
    if (status == SPACEPDHCG_CUDA_SUCCESS) status = compile_conversion(result.get(), *problem, stream);
    if (status == SPACEPDHCG_CUDA_SUCCESS) status = refresh_conversion(result.get(), *problem, stream);
    result->report.conversion_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - conversion_start
    ).count();
    if (status != SPACEPDHCG_CUDA_SUCCESS) return status;
    const std::size_t variables =
        static_cast<std::size_t>(problem->canonical_structure.variables);
    const std::size_t duals =
        static_cast<std::size_t>(
            problem->canonical_structure.scalar_rows
            + problem->canonical_structure.affine_rows
        );
    if (!result->set_device_io) {
        result->primal.assign(variables, 0.0);
        result->accepted_primal.assign(variables, 0.0);
    }
    result->dual.assign(duals, 0.0);
    result->solver = static_cast<SolverAbi*>(std::calloc(1U, sizeof(SolverAbi)));
    if (result->solver == nullptr) {
        return SPACEPDHCG_CUDA_OUT_OF_MEMORY;
    }
    const bool low_thrust =
        problem->dynamics.model == SPACEPDHCG_CUDA_DYNAMICS_LOW_THRUST;
    // Low-thrust mass-flow equalities and virtual penalties span many orders
    // of magnitude. Near-zero factor regularization makes identical QPs vary
    // with cuDSS accumulation order. Stabilize the factors, then refine against
    // the original, unregularized equations at the existing strict tolerance.
    // ruiz_iters is the caller's choice (amendment single-gpu-v1.2 selects
    // QOCO's own Ruiz equilibration for IPM attempts; 0 keeps the pinned
    // QOCO commit's default of no equilibration).
    SettingsAbi settings{
        200, std::max(0, ruiz_iterations), low_thrust ? 20 : 5,
        low_thrust ? 1.0e-12 : 1.0e-6,
        low_thrust ? 1.0e-8 : 1.0e-13, low_thrust ? 1.0e-8 : 1.0e-8,
        low_thrust ? 1.0e-8 : 1.0e-13,
        low_thrust ? 1.0e-13 : 1.0e-11,
        tolerance, tolerance, 1.0e-5, 1.0e-5, 0,
    };
    // Diagnostic only: QOCO's own iteration log on stderr. Never set by the
    // campaign scheduler; timing records are not affected in its absence.
    if (const char* verbose = std::getenv("SPACEPDHCG_QOCO_VERBOSE");
        verbose != nullptr && verbose[0] == '1') {
        settings.verbose = 1;
    }
    result->report.ruiz_iterations = settings.ruiz_iters;
    result->report.status_code = -1;
    result->configured_settings = settings;
    result->variables = problem->canonical_structure.variables;
    const int code = setup_solver(result.get());
    if (code != 0) {
        return code == 5 ? SPACEPDHCG_CUDA_OUT_OF_MEMORY
                         : SPACEPDHCG_CUDA_NUMERICAL_FAILURE;
    }
    // The first setup completed metadata copies. Subsequent recovery setups
    // borrow neither the caller's index arrays nor its original stream handle.
    result->trajectory_stream = nullptr;
    const auto audit_start = std::chrono::steady_clock::now();
    result->dual_transfer = make_dual_transfer(result->formulation, problem->canonical_structure);
    const auto audit_status = qoco_gpu_audit_create(audit_input(result.get()),
        result->device_solution == nullptr, stream, &result->gpu_audit);
    result->report.setup_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - audit_start).count();
    if (audit_status != cudaSuccess) return audit_status == cudaErrorMemoryAllocation
        ? SPACEPDHCG_CUDA_OUT_OF_MEMORY : SPACEPDHCG_CUDA_RUNTIME_ERROR;
    result->report.workspace_creations = 1U;
    *workspace = result.release();
    return SPACEPDHCG_CUDA_SUCCESS;
}

spacepdhcg_cuda_status copy_native_report(spacepdhcg_native_qoco* workspace,
    spacepdhcg_native_qoco_report* report, spacepdhcg_cuda_status status) {
        workspace->report.producer_invalid=(workspace->validation_flags & 16)!=0;
        *report = workspace->report;
        const auto transfers = qoco_gpu_audit_transfers(workspace->gpu_audit);
        report->h2d_copy_count += transfers.h2d_count;
        report->h2d_bytes += transfers.h2d_bytes;
        report->d2h_copy_count += transfers.d2h_count;
        report->d2h_bytes += transfers.d2h_bytes;
        const auto memory = qoco_gpu_audit_memory(workspace->gpu_audit);
        report->audit_allocations = memory.allocations;
        report->audit_peak_bytes = memory.peak_bytes;
        const auto topology_transfers = qoco_gpu_topology_transfers(workspace->topology.device);
        report->h2d_copy_count += topology_transfers.h2d_count;
        report->h2d_bytes += topology_transfers.h2d_bytes;
        report->d2h_copy_count += topology_transfers.d2h_count;
        report->d2h_bytes += topology_transfers.d2h_bytes;
        const auto topology_memory = qoco_gpu_topology_memory(workspace->topology.device);
        report->audit_allocations += topology_memory.allocations;
        report->audit_peak_bytes += topology_memory.peak_bytes;
        const auto conversion_transfers = qoco_gpu_conversion_transfers(workspace->conversion.device);
        report->h2d_copy_count += conversion_transfers.h2d_count;
        report->h2d_bytes += conversion_transfers.h2d_bytes;
        report->d2h_copy_count += conversion_transfers.d2h_count;
        report->d2h_bytes += conversion_transfers.d2h_bytes;
        const auto conversion_memory = qoco_gpu_conversion_memory(workspace->conversion.device);
        report->audit_allocations += conversion_memory.allocations;
        report->audit_peak_bytes += conversion_memory.peak_bytes;
        report->audit_allocations += workspace->trajectory_indices != nullptr ? 1U : 0U;
        report->audit_peak_bytes += workspace->trajectory_bytes;
        if(workspace->graph_progress) { ++report->audit_allocations;report->audit_peak_bytes+=sizeof(QocoGraphProgress); }
        return status;
}

spacepdhcg_cuda_status finish_native_solve(spacepdhcg_native_qoco* workspace,
    const spacepdhcg_cuda_scvx_problem* problem, cudaStream_t stream,
    double* device_primal, double* device_dual, spacepdhcg_native_qoco_report* report,
    spacepdhcg_native_qoco_consumer consumer, void* context, bool warm,
    int status_code, QocoAuditResult replay_audit, bool replay_audited, bool invoked) {
    const auto finish = [&](spacepdhcg_cuda_status status) {
        return copy_native_report(workspace,report,status);
    };
    if (workspace->validation_flags) {
        // Invalid canonical data may already have reached numeric buffers, but
        // the device guard prevented IPM/acceptance. Rebuild before the next try.
        workspace->needs_fresh_solver=true;
        workspace->report.status_code=status_code;
        workspace->report.iterations=0;
        const auto failure=canonical_validation_status(workspace->validation_flags);
        workspace->report.failure=failure==SPACEPDHCG_CUDA_NUMERICAL_FAILURE
            ? SPACEPDHCG_CUDA_QOCO_FAILURE_NUMERICAL : failure==SPACEPDHCG_CUDA_UNSUPPORTED
            ? SPACEPDHCG_CUDA_QOCO_FAILURE_UNSUPPORTED : SPACEPDHCG_CUDA_QOCO_FAILURE_ABI;
        return finish(failure);
    }
    if (!invoked) {
        if (workspace->numeric_update_invalid) {
            workspace->needs_fresh_solver=true;
            workspace->report.failure=SPACEPDHCG_CUDA_QOCO_FAILURE_NUMERICAL;
            return finish(SPACEPDHCG_CUDA_NUMERICAL_FAILURE);
        }
        workspace->report.failure = SPACEPDHCG_CUDA_QOCO_FAILURE_ABI;
        return finish(SPACEPDHCG_CUDA_RUNTIME_ERROR);
    }
    ++workspace->report.solves;
    workspace->report.status_code = status_code;
    if (workspace->solver->sol == nullptr
        || workspace->solver->sol->status != status_code) {
        workspace->report.failure = SPACEPDHCG_CUDA_QOCO_FAILURE_ABI;
        return finish(SPACEPDHCG_CUDA_INTERNAL_ERROR);
    }
    workspace->report.iterations = workspace->solver->sol->iters;
    if (warm && status_code == 2) {
        // QOCO_SOLVED_INACCURATE off a warm start: an interior-point method started at the
        // previous (boundary) optimum stalls after a single iteration and hands back a point
        // that only meets the loose 1e-5 tolerances (pd3 polish: dual residual 2.5e-3 from
        // a 1e-10 cold solve). Re-solve cold once and independently audit the result.
        // cuDSS factor accumulation can vary between cold solves too; the cold
        // start is a recovery attempt, not a guarantee of numerical qualification.
        // `dual_discarded` describes how the *request* was handled (the requested dual
        // was never usable by QOCO) and stays set; only the primal-warm flag flips.
        if (workspace->primal_start) {
            if (workspace->primal_start(workspace->solver, 0) != 0)
                return finish(SPACEPDHCG_CUDA_INVALID_STATE);
        } else {
            workspace->set_x0(workspace->solver, nullptr);
        }
        workspace->report.warm_primal_accepted = 0;
        ++workspace->report.warm_inaccurate_cold_retries;
        const auto retry_start = std::chrono::steady_clock::now();
        replay_audited=false;
        const bool retried = solve_with_reduction_scope(workspace, &status_code);
        workspace->report.solve_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - retry_start
        ).count();
        if (!retried) {
            workspace->report.failure = SPACEPDHCG_CUDA_QOCO_FAILURE_ABI;
            return finish(SPACEPDHCG_CUDA_RUNTIME_ERROR);
        }
        ++workspace->report.solves;
        if (workspace->solver->sol->status != status_code) {
            workspace->report.failure = SPACEPDHCG_CUDA_QOCO_FAILURE_ABI;
            return finish(SPACEPDHCG_CUDA_INTERNAL_ERROR);
        }
        workspace->report.iterations += workspace->solver->sol->iters;
        // `status_code` is the raw status of the *last* solve: the cold retry, not the
        // stalled warm attempt it replaced.
        workspace->report.status_code = status_code;
    }
    workspace->report.last_status_inaccurate = status_code == 2 ? 1 : 0;
    if (status_code != 1 && status_code != 2) {
        workspace->needs_fresh_solver = true;
        workspace->report.failure = status_code == 4
            ? SPACEPDHCG_CUDA_QOCO_FAILURE_MAX_ITERATIONS
            : SPACEPDHCG_CUDA_QOCO_FAILURE_NUMERICAL;
        return finish(SPACEPDHCG_CUDA_NUMERICAL_FAILURE);
    }
    if (!workspace->set_device_io)
        std::copy_n(workspace->solver->sol->x, workspace->variables, workspace->primal.begin());
    const auto audit_start = std::chrono::steady_clock::now();
    const double *x{}, *y{}, *z{};
    auto cuda_status = cudaSuccess;
    QocoAuditResult audit=replay_audit;
    if (!replay_audited) {
    if (workspace->device_solution) {
        if (workspace->device_solution(workspace->solver, workspace->variables,
                static_cast<int>(workspace->formulation.b.size()),
                static_cast<int>(workspace->formulation.h.size()), &x, &y, &z) != 0) {
            workspace->report.failure = SPACEPDHCG_CUDA_QOCO_FAILURE_ABI;
            return finish(SPACEPDHCG_CUDA_INVALID_STATE);
        }
    } else {
        cuda_status = qoco_gpu_audit_upload_solution(workspace->gpu_audit,
            workspace->solver->sol->x, workspace->solver->sol->y, workspace->solver->sol->z,
            stream, &x, &y, &z);
    }
    if (cuda_status == cudaSuccess) {
        cuda_status = workspace->origin_mode
            ? qoco_gpu_audit_reconstruct(workspace->gpu_audit,x,device_primal,stream)
            : cudaMemcpyAsync(device_primal, x, workspace->variables * sizeof(double),cudaMemcpyDeviceToDevice,stream);
        if (cuda_status == cudaSuccess && !workspace->origin_mode) {
            ++workspace->report.d2d_copy_count;
            workspace->report.d2d_bytes += workspace->variables * sizeof(double);
        }
        if (workspace->origin_mode) x=device_primal;
    }
    if (cuda_status == cudaSuccess) cuda_status = qoco_gpu_audit_run(workspace->gpu_audit,
        x, y, z, device_dual, stream, &audit);
    }
    workspace->report.residual_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - audit_start).count();
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
    // Explicit test oracle only; production audit and dual mapping run on CUDA.
    if (const char* compare = std::getenv("SPACEPDHCG_TEST_QOCO_GPU_AUDIT_COMPARE");
        problem && compare != nullptr && compare[0] == '1') {
        if (workspace->download_solution) {
            if (workspace->download_solution(workspace->solver) != 0)
                return finish(SPACEPDHCG_CUDA_RUNTIME_ERROR);
            workspace->primal.resize(workspace->variables);
            if (workspace->origin_mode) {
                if (cudaMemcpyAsync(workspace->primal.data(),device_primal,workspace->variables*sizeof(double),
                    cudaMemcpyDeviceToHost,stream)!=cudaSuccess || cudaStreamSynchronize(stream)!=cudaSuccess)
                    return finish(SPACEPDHCG_CUDA_RUNTIME_ERROR);
                ++workspace->report.d2h_copy_count;workspace->report.d2h_bytes+=workspace->variables*sizeof(double);
            } else std::copy_n(workspace->solver->sol->x, workspace->variables, workspace->primal.begin());
        }
        residuals(workspace);
        map_dual(workspace, problem->canonical_structure);
        const double reference[]{workspace->report.primal_residual, workspace->report.dual_residual,
            workspace->report.absolute_primal_residual, workspace->report.absolute_dual_residual,
            workspace->report.dual_cone_residual, workspace->report.complementarity_residual};
        const double actual[]{audit.primal, audit.dual, audit.absolute_primal, audit.absolute_dual,
                              audit.dual_cone, audit.complementarity};
        for (int i = 0; i < 6; ++i) {
            if (!std::isfinite(actual[i]) || std::abs(reference[i] - actual[i]) > 1e-11 * (1 + std::abs(reference[i])))
                return finish(SPACEPDHCG_CUDA_INTERNAL_ERROR);
        }
        std::vector<double> mapped(workspace->dual.size());
        if (cudaMemcpyAsync(mapped.data(), device_dual, mapped.size() * sizeof(double),
                            cudaMemcpyDeviceToHost, stream) != cudaSuccess
            || cudaStreamSynchronize(stream) != cudaSuccess) return finish(SPACEPDHCG_CUDA_RUNTIME_ERROR);
        ++workspace->report.d2h_copy_count;
        workspace->report.d2h_bytes += mapped.size() * sizeof(double);
        for (std::size_t i = 0; i < mapped.size(); ++i)
            if (std::abs(mapped[i] - workspace->dual[i]) > 1e-11 * (1 + std::abs(workspace->dual[i])))
                return finish(SPACEPDHCG_CUDA_INTERNAL_ERROR);
    }
    workspace->report.primal_residual = audit.primal;
    workspace->report.dual_residual = audit.dual;
    workspace->report.absolute_primal_residual = audit.absolute_primal;
    workspace->report.absolute_dual_residual = audit.absolute_dual;
    workspace->report.dual_cone_residual = audit.dual_cone;
    workspace->report.complementarity_residual = audit.complementarity;
    if (const char* dump = std::getenv("SPACEPDHCG_QOCO_DUMP_PRIMAL");
        dump && dump[0] == '1') {
        // Explicit diagnostic export of the IPM output, not the unused PDHCG
        // workspace. Never enabled by a timed campaign or production caller.
        std::vector<double> snapshot(workspace->variables);
        if (cudaMemcpyAsync(snapshot.data(), device_primal,
                snapshot.size()*sizeof(double), cudaMemcpyDeviceToHost, stream)!=cudaSuccess
            || cudaStreamSynchronize(stream)!=cudaSuccess) return finish(SPACEPDHCG_CUDA_RUNTIME_ERROR);
        ++workspace->report.d2h_copy_count;
        workspace->report.d2h_bytes += snapshot.size()*sizeof(double);
        std::printf("PD3DATA qoco_primal");
        for (double value : snapshot) std::printf(" %.17g",value);
        std::printf("\nQOCOAUDIT %.17g %.17g %.17g %.17g\n",
            audit.absolute_primal,audit.absolute_dual,audit.dual_cone,audit.complementarity);
    }
    if (!std::isfinite(audit.primal) || !std::isfinite(audit.dual)) {
        workspace->report.failure = SPACEPDHCG_CUDA_QOCO_FAILURE_NUMERICAL;
        return finish(SPACEPDHCG_CUDA_NUMERICAL_FAILURE);
    }
    workspace->report.failure = SPACEPDHCG_CUDA_QOCO_FAILURE_NONE;
    if (consumer && !replay_audited) {
        const QocoReplayStatus* device_status{};
        auto result=qoco_gpu_audit_publish_status(workspace->gpu_audit,status_code,
            workspace->report.iterations,stream,&device_status);
        if (result==cudaSuccess) result=consumer(context,device_status,
            qoco_gpu_audit_device_result(workspace->gpu_audit),stream);
        // Drain even after a failed launch: the consumer context is borrowed.
        const auto waited=cudaStreamSynchronize(stream);
        if (result!=cudaSuccess || waited!=cudaSuccess) return finish(SPACEPDHCG_CUDA_RUNTIME_ERROR);
    }
    return finish(SPACEPDHCG_CUDA_SUCCESS);
}

spacepdhcg_cuda_status native_solver_values(spacepdhcg_native_qoco* w,cudaStream_t stream,const double** values) {
    *values=qoco_gpu_conversion_values(w->conversion.device);
    if (!w->origin_mode) return SPACEPDHCG_CUDA_SUCCESS;
    const auto code=qoco_gpu_audit_origin_values(w->gpu_audit,*values,stream,values);
    if (code!=cudaSuccess) return SPACEPDHCG_CUDA_RUNTIME_ERROR;
    const size_t nnz=w->formulation.p.values.size()+w->formulation.a.values.size()+w->formulation.g.values.size();
    if (nnz) { ++w->report.d2d_copy_count;w->report.d2d_bytes+=nnz*sizeof(double); }
    return SPACEPDHCG_CUDA_SUCCESS;
}

// Explicit diagnostic only. Decimal round-trip values and retained topology
// permit standalone replays without reconstructing a trajectory or its seed.
// The destination directory must already exist; existing files are never replaced.
spacepdhcg_cuda_status snapshot_qoco_problem(spacepdhcg_native_qoco* w,cudaStream_t stream) {
    const auto* directory=std::getenv("SPACEPDHCG_QOCO_SNAPSHOT_DIRECTORY");
    if (!directory || !directory[0]) return SPACEPDHCG_CUDA_SUCCESS;
    const auto& f=w->formulation;
    const size_t count=f.p.values.size()+f.a.values.size()+f.g.values.size()
        +f.c.size()+f.b.size()+f.h.size();
    std::vector<double> original(count),shifted(w->origin_mode ? count : 0),origin(w->origin_mode ? w->variables : 0);
    double offset=0;
    const double* solver_values{};
    if (native_solver_values(w,stream,&solver_values)!=SPACEPDHCG_CUDA_SUCCESS)
        return SPACEPDHCG_CUDA_RUNTIME_ERROR;
    const auto download=[&](double* target,const double* source,size_t n) {
        if (!n) return true;
        if (cudaMemcpyAsync(target,source,n*sizeof(double),cudaMemcpyDeviceToHost,stream)!=cudaSuccess) return false;
        ++w->report.d2h_copy_count; w->report.d2h_bytes+=n*sizeof(double);return true;
    };
    const bool copied=download(original.data(),qoco_gpu_conversion_values(w->conversion.device),count)
        && (!w->origin_mode || (download(shifted.data(),solver_values,count)
            && download(origin.data(),qoco_gpu_audit_origin(w->gpu_audit),origin.size())
            && download(&offset,qoco_gpu_audit_origin_offset(w->gpu_audit),1)));
    const auto waited=cudaStreamSynchronize(stream);
    if (!copied || waited!=cudaSuccess) return SPACEPDHCG_CUDA_RUNTIME_ERROR;
    static std::atomic<unsigned long long> sequence{0};
    const auto id=sequence.fetch_add(1);
    char path[4096];
    const int length=std::snprintf(path,sizeof(path),"%s/qp-%d-%06llu.txt",directory,int(getpid()),id);
    if (length<0 || length>=int(sizeof(path))) return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    FILE* file=std::fopen(path,"wx");
    if (!file) return SPACEPDHCG_CUDA_RUNTIME_ERROR;
    std::fprintf(file,"SPACEPDHCG_QOCO_QP_V1\n%d %zu %zu %zu %zu %zu %d %zu %d\n",w->variables,
        f.b.size(),f.h.size(),f.p.values.size(),f.a.values.size(),f.g.values.size(),f.nonnegative,f.soc.size(),int(w->origin_mode));
    const auto& p=w->configured_settings;
    std::fprintf(file,"%d %d %d %u\n%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
        p.max_iters,p.ruiz_iters,p.max_ir_iters,unsigned(p.verbose),p.ir_tol,p.kkt_static_reg_p,
        p.kkt_static_reg_a,p.kkt_static_reg_g,p.kkt_dynamic_reg,p.abstol,p.reltol,p.abstol_inaccurate,p.reltol_inaccurate);
    const auto integers=[&](const std::vector<int>& a) { std::fprintf(file,"%zu",a.size());for(int v:a) std::fprintf(file," %d",v);std::fprintf(file,"\n"); };
    for (const auto* matrix:{&f.p,&f.a,&f.g}) { integers(matrix->offsets);integers(matrix->indices); }
    integers(f.soc);
    const auto numbers=[&](const std::vector<double>& a) { std::fprintf(file,"%zu",a.size());for(double v:a) std::fprintf(file," %.17g",v);std::fprintf(file,"\n"); };
    numbers(original);numbers(shifted);numbers(origin);std::fprintf(file,"%.17g\n",offset);
    const bool failed=std::ferror(file)!=0;
    if (std::fclose(file)!=0 || failed) return SPACEPDHCG_CUDA_RUNTIME_ERROR;
    std::fprintf(stderr,"QOCO_SNAPSHOT id=%llu prior_solves=%llu origin=%d path=%s\n",id,
        static_cast<unsigned long long>(w->report.solves),int(w->origin_mode),path);
    return SPACEPDHCG_CUDA_SUCCESS;
}

spacepdhcg_cuda_status native_qoco_update_solve_impl(
    spacepdhcg_native_qoco* workspace,
    const spacepdhcg_cuda_scvx_problem* problem,
    cudaStream_t stream,
    spacepdhcg_cuda_warm_start_mode requested_warm,
    double* device_primal,
    double* device_dual,
    spacepdhcg_native_qoco_report* report,
    spacepdhcg_native_qoco_consumer consumer=nullptr, void* context=nullptr,
    const int* producer_invalid=nullptr, bool defer=false
) {
    if (workspace == nullptr || problem == nullptr || device_primal == nullptr
        || device_dual == nullptr || report == nullptr) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    if (workspace->deferred_active || workspace->graph_active) return SPACEPDHCG_CUDA_INVALID_STATE;
    if (workspace->origin_mode && requested_warm!=SPACEPDHCG_CUDA_WARM_START_NONE) return SPACEPDHCG_CUDA_INVALID_STATE;
    workspace->numeric_update_invalid=false;
    workspace->validation_flags=0;
    workspace->report.producer_invalid=workspace->report.producer_validation_queued=0;
    const auto enabled=[](const char* name) { const auto* value=std::getenv(name); return value && value[0]=='1'; };
    workspace->queue_validation_allowed=enabled("SPACEPDHCG_TEST_QOCO_DEVICE_VALIDATION")
        && enabled("SPACEPDHCG_TEST_QOCO_NATIVE_NUMERIC_REPLAY") && enabled("SPACEPDHCG_TEST_QOCO_NATIVE_REPLAY")
        && requested_warm==SPACEPDHCG_CUDA_WARM_START_NONE && workspace->queued_numeric_update
        && workspace->finish_numeric_update && workspace->replay_updated;
    // Drain update work on all error/exception exits, before the caller can
    // reuse inputs. Successful replay clears the borrowed result itself.
    struct PendingUpdate {
        spacepdhcg_native_qoco* w;
        cudaStream_t stream;
        ~PendingUpdate() {
            if (w->deferred_active) return;
            if (w->queued_numeric_result) {
                w->finish_numeric_update(w->numeric_update_context,1);
                w->queued_numeric_result=nullptr;
            }
            if (w->validation_pending) cudaStreamSynchronize(stream);
            w->validation_pending=false; w->device_validation=nullptr;
        }
    } pending_update{workspace,stream};
    const auto finish = [&](spacepdhcg_cuda_status status) {
        return copy_native_report(workspace,report,status);
    };
    if (workspace->solver == nullptr) {
        // A previous solver rebuild failed; this workspace is unusable.
        workspace->report.failure = SPACEPDHCG_CUDA_QOCO_FAILURE_ABI;
        return finish(SPACEPDHCG_CUDA_INVALID_STATE);
    }
    if (producer_invalid && (!workspace->queue_validation_allowed
        || workspace->report.solves==0 || conversion_needs_host(workspace))) {
        workspace->validation_pending=true;
        const auto copied=cudaMemcpyAsync(&workspace->validation_flags,producer_invalid,sizeof(int),cudaMemcpyDeviceToHost,stream);
        if (copied==cudaSuccess) { ++workspace->report.d2h_copy_count; workspace->report.d2h_bytes+=sizeof(int); }
        const auto waited=cudaStreamSynchronize(stream);
        workspace->validation_pending=false;
        if (copied!=cudaSuccess || waited!=cudaSuccess) return finish(SPACEPDHCG_CUDA_RUNTIME_ERROR);
        workspace->validation_flags=workspace->validation_flags ? 16 : 0;
        producer_invalid=nullptr;
        if (workspace->validation_flags) {
            workspace->report.failure=SPACEPDHCG_CUDA_QOCO_FAILURE_NUMERICAL;
            workspace->report.status_code=3; workspace->report.iterations=0;
            return finish(SPACEPDHCG_CUDA_NUMERICAL_FAILURE);
        }
    }
    if (workspace->report.solves != 0U || workspace->needs_fresh_solver || workspace->refresh_new_leg) {
        const auto update_start = std::chrono::steady_clock::now();
        auto status = refresh_conversion(workspace, *problem, stream,producer_invalid);
        if (status != SPACEPDHCG_CUDA_SUCCESS) {
            workspace->report.failure =
                status == SPACEPDHCG_CUDA_UNSUPPORTED
                ? SPACEPDHCG_CUDA_QOCO_FAILURE_UNSUPPORTED
                : status == SPACEPDHCG_CUDA_NUMERICAL_FAILURE
                ? SPACEPDHCG_CUDA_QOCO_FAILURE_NUMERICAL
                : SPACEPDHCG_CUDA_QOCO_FAILURE_ABI;
            return finish(status);
        }
        const auto audit_status = qoco_gpu_audit_update_device(workspace->gpu_audit,
            qoco_gpu_conversion_values(workspace->conversion.device), stream);
        if (audit_status != cudaSuccess) return finish(SPACEPDHCG_CUDA_RUNTIME_ERROR);
        for (const auto* values : {&workspace->formulation.p.values, &workspace->formulation.a.values,
                                  &workspace->formulation.g.values, &workspace->formulation.c,
                                  &workspace->formulation.b, &workspace->formulation.h}) if (!values->empty()) {
            ++workspace->report.d2d_copy_count;
            workspace->report.d2d_bytes += values->size() * sizeof(double);
        }
        if (workspace->needs_fresh_solver) {
            workspace->deferred_ready=false;
            workspace->numeric_scale_primed=false;
            workspace->retained_replay_ready=false;
            // The previous solve failed. QOCO carries its best-iterate tracker and
            // the stall-escalated kkt_dynamic_reg across solves, so a numeric update
            // would not give an independent attempt (observed: 101, 62, then 1
            // iteration per solve on the same data). Rebuild the solver instead and
            // report the extra workspace creation.
            if (workspace->numeric_update_context) {
                workspace->destroy_numeric_update(workspace->numeric_update_context);
                workspace->numeric_update_context = nullptr;
            }
            workspace->cleanup(workspace->solver);
            workspace->solver =
                static_cast<SolverAbi*>(std::calloc(1U, sizeof(SolverAbi)));
            if (workspace->solver == nullptr) {
                workspace->report.failure = SPACEPDHCG_CUDA_QOCO_FAILURE_OUT_OF_MEMORY;
                return finish(SPACEPDHCG_CUDA_OUT_OF_MEMORY);
            }
            const int code = setup_solver(workspace);
            if (code != 0) {
                workspace->report.failure =
                    code == 5 ? SPACEPDHCG_CUDA_QOCO_FAILURE_OUT_OF_MEMORY
                              : SPACEPDHCG_CUDA_QOCO_FAILURE_ABI;
                return finish(
                    code == 5 ? SPACEPDHCG_CUDA_OUT_OF_MEMORY
                              : SPACEPDHCG_CUDA_INTERNAL_ERROR
                );
            }
            ++workspace->report.workspace_creations;
            workspace->needs_fresh_solver = false;
            workspace->has_accepted = false;
        } else {
            if (workspace->numeric_update_context) {
                const double* solver_values{};
                if (native_solver_values(workspace,stream,&solver_values)!=SPACEPDHCG_CUDA_SUCCESS)
                    return finish(SPACEPDHCG_CUDA_RUNTIME_ERROR);
                const char* queue=std::getenv("SPACEPDHCG_TEST_QOCO_NATIVE_NUMERIC_REPLAY");
                const char* replay=std::getenv("SPACEPDHCG_TEST_QOCO_NATIVE_REPLAY");
                const bool queued=queue && queue[0]=='1' && replay && replay[0]=='1'
                    && requested_warm==SPACEPDHCG_CUDA_WARM_START_NONE;
                if (queued && (!workspace->queued_numeric_update || !workspace->finish_numeric_update
                    || !workspace->replay_updated)) return finish(SPACEPDHCG_CUDA_UNSUPPORTED);
                int updated=queued
                    ? workspace->queued_numeric_update(workspace->numeric_update_context,
                        solver_values,stream,&workspace->queued_numeric_result)
                    : workspace->device_numeric_update(workspace->numeric_update_context,
                        solver_values,stream);
                // Zero-Ruiz setup has not populated the retained numeric scale
                // packet yet. Prime that packet once using the existing update.
                // Subsequent updates stay queued; setup/priming is explicit.
                if (queued && updated==2 && !workspace->queued_numeric_result && !defer) {
                    if (!collect_validation(workspace,stream)) return finish(SPACEPDHCG_CUDA_RUNTIME_ERROR);
                    const auto validation=canonical_validation_status(workspace->validation_flags);
                    if (validation!=SPACEPDHCG_CUDA_SUCCESS) {
                        workspace->report.failure=validation==SPACEPDHCG_CUDA_NUMERICAL_FAILURE
                            ? SPACEPDHCG_CUDA_QOCO_FAILURE_NUMERICAL : validation==SPACEPDHCG_CUDA_UNSUPPORTED
                            ? SPACEPDHCG_CUDA_QOCO_FAILURE_UNSUPPORTED : SPACEPDHCG_CUDA_QOCO_FAILURE_ABI;
                        return finish(validation);
                    }
                    updated=workspace->device_numeric_update(workspace->numeric_update_context,
                        solver_values,stream);
                    if(updated==0)workspace->numeric_scale_primed=true;
                }
                // A partial enqueue failure may not publish an output pointer.
                if (queued && updated) workspace->finish_numeric_update(workspace->numeric_update_context,1);
                if (updated != 0) {
                    workspace->needs_fresh_solver = true;
                    workspace->report.failure = updated == 3 ? SPACEPDHCG_CUDA_QOCO_FAILURE_NUMERICAL : SPACEPDHCG_CUDA_QOCO_FAILURE_ABI;
                    return finish(updated == 3 ? SPACEPDHCG_CUDA_NUMERICAL_FAILURE : SPACEPDHCG_CUDA_RUNTIME_ERROR);
                }
                ++workspace->report.device_numeric_updates;
                workspace->origin_applied=workspace->origin_mode;
            } else {
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
            }
            // Restore the configured settings: QOCO's stall handler mutates
            // solver->settings->kkt_dynamic_reg in place. A settings validation
            // failure here is an adapter/ABI fault, never a solver outcome.
            if (workspace->update_settings(
                    workspace->solver, &workspace->configured_settings
                ) != 0) {
                workspace->report.failure = SPACEPDHCG_CUDA_QOCO_FAILURE_ABI;
                return finish(SPACEPDHCG_CUDA_INTERNAL_ERROR);
            }
        }
        workspace->report.update_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - update_start
        ).count();
        ++workspace->report.numeric_updates;
        workspace->refresh_new_leg=false;
    }
    if (workspace->origin_mode && !workspace->origin_applied) {
        // Initial/rebuilt solver storage is compiled from the original matrices.
        // Apply the GPU translation before its first numerical solve, once.
        const double* values{};
        if (native_solver_values(workspace,stream,&values)!=SPACEPDHCG_CUDA_SUCCESS)
            return finish(SPACEPDHCG_CUDA_RUNTIME_ERROR);
        const int updated=workspace->device_numeric_update(workspace->numeric_update_context,values,stream);
        if (updated) {
            workspace->needs_fresh_solver=true;
            workspace->report.status_code=updated==3 ? 3 : -1;
            workspace->report.iterations=0;
            workspace->report.failure=updated==3 ? SPACEPDHCG_CUDA_QOCO_FAILURE_NUMERICAL : SPACEPDHCG_CUDA_QOCO_FAILURE_ABI;
            return finish(updated==3 ? SPACEPDHCG_CUDA_NUMERICAL_FAILURE : SPACEPDHCG_CUDA_RUNTIME_ERROR);
        }
        ++workspace->report.device_numeric_updates; workspace->origin_applied=true;
    }
    if (const auto snapshot=snapshot_qoco_problem(workspace,stream); snapshot!=SPACEPDHCG_CUDA_SUCCESS)
        return finish(snapshot);
    if (workspace->configured_settings.verbose != 0) {
        // Diagnostic only (SPACEPDHCG_QOCO_VERBOSE=1): content hash of the data
        // handed to QOCO, so identical-data re-solves can be told from drift.
        std::uint64_t hash = 1469598103934665603ULL;
        const auto mix = [&hash](const std::vector<double>& values) {
            for (const double value : values) {
                std::uint64_t bits{};
                std::memcpy(&bits, &value, sizeof(bits));
                for (int shift = 0; shift < 64; shift += 8) {
                    hash ^= (bits >> shift) & 0xffULL;
                    hash *= 1099511628211ULL;
                }
            }
        };
        mix(workspace->formulation.c);
        mix(workspace->formulation.b);
        mix(workspace->formulation.h);
        mix(workspace->formulation.p.values);
        mix(workspace->formulation.a.values);
        mix(workspace->formulation.g.values);
        // Structural emptiness that QOCO's Ruiz equilibration cannot handle
        // (safe_div(1, 0) = DBL_MAX): all-zero rows of A/G and variables that
        // appear in no row of [P; A; G] with a zero cost.
        const auto zero_rows = [](const Csc& matrix) {
            std::vector<char> touched(static_cast<std::size_t>(matrix.rows), 0);
            for (std::size_t k = 0; k < matrix.values.size(); ++k) {
                if (matrix.values[k] != 0.0) {
                    touched[static_cast<std::size_t>(matrix.indices[k])] = 1;
                }
            }
            return std::count(touched.begin(), touched.end(), 0);
        };
        const auto touched_columns = [](const Csc& matrix, std::vector<char>* out) {
            for (int column = 0; column < matrix.columns; ++column) {
                for (int k = matrix.offsets[column]; k < matrix.offsets[column + 1]; ++k) {
                    if (matrix.values[k] != 0.0) {
                        (*out)[static_cast<std::size_t>(column)] = 1;
                    }
                }
            }
        };
        std::vector<char> column_touched(workspace->formulation.c.size(), 0);
        touched_columns(workspace->formulation.p, &column_touched);
        touched_columns(workspace->formulation.a, &column_touched);
        touched_columns(workspace->formulation.g, &column_touched);
        for (std::size_t j = 0; j < workspace->formulation.c.size(); ++j) {
            if (workspace->formulation.c[j] != 0.0) {
                column_touched[j] = 1;
            }
        }
        // Where the zero G rows come from and what their offsets are.
        std::vector<char> g_touched(static_cast<std::size_t>(workspace->formulation.g.rows), 0);
        for (std::size_t k = 0; k < workspace->formulation.g.values.size(); ++k) {
            if (workspace->formulation.g.values[k] != 0.0) {
                g_touched[static_cast<std::size_t>(workspace->formulation.g.indices[k])] = 1;
            }
        }
        long zero_scalar = 0, zero_variable = 0, zero_affine = 0, zero_cone = 0;
        double h_min = std::numeric_limits<double>::infinity();
        double h_max = -std::numeric_limits<double>::infinity();
        for (std::size_t row = 0; row < g_touched.size(); ++row) {
            if (g_touched[row] != 0) {
                continue;
            }
            const auto& map = workspace->formulation.conic_map[row];
            (map.source == Source::scalar ? zero_scalar
             : map.source == Source::variable ? zero_variable
             : map.source == Source::affine ? zero_affine : zero_cone) += 1;
            h_min = std::min(h_min, workspace->formulation.h[row]);
            h_max = std::max(h_max, workspace->formulation.h[row]);
        }
        std::fprintf(
            stderr,
            "{\"case\":\"qoco_formulation\",\"solve_ordinal\":%llu,"
            "\"data_fnv1a64\":\"%016llx\",\"fresh_solver\":%d,"
            "\"zero_rows_a\":%ld,\"zero_rows_g\":%ld,\"zero_columns\":%ld,"
            "\"zero_g_by_source\":{\"scalar\":%ld,\"variable\":%ld,\"affine_cone\":%ld,"
            "\"variable_cone\":%ld},\"zero_g_h_range\":[%.6g,%.6g],\"ruiz_iters\":%d}\n",
            static_cast<unsigned long long>(workspace->report.solves),
            static_cast<unsigned long long>(hash),
            workspace->report.numeric_updates == 0U ? 1 : 0,
            static_cast<long>(zero_rows(workspace->formulation.a)),
            static_cast<long>(zero_rows(workspace->formulation.g)),
            static_cast<long>(std::count(column_touched.begin(), column_touched.end(), 0)),
            zero_scalar, zero_variable, zero_affine, zero_cone, h_min, h_max,
            workspace->configured_settings.ruiz_iters
        );
    }
    const bool warm = workspace->has_accepted
        && requested_warm != SPACEPDHCG_CUDA_WARM_START_NONE;
    if (workspace->primal_start) {
        if (workspace->primal_start(workspace->solver, warm ? 1 : 0) != 0)
            return finish(SPACEPDHCG_CUDA_INVALID_STATE);
    } else {
        workspace->set_x0(workspace->solver, warm ? workspace->accepted_primal.data() : nullptr);
    }
    workspace->report.warm_primal_accepted = warm ? 1 : 0;
    workspace->report.dual_discarded =
        warm
        && (requested_warm == SPACEPDHCG_CUDA_WARM_START_PRIMAL_DUAL
            || requested_warm == SPACEPDHCG_CUDA_WARM_START_FULL_RETAINED)
        ? 1 : 0;
    const auto solve_start = std::chrono::steady_clock::now();
    int status_code = 0;
    QocoAuditResult replay_audit{};
    bool replay_audited=false;
    // The existing warm-inaccurate cold retry consumes mutated host solver
    // settings. Retain that path until its retry policy is device-controlled.
    const bool invoked = warm ? solve_with_reduction_scope(workspace,&status_code)
        : solve_with_device_audit(workspace,stream,device_primal,device_dual,
            &status_code,&replay_audit,&replay_audited,consumer,context,defer);
    if (!replay_audited && !workspace->deferred_active) workspace->report.solve_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - solve_start
    ).count();
    if (workspace->deferred_active) return SPACEPDHCG_CUDA_SUCCESS;
    return finish_native_solve(workspace,problem,stream,device_primal,device_dual,report,
        consumer,context,warm,status_code,replay_audit,replay_audited,invoked);
}

spacepdhcg_cuda_status spacepdhcg_native_qoco_update_solve_with_consumer(
    spacepdhcg_native_qoco* w, const spacepdhcg_cuda_scvx_problem* p, cudaStream_t stream,
    double* primal, double* dual, spacepdhcg_native_qoco_report* report,
    spacepdhcg_native_qoco_consumer consumer, void* context) {
    return spacepdhcg_native_qoco_update_solve_with_input_guard(w,p,stream,primal,dual,report,consumer,context,nullptr);
}
spacepdhcg_cuda_status spacepdhcg_native_qoco_update_solve_with_input_guard(
    spacepdhcg_native_qoco* w, const spacepdhcg_cuda_scvx_problem* p, cudaStream_t stream,
    double* primal, double* dual, spacepdhcg_native_qoco_report* report,
    spacepdhcg_native_qoco_consumer consumer, void* context, const int* producer_invalid) {
    if (report) report->producer_invalid=report->producer_validation_queued=0;
    try {
        return native_qoco_update_solve_impl(w,p,stream,SPACEPDHCG_CUDA_WARM_START_NONE,
            primal,dual,report,consumer,context,producer_invalid);
    } catch (const std::bad_alloc&) {
        cudaStreamSynchronize(stream);
        return SPACEPDHCG_CUDA_OUT_OF_MEMORY;
    } catch (...) {
        cudaStreamSynchronize(stream);
        return SPACEPDHCG_CUDA_INTERNAL_ERROR;
    }
}

spacepdhcg_cuda_status spacepdhcg_native_qoco_create(
    const spacepdhcg_cuda_scvx_problem* problem,
    cudaStream_t stream,
    int ruiz_iterations,
    spacepdhcg_native_qoco** workspace
) {
    try {
        return native_qoco_create_impl(problem, stream, ruiz_iterations, workspace, 1.0e-8, false);
    } catch (const std::bad_alloc&) {
        return SPACEPDHCG_CUDA_OUT_OF_MEMORY;
    } catch (...) {
        return SPACEPDHCG_CUDA_INTERNAL_ERROR;
    }
}

spacepdhcg_cuda_status spacepdhcg_native_qoco_create_configured(
    const spacepdhcg_cuda_scvx_problem* problem, cudaStream_t stream,
    int ruiz_iterations, double tolerance, bool require_device_extensions,
    spacepdhcg_native_qoco** workspace
) {
    try {
        return native_qoco_create_impl(problem, stream, ruiz_iterations, workspace,
            tolerance, require_device_extensions);
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

spacepdhcg_cuda_status spacepdhcg_native_qoco_accept(
    spacepdhcg_native_qoco* workspace, spacepdhcg_native_qoco_report* report) {
    if (workspace && (workspace->deferred_active || workspace->graph_active)) return SPACEPDHCG_CUDA_INVALID_STATE;
    if (workspace && workspace->origin_mode) return SPACEPDHCG_CUDA_INVALID_STATE;
    if (workspace != nullptr) {
        if (workspace->primal_start) {
            if (workspace->primal_start(workspace->solver, 2) != 0)
                return SPACEPDHCG_CUDA_RUNTIME_ERROR;
            ++workspace->report.d2d_copy_count;
            workspace->report.d2d_bytes += workspace->variables * sizeof(double);
            // The final accepted step may have no subsequent solve to refresh
            // the caller's report. Include this transfer immediately as well.
            if (report) {
                ++report->d2d_copy_count;
                report->d2d_bytes += workspace->variables * sizeof(double);
            }
        } else {
            workspace->accepted_primal = workspace->primal;
        }
        workspace->has_accepted = true;
    }
    return SPACEPDHCG_CUDA_SUCCESS;
}

spacepdhcg_cuda_status spacepdhcg_native_qoco_reset_warm_state(
    spacepdhcg_native_qoco* workspace,
    const bool retain_primal
) {
    if (workspace == nullptr) {
        return SPACEPDHCG_CUDA_SUCCESS;
    }
    if (workspace->deferred_active || workspace->graph_active) return SPACEPDHCG_CUDA_INVALID_STATE;
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
    if (workspace->solver != nullptr && workspace->primal_start) {
        if (workspace->primal_start(workspace->solver,
                retain_primal && workspace->has_accepted ? 1 : 0) != 0)
            return SPACEPDHCG_CUDA_INVALID_STATE;
    } else if (workspace->solver != nullptr && workspace->set_x0 != nullptr) {
        workspace->set_x0(
            workspace->solver,
            retain_primal && workspace->has_accepted
                ? workspace->accepted_primal.data()
                : nullptr
        );
    }
    return SPACEPDHCG_CUDA_SUCCESS;
}

void spacepdhcg_native_qoco_destroy(spacepdhcg_native_qoco* workspace) {
    if (workspace && workspace->graph_active) { workspace->destroy_after_graph=true; return; }
    delete workspace;
}

spacepdhcg_cuda_status spacepdhcg_native_qoco_restart_leg(spacepdhcg_native_qoco* w) {
    int device=-1;
    if(!w || w->deferred_active || w->graph_active || w->destroy_after_graph
        || std::this_thread::get_id()!=w->creation_owner || cudaGetDevice(&device)!=cudaSuccess
        || device!=w->creation_device) return SPACEPDHCG_CUDA_INVALID_STATE;
    // Scaled reuse requires the actual workspace policy, not merely an env flag
    // that an older library may ignore. Unit objective scaling removes the old
    // objective's influence on the next leg's Ruiz cost normalization.
    const auto* scaled=std::getenv("SPACEPDHCG_TEST_GTOC12_SCALED_QOCO_POOL");
    if(w->configured_settings.ruiz_iters &&
        (!scaled || scaled[0]!='1' || !w->numeric_preserves_objective
         || !w->numeric_update_context || w->numeric_preserves_objective(w->numeric_update_context)!=1))
        return SPACEPDHCG_CUDA_UNSUPPORTED;
    if(!w->solver || !w->numeric_update_context
        || !w->finish_replay || !w->finish_numeric_update || !w->update_settings
        || !w->primal_start || !w->emit_graph) return SPACEPDHCG_CUDA_UNSUPPORTED;
    if(w->finish_replay(w->solver) || w->finish_numeric_update(w->numeric_update_context,1)
        || w->update_settings(w->solver,&w->configured_settings) || w->primal_start(w->solver,0))
        return SPACEPDHCG_CUDA_RUNTIME_ERROR;
    // The retained IPM graph initializes x/y/z/s, best/stall controls and all
    // iteration counters before solving. Do not carry an accepted warm start.
    w->needs_fresh_solver=false;w->refresh_new_leg=true;w->has_accepted=false;
    w->deferred_ready=false;w->numeric_scale_primed=false;w->origin_applied=false;
    const auto* retained=std::getenv("SPACEPDHCG_TEST_QOCO_RETAINED_REPLAY");
    w->retained_replay_ready=retained && retained[0]=='1';
    w->queued_numeric_result=nullptr;w->numeric_update_invalid=false;
    w->validation_pending=false;w->device_validation=nullptr;w->validation_flags=0;
    w->report={};w->report.status_code=-1;
    return SPACEPDHCG_CUDA_SUCCESS;
}

bool spacepdhcg_native_qoco_can_enqueue(const spacepdhcg_native_qoco* w) {
    const auto enabled=[](const char* name) { const auto* v=std::getenv(name); return v && v[0]=='1'; };
    return w && !w->deferred_active && !w->graph_active && w->deferred_ready && w->solver && w->report.solves
        && !conversion_needs_host(w) && w->replay_updated && w->finish_replay && w->finish_numeric_update
        && w->queued_numeric_update && w->primal_start && w->set_device_io
        && enabled("SPACEPDHCG_TEST_QOCO_DEVICE_VALIDATION")
        && enabled("SPACEPDHCG_TEST_QOCO_NATIVE_NUMERIC_REPLAY")
        && enabled("SPACEPDHCG_TEST_QOCO_NATIVE_REPLAY") && enabled("SPACEPDHCG_TEST_QOCO_IPM_GRAPH");
}
spacepdhcg_cuda_status spacepdhcg_native_qoco_enqueue(
    spacepdhcg_native_qoco* w, const spacepdhcg_cuda_scvx_problem* p, cudaStream_t stream,
    double* primal, double* dual, spacepdhcg_native_qoco_consumer consumer,
    void* context, const int* producer_invalid) {
    if (w && w->deferred_active) return SPACEPDHCG_CUDA_INVALID_STATE;
    if (!spacepdhcg_native_qoco_can_enqueue(w)) return SPACEPDHCG_CUDA_UNSUPPORTED;
    cudaStreamCaptureStatus capture{};
    if (cudaStreamIsCapturing(stream,&capture)!=cudaSuccess || capture!=cudaStreamCaptureStatusNone)
        return SPACEPDHCG_CUDA_INVALID_STATE;
    spacepdhcg_native_qoco_report unused{};
    try {
        const auto code=native_qoco_update_solve_impl(w,p,stream,SPACEPDHCG_CUDA_WARM_START_NONE,
            primal,dual,&unused,consumer,context,producer_invalid,true);
        if (code==SPACEPDHCG_CUDA_SUCCESS && !w->deferred_active) return SPACEPDHCG_CUDA_INTERNAL_ERROR;
        return code;
    } catch (...) {
        cudaStreamSynchronize(stream);
        return SPACEPDHCG_CUDA_INTERNAL_ERROR;
    }
}
spacepdhcg_cuda_status spacepdhcg_native_qoco_finish(
    spacepdhcg_native_qoco* w, cudaStream_t stream, spacepdhcg_native_qoco_report* report) {
    if (!w || !report) return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    int device=-1;
    if (!w->deferred_active || stream!=w->deferred_stream || cudaGetDevice(&device)!=cudaSuccess
        || device!=w->deferred_device || std::this_thread::get_id()!=w->deferred_owner)
        return SPACEPDHCG_CUDA_INVALID_STATE;
    int status=-1; QocoAuditResult audit{};
    const bool collected=collect_device_replay(w,stream,w->deferred_status,&status,&audit);
    if (!collected) {
        cudaStreamSynchronize(stream); w->finish_replay(w->solver);
        w->finish_numeric_update(w->numeric_update_context,0);
        w->queued_numeric_result=nullptr; w->validation_pending=false; w->device_validation=nullptr;
        w->needs_fresh_solver=true;
    }
    w->deferred_active=false; w->deferred_status=nullptr;
    // Replay already emitted the audit and consumer. No host oracle or warm
    // retry is eligible for this API, so no borrowed problem/host callback is retained.
    return finish_native_solve(w,nullptr,stream,nullptr,nullptr,report,nullptr,nullptr,
        false,status,audit,true,collected);
}

spacepdhcg_cuda_status spacepdhcg_native_qoco_set_origin(
    spacepdhcg_native_qoco* w,const double* origin,int count,cudaStream_t stream) {
    if (!w || !origin || count<1 || count>w->variables) return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    if (w->deferred_active || w->graph_active) return SPACEPDHCG_CUDA_INVALID_STATE;
    if (!w->numeric_update_context || !w->device_numeric_update || !w->device_solution || !w->set_device_io)
        return SPACEPDHCG_CUDA_UNSUPPORTED;
    const auto allocated=qoco_gpu_audit_enable_origin(w->gpu_audit);
    if (allocated!=cudaSuccess) return allocated==cudaErrorMemoryAllocation
        ? SPACEPDHCG_CUDA_OUT_OF_MEMORY : SPACEPDHCG_CUDA_RUNTIME_ERROR;
    if (w->set_stopping_origin && w->set_stopping_origin(w->solver,
        qoco_gpu_audit_origin(w->gpu_audit),qoco_gpu_audit_origin_offset(w->gpu_audit)) != 0)
        return SPACEPDHCG_CUDA_INVALID_STATE;
    const auto copied=qoco_gpu_audit_set_origin(w->gpu_audit,origin,count,stream);
    if (copied!=cudaSuccess) { cudaStreamSynchronize(stream); return SPACEPDHCG_CUDA_RUNTIME_ERROR; }
    w->origin_mode=true; w->origin_applied=false;
    ++w->report.d2d_copy_count; w->report.d2d_bytes+=count*sizeof(double);
    return SPACEPDHCG_CUDA_SUCCESS;
}

namespace {
bool graph_owner(const spacepdhcg_native_qoco* w,cudaStream_t stream) {
    int device=-1;
    return w && w->graph_active && stream==w->graph_stream
        && std::this_thread::get_id()==w->creation_owner
        && cudaGetDevice(&device)==cudaSuccess && device==w->creation_device;
}
}
spacepdhcg_cuda_status spacepdhcg_native_qoco_begin_graph(
    spacepdhcg_native_qoco* w,cudaStream_t stream,const QocoGraphProgress** progress) {
    if (progress) *progress=nullptr;
    if (!w || !progress || !stream || stream==cudaStreamLegacy) return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    // Snapshotting synchronizes and writes host files; never silently omit graph iterations.
    const char* snapshot_directory=std::getenv("SPACEPDHCG_QOCO_SNAPSHOT_DIRECTORY");
    if (snapshot_directory && *snapshot_directory) return SPACEPDHCG_CUDA_UNSUPPORTED;
    if (w->graph_active || w->deferred_active || std::this_thread::get_id()!=w->creation_owner)
        return SPACEPDHCG_CUDA_INVALID_STATE;
    int device=-1;cudaStreamCaptureStatus capture{};
    if (cudaGetDevice(&device)!=cudaSuccess || device!=w->creation_device
        || cudaStreamIsCapturing(stream,&capture)!=cudaSuccess || capture!=cudaStreamCaptureStatusNone)
        return SPACEPDHCG_CUDA_INVALID_STATE;
    if (!spacepdhcg_native_qoco_can_enqueue(w) || !w->emit_graph || !w->capture_numeric_update)
        return SPACEPDHCG_CUDA_UNSUPPORTED;
    if (!w->graph_progress && cudaMalloc(&w->graph_progress,sizeof(QocoGraphProgress))!=cudaSuccess)
        return SPACEPDHCG_CUDA_OUT_OF_MEMORY;
    if (cudaMemsetAsync(w->graph_progress,0,sizeof(QocoGraphProgress),stream)!=cudaSuccess
        || cudaStreamSynchronize(stream)!=cudaSuccess) return SPACEPDHCG_CUDA_RUNTIME_ERROR;
    w->graph_active=true;w->graph_stream=stream;*progress=w->graph_progress;
    return SPACEPDHCG_CUDA_SUCCESS;
}

spacepdhcg_cuda_status spacepdhcg_native_qoco_emit_graph(
    spacepdhcg_native_qoco* w,const spacepdhcg_cuda_scvx_problem* problem,cudaGraph_t graph,
    const cudaGraphNode_t* dependencies,size_t count,cudaStream_t stream,double* primal,double* dual,
    const double* origin,int origin_count,const int* producer_invalid,
    spacepdhcg_native_qoco_consumer consumer,void* context,cudaGraphNode_t* completion) {
    if (completion) *completion=nullptr;
    if (!problem || !graph || !primal || !dual || !completion || (count && !dependencies))
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    if (!graph_owner(w,stream) || w->destroy_after_graph) return SPACEPDHCG_CUDA_INVALID_STATE;
    if (w->origin_mode ? (!origin || origin_count<1 || origin_count>w->variables) : (origin || origin_count))
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    // Recording must not count as execution or leave imperative pending flags.
    struct Restore {
        spacepdhcg_native_qoco* w;
        spacepdhcg_native_qoco_report report;
        ~Restore() { w->report=report;w->validation_pending=false;w->device_validation=nullptr; }
    } restore{w,w->report};
    try {
        w->queue_validation_allowed=true;
        const double* numeric{};const int* validation{};
        std::uint64_t copies=0,bytes=0;
        cudaGraphNode_t prefix{};
        auto error=spacepdhcg_graph_append(stream,graph,dependencies,count,[&]() {
            if (w->origin_mode) {
                auto e=qoco_gpu_audit_set_origin(w->gpu_audit,origin,origin_count,stream);
                if(e!=cudaSuccess) return e;
                ++copies;bytes+=origin_count*sizeof(double);
            }
            if (refresh_conversion(w,*problem,stream,producer_invalid)!=SPACEPDHCG_CUDA_SUCCESS)
                return cudaErrorInvalidValue;
            validation=w->device_validation;
            auto e=qoco_gpu_audit_update_device(w->gpu_audit,qoco_gpu_conversion_values(w->conversion.device),stream);
            if(e!=cudaSuccess) return e;
            for (const auto* values : {&w->formulation.p.values,&w->formulation.a.values,&w->formulation.g.values,
                    &w->formulation.c,&w->formulation.b,&w->formulation.h}) if(!values->empty()) {
                ++copies;bytes+=values->size()*sizeof(double);
            }
            const double* values{};
            if(native_solver_values(w,stream,&values)!=SPACEPDHCG_CUDA_SUCCESS) return cudaErrorUnknown;
            copies+=w->report.d2d_copy_count-restore.report.d2d_copy_count;
            bytes+=w->report.d2d_bytes-restore.report.d2d_bytes;
            const int updated=w->capture_numeric_update(w->numeric_update_context,values,stream,&numeric);
            if(updated || !numeric) return cudaErrorInvalidValue;
            return qoco_gpu_conversion_guard_numeric(w->conversion.device,numeric,stream,&numeric);
        },&prefix);
        if(error!=cudaSuccess) return SPACEPDHCG_CUDA_RUNTIME_ERROR;
        ReplayOutputAbi output{};cudaGraphNode_t solved{};
        const int emitted=w->emit_graph(w->solver,graph,&prefix,1,numeric,&output,&solved);
        if(emitted) return emitted==2 ? SPACEPDHCG_CUDA_INVALID_STATE : SPACEPDHCG_CUDA_RUNTIME_ERROR;
        if(!output.completion || output.n!=w->variables || output.p!=int(w->formulation.b.size())
            || output.m!=int(w->formulation.h.size())) return SPACEPDHCG_CUDA_INTERNAL_ERROR;
        error=spacepdhcg_graph_append(stream,graph,&solved,1,[&]() {
            const QocoAuditResult* audit{};const QocoReplayStatus* status{};
            if(w->origin_mode) {
                auto e=qoco_gpu_audit_reconstruct(w->gpu_audit,output.x,primal,stream);
                if(e!=cudaSuccess) return e;
            }
            auto e=qoco_gpu_audit_run_device(w->gpu_audit,w->origin_mode ? primal : output.x,
                output.y,output.z,dual,stream,&audit);
            if(e!=cudaSuccess) return e;
            e=qoco_gpu_audit_replay_status(w->gpu_audit,reinterpret_cast<const int*>(output.completion),stream,&status);
            if(e!=cudaSuccess) return e;
            if(!w->origin_mode) {
                e=cudaMemcpyAsync(primal,output.x,w->variables*sizeof(double),cudaMemcpyDeviceToDevice,stream);
                if(e!=cudaSuccess) return e;
                ++copies;bytes+=w->variables*sizeof(double);
            }
            e=qoco_gpu_graph_record(w->graph_progress,status,audit,validation,copies,bytes,stream);
            if(e!=cudaSuccess) return e;
            return consumer ? consumer(context,status,audit,stream) : cudaSuccess;
        },completion);
        return error==cudaSuccess ? SPACEPDHCG_CUDA_SUCCESS : SPACEPDHCG_CUDA_RUNTIME_ERROR;
    } catch(const std::bad_alloc&) { return SPACEPDHCG_CUDA_OUT_OF_MEMORY; }
    catch(...) { return SPACEPDHCG_CUDA_INTERNAL_ERROR; }
}

spacepdhcg_cuda_status spacepdhcg_native_qoco_end_graph(
    spacepdhcg_native_qoco* w,cudaStream_t stream,spacepdhcg_native_qoco_report* report) {
    if (!report) return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    if (!graph_owner(w,stream)) return SPACEPDHCG_CUDA_INVALID_STATE;
    QocoGraphProgress progress{};
    const auto copied=cudaMemcpyAsync(&progress,w->graph_progress,sizeof(progress),cudaMemcpyDeviceToHost,stream);
    const auto waited=cudaStreamSynchronize(stream);
    w->graph_active=false;w->deferred_ready=false;w->needs_fresh_solver=true;
    w->has_accepted=false;w->origin_applied=false;
    auto code=SPACEPDHCG_CUDA_RUNTIME_ERROR;
    if(copied==cudaSuccess && waited==cudaSuccess) {
        ++w->report.d2h_copy_count;w->report.d2h_bytes+=sizeof(progress);
        w->report.graph_attempts+=progress.attempts;w->report.graph_iterations+=progress.iterations;
        w->report.numeric_updates+=progress.attempts;w->report.device_numeric_updates+=progress.attempts;
        w->report.solves+=progress.solver_runs;
        w->report.d2d_copy_count+=progress.d2d_count;w->report.d2d_bytes+=progress.d2d_bytes;
        if(progress.attempts) {
            w->report.status_code=progress.last_status.status;w->report.iterations=progress.last_status.iterations;
            const auto& a=progress.last_audit;
            w->report.primal_residual=a.primal;w->report.dual_residual=a.dual;
            w->report.absolute_primal_residual=a.absolute_primal;w->report.absolute_dual_residual=a.absolute_dual;
            w->report.dual_cone_residual=a.dual_cone;w->report.complementarity_residual=a.complementarity;
            w->validation_flags=progress.last_validation;
            w->report.failure=progress.last_validation || progress.last_status.status<1 || progress.last_status.status>2
                ? SPACEPDHCG_CUDA_QOCO_FAILURE_NUMERICAL : SPACEPDHCG_CUDA_QOCO_FAILURE_NONE;
        }
        code=SPACEPDHCG_CUDA_SUCCESS; // Collected; caller must still qualify each result.
    }
    copy_native_report(w,report,code);
    if(w->destroy_after_graph) delete w;
    return code;
}
