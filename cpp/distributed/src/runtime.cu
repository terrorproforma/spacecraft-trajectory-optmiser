/*
 * Persistent MPI/NCCL ownership and local distributed algebra for Gate G5.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "spacepdhcg/distributed/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <utility>

namespace spacepdhcg::distributed::g5 {
namespace {

[[noreturn]] void throw_cuda(cudaError_t status, std::string_view operation) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(status)
    );
}

[[noreturn]] void throw_nccl(ncclResult_t status, std::string_view operation) {
    throw std::runtime_error(
        std::string(operation) + ": " + ncclGetErrorString(status)
    );
}

[[noreturn]] void throw_mpi(int status, std::string_view operation) {
    std::array<char, MPI_MAX_ERROR_STRING> buffer{};
    int length = 0;
    MPI_Error_string(status, buffer.data(), &length);
    throw std::runtime_error(
        std::string(operation) + ": "
        + std::string(buffer.data(), static_cast<std::size_t>(length))
    );
}

void check_cuda(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) {
        throw_cuda(status, operation);
    }
}

void check_nccl(ncclResult_t status, std::string_view operation) {
    if (status != ncclSuccess) {
        throw_nccl(status, operation);
    }
}

void check_mpi(int status, std::string_view operation) {
    if (status != MPI_SUCCESS) {
        throw_mpi(status, operation);
    }
}

__global__ void csc_forward_kernel(
    int rows,
    int columns,
    const int* offsets,
    const int* indices,
    const double* values,
    const double* input,
    double* output
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }
    double sum = 0.0;
    for (int column = 0; column < columns; ++column) {
        for (int position = offsets[column]; position < offsets[column + 1]; ++position) {
            if (indices[position] == row) {
                sum += values[position] * input[column];
            }
        }
    }
    output[row] = sum;
}

__global__ void csc_transpose_kernel(
    int columns,
    const int* offsets,
    const int* indices,
    const double* values,
    const double* input,
    double* output
) {
    const int column = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (column >= columns) {
        return;
    }
    double sum = 0.0;
    for (int position = offsets[column]; position < offsets[column + 1]; ++position) {
        sum += values[position] * input[indices[position]];
    }
    output[column] = sum;
}

__global__ void project_soc_kernel(
    double* values,
    const int* starts,
    const int* dimensions,
    int cone_count
) {
    const int cone = static_cast<int>(blockIdx.x);
    if (cone >= cone_count || threadIdx.x != 0) {
        return;
    }
    const int start = starts[cone];
    const int dimension = dimensions[cone];
    if (dimension <= 0) {
        return;
    }
    const double scalar = values[start];
    double norm_squared = 0.0;
    for (int component = 1; component < dimension; ++component) {
        const double value = values[start + component];
        norm_squared += value * value;
    }
    const double norm = sqrt(norm_squared);
    if (norm <= scalar) {
        return;
    }
    if (norm <= -scalar) {
        for (int component = 0; component < dimension; ++component) {
            values[start + component] = 0.0;
        }
        return;
    }
    const double projected_scalar = 0.5 * (norm + scalar);
    const double scale = norm > 0.0 ? projected_scalar / norm : 0.0;
    values[start] = projected_scalar;
    for (int component = 1; component < dimension; ++component) {
        values[start + component] *= scale;
    }
}

}  // namespace

MpiNcclRuntime::MpiNcclRuntime(RuntimeOptions options) : options_(options) {
    int initialised = 0;
    check_mpi(MPI_Initialized(&initialised), "MPI_Initialized");
    if (initialised == 0) {
        throw std::invalid_argument("MPI must be initialised before the distributed runtime");
    }

    try {
        check_mpi(MPI_Comm_dup(options_.communicator, &communicator_), "MPI_Comm_dup");
        check_mpi(
            MPI_Comm_set_errhandler(communicator_, MPI_ERRORS_RETURN),
            "MPI_Comm_set_errhandler"
        );
        check_mpi(MPI_Comm_rank(communicator_, &telemetry_.rank), "MPI_Comm_rank");
        check_mpi(MPI_Comm_size(communicator_, &telemetry_.world_size), "MPI_Comm_size");
        check_mpi(
            MPI_Comm_split_type(
                communicator_,
                MPI_COMM_TYPE_SHARED,
                telemetry_.rank,
                MPI_INFO_NULL,
                &local_communicator_
            ),
            "MPI_Comm_split_type"
        );
        check_mpi(
            MPI_Comm_rank(local_communicator_, &telemetry_.local_rank),
            "MPI local rank"
        );
        int local_size = 0;
        check_mpi(MPI_Comm_size(local_communicator_, &local_size), "MPI local size");

        int device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count <= 0 || local_size > device_count) {
            throw std::runtime_error(
                "one MPI rank per physical GPU is required; MPS and rank oversubscription are unsupported"
            );
        }
        telemetry_.device = telemetry_.local_rank;
        telemetry_.deterministic = options_.deterministic;
        telemetry_.overlap_enabled = options_.enable_overlap;
        check_cuda(cudaSetDevice(telemetry_.device), "cudaSetDevice");
        check_cuda(
            cudaStreamCreateWithFlags(&compute_stream_, cudaStreamNonBlocking),
            "create compute stream"
        );
        if (options_.enable_overlap) {
            check_cuda(
                cudaStreamCreateWithFlags(&collective_stream_, cudaStreamNonBlocking),
                "create collective stream"
            );
        }
        check_cuda(
            cudaEventCreateWithFlags(&local_ready_, cudaEventDisableTiming),
            "create local-ready event"
        );
        check_cuda(
            cudaEventCreateWithFlags(&collective_complete_, cudaEventDisableTiming),
            "create collective-complete event"
        );

        ncclUniqueId identifier{};
        if (telemetry_.rank == 0) {
            check_nccl(ncclGetUniqueId(&identifier), "ncclGetUniqueId");
        }
        check_mpi(
            MPI_Bcast(
                &identifier,
                static_cast<int>(sizeof(identifier)),
                MPI_BYTE,
                0,
                communicator_
            ),
            "MPI_Bcast NCCL identifier"
        );
        check_nccl(
            ncclCommInitRank(
                &nccl_communicator_,
                telemetry_.world_size,
                identifier,
                telemetry_.rank
            ),
            "ncclCommInitRank"
        );
        state_ = RuntimeState::created;
    } catch (...) {
        fail("distributed runtime creation failed");
        if (nccl_communicator_ != nullptr) {
            ncclCommDestroy(nccl_communicator_);
            nccl_communicator_ = nullptr;
        }
        if (collective_complete_ != nullptr) {
            cudaEventDestroy(collective_complete_);
            collective_complete_ = nullptr;
        }
        if (local_ready_ != nullptr) {
            cudaEventDestroy(local_ready_);
            local_ready_ = nullptr;
        }
        if (collective_stream_ != nullptr) {
            cudaStreamDestroy(collective_stream_);
            collective_stream_ = nullptr;
        }
        if (compute_stream_ != nullptr) {
            cudaStreamDestroy(compute_stream_);
            compute_stream_ = nullptr;
        }
        if (local_communicator_ != MPI_COMM_NULL) {
            MPI_Comm_free(&local_communicator_);
        }
        if (communicator_ != MPI_COMM_NULL) {
            MPI_Comm_free(&communicator_);
        }
        throw;
    }
}

MpiNcclRuntime::~MpiNcclRuntime() {
    if (state_ != RuntimeState::cancelled && compute_stream_ != nullptr) {
        cudaStreamSynchronize(compute_stream_);
    }
    for (auto& timing : pending_timings_) {
        if (timing.start != nullptr) {
            cudaEventDestroy(timing.start);
        }
        if (timing.stop != nullptr) {
            cudaEventDestroy(timing.stop);
        }
    }
    pending_timings_.clear();
    if (nccl_communicator_ != nullptr) {
        ncclCommDestroy(nccl_communicator_);
        nccl_communicator_ = nullptr;
    }
    if (collective_complete_ != nullptr) {
        cudaEventDestroy(collective_complete_);
    }
    if (local_ready_ != nullptr) {
        cudaEventDestroy(local_ready_);
    }
    if (collective_stream_ != nullptr) {
        cudaStreamDestroy(collective_stream_);
    }
    if (compute_stream_ != nullptr) {
        cudaStreamDestroy(compute_stream_);
    }
    if (local_communicator_ != MPI_COMM_NULL) {
        MPI_Comm_free(&local_communicator_);
    }
    if (communicator_ != MPI_COMM_NULL) {
        MPI_Comm_free(&communicator_);
    }
    state_ = RuntimeState::destroyed;
}

void MpiNcclRuntime::allreduce_sum(
    double* device_values,
    std::size_t count,
    CollectiveKind kind,
    std::uint64_t frequency,
    std::string_view purpose
) {
    allreduce(device_values, count, ncclSum, kind, frequency, purpose);
}

void MpiNcclRuntime::allreduce_max(
    double* device_values,
    std::size_t count,
    CollectiveKind kind,
    std::uint64_t frequency,
    std::string_view purpose
) {
    allreduce(device_values, count, ncclMax, kind, frequency, purpose);
}

void MpiNcclRuntime::allreduce(
    double* device_values,
    std::size_t count,
    ncclRedOp_t operation,
    CollectiveKind kind,
    std::uint64_t frequency,
    std::string_view purpose
) {
    if (state_ == RuntimeState::failed || state_ == RuntimeState::cancelled
        || state_ == RuntimeState::destroyed) {
        throw std::logic_error("collective requested from a terminal runtime state");
    }
    if ((count > 0 && device_values == nullptr) || frequency == 0 || purpose.empty()) {
        throw std::invalid_argument("collective payload, frequency, and purpose must be explicit");
    }
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("collective count exceeds the supported NCCL range");
    }

    auto telemetry_iterator = std::find_if(
        telemetry_.collectives.begin(),
        telemetry_.collectives.end(),
        [kind, frequency, purpose](const CollectiveTelemetry& entry) {
            return entry.kind == kind && entry.frequency == frequency
                   && entry.purpose == purpose;
        }
    );
    if (telemetry_iterator == telemetry_.collectives.end()) {
        telemetry_.collectives.push_back(
            CollectiveTelemetry{kind, 0, 0, 0, 0, frequency, std::string(purpose)}
        );
        telemetry_iterator = std::prev(telemetry_.collectives.end());
    }
    const auto telemetry_index = static_cast<std::size_t>(
        std::distance(telemetry_.collectives.begin(), telemetry_iterator)
    );
    const auto payload = static_cast<std::uint64_t>(count * sizeof(double));
    ++telemetry_iterator->call_count;
    telemetry_iterator->element_count += count;
    telemetry_iterator->payload_bytes += payload;
    telemetry_iterator->wire_bytes_estimate +=
        telemetry_.world_size > 1
            ? 2ULL * static_cast<std::uint64_t>(telemetry_.world_size - 1) * payload
                  / static_cast<std::uint64_t>(telemetry_.world_size)
            : 0ULL;
    if (count == 0) {
        return;
    }

    PendingTiming timing{telemetry_index, nullptr, nullptr};
    try {
        ordering_.begin();
        check_cuda(
            cudaEventRecord(local_ready_, compute_stream_),
            "record local-ready event"
        );
        ordering_.collective_wait();
        check_cuda(
            cudaStreamWaitEvent(collective_stream(), local_ready_, 0),
            "collective stream wait"
        );

        check_cuda(cudaEventCreate(&timing.start), "create collective start timing");
        check_cuda(cudaEventCreate(&timing.stop), "create collective stop timing");
        check_cuda(cudaEventRecord(timing.start, collective_stream()), "record collective start");
        ordering_.enqueue();
        check_nccl(
            ncclAllReduce(
                device_values,
                device_values,
                count,
                ncclDouble,
                operation,
                nccl_communicator_,
                collective_stream()
            ),
            "ncclAllReduce"
        );
        check_cuda(cudaEventRecord(timing.stop, collective_stream()), "record collective stop");
        pending_timings_.push_back(timing);
        timing.start = nullptr;
        timing.stop = nullptr;
        ordering_.collective_complete();
        check_cuda(
            cudaEventRecord(collective_complete_, collective_stream()),
            "record collective-complete event"
        );
        ordering_.compute_wait();
        check_cuda(
            cudaStreamWaitEvent(compute_stream_, collective_complete_, 0),
            "compute stream wait"
        );
        ordering_.finish();
    } catch (const std::exception& error) {
        if (timing.start != nullptr) {
            cudaEventDestroy(timing.start);
        }
        if (timing.stop != nullptr) {
            cudaEventDestroy(timing.stop);
        }
        ordering_.fail();
        fail(error.what());
        throw;
    }
}

void MpiNcclRuntime::synchronize() {
    check_cuda(cudaStreamSynchronize(compute_stream_), "synchronize compute stream");
    if (options_.enable_overlap) {
        check_cuda(cudaStreamSynchronize(collective_stream_), "synchronize collective stream");
    }
    for (auto& timing : pending_timings_) {
        float milliseconds = 0.0F;
        check_cuda(
            cudaEventElapsedTime(&milliseconds, timing.start, timing.stop),
            "measure collective"
        );
        const double seconds = static_cast<double>(milliseconds) * 1.0e-3;
        auto& collective = telemetry_.collectives.at(timing.telemetry_index);
        collective.collective_seconds += seconds;
        if (!options_.enable_overlap) {
            collective.exposed_seconds += seconds;
            telemetry_.exposed_communication_seconds += seconds;
        }
        cudaEventDestroy(timing.start);
        cudaEventDestroy(timing.stop);
    }
    pending_timings_.clear();
}

void MpiNcclRuntime::record_local_compute(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0.0) {
        throw std::invalid_argument("local compute time must be finite and non-negative");
    }
    telemetry_.local_compute_seconds += seconds;
}

void MpiNcclRuntime::record_overlap(double seconds) {
    if (!options_.enable_overlap || !std::isfinite(seconds) || seconds < 0.0) {
        throw std::invalid_argument("overlap time requires overlap mode and a finite duration");
    }
    telemetry_.overlapped_communication_seconds += seconds;
}

void MpiNcclRuntime::mark_values_updated() {
    if (state_ != RuntimeState::created && state_ != RuntimeState::solved
        && state_ != RuntimeState::values_updated && state_ != RuntimeState::warm_started) {
        throw std::logic_error("values cannot update in the current distributed state");
    }
    state_ = RuntimeState::values_updated;
}

void MpiNcclRuntime::mark_warm_started() {
    if (state_ != RuntimeState::created && state_ != RuntimeState::solved
        && state_ != RuntimeState::values_updated && state_ != RuntimeState::warm_started) {
        throw std::logic_error("warm state cannot update in the current distributed state");
    }
    state_ = RuntimeState::warm_started;
}

void MpiNcclRuntime::begin_solve() {
    if (state_ != RuntimeState::created && state_ != RuntimeState::values_updated
        && state_ != RuntimeState::warm_started && state_ != RuntimeState::solved) {
        throw std::logic_error("distributed solve cannot begin in the current state");
    }
    state_ = RuntimeState::solving;
}

void MpiNcclRuntime::finish_solve() {
    if (state_ != RuntimeState::solving) {
        throw std::logic_error("distributed solve is not active");
    }
    state_ = RuntimeState::solved;
}

void MpiNcclRuntime::cancel() noexcept {
    telemetry_.rank_status = RankStatus::cancelled;
    state_ = RuntimeState::cancelled;
    ordering_.cancel();
    if (nccl_communicator_ != nullptr) {
        ncclCommAbort(nccl_communicator_);
        nccl_communicator_ = nullptr;
    }
}

void MpiNcclRuntime::fail(std::string message) noexcept {
    last_error_ = std::move(message);
    telemetry_.rank_status = RankStatus::failed;
    state_ = RuntimeState::failed;
    ordering_.fail();
}

RankStatus MpiNcclRuntime::synchronize_status() {
    auto local = static_cast<std::int32_t>(telemetry_.rank_status);
    std::int32_t global = local;
    const double start = MPI_Wtime();
    const int result = MPI_Allreduce(
        &local,
        &global,
        1,
        MPI_INT32_T,
        MPI_MAX,
        communicator_
    );
    const double seconds = MPI_Wtime() - start;
    auto iterator = std::find_if(
        telemetry_.collectives.begin(),
        telemetry_.collectives.end(),
        [](const CollectiveTelemetry& entry) {
            return entry.kind == CollectiveKind::status_max;
        }
    );
    if (iterator == telemetry_.collectives.end()) {
        telemetry_.collectives.push_back(CollectiveTelemetry{
            CollectiveKind::status_max,
            0,
            0,
            0,
            0,
            1,
            "global cancellation/failure/rank-loss status",
        });
        iterator = std::prev(telemetry_.collectives.end());
    }
    ++iterator->call_count;
    ++iterator->element_count;
    iterator->payload_bytes += sizeof(std::int32_t);
    iterator->wire_bytes_estimate +=
        telemetry_.world_size > 1
            ? 2ULL * static_cast<std::uint64_t>(telemetry_.world_size - 1)
                  * sizeof(std::int32_t)
                  / static_cast<std::uint64_t>(telemetry_.world_size)
            : 0ULL;
    iterator->collective_seconds += seconds;
    iterator->exposed_seconds += seconds;
    telemetry_.exposed_communication_seconds += seconds;
    if (result != MPI_SUCCESS) {
        telemetry_.rank_status = RankStatus::rank_lost;
        state_ = RuntimeState::failed;
        last_error_ = "MPI status reduction failed; rank loss or communicator failure";
        return telemetry_.rank_status;
    }
    telemetry_.rank_status = static_cast<RankStatus>(global);
    if (telemetry_.rank_status == RankStatus::cancelled) {
        state_ = RuntimeState::cancelled;
    } else if (telemetry_.rank_status != RankStatus::healthy) {
        state_ = RuntimeState::failed;
    }
    return telemetry_.rank_status;
}

cudaError_t csc_forward_async(
    int rows,
    int columns,
    const int* offsets,
    const int* indices,
    const double* values,
    const double* input,
    double* output,
    cudaStream_t stream
) noexcept {
    if (rows < 0 || columns < 0 || offsets == nullptr || (rows > 0 && output == nullptr)
        || (columns > 0 && input == nullptr)) {
        return cudaErrorInvalidValue;
    }
    constexpr int threads = 128;
    if (rows > 0) {
        csc_forward_kernel<<<(rows + threads - 1) / threads, threads, 0, stream>>>(
            rows,
            columns,
            offsets,
            indices,
            values,
            input,
            output
        );
    }
    return cudaPeekAtLastError();
}

cudaError_t csc_transpose_async(
    int rows,
    int columns,
    const int* offsets,
    const int* indices,
    const double* values,
    const double* input,
    double* output,
    cudaStream_t stream
) noexcept {
    if (rows < 0 || columns < 0 || offsets == nullptr || (rows > 0 && input == nullptr)
        || (columns > 0 && output == nullptr)) {
        return cudaErrorInvalidValue;
    }
    constexpr int threads = 128;
    if (columns > 0) {
        csc_transpose_kernel<<<(columns + threads - 1) / threads, threads, 0, stream>>>(
            columns,
            offsets,
            indices,
            values,
            input,
            output
        );
    }
    return cudaPeekAtLastError();
}

cudaError_t project_soc_blocks_async(
    double* values,
    const int* starts,
    const int* dimensions,
    int cone_count,
    cudaStream_t stream
) noexcept {
    if (cone_count < 0
        || (cone_count > 0 && (values == nullptr || starts == nullptr || dimensions == nullptr))) {
        return cudaErrorInvalidValue;
    }
    if (cone_count > 0) {
        project_soc_kernel<<<cone_count, 1, 0, stream>>>(
            values,
            starts,
            dimensions,
            cone_count
        );
    }
    return cudaPeekAtLastError();
}

}  // namespace spacepdhcg::distributed::g5
