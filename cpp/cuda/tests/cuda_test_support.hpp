#pragma once

#include "spacepdhcg/cuda/persistent_pdhcg_c_api.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace spacepdhcg::cuda::test {

inline void require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

inline void cuda_require(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA FAIL (%s): %s\n", operation, cudaGetErrorString(status));
        std::exit(2);
    }
}

inline void status_require(spacepdhcg_cuda_status status, const char* operation) {
    if (status != SPACEPDHCG_CUDA_SUCCESS) {
        std::fprintf(stderr, "API FAIL (%s): %d\n", operation, static_cast<int>(status));
        std::exit(3);
    }
}

template <typename T>
class CudaBuffer {
  public:
    CudaBuffer() = default;
    CudaBuffer(std::size_t elements, bool managed) : elements_(elements), managed_(managed) {
        if (elements_ == 0U) {
            return;
        }
        if (managed_) {
            cuda_require(
                cudaMallocManaged(&pointer_, elements_ * sizeof(T), cudaMemAttachGlobal),
                "cudaMallocManaged"
            );
        } else {
            cuda_require(cudaMalloc(&pointer_, elements_ * sizeof(T)), "cudaMalloc");
        }
    }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    CudaBuffer(CudaBuffer&& other) noexcept
        : pointer_(std::exchange(other.pointer_, nullptr)),
          elements_(std::exchange(other.elements_, 0U)),
          managed_(other.managed_) {}

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            release();
            pointer_ = std::exchange(other.pointer_, nullptr);
            elements_ = std::exchange(other.elements_, 0U);
            managed_ = other.managed_;
        }
        return *this;
    }

    ~CudaBuffer() { release(); }

    void upload(const std::vector<T>& values, cudaStream_t stream) {
        require(values.size() == elements_, "upload size mismatch");
        if (elements_ == 0U) {
            return;
        }
        if (managed_) {
            std::memcpy(pointer_, values.data(), elements_ * sizeof(T));
        } else {
            cuda_require(
                cudaMemcpyAsync(
                    pointer_,
                    values.data(),
                    elements_ * sizeof(T),
                    cudaMemcpyHostToDevice,
                    stream
                ),
                "upload"
            );
        }
    }

    [[nodiscard]] std::vector<T> download(cudaStream_t stream) const {
        std::vector<T> values(elements_);
        if (elements_ > 0U) {
            cuda_require(
                cudaMemcpyAsync(
                    values.data(),
                    pointer_,
                    elements_ * sizeof(T),
                    cudaMemcpyDeviceToHost,
                    stream
                ),
                "download"
            );
            cuda_require(cudaStreamSynchronize(stream), "download synchronize");
        }
        return values;
    }

    [[nodiscard]] T* get() noexcept { return pointer_; }
    [[nodiscard]] const T* get() const noexcept { return pointer_; }
    [[nodiscard]] std::size_t size() const noexcept { return elements_; }

  private:
    T* pointer_{nullptr};
    std::size_t elements_{0U};
    bool managed_{false};

    void release() noexcept {
        if (pointer_ != nullptr) {
            static_cast<void>(cudaFree(pointer_));
            pointer_ = nullptr;
        }
    }
};

inline spacepdhcg_accelerator_buffer_view view(
    void* pointer,
    std::size_t elements,
    bool managed,
    spacepdhcg_accelerator_scalar_type type,
    spacepdhcg_accelerator_access access
) {
    return spacepdhcg_accelerator_buffer_view{
        pointer,
        spacepdhcg_accelerator_device{
            managed ? SPACEPDHCG_DEVICE_CUDA_MANAGED : SPACEPDHCG_DEVICE_CUDA,
            managed ? 0 : 0,
        },
        type,
        elements,
        0U,
        1,
        access,
    };
}

struct ProblemStorage {
    enum class Fixture {
        box,
        soc,
    };

    bool managed{false};
    cudaStream_t stream{nullptr};
    bool owns_stream{false};
    std::uint64_t fingerprint{0x7cd07e0a4f9e0b61ULL};
    int variables{0};
    int scalar_rows{0};
    int affine_rows{0};

    std::vector<int> h_q_offsets{};
    std::vector<int> h_q_indices{};
    std::vector<int> h_a_offsets{};
    std::vector<int> h_a_indices{};
    std::vector<int> h_f_offsets{};
    std::vector<int> h_f_indices{};
    std::vector<double> h_q{};
    std::vector<double> h_a{};
    std::vector<double> h_f{};
    std::vector<double> h_c{};
    std::vector<double> h_scalar_lower{};
    std::vector<double> h_scalar_upper{};
    std::vector<double> h_affine_offset{};
    std::vector<double> h_variable_lower{};
    std::vector<double> h_variable_upper{};
    std::vector<spacepdhcg_cuda_cone_descriptor> affine_cones{};
    std::vector<spacepdhcg_cuda_cone_descriptor> variable_cones{};

    CudaBuffer<int> q_offsets{};
    CudaBuffer<int> q_indices{};
    CudaBuffer<int> a_offsets{};
    CudaBuffer<int> a_indices{};
    CudaBuffer<int> f_offsets{};
    CudaBuffer<int> f_indices{};
    CudaBuffer<double> q{};
    CudaBuffer<double> a{};
    CudaBuffer<double> f{};
    CudaBuffer<double> c{};
    CudaBuffer<double> scalar_lower{};
    CudaBuffer<double> scalar_upper{};
    CudaBuffer<double> affine_offset{};
    CudaBuffer<double> variable_lower{};
    CudaBuffer<double> variable_upper{};
    CudaBuffer<double> primal{};
    CudaBuffer<double> dual{};

    spacepdhcg_cuda_structure structure{};
    spacepdhcg_cqp_accelerator_exchange exchange{};

    ProblemStorage(bool use_managed, bool non_default_stream)
        : managed(use_managed), owns_stream(non_default_stream) {
        if (owns_stream) {
            cuda_require(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream create");
        }
    }

    ProblemStorage(bool use_managed, bool non_default_stream, Fixture fixture)
        : ProblemStorage(use_managed, non_default_stream) {
        variables = 2;
        h_q_offsets = {0, 1, 2};
        h_q_indices = {0, 1};
        if (fixture == Fixture::box) {
            scalar_rows = 1;
            affine_rows = 0;
            h_a_offsets = {0, 1, 2};
            h_a_indices = {0, 0};
            h_f_offsets = {};
            h_f_indices = {};
            h_q = {1.0, 2.0};
            h_a = {1.0, 1.0};
            h_f = {};
            h_c = {-1.0, -1.0};
            h_scalar_lower = {1.0};
            h_scalar_upper = {1.0};
            h_affine_offset = {};
            h_variable_lower = {0.0, 0.0};
            h_variable_upper = {1.0, 1.0};
        } else {
            fingerprint = 0x195f87b8e20c61d3ULL;
            scalar_rows = 0;
            affine_rows = 3;
            h_a_offsets = {0, 0, 0};
            h_a_indices = {};
            h_f_offsets = {0, 1, 2};
            h_f_indices = {0, 1};
            h_q = {1.0, 1.0};
            h_a = {};
            h_f = {1.0, 1.0};
            h_c = {-2.0, 0.0};
            h_scalar_lower = {};
            h_scalar_upper = {};
            h_affine_offset = {0.0, 0.0, 1.0};
            h_variable_lower = {
                -std::numeric_limits<double>::infinity(),
                -std::numeric_limits<double>::infinity(),
            };
            h_variable_upper = {
                std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity(),
            };
            affine_cones = {
                {SPACEPDHCG_CUDA_CONE_SECOND_ORDER, 0, 1, 0.0},
            };
        }
        materialise();
    }

    ProblemStorage(const ProblemStorage&) = delete;
    ProblemStorage& operator=(const ProblemStorage&) = delete;

    ~ProblemStorage() {
        if (owns_stream && stream != nullptr) {
            static_cast<void>(cudaStreamSynchronize(stream));
            static_cast<void>(cudaStreamDestroy(stream));
        }
    }

    void materialise() {
        q_offsets = CudaBuffer<int>(h_q_offsets.size(), managed);
        q_indices = CudaBuffer<int>(h_q_indices.size(), managed);
        a_offsets = CudaBuffer<int>(h_a_offsets.size(), managed);
        a_indices = CudaBuffer<int>(h_a_indices.size(), managed);
        f_offsets = CudaBuffer<int>(h_f_offsets.size(), managed);
        f_indices = CudaBuffer<int>(h_f_indices.size(), managed);
        q = CudaBuffer<double>(h_q.size(), managed);
        a = CudaBuffer<double>(h_a.size(), managed);
        f = CudaBuffer<double>(h_f.size(), managed);
        c = CudaBuffer<double>(h_c.size(), managed);
        scalar_lower = CudaBuffer<double>(h_scalar_lower.size(), managed);
        scalar_upper = CudaBuffer<double>(h_scalar_upper.size(), managed);
        affine_offset = CudaBuffer<double>(h_affine_offset.size(), managed);
        variable_lower = CudaBuffer<double>(h_variable_lower.size(), managed);
        variable_upper = CudaBuffer<double>(h_variable_upper.size(), managed);
        primal = CudaBuffer<double>(static_cast<std::size_t>(variables), managed);
        dual = CudaBuffer<double>(static_cast<std::size_t>(scalar_rows + affine_rows), managed);

        q_offsets.upload(h_q_offsets, stream);
        q_indices.upload(h_q_indices, stream);
        a_offsets.upload(h_a_offsets, stream);
        a_indices.upload(h_a_indices, stream);
        f_offsets.upload(h_f_offsets, stream);
        f_indices.upload(h_f_indices, stream);
        upload_numeric();
        primal.upload(std::vector<double>(static_cast<std::size_t>(variables), 0.0), stream);
        dual.upload(
            std::vector<double>(static_cast<std::size_t>(scalar_rows + affine_rows), 0.0),
            stream
        );
        cuda_require(cudaStreamSynchronize(stream), "problem materialise");

        const auto storage_device = spacepdhcg_accelerator_device{
            managed ? SPACEPDHCG_DEVICE_CUDA_MANAGED : SPACEPDHCG_DEVICE_CUDA,
            0,
        };
        structure = spacepdhcg_cuda_structure{
            SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,
            fingerprint,
            variables,
            scalar_rows,
            affine_rows,
            h_q.size(),
            h_a.size(),
            h_f.size(),
            affine_cones.data(),
            affine_cones.size(),
            variable_cones.data(),
            variable_cones.size(),
        };
        exchange = spacepdhcg_cqp_accelerator_exchange{};
        exchange.abi_version = SPACEPDHCG_ACCELERATOR_EXCHANGE_ABI_VERSION;
        exchange.topology_fingerprint = fingerprint;
        exchange.consumer_stream = spacepdhcg_accelerator_stream{
            {SPACEPDHCG_DEVICE_CUDA, 0},
            reinterpret_cast<std::uintptr_t>(stream),
        };
        exchange.topology = {
            view(q_offsets.get(), q_offsets.size(), managed, SPACEPDHCG_SCALAR_INT32,
                 SPACEPDHCG_ACCESS_READ_ONLY),
            view(q_indices.get(), q_indices.size(), managed, SPACEPDHCG_SCALAR_INT32,
                 SPACEPDHCG_ACCESS_READ_ONLY),
            view(a_offsets.get(), a_offsets.size(), managed, SPACEPDHCG_SCALAR_INT32,
                 SPACEPDHCG_ACCESS_READ_ONLY),
            view(a_indices.get(), a_indices.size(), managed, SPACEPDHCG_SCALAR_INT32,
                 SPACEPDHCG_ACCESS_READ_ONLY),
            view(f_offsets.get(), f_offsets.size(), managed, SPACEPDHCG_SCALAR_INT32,
                 SPACEPDHCG_ACCESS_READ_ONLY),
            view(f_indices.get(), f_indices.size(), managed, SPACEPDHCG_SCALAR_INT32,
                 SPACEPDHCG_ACCESS_READ_ONLY),
        };
        exchange.numeric = numeric_views();
        exchange.iterates = {
            view(primal.get(), primal.size(), managed, SPACEPDHCG_SCALAR_FLOAT64,
                 SPACEPDHCG_ACCESS_READ_WRITE),
            view(dual.get(), dual.size(), managed, SPACEPDHCG_SCALAR_FLOAT64,
                 SPACEPDHCG_ACCESS_READ_WRITE),
        };
        static_cast<void>(storage_device);
    }

    void upload_numeric() {
        q.upload(h_q, stream);
        a.upload(h_a, stream);
        f.upload(h_f, stream);
        c.upload(h_c, stream);
        scalar_lower.upload(h_scalar_lower, stream);
        scalar_upper.upload(h_scalar_upper, stream);
        affine_offset.upload(h_affine_offset, stream);
        variable_lower.upload(h_variable_lower, stream);
        variable_upper.upload(h_variable_upper, stream);
    }

    [[nodiscard]] spacepdhcg_cqp_numeric_accelerator_views numeric_views() {
        return {
            view(q.get(), q.size(), managed, SPACEPDHCG_SCALAR_FLOAT64,
                 SPACEPDHCG_ACCESS_READ_WRITE),
            view(a.get(), a.size(), managed, SPACEPDHCG_SCALAR_FLOAT64,
                 SPACEPDHCG_ACCESS_READ_WRITE),
            view(f.get(), f.size(), managed, SPACEPDHCG_SCALAR_FLOAT64,
                 SPACEPDHCG_ACCESS_READ_WRITE),
            view(c.get(), c.size(), managed, SPACEPDHCG_SCALAR_FLOAT64,
                 SPACEPDHCG_ACCESS_READ_WRITE),
            view(scalar_lower.get(), scalar_lower.size(), managed, SPACEPDHCG_SCALAR_FLOAT64,
                 SPACEPDHCG_ACCESS_READ_WRITE),
            view(scalar_upper.get(), scalar_upper.size(), managed, SPACEPDHCG_SCALAR_FLOAT64,
                 SPACEPDHCG_ACCESS_READ_WRITE),
            view(affine_offset.get(), affine_offset.size(), managed, SPACEPDHCG_SCALAR_FLOAT64,
                 SPACEPDHCG_ACCESS_READ_WRITE),
            view(variable_lower.get(), variable_lower.size(), managed, SPACEPDHCG_SCALAR_FLOAT64,
                 SPACEPDHCG_ACCESS_READ_WRITE),
            view(variable_upper.get(), variable_upper.size(), managed, SPACEPDHCG_SCALAR_FLOAT64,
                 SPACEPDHCG_ACCESS_READ_WRITE),
        };
    }
};

inline ProblemStorage make_box_problem(bool managed = false, bool non_default_stream = true) {
    return ProblemStorage(managed, non_default_stream, ProblemStorage::Fixture::box);
}

inline ProblemStorage make_soc_problem(bool managed = false, bool non_default_stream = true) {
    return ProblemStorage(managed, non_default_stream, ProblemStorage::Fixture::soc);
}

inline spacepdhcg_cuda_create_options create_options() {
    return spacepdhcg_cuda_create_options{
        SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,
        SPACEPDHCG_CUDA_SCALING_REFRESH_IF_NEEDED,
        0.25,
        0.50,
        4U,
        1,
        nullptr,
        nullptr,
        nullptr,
    };
}

inline spacepdhcg_cuda_solve_options solve_options(
    double tolerance = 2.0e-6,
    std::uint64_t iterations = 100'000U
) {
    return spacepdhcg_cuda_solve_options{
        SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,
        tolerance,
        tolerance,
        iterations,
        25U,
    };
}

inline spacepdhcg_cuda_workspace* create_workspace(
    ProblemStorage& problem,
    spacepdhcg_cuda_create_options options = create_options()
) {
    spacepdhcg_cuda_workspace* workspace{nullptr};
    const auto status = spacepdhcg_cuda_workspace_create(
        &problem.structure,
        &problem.exchange,
        &options,
        &workspace
    );
    if (status != SPACEPDHCG_CUDA_SUCCESS) {
        std::fprintf(stderr, "workspace create failed: %d\n", static_cast<int>(status));
        std::exit(4);
    }
    return workspace;
}

inline spacepdhcg_cuda_diagnostics solve_and_wait(
    spacepdhcg_cuda_workspace* workspace,
    ProblemStorage& problem,
    spacepdhcg_cuda_solve_options options = solve_options()
) {
    status_require(
        spacepdhcg_cuda_workspace_solve_async(
            workspace,
            &options,
            problem.exchange.consumer_stream
        ),
        "solve async"
    );
    status_require(spacepdhcg_cuda_workspace_wait(workspace), "solve wait");
    spacepdhcg_cuda_diagnostics diagnostics{};
    status_require(
        spacepdhcg_cuda_workspace_diagnostics(workspace, &diagnostics),
        "diagnostics"
    );
    return diagnostics;
}

inline void destroy_workspace(spacepdhcg_cuda_workspace*& workspace) {
    status_require(spacepdhcg_cuda_workspace_destroy(&workspace), "workspace destroy");
    require(workspace == nullptr, "destroy did not null workspace");
}

inline void require_close(double actual, double expected, double tolerance, const char* message) {
    if (std::abs(actual - expected) > tolerance) {
        std::fprintf(
            stderr,
            "FAIL: %s: actual=%.17g expected=%.17g tolerance=%.3g\n",
            message,
            actual,
            expected,
            tolerance
        );
        std::exit(5);
    }
}

}  // namespace spacepdhcg::cuda::test
