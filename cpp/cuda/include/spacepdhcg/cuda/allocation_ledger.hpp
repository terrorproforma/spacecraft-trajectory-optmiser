/*
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace spacepdhcg::cuda {

enum class AllocationCategory : std::uint8_t {
    topology,
    numeric,
    iterate,
    residual,
    scaling,
    cone,
    descriptor_scratch,
    diagnostics,
};

struct AllocationRecord {
    void* pointer{nullptr};
    std::size_t bytes{0};
    AllocationCategory category{AllocationCategory::numeric};
    std::uint64_t creation_epoch{0};
    std::uint64_t free_epoch{0};
    bool host_pinned{false};
};

class AllocationLedger {
  public:
    cudaError_t allocate(
        void** pointer,
        std::size_t bytes,
        AllocationCategory category,
        std::uint64_t epoch
    ) {
        if (bytes == 0U) {
            *pointer = nullptr;
            return cudaSuccess;
        }
        const auto result = cudaMalloc(pointer, bytes);
        if (result == cudaSuccess) {
            record(*pointer, bytes, category, epoch, false);
        }
        return result;
    }

    cudaError_t allocate_pinned(
        void** pointer,
        std::size_t bytes,
        AllocationCategory category,
        std::uint64_t epoch
    ) {
        if (bytes == 0U) {
            *pointer = nullptr;
            return cudaSuccess;
        }
        const auto result = cudaHostAlloc(pointer, bytes, cudaHostAllocPortable);
        if (result == cudaSuccess) {
            record(*pointer, bytes, category, epoch, true);
        }
        return result;
    }

    cudaError_t allocate_mapped(
        void** host_pointer,
        void** device_pointer,
        std::size_t bytes,
        AllocationCategory category,
        std::uint64_t epoch
    ) {
        if (bytes == 0U) {
            *host_pointer = nullptr;
            *device_pointer = nullptr;
            return cudaSuccess;
        }
        auto result = cudaHostAlloc(
            host_pointer,
            bytes,
            cudaHostAllocPortable | cudaHostAllocMapped
        );
        if (result != cudaSuccess) {
            return result;
        }
        result = cudaHostGetDevicePointer(device_pointer, *host_pointer, 0U);
        if (result != cudaSuccess) {
            static_cast<void>(cudaFreeHost(*host_pointer));
            *host_pointer = nullptr;
            *device_pointer = nullptr;
            return result;
        }
        record(*host_pointer, bytes, category, epoch, true);
        return cudaSuccess;
    }

    cudaError_t release(void* pointer, std::uint64_t epoch) noexcept {
        if (pointer == nullptr) {
            return cudaSuccess;
        }
        for (auto& record : records_) {
            if (record.pointer == pointer && record.free_epoch == 0U) {
                const auto result =
                    record.host_pinned ? cudaFreeHost(pointer) : cudaFree(pointer);
                if (result == cudaSuccess) {
                    record.free_epoch = epoch == 0U ? 1U : epoch;
                    ++free_count_;
                    --active_count_;
                    active_bytes_ -= record.bytes;
                }
                return result;
            }
        }
        return cudaErrorInvalidDevicePointer;
    }

    [[nodiscard]] std::uint64_t allocation_count() const noexcept {
        return allocation_count_;
    }
    [[nodiscard]] std::uint64_t free_count() const noexcept { return free_count_; }
    [[nodiscard]] std::uint64_t active_count() const noexcept { return active_count_; }
    [[nodiscard]] std::uint64_t active_bytes() const noexcept { return active_bytes_; }
    [[nodiscard]] std::uint64_t peak_active_bytes() const noexcept {
        return peak_active_bytes_;
    }
    [[nodiscard]] std::uint64_t topology_allocation_count() const noexcept {
        return topology_allocation_count_;
    }
    [[nodiscard]] const std::vector<AllocationRecord>& records() const noexcept {
        return records_;
    }

  private:
    std::vector<AllocationRecord> records_{};
    std::uint64_t allocation_count_{0};
    std::uint64_t free_count_{0};
    std::uint64_t active_count_{0};
    std::uint64_t active_bytes_{0};
    std::uint64_t peak_active_bytes_{0};
    std::uint64_t topology_allocation_count_{0};

    void record(
        void* pointer,
        std::size_t bytes,
        AllocationCategory category,
        std::uint64_t epoch,
        bool host_pinned
    ) {
        records_.push_back(AllocationRecord{
            pointer,
            bytes,
            category,
            epoch,
            0U,
            host_pinned,
        });
        ++allocation_count_;
        ++active_count_;
        active_bytes_ += bytes;
        if (active_bytes_ > peak_active_bytes_) {
            peak_active_bytes_ = active_bytes_;
        }
        if (category == AllocationCategory::topology) {
            ++topology_allocation_count_;
        }
    }
};

}  // namespace spacepdhcg::cuda
