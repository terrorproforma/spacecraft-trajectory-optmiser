/*
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "spacepdhcg/accelerator_c_api.h"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace spacepdhcg::cuda {

[[nodiscard]] inline cudaStream_t native_stream(
    const spacepdhcg_accelerator_stream stream
) noexcept {
    return reinterpret_cast<cudaStream_t>(stream.native_handle);
}

[[nodiscard]] inline bool same_stream(
    const spacepdhcg_accelerator_stream left,
    const spacepdhcg_accelerator_stream right
) noexcept {
    return left.device.type == right.device.type
        && left.device.id == right.device.id
        && left.native_handle == right.native_handle;
}

class StreamEvent {
  public:
    StreamEvent() = default;
    StreamEvent(const StreamEvent&) = delete;
    StreamEvent& operator=(const StreamEvent&) = delete;

    ~StreamEvent() {
        if (event_ != nullptr) {
            static_cast<void>(cudaEventDestroy(event_));
        }
    }

    cudaError_t create() noexcept {
        return cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
    }

    cudaError_t record(cudaStream_t stream) noexcept {
        return cudaEventRecord(event_, stream);
    }

    cudaError_t query() const noexcept { return cudaEventQuery(event_); }
    cudaError_t wait() const noexcept { return cudaEventSynchronize(event_); }

    [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

  private:
    cudaEvent_t event_{nullptr};
};

class TimingEvents {
  public:
    TimingEvents() = default;
    TimingEvents(const TimingEvents&) = delete;
    TimingEvents& operator=(const TimingEvents&) = delete;

    ~TimingEvents() {
        if (start_ != nullptr) {
            static_cast<void>(cudaEventDestroy(start_));
        }
        if (stop_ != nullptr) {
            static_cast<void>(cudaEventDestroy(stop_));
        }
    }

    cudaError_t create() noexcept {
        auto result = cudaEventCreate(&start_);
        if (result != cudaSuccess) {
            return result;
        }
        return cudaEventCreate(&stop_);
    }

    cudaError_t begin(cudaStream_t stream) noexcept {
        return cudaEventRecord(start_, stream);
    }
    cudaError_t end(cudaStream_t stream) noexcept {
        return cudaEventRecord(stop_, stream);
    }

    [[nodiscard]] double elapsed_seconds() const noexcept {
        float milliseconds{0.0F};
        if (cudaEventElapsedTime(&milliseconds, start_, stop_) != cudaSuccess) {
            return 0.0;
        }
        return static_cast<double>(milliseconds) * 1.0e-3;
    }

  private:
    cudaEvent_t start_{nullptr};
    cudaEvent_t stop_{nullptr};
};

}  // namespace spacepdhcg::cuda
