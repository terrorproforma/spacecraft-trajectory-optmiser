#pragma once

#include "spacepdhcg/core/accelerator_view.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

#if __has_include(<dlpack/dlpack.h>)
#include <dlpack/dlpack.h>
#define SPACEPDHCG_HAS_DLPACK 1
#else
#define SPACEPDHCG_HAS_DLPACK 0
#endif

namespace spacepdhcg::core {

#if SPACEPDHCG_HAS_DLPACK

namespace dlpack_detail {

inline AcceleratorDeviceType device_type(const DLDeviceType type) {
    switch (type) {
        case kDLCPU:
            return AcceleratorDeviceType::cpu;
        case kDLCUDA:
            return AcceleratorDeviceType::cuda;
        case kDLCUDAHost:
            return AcceleratorDeviceType::cuda_host;
        case kDLCUDAManaged:
            return AcceleratorDeviceType::cuda_managed;
        default:
            throw std::invalid_argument(
                "DLPack tensor uses an unsupported device type"
            );
    }
}

inline AcceleratorScalarType scalar_type(const DLDataType dtype) {
    if (dtype.lanes != 1U) {
        throw std::invalid_argument(
            "SpacePDHCG requires scalar DLPack tensors with lanes equal to one"
        );
    }
    if (dtype.code == kDLInt && dtype.bits == 32U) {
        return AcceleratorScalarType::int32;
    }
    if (dtype.code == kDLInt && dtype.bits == 64U) {
        return AcceleratorScalarType::int64;
    }
    if (dtype.code == kDLFloat && dtype.bits == 32U) {
        return AcceleratorScalarType::float32;
    }
    if (dtype.code == kDLFloat && dtype.bits == 64U) {
        return AcceleratorScalarType::float64;
    }
    throw std::invalid_argument(
        "DLPack tensor dtype is not int32, int64, float32, or float64"
    );
}

inline AcceleratorBufferView view(const DLManagedTensorVersioned& managed) {
    if (managed.version.major != DLPACK_MAJOR_VERSION) {
        throw std::invalid_argument(
            "DLPack major ABI version is incompatible with this build"
        );
    }
    const auto& tensor = managed.dl_tensor;
    if (tensor.ndim != 1 || tensor.shape == nullptr || tensor.strides == nullptr) {
        throw std::invalid_argument(
            "SpacePDHCG requires a rank-one DLPack tensor with explicit shape and stride"
        );
    }
    if (tensor.shape[0] < 0 || tensor.strides[0] <= 0) {
        throw std::invalid_argument(
            "DLPack tensor shape and stride must be non-negative/positive"
        );
    }
    AcceleratorBufferView result{
        tensor.data,
        AcceleratorDevice{
            device_type(tensor.device.device_type),
            tensor.device.device_id,
        },
        scalar_type(tensor.dtype),
        static_cast<std::size_t>(tensor.shape[0]),
        static_cast<std::size_t>(tensor.byte_offset),
        static_cast<std::ptrdiff_t>(tensor.strides[0]),
        (managed.flags & DLPACK_FLAG_BITMASK_READ_ONLY) != 0U
            ? AcceleratorAccess::read_only
            : AcceleratorAccess::read_write,
    };
    result.validate();
    return result;
}

}  // namespace dlpack_detail

/// RAII owner for a borrowed versioned DLPack tensor.
///
/// The producer-provided deleter is called exactly once. The caller must perform the DLPack
/// producer/consumer stream handoff before constructing a workspace exchange; this class does not
/// synchronize CUDA and deliberately cannot guess the producer's stream semantics.
class DLPackBorrow {
  public:
    explicit DLPackBorrow(DLManagedTensorVersioned* managed) : managed_(managed) {
        if (managed_ == nullptr) {
            throw std::invalid_argument("DLPack managed tensor pointer may not be null");
        }
        if (managed_->version.major != DLPACK_MAJOR_VERSION) {
            auto* incompatible = std::exchange(managed_, nullptr);
            if (incompatible->deleter != nullptr) {
                incompatible->deleter(incompatible);
            }
            throw std::invalid_argument(
                "DLPack major ABI version is incompatible with this build"
            );
        }
    }

    DLPackBorrow(const DLPackBorrow&) = delete;
    DLPackBorrow& operator=(const DLPackBorrow&) = delete;

    DLPackBorrow(DLPackBorrow&& other) noexcept
        : managed_(std::exchange(other.managed_, nullptr)) {}

    DLPackBorrow& operator=(DLPackBorrow&& other) noexcept {
        if (this != &other) {
            reset();
            managed_ = std::exchange(other.managed_, nullptr);
        }
        return *this;
    }

    ~DLPackBorrow() { reset(); }

    [[nodiscard]] AcceleratorBufferView view() const {
        if (managed_ == nullptr) {
            throw std::logic_error("DLPack borrow has no managed tensor");
        }
        return dlpack_detail::view(*managed_);
    }

    [[nodiscard]] DLManagedTensorVersioned* get() const noexcept { return managed_; }

    [[nodiscard]] DLManagedTensorVersioned* release() noexcept {
        return std::exchange(managed_, nullptr);
    }

    void reset() noexcept {
        auto* managed = std::exchange(managed_, nullptr);
        if (managed != nullptr && managed->deleter != nullptr) {
            managed->deleter(managed);
        }
    }

  private:
    DLManagedTensorVersioned* managed_{nullptr};
};

#endif  // SPACEPDHCG_HAS_DLPACK

}  // namespace spacepdhcg::core
