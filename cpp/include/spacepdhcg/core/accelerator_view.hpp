#pragma once

#include "spacepdhcg/core/fixed_cqp.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace spacepdhcg::core {

/// Device codes intentionally match the corresponding stable DLPack device values for the
/// supported memory classes. This permits structural conversion without making DLPack a mandatory
/// build dependency.
enum class AcceleratorDeviceType : std::int32_t {
    cpu = 1,
    cuda = 2,
    cuda_host = 3,
    cuda_managed = 13,
};

enum class AcceleratorScalarType : std::uint8_t {
    int32,
    int64,
    float32,
    float64,
};

enum class AcceleratorAccess : std::uint8_t {
    read_only,
    read_write,
};

struct AcceleratorDevice {
    AcceleratorDeviceType type{AcceleratorDeviceType::cpu};
    std::int32_t id{0};

    void validate() const {
        if (id < 0) {
            throw std::invalid_argument("accelerator device id must be non-negative");
        }
        if ((type == AcceleratorDeviceType::cpu
             || type == AcceleratorDeviceType::cuda_host
             || type == AcceleratorDeviceType::cuda_managed)
            && id != 0) {
            throw std::invalid_argument(
                "CPU, CUDA-host and CUDA-managed views must use device id zero"
            );
        }
    }

    [[nodiscard]] bool operator==(const AcceleratorDevice&) const noexcept = default;
};

/// CUDA execution context. Storage may be ordinary CUDA memory on this device or CUDA-managed
/// memory whose DLPack device id is necessarily zero.
struct AcceleratorStream {
    AcceleratorDevice device{AcceleratorDeviceType::cuda, 0};
    std::uintptr_t native_handle{0U};

    void validate() const {
        device.validate();
        if (device.type != AcceleratorDeviceType::cuda) {
            throw std::invalid_argument(
                "accelerator consumer stream must identify a CUDA execution device"
            );
        }
    }
};

struct AcceleratorBufferView {
    void* data{nullptr};
    AcceleratorDevice device{};
    AcceleratorScalarType scalar_type{AcceleratorScalarType::float64};
    std::size_t elements{0U};
    std::size_t byte_offset{0U};
    std::ptrdiff_t element_stride{1};
    AcceleratorAccess access{AcceleratorAccess::read_only};

    [[nodiscard]] std::size_t scalar_bytes() const {
        switch (scalar_type) {
            case AcceleratorScalarType::int32:
            case AcceleratorScalarType::float32:
                return 4U;
            case AcceleratorScalarType::int64:
            case AcceleratorScalarType::float64:
                return 8U;
        }
        throw std::invalid_argument("unknown accelerator scalar type");
    }

    [[nodiscard]] std::size_t required_bytes() const {
        if (elements == 0U) {
            return 0U;
        }
        if (element_stride <= 0) {
            throw std::invalid_argument(
                "accelerator buffer stride must be positive"
            );
        }
        const auto stride = static_cast<std::size_t>(element_stride);
        if (elements - 1U > std::numeric_limits<std::size_t>::max() / stride) {
            throw std::overflow_error("accelerator buffer element extent overflows size_t");
        }
        const auto last_element = stride * (elements - 1U);
        const auto bytes = scalar_bytes();
        if (last_element >
            (std::numeric_limits<std::size_t>::max() / bytes) - 1U) {
            throw std::overflow_error("accelerator buffer byte size overflows size_t");
        }
        return (last_element + 1U) * bytes;
    }

    void validate() const {
        device.validate();
        if (elements == 0U) {
            if (data != nullptr) {
                throw std::invalid_argument(
                    "zero-length accelerator buffers must use a null data pointer"
                );
            }
            if (byte_offset != 0U) {
                throw std::invalid_argument(
                    "zero-length accelerator buffers must use byte_offset zero"
                );
            }
            if (element_stride <= 0) {
                throw std::invalid_argument(
                    "zero-length accelerator buffer stride must still be positive"
                );
            }
            return;
        }
        if (data == nullptr) {
            throw std::invalid_argument(
                "non-empty accelerator buffer has a null data pointer"
            );
        }
        if (element_stride <= 0) {
            throw std::invalid_argument(
                "accelerator buffer stride must be positive"
            );
        }
        if (byte_offset % scalar_bytes() != 0U) {
            throw std::invalid_argument(
                "accelerator buffer byte_offset is not scalar aligned"
            );
        }
        static_cast<void>(required_bytes());
    }

    [[nodiscard]] bool contiguous() const noexcept {
        return elements == 0U || element_stride == 1;
    }
};

struct CqpTopologyAcceleratorViews {
    AcceleratorBufferView quadratic_offsets{};
    AcceleratorBufferView quadratic_indices{};
    AcceleratorBufferView scalar_offsets{};
    AcceleratorBufferView scalar_indices{};
    AcceleratorBufferView affine_offsets{};
    AcceleratorBufferView affine_indices{};
};

struct CqpNumericAcceleratorViews {
    AcceleratorBufferView quadratic{};
    AcceleratorBufferView scalar_constraint{};
    AcceleratorBufferView affine_cone{};
    AcceleratorBufferView linear_objective{};
    AcceleratorBufferView scalar_lower{};
    AcceleratorBufferView scalar_upper{};
    AcceleratorBufferView affine_offset{};
    AcceleratorBufferView variable_lower{};
    AcceleratorBufferView variable_upper{};
};

struct CqpIterateAcceleratorViews {
    AcceleratorBufferView primal{};
    AcceleratorBufferView dual{};
};

struct CqpAcceleratorExchange {
    std::uint32_t abi_version{1U};
    std::uint64_t topology_fingerprint{0U};
    AcceleratorStream consumer_stream{};
    CqpTopologyAcceleratorViews topology{};
    CqpNumericAcceleratorViews numeric{};
    CqpIterateAcceleratorViews iterates{};
};

namespace accelerator_view_detail {

inline void require_view(
    const AcceleratorBufferView& view,
    const AcceleratorDevice expected_storage,
    const AcceleratorScalarType expected_type,
    const std::size_t expected_elements,
    const AcceleratorAccess expected_access,
    const std::string_view name
) {
    view.validate();
    if (view.device != expected_storage) {
        throw std::invalid_argument(
            std::string(name) + " is in a different storage class or device"
        );
    }
    if (view.scalar_type != expected_type) {
        throw std::invalid_argument(
            std::string(name) + " has the wrong scalar type"
        );
    }
    if (view.elements != expected_elements) {
        throw std::invalid_argument(
            std::string(name) + " has " + std::to_string(view.elements)
            + " elements; expected " + std::to_string(expected_elements)
        );
    }
    if (!view.contiguous()) {
        throw std::invalid_argument(
            std::string(name) + " must be contiguous"
        );
    }
    if (view.access != expected_access) {
        throw std::invalid_argument(
            std::string(name)
            + (expected_access == AcceleratorAccess::read_write
                   ? " must be writable"
                   : " must be read-only")
        );
    }
}

inline std::size_t affine_nonzeros(const FixedStructure& structure) noexcept {
    return structure.affine_cone.has_value()
               ? structure.affine_cone->nonzeros()
               : 0U;
}

inline std::size_t affine_offset_entries(const FixedStructure& structure) noexcept {
    return structure.affine_cone.has_value()
               ? structure.affine_cone->offsets.size()
               : 0U;
}

inline AcceleratorDevice storage_device(const CqpAcceleratorExchange& exchange) {
    const auto storage = exchange.topology.quadratic_offsets.device;
    storage.validate();
    if (storage.type != AcceleratorDeviceType::cuda
        && storage.type != AcceleratorDeviceType::cuda_managed) {
        throw std::invalid_argument(
            "persistent accelerator storage must be CUDA device or CUDA-managed memory"
        );
    }
    if (storage.type == AcceleratorDeviceType::cuda
        && storage.id != exchange.consumer_stream.device.id) {
        throw std::invalid_argument(
            "CUDA storage device does not match the consumer stream device"
        );
    }
    return storage;
}

}  // namespace accelerator_view_detail

/// Validate a complete zero-copy workspace exchange against one immutable CQP topology.
///
/// Topology buffers are exactly read-only and borrowed once. Numerical values and iterates are
/// exactly writable because a persistent workspace updates coefficients and warm starts in place.
/// All storage views use one memory class: either ordinary CUDA memory on the consumer-stream GPU,
/// or CUDA-managed memory (DLPack device id zero) consumed by that GPU's stream. CUDA-host staging
/// buffers require a separate transfer operation and are not accepted as persistent solver state.
inline void validate_cqp_accelerator_exchange(
    const FixedStructure& structure,
    const CqpAcceleratorExchange& exchange
) {
    structure.validate();
    if (exchange.abi_version != 1U) {
        throw std::invalid_argument("unsupported accelerator exchange ABI version");
    }
    if (exchange.topology_fingerprint != structure.fingerprint()) {
        throw std::invalid_argument(
            "accelerator exchange topology fingerprint does not match the CQP"
        );
    }
    exchange.consumer_stream.validate();
    const auto storage = accelerator_view_detail::storage_device(exchange);
    using accelerator_view_detail::require_view;
    const auto index_type = AcceleratorScalarType::int32;
    require_view(
        exchange.topology.quadratic_offsets,
        storage,
        index_type,
        structure.quadratic.offsets.size(),
        AcceleratorAccess::read_only,
        "quadratic offsets"
    );
    require_view(
        exchange.topology.quadratic_indices,
        storage,
        index_type,
        structure.quadratic.indices.size(),
        AcceleratorAccess::read_only,
        "quadratic indices"
    );
    require_view(
        exchange.topology.scalar_offsets,
        storage,
        index_type,
        structure.scalar_constraint.offsets.size(),
        AcceleratorAccess::read_only,
        "scalar offsets"
    );
    require_view(
        exchange.topology.scalar_indices,
        storage,
        index_type,
        structure.scalar_constraint.indices.size(),
        AcceleratorAccess::read_only,
        "scalar indices"
    );
    require_view(
        exchange.topology.affine_offsets,
        storage,
        index_type,
        accelerator_view_detail::affine_offset_entries(structure),
        AcceleratorAccess::read_only,
        "affine offsets"
    );
    require_view(
        exchange.topology.affine_indices,
        storage,
        index_type,
        accelerator_view_detail::affine_nonzeros(structure),
        AcceleratorAccess::read_only,
        "affine indices"
    );

    const auto floating = AcceleratorScalarType::float64;
    const auto writable = AcceleratorAccess::read_write;
    require_view(
        exchange.numeric.quadratic,
        storage,
        floating,
        structure.quadratic.nonzeros(),
        writable,
        "quadratic values"
    );
    require_view(
        exchange.numeric.scalar_constraint,
        storage,
        floating,
        structure.scalar_constraint.nonzeros(),
        writable,
        "scalar constraint values"
    );
    require_view(
        exchange.numeric.affine_cone,
        storage,
        floating,
        accelerator_view_detail::affine_nonzeros(structure),
        writable,
        "affine cone values"
    );
    require_view(
        exchange.numeric.linear_objective,
        storage,
        floating,
        static_cast<std::size_t>(structure.variables()),
        writable,
        "linear objective"
    );
    require_view(
        exchange.numeric.scalar_lower,
        storage,
        floating,
        static_cast<std::size_t>(structure.scalar_rows()),
        writable,
        "scalar lower bounds"
    );
    require_view(
        exchange.numeric.scalar_upper,
        storage,
        floating,
        static_cast<std::size_t>(structure.scalar_rows()),
        writable,
        "scalar upper bounds"
    );
    require_view(
        exchange.numeric.affine_offset,
        storage,
        floating,
        static_cast<std::size_t>(structure.affine_rows()),
        writable,
        "affine offsets"
    );
    require_view(
        exchange.numeric.variable_lower,
        storage,
        floating,
        static_cast<std::size_t>(structure.variables()),
        writable,
        "variable lower bounds"
    );
    require_view(
        exchange.numeric.variable_upper,
        storage,
        floating,
        static_cast<std::size_t>(structure.variables()),
        writable,
        "variable upper bounds"
    );
    require_view(
        exchange.iterates.primal,
        storage,
        floating,
        static_cast<std::size_t>(structure.variables()),
        writable,
        "primal iterate"
    );
    require_view(
        exchange.iterates.dual,
        storage,
        floating,
        static_cast<std::size_t>(structure.duals()),
        writable,
        "dual iterate"
    );
}

}  // namespace spacepdhcg::core
