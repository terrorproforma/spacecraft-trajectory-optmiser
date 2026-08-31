#include "spacepdhcg/accelerator_c_api.h"
#include "spacepdhcg/core/accelerator_view.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

using namespace spacepdhcg::core;

AcceleratorBufferView view(
    void* data,
    const AcceleratorDevice device,
    const AcceleratorScalarType scalar_type,
    const std::size_t elements,
    const AcceleratorAccess access
) {
    return AcceleratorBufferView{
        data,
        device,
        scalar_type,
        elements,
        0U,
        1,
        access,
    };
}

FixedStructure structure() {
    FixedStructure result{};
    result.quadratic = CscPattern{
        2,
        2,
        {0, 1, 2},
        {0, 1},
    };
    result.scalar_constraint = CscPattern{
        1,
        2,
        {0, 1, 2},
        {0, 0},
    };
    result.validate();
    return result;
}

template <typename Function>
bool rejects(Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}  // namespace

int main() {
    static_assert(
        static_cast<std::int32_t>(AcceleratorDeviceType::cuda)
        == SPACEPDHCG_DEVICE_CUDA
    );
    static_assert(SPACEPDHCG_ACCELERATOR_EXCHANGE_ABI_VERSION == 1U);

    const auto problem = structure();
    const AcceleratorDevice device{AcceleratorDeviceType::cuda, 0};
    std::array<std::int32_t, 3U> q_offsets{0, 1, 2};
    std::array<std::int32_t, 2U> q_indices{0, 1};
    std::array<std::int32_t, 3U> a_offsets{0, 1, 2};
    std::array<std::int32_t, 2U> a_indices{0, 0};
    std::array<double, 2U> q_values{1.0, 1.0};
    std::array<double, 2U> a_values{1.0, 1.0};
    std::array<double, 2U> linear{0.0, 0.0};
    std::array<double, 1U> scalar_lower{0.0};
    std::array<double, 1U> scalar_upper{0.0};
    std::array<double, 2U> variable_lower{-1.0, -1.0};
    std::array<double, 2U> variable_upper{1.0, 1.0};
    std::array<double, 2U> primal{0.0, 0.0};
    std::array<double, 1U> dual{0.0};

    const AcceleratorBufferView empty{
        nullptr,
        device,
        AcceleratorScalarType::float64,
        0U,
        0U,
        1,
        AcceleratorAccess::read_write,
    };
    CqpAcceleratorExchange exchange{
        1U,
        problem.fingerprint(),
        AcceleratorStream{device, 0x1234U},
        CqpTopologyAcceleratorViews{
            view(
                q_offsets.data(),
                device,
                AcceleratorScalarType::int32,
                q_offsets.size(),
                AcceleratorAccess::read_only
            ),
            view(
                q_indices.data(),
                device,
                AcceleratorScalarType::int32,
                q_indices.size(),
                AcceleratorAccess::read_only
            ),
            view(
                a_offsets.data(),
                device,
                AcceleratorScalarType::int32,
                a_offsets.size(),
                AcceleratorAccess::read_only
            ),
            view(
                a_indices.data(),
                device,
                AcceleratorScalarType::int32,
                a_indices.size(),
                AcceleratorAccess::read_only
            ),
            AcceleratorBufferView{
                nullptr,
                device,
                AcceleratorScalarType::int32,
                0U,
                0U,
                1,
                AcceleratorAccess::read_only,
            },
            AcceleratorBufferView{
                nullptr,
                device,
                AcceleratorScalarType::int32,
                0U,
                0U,
                1,
                AcceleratorAccess::read_only,
            },
        },
        CqpNumericAcceleratorViews{
            view(q_values.data(), device, AcceleratorScalarType::float64, 2U,
                 AcceleratorAccess::read_write),
            view(a_values.data(), device, AcceleratorScalarType::float64, 2U,
                 AcceleratorAccess::read_write),
            empty,
            view(linear.data(), device, AcceleratorScalarType::float64, 2U,
                 AcceleratorAccess::read_write),
            view(scalar_lower.data(), device, AcceleratorScalarType::float64, 1U,
                 AcceleratorAccess::read_write),
            view(scalar_upper.data(), device, AcceleratorScalarType::float64, 1U,
                 AcceleratorAccess::read_write),
            empty,
            view(variable_lower.data(), device, AcceleratorScalarType::float64, 2U,
                 AcceleratorAccess::read_write),
            view(variable_upper.data(), device, AcceleratorScalarType::float64, 2U,
                 AcceleratorAccess::read_write),
        },
        CqpIterateAcceleratorViews{
            view(primal.data(), device, AcceleratorScalarType::float64, 2U,
                 AcceleratorAccess::read_write),
            view(dual.data(), device, AcceleratorScalarType::float64, 1U,
                 AcceleratorAccess::read_write),
        },
    };
    validate_cqp_accelerator_exchange(problem, exchange);

    auto wrong_fingerprint = exchange;
    ++wrong_fingerprint.topology_fingerprint;
    if (!rejects([&] {
            validate_cqp_accelerator_exchange(problem, wrong_fingerprint);
        })) {
        return 1;
    }

    auto wrong_device = exchange;
    wrong_device.iterates.primal.device.id = 1;
    if (!rejects([&] {
            validate_cqp_accelerator_exchange(problem, wrong_device);
        })) {
        return 2;
    }

    auto read_only_values = exchange;
    read_only_values.numeric.quadratic.access = AcceleratorAccess::read_only;
    if (!rejects([&] {
            validate_cqp_accelerator_exchange(problem, read_only_values);
        })) {
        return 3;
    }

    auto strided_values = exchange;
    strided_values.numeric.scalar_constraint.element_stride = 2;
    if (!rejects([&] {
            validate_cqp_accelerator_exchange(problem, strided_values);
        })) {
        return 4;
    }

    auto wrong_type = exchange;
    wrong_type.topology.scalar_indices.scalar_type = AcceleratorScalarType::int64;
    if (!rejects([&] {
            validate_cqp_accelerator_exchange(problem, wrong_type);
        })) {
        return 5;
    }

    auto bad_zero = exchange;
    bad_zero.numeric.affine_cone.data = q_values.data();
    if (!rejects([&] {
            validate_cqp_accelerator_exchange(problem, bad_zero);
        })) {
        return 6;
    }

    auto misaligned = exchange;
    misaligned.iterates.dual.byte_offset = 1U;
    if (!rejects([&] {
            validate_cqp_accelerator_exchange(problem, misaligned);
        })) {
        return 7;
    }
    return 0;
}
