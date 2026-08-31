#include "cuda_test_support.hpp"
#include "spacepdhcg/cuda/persistent_pdhcg_dlpack_c_api.h"

#include <cstdint>
#include <cstdio>

namespace test = spacepdhcg::cuda::test;

namespace {

struct DLDevice {
    std::int32_t type;
    std::int32_t id;
};

struct DLDataType {
    std::uint8_t code;
    std::uint8_t bits;
    std::uint16_t lanes;
};

struct DLTensor {
    void* data;
    DLDevice device;
    std::int32_t ndim;
    DLDataType dtype;
    std::int64_t* shape;
    std::int64_t* strides;
    std::uint64_t byte_offset;
};

struct LegacyManaged {
    DLTensor tensor;
    void* manager_context;
    void (*deleter)(LegacyManaged*);
};

struct VersionedManaged {
    std::uint32_t major;
    std::uint32_t minor;
    void* manager_context;
    void (*deleter)(VersionedManaged*);
    std::uint64_t flags;
    DLTensor tensor;
};

struct ProducerContext {
    int* delete_count;
    std::int64_t shape;
    std::int64_t stride;
    LegacyManaged legacy{};
    VersionedManaged versioned{};
};

void delete_legacy(LegacyManaged* managed) {
    auto* context = static_cast<ProducerContext*>(managed->manager_context);
    ++*context->delete_count;
    delete context;
}

void delete_versioned(VersionedManaged* managed) {
    auto* context = static_cast<ProducerContext*>(managed->manager_context);
    ++*context->delete_count;
    delete context;
}

spacepdhcg_dlpack_managed_tensor produce(
    const spacepdhcg_accelerator_buffer_view& view,
    int* delete_count,
    bool versioned
) {
    auto* context = new ProducerContext{};
    context->delete_count = delete_count;
    context->shape = static_cast<std::int64_t>(view.elements);
    context->stride = view.element_stride;
    const auto dtype = DLDataType{
        static_cast<std::uint8_t>(
            view.scalar_type == SPACEPDHCG_SCALAR_INT32
                || view.scalar_type == SPACEPDHCG_SCALAR_INT64
            ? 0U
            : 2U
        ),
        static_cast<std::uint8_t>(
            view.scalar_type == SPACEPDHCG_SCALAR_INT32
                || view.scalar_type == SPACEPDHCG_SCALAR_FLOAT32
            ? 32U
            : 64U
        ),
        1U,
    };
    const auto tensor = DLTensor{
        view.data,
        {
            view.device.type == SPACEPDHCG_DEVICE_CUDA_MANAGED ? 13 : 2,
            view.device.id,
        },
        1,
        dtype,
        &context->shape,
        &context->stride,
        view.byte_offset,
    };
    if (versioned) {
        context->versioned = {
            1U,
            3U,
            context,
            delete_versioned,
            view.access == SPACEPDHCG_ACCESS_READ_ONLY ? 1ULL : 0ULL,
            tensor,
        };
        return {
            &context->versioned,
            SPACEPDHCG_DLPACK_VERSIONED,
            view.access,
        };
    }
    context->legacy = {tensor, context, delete_legacy};
    return {
        &context->legacy,
        SPACEPDHCG_DLPACK_LEGACY,
        view.access,
    };
}

spacepdhcg_cqp_dlpack_exchange produce_exchange(
    const test::ProblemStorage& problem,
    int* delete_count
) {
    const auto& source = problem.exchange;
    spacepdhcg_cqp_dlpack_exchange result{};
    result.abi_version = SPACEPDHCG_ACCELERATOR_EXCHANGE_ABI_VERSION;
    result.topology_fingerprint = source.topology_fingerprint;
    result.consumer_stream = source.consumer_stream;
    bool versioned = false;
#define PRODUCE_CREATE(PATH) \
    result.PATH = produce(source.PATH, delete_count, versioned = !versioned)
    PRODUCE_CREATE(topology.quadratic_offsets);
    PRODUCE_CREATE(topology.quadratic_indices);
    PRODUCE_CREATE(topology.scalar_offsets);
    PRODUCE_CREATE(topology.scalar_indices);
    PRODUCE_CREATE(topology.affine_offsets);
    PRODUCE_CREATE(topology.affine_indices);
    PRODUCE_CREATE(numeric.quadratic);
    PRODUCE_CREATE(numeric.scalar_constraint);
    PRODUCE_CREATE(numeric.affine_cone);
    PRODUCE_CREATE(numeric.linear_objective);
    PRODUCE_CREATE(numeric.scalar_lower);
    PRODUCE_CREATE(numeric.scalar_upper);
    PRODUCE_CREATE(numeric.affine_offset);
    PRODUCE_CREATE(numeric.variable_lower);
    PRODUCE_CREATE(numeric.variable_upper);
    PRODUCE_CREATE(iterates.primal);
    PRODUCE_CREATE(iterates.dual);
#undef PRODUCE_CREATE
    return result;
}

spacepdhcg_cqp_numeric_dlpack_tensors produce_numeric(
    test::ProblemStorage& problem,
    int* delete_count
) {
    const auto source = problem.numeric_views();
    spacepdhcg_cqp_numeric_dlpack_tensors result{};
    bool versioned = false;
#define PRODUCE_UPDATE(NAME) \
    result.NAME = produce(source.NAME, delete_count, versioned = !versioned)
    PRODUCE_UPDATE(quadratic);
    PRODUCE_UPDATE(scalar_constraint);
    PRODUCE_UPDATE(affine_cone);
    PRODUCE_UPDATE(linear_objective);
    PRODUCE_UPDATE(scalar_lower);
    PRODUCE_UPDATE(scalar_upper);
    PRODUCE_UPDATE(affine_offset);
    PRODUCE_UPDATE(variable_lower);
    PRODUCE_UPDATE(variable_upper);
#undef PRODUCE_UPDATE
    return result;
}

enum class InvalidCase {
    dtype,
    rank,
    stride,
    alignment,
    device_index,
    host_storage,
    read_only,
    major_version,
};

void require_rejected(
    const test::ProblemStorage& problem,
    const InvalidCase invalid_case,
    const spacepdhcg_cuda_status expected
) {
    int delete_count = 0;
    auto exchange = produce_exchange(problem, &delete_count);
    auto* tensor = static_cast<VersionedManaged*>(
        exchange.numeric.quadratic.managed_tensor
    );
    switch (invalid_case) {
        case InvalidCase::dtype:
            tensor->tensor.dtype.bits = 32U;
            break;
        case InvalidCase::rank:
            tensor->tensor.ndim = 2;
            break;
        case InvalidCase::stride:
            tensor->tensor.strides[0] = 2;
            break;
        case InvalidCase::alignment:
            tensor->tensor.byte_offset = 1U;
            break;
        case InvalidCase::device_index:
            tensor->tensor.device.id = 1;
            break;
        case InvalidCase::host_storage:
            tensor->tensor.device.type = 1;
            break;
        case InvalidCase::read_only:
            tensor->flags = 1U;
            break;
        case InvalidCase::major_version:
            tensor->major = 2U;
            break;
    }
    auto options = test::create_options();
    spacepdhcg_cuda_workspace* workspace{nullptr};
    const auto status = spacepdhcg_cuda_workspace_create_from_dlpack(
        &problem.structure,
        &exchange,
        &options,
        &workspace
    );
    test::require(status == expected && workspace == nullptr, "invalid DLPack tensor accepted");
    test::require(
        delete_count == 17,
        "DLPack rejection did not release every managed tensor exactly once"
    );
}

}  // namespace

int main() {
    auto problem = test::make_box_problem(false, true);
    int delete_count = 0;
    auto exchange = produce_exchange(problem, &delete_count);
    auto options = test::create_options();
    spacepdhcg_cuda_workspace* workspace{nullptr};
    test::status_require(
        spacepdhcg_cuda_workspace_create_from_dlpack(
            &problem.structure,
            &exchange,
            &options,
            &workspace
        ),
        "DLPack workspace create"
    );
    test::require(delete_count == 0, "persistent DLPack borrows released early");

    spacepdhcg_cuda_pointer_snapshot before{};
    test::status_require(
        spacepdhcg_cuda_workspace_pointer_snapshot(workspace, &before),
        "pointer snapshot before DLPack update"
    );
    auto update = produce_numeric(problem, &delete_count);
    test::status_require(
        spacepdhcg_cuda_workspace_update_from_dlpack_async(
            workspace,
            problem.fingerprint,
            &update,
            problem.exchange.consumer_stream
        ),
        "DLPack values update"
    );
    test::require(delete_count == 0, "asynchronous DLPack borrow released early");
    test::status_require(spacepdhcg_cuda_workspace_wait(workspace), "DLPack update wait");
    test::require(delete_count == 9, "completed DLPack update did not release exactly once");

    spacepdhcg_cuda_pointer_snapshot after{};
    test::status_require(
        spacepdhcg_cuda_workspace_pointer_snapshot(workspace, &after),
        "pointer snapshot after DLPack update"
    );
    test::require(
        before.quadratic_offsets == after.quadratic_offsets
            && before.quadratic_values == after.quadratic_values
            && before.primal == after.primal,
        "DLPack update changed persistent pointers"
    );
    const auto diagnostics = test::solve_and_wait(workspace, problem);
    test::require(
        diagnostics.hidden_cpu_fallback == 0 && diagnostics.used_declared_stream == 1,
        "DLPack solve violated execution contract"
    );
    test::destroy_workspace(workspace);
    test::require(delete_count == 26, "persistent DLPack deleter count is not exact");

    auto managed_problem = test::make_box_problem(true, true);
    int managed_delete_count = 0;
    auto managed_exchange = produce_exchange(managed_problem, &managed_delete_count);
    test::status_require(
        spacepdhcg_cuda_workspace_create_from_dlpack(
            &managed_problem.structure,
            &managed_exchange,
            &options,
            &workspace
        ),
        "managed-memory DLPack workspace create"
    );
    test::destroy_workspace(workspace);
    test::require(
        managed_delete_count == 17,
        "managed-memory DLPack borrows were not released exactly once"
    );

    require_rejected(problem, InvalidCase::dtype, SPACEPDHCG_CUDA_POINTER_CONTRACT);
    require_rejected(problem, InvalidCase::rank, SPACEPDHCG_CUDA_POINTER_CONTRACT);
    require_rejected(problem, InvalidCase::stride, SPACEPDHCG_CUDA_POINTER_CONTRACT);
    require_rejected(problem, InvalidCase::alignment, SPACEPDHCG_CUDA_POINTER_CONTRACT);
    require_rejected(problem, InvalidCase::device_index, SPACEPDHCG_CUDA_POINTER_CONTRACT);
    require_rejected(problem, InvalidCase::host_storage, SPACEPDHCG_CUDA_POINTER_CONTRACT);
    require_rejected(problem, InvalidCase::read_only, SPACEPDHCG_CUDA_POINTER_CONTRACT);
    require_rejected(problem, InvalidCase::major_version, SPACEPDHCG_CUDA_INVALID_ARGUMENT);
    int fingerprint_delete_count = 0;
    auto fingerprint_exchange = produce_exchange(problem, &fingerprint_delete_count);
    fingerprint_exchange.topology_fingerprint ^= 1U;
    const auto fingerprint_status = spacepdhcg_cuda_workspace_create_from_dlpack(
        &problem.structure,
        &fingerprint_exchange,
        &options,
        &workspace
    );
    test::require(
        fingerprint_status == SPACEPDHCG_CUDA_TOPOLOGY_MISMATCH && workspace == nullptr,
        "DLPack topology mismatch was accepted"
    );
    test::require(
        fingerprint_delete_count == 17,
        "DLPack topology rejection did not release every managed tensor"
    );

    std::printf(
        "{\"case\":\"dlpack_contract\",\"producer\":\"independent\","
        "\"non_default_stream\":true,\"persistent_deletes\":17,"
        "\"update_deletes\":9,\"invalid_cases\":8,"
        "\"topology_rejected\":true,"
        "\"host_span_rejected\":true,\"pointer_stable\":true,"
        "\"managed_memory\":true,\"exact_once_deletion\":true}\n"
    );
    return 0;
}
