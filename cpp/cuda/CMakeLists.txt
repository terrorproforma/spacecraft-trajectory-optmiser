cmake_minimum_required(VERSION 3.24)

enable_language(C)
enable_language(CUDA)
# Conditional graph loops keep GTOC12 price/mass decisions on the device.
find_package(CUDAToolkit 12.4 REQUIRED)
find_package(Git REQUIRED)
find_package(ZLIB REQUIRED)
find_program(SPACEPDHCG_PATCH_EXECUTABLE patch REQUIRED)

set(SPACEPDHCG_REPOSITORY_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")
set(
    SPACEPDHCG_PDHCG_SOURCE_ROOT
    "${SPACEPDHCG_REPOSITORY_ROOT}/_upstream/pdhcg"
    CACHE PATH
    "Read-only path to the pinned upstream PDHCG checkout"
)
set(SPACEPDHCG_PDHCG_COMMIT "167c8b72b4b96d2f94d405b8763e485514192b81")
set(SPACEPDHCG_PDHCG_TREE "62b05e6c1bedd385f6c267af3645ae4aae0421b4")

if(NOT EXISTS "${SPACEPDHCG_PDHCG_SOURCE_ROOT}/.git")
    message(FATAL_ERROR
        "Pinned PDHCG checkout is missing. Run scripts/gpu/checkout_pinned_pdhcg.sh first."
    )
endif()

execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${SPACEPDHCG_PDHCG_SOURCE_ROOT}" rev-parse HEAD
    OUTPUT_VARIABLE SPACEPDHCG_ACTUAL_PDHCG_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${SPACEPDHCG_PDHCG_SOURCE_ROOT}" rev-parse "HEAD^{tree}"
    OUTPUT_VARIABLE SPACEPDHCG_ACTUAL_PDHCG_TREE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${SPACEPDHCG_PDHCG_SOURCE_ROOT}" status --porcelain=v1
    OUTPUT_VARIABLE SPACEPDHCG_PDHCG_DIRTY
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${SPACEPDHCG_REPOSITORY_ROOT}" rev-parse HEAD
    OUTPUT_VARIABLE SPACEPDHCG_SOURCE_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
if(NOT SPACEPDHCG_ACTUAL_PDHCG_COMMIT STREQUAL SPACEPDHCG_PDHCG_COMMIT)
    message(FATAL_ERROR
        "Pinned PDHCG commit mismatch: ${SPACEPDHCG_ACTUAL_PDHCG_COMMIT}"
    )
endif()
if(NOT SPACEPDHCG_ACTUAL_PDHCG_TREE STREQUAL SPACEPDHCG_PDHCG_TREE)
    message(FATAL_ERROR
        "Pinned PDHCG tree mismatch: ${SPACEPDHCG_ACTUAL_PDHCG_TREE}"
    )
endif()
if(NOT SPACEPDHCG_PDHCG_DIRTY STREQUAL "")
    message(FATAL_ERROR "Pinned PDHCG checkout is dirty; refusing CUDA build")
endif()

message(STATUS "SpacePDHCG persistent adapter strategy: linked_internal_adapter")
message(STATUS "Pinned PDHCG commit: ${SPACEPDHCG_ACTUAL_PDHCG_COMMIT}")
message(STATUS "Pinned PDHCG tree: ${SPACEPDHCG_ACTUAL_PDHCG_TREE}")
set(
    SPACEPDHCG_PDHCG_CLEANUP_PATCH
    "${SPACEPDHCG_REPOSITORY_ROOT}/third_party/patches/pdhcg/0001-free-quadratic-state.patch"
)
file(SHA256 "${SPACEPDHCG_PDHCG_CLEANUP_PATCH}" SPACEPDHCG_PDHCG_CLEANUP_PATCH_SHA256)
set(SPACEPDHCG_PDHCG_ROOT "${CMAKE_CURRENT_BINARY_DIR}/pdhcg-patched")
file(REMOVE_RECURSE "${SPACEPDHCG_PDHCG_ROOT}")
file(MAKE_DIRECTORY "${SPACEPDHCG_PDHCG_ROOT}")
file(
    COPY "${SPACEPDHCG_PDHCG_SOURCE_ROOT}/"
    DESTINATION "${SPACEPDHCG_PDHCG_ROOT}"
    PATTERN ".git" EXCLUDE
)
execute_process(
    COMMAND "${SPACEPDHCG_PATCH_EXECUTABLE}" --dry-run -p1
            --input "${SPACEPDHCG_PDHCG_CLEANUP_PATCH}"
    WORKING_DIRECTORY "${SPACEPDHCG_PDHCG_ROOT}"
    COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
    COMMAND "${SPACEPDHCG_PATCH_EXECUTABLE}" -p1
            --input "${SPACEPDHCG_PDHCG_CLEANUP_PATCH}"
    WORKING_DIRECTORY "${SPACEPDHCG_PDHCG_ROOT}"
    COMMAND_ERROR_IS_FATAL ANY
)
message(STATUS
    "Pinned PDHCG patch: 0001-free-quadratic-state.patch "
    "sha256=${SPACEPDHCG_PDHCG_CLEANUP_PATCH_SHA256}"
)

add_library(
    spacepdhcg_cuda
    SHARED
        src/device_scvx.cu
        src/gtoc12_discretisation.cu
        src/gtoc12_conic.cu
        src/gtoc12_qoco.cu
        src/gtoc12_scvx.cu
        src/gtoc12_verification.cu
        src/gtoc12_neighbours.cu
        src/gtoc12_collection.cu
        src/gtoc12_retime.cu
        src/gtoc12_collect_dp.cu
        src/gtoc12_joint.cu
        src/native_qoco_adapter.cpp
        src/native_qoco_gpu.cu
        src/orbitweaver_gpu.cu
        src/persistent_pdhcg.cu
)
add_library(SpacePDHCG::cuda ALIAS spacepdhcg_cuda)
set_source_files_properties(src/gtoc12_joint.cu PROPERTIES COMPILE_OPTIONS "--fmad=false")
set_source_files_properties(src/gtoc12_discretisation.cu src/gtoc12_conic.cu src/gtoc12_qoco.cu src/gtoc12_scvx.cu src/gtoc12_verification.cu src/gtoc12_neighbours.cu src/gtoc12_collection.cu src/gtoc12_retime.cu src/gtoc12_collect_dp.cu PROPERTIES
    COMPILE_OPTIONS "--fmad=false"
)
target_include_directories(
    spacepdhcg_cuda
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/internal"
)
target_link_libraries(
    spacepdhcg_cuda
    PUBLIC
        CUDA::cudart
        CUDA::cusparse
    PRIVATE
        ${CMAKE_DL_LIBS}
)
target_compile_features(spacepdhcg_cuda PUBLIC cxx_std_20)
target_compile_options(
    spacepdhcg_cuda
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xcompiler=-Wall,-Wextra>
)
if(SPACEPDHCG_NATIVE_WARNINGS_AS_ERRORS)
    target_compile_options(
        spacepdhcg_cuda
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Werror>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>
    )
endif()
set_target_properties(
    spacepdhcg_cuda
    PROPERTIES
        OUTPUT_NAME spacepdhcg_cuda
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
)

# The exact pinned upstream sources are linked as the one-shot GPU comparator.
# They are never called by the persistent hot path.
file(GLOB SPACEPDHCG_PINNED_PDHCG_C_SOURCES
    CONFIGURE_DEPENDS
    "${SPACEPDHCG_PDHCG_ROOT}/src/*.c"
)
file(GLOB SPACEPDHCG_PINNED_PDHCG_CUDA_SOURCES
    CONFIGURE_DEPENDS
    "${SPACEPDHCG_PDHCG_ROOT}/src/*.cu"
    "${SPACEPDHCG_PDHCG_ROOT}/src/kernels/*.cu"
)
list(REMOVE_ITEM
    SPACEPDHCG_PINNED_PDHCG_C_SOURCES
    "${SPACEPDHCG_PDHCG_ROOT}/src/cli.c"
)
add_library(
    spacepdhcg_pinned_reference
    STATIC
        ${SPACEPDHCG_PINNED_PDHCG_C_SOURCES}
        ${SPACEPDHCG_PINNED_PDHCG_CUDA_SOURCES}
        "${SPACEPDHCG_PDHCG_ROOT}/distributed/distributed_ops_stub.c"
        "${SPACEPDHCG_PDHCG_ROOT}/distributed/distributed_conic_stub.c"
)
target_include_directories(
    spacepdhcg_pinned_reference
    PUBLIC "${SPACEPDHCG_PDHCG_ROOT}/include"
    PRIVATE
        "${SPACEPDHCG_PDHCG_ROOT}/internal"
        "${SPACEPDHCG_PDHCG_ROOT}/distributed"
)
target_compile_definitions(
    spacepdhcg_pinned_reference
    PUBLIC
        CUSPARSE_ENABLE_EXPERIMENTAL_API
        PDHCG_VERSION="0.3.0-pinned"
)
target_link_libraries(
    spacepdhcg_pinned_reference
    PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
        CUDA::cusparse
        ZLIB::ZLIB
)
set_target_properties(
    spacepdhcg_pinned_reference
    PROPERTIES
        C_STANDARD 99
        CXX_STANDARD 17
        CUDA_STANDARD 17
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
)

if(BUILD_TESTING)
    # The importer deliberately has a CPU-only conversion/KKT test. It never
    # initializes CUDA, even though this target is configured in a CUDA build.
    add_executable(persistent_snapshot_conversion_test tests/persistent_snapshot_conversion_test.cpp)
    target_include_directories(persistent_snapshot_conversion_test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_features(persistent_snapshot_conversion_test PRIVATE cxx_std_20)
    spacepdhcg_apply_native_options(persistent_snapshot_conversion_test)
    add_test(NAME persistent_snapshot_conversion_test COMMAND persistent_snapshot_conversion_test)
    set_tests_properties(persistent_snapshot_conversion_test PROPERTIES TIMEOUT 30)

    add_executable(persistent_snapshot_replay tests/persistent_snapshot_replay.cu)
    find_package(Threads REQUIRED)
    target_link_libraries(persistent_snapshot_replay PRIVATE spacepdhcg_cuda Threads::Threads ${CMAKE_DL_LIBS})
    set_target_properties(persistent_snapshot_replay PROPERTIES CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/cuda-tests")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        tests/persistent_snapshot.hpp tests/persistent_snapshot_replay.cu tests/cuda_test_support.hpp
        include/spacepdhcg/cuda/persistent_pdhcg_c_api.h)
    set(SPACEPDHCG_SNAPSHOT_SOURCE_HASHES "")
    foreach(snapshot_source tests/persistent_snapshot.hpp tests/persistent_snapshot_replay.cu tests/cuda_test_support.hpp
            include/spacepdhcg/cuda/persistent_pdhcg_c_api.h)
        file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${snapshot_source}" snapshot_hash)
        string(APPEND SPACEPDHCG_SNAPSHOT_SOURCE_HASHES "${snapshot_source}:${snapshot_hash}\n")
    endforeach()
    string(SHA256 SPACEPDHCG_SNAPSHOT_SOURCE_SHA256 "${SPACEPDHCG_SNAPSHOT_SOURCE_HASHES}")
    target_compile_definitions(persistent_snapshot_replay PRIVATE
        SPACEPDHCG_SOURCE_COMMIT="${SPACEPDHCG_SOURCE_COMMIT}"
        SPACEPDHCG_SNAPSHOT_SOURCE_SHA256="${SPACEPDHCG_SNAPSHOT_SOURCE_SHA256}")

    # The original algorithm remains a separate one-shot diagnostic baseline.
    # It imports the identical capture and exports vectors for the same KKT audit.
    add_executable(upstream_snapshot_replay tests/upstream_snapshot_replay.cu)
    target_include_directories(upstream_snapshot_replay PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_link_libraries(upstream_snapshot_replay PRIVATE spacepdhcg_pinned_reference)
    set_target_properties(upstream_snapshot_replay PROPERTIES CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON CUDA_RESOLVE_DEVICE_SYMBOLS ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/cuda-tests")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        tests/upstream_snapshot_replay.cu
        "${CMAKE_CURRENT_SOURCE_DIR}/../include/spacepdhcg/accelerator_c_api.h"
        "${SPACEPDHCG_PDHCG_CLEANUP_PATCH}")
    set(SPACEPDHCG_UPSTREAM_REPLAY_HASHES
        "upstream:${SPACEPDHCG_PDHCG_COMMIT}\ntree:${SPACEPDHCG_PDHCG_TREE}\npatch:${SPACEPDHCG_PDHCG_CLEANUP_PATCH_SHA256}\n")
    foreach(replay_source tests/persistent_snapshot.hpp tests/upstream_snapshot_replay.cu
            include/spacepdhcg/cuda/persistent_pdhcg_c_api.h)
        file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${replay_source}" replay_hash)
        string(APPEND SPACEPDHCG_UPSTREAM_REPLAY_HASHES "${replay_source}:${replay_hash}\n")
    endforeach()
    file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/../include/spacepdhcg/accelerator_c_api.h" replay_hash)
    string(APPEND SPACEPDHCG_UPSTREAM_REPLAY_HASHES "accelerator_c_api.h:${replay_hash}\n")
    string(SHA256 SPACEPDHCG_UPSTREAM_REPLAY_SOURCE_SHA256 "${SPACEPDHCG_UPSTREAM_REPLAY_HASHES}")
    target_compile_definitions(upstream_snapshot_replay PRIVATE
        SPACEPDHCG_SOURCE_COMMIT="${SPACEPDHCG_SOURCE_COMMIT}"
        SPACEPDHCG_UPSTREAM_REPLAY_SOURCE_SHA256="${SPACEPDHCG_UPSTREAM_REPLAY_SOURCE_SHA256}")

    file(
        GLOB SPACEPDHCG_CUDA_TEST_SOURCES
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*_test.cu"
    )
    list(APPEND SPACEPDHCG_CUDA_TEST_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/gtoc12_joint_smoke.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/gtoc12_joint_selection_test.cu"
    )
    list(REMOVE_DUPLICATES SPACEPDHCG_CUDA_TEST_SOURCES)
    foreach(source IN LISTS SPACEPDHCG_CUDA_TEST_SOURCES)
        get_filename_component(test_name "${source}" NAME_WE)
        # This standalone test needs the isolated QOCO source headers/library;
        # see docs/GPU_QOCO_LOCAL_BACKEND.md for its separate build command.
        if(test_name STREQUAL "qoco_gpu_operators_test"
           OR test_name STREQUAL "qoco_fused_kkt_product_test"
           OR test_name STREQUAL "qoco_gpu_numeric_update_test"
           OR test_name STREQUAL "qoco_gpu_scalar_test"
           OR test_name STREQUAL "qoco_gpu_stopping_test"
           OR test_name STREQUAL "qoco_gpu_step_control_test"
           OR test_name STREQUAL "qoco_gpu_ordering_test"
           OR test_name STREQUAL "qoco_trajectory_ordering_test")
            continue()
        endif()
        add_executable("${test_name}" "${source}")
        target_include_directories(
            "${test_name}"
            PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/tests"
                "${SPACEPDHCG_PDHCG_ROOT}/include"
        )
        target_link_libraries(
            "${test_name}"
            PRIVATE
                spacepdhcg_cuda
                spacepdhcg_pinned_reference
                ${CMAKE_DL_LIBS}
        )
        target_compile_definitions(
            "${test_name}"
            PRIVATE SPACEPDHCG_SOURCE_COMMIT="${SPACEPDHCG_SOURCE_COMMIT}"
        )
        if(test_name STREQUAL "gtoc12_joint_smoke"
           OR test_name STREQUAL "gtoc12_joint_selection_test")
            # These tests perform their native API calls inside assert().
            target_compile_options("${test_name}" PRIVATE -UNDEBUG)
        endif()
        set_target_properties(
            "${test_name}"
            PROPERTIES
                CUDA_STANDARD 20
                CUDA_STANDARD_REQUIRED ON
                CUDA_RESOLVE_DEVICE_SYMBOLS ON
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/cuda-tests"
        )
        if(test_name STREQUAL "persistent_soc_test")
            add_test(NAME "${test_name}" COMMAND "${test_name}" --with-one-shot)
        elseif(test_name STREQUAL "persistent_halpern_test")
            add_test(NAME "${test_name}" COMMAND "${test_name}" plain)
            add_test(NAME persistent_halpern_adaptive_test COMMAND "${test_name}" adaptive)
            set_tests_properties(persistent_halpern_adaptive_test PROPERTIES TIMEOUT 180)
        else()
            add_test(NAME "${test_name}" COMMAND "${test_name}")
        endif()
        set_tests_properties("${test_name}" PROPERTIES TIMEOUT 180)
        if(test_name STREQUAL "device_scvx_integration_test")
            add_test(
                NAME device_scvx_production_test
                COMMAND "${test_name}" --production-outer
            )
            add_test(
                NAME device_scvx_p1d_path_audit_test
                COMMAND "${test_name}" --p1d-path-audit
            )
            add_test(
                NAME device_scvx_qoco_handback_test
                COMMAND "${test_name}" --qoco-handback
            )
            add_test(
                NAME device_scvx_p1c_qoco_repeatability_test
                COMMAND "${test_name}" --p1c-qoco-repeatability
            )
            add_test(
                NAME device_scvx_qoco_unavailable_test
                COMMAND
                    "${CMAKE_COMMAND}" -E env
                    --unset=SPACEPDHCG_QOCO_LIBRARY
                    "$<TARGET_FILE:${test_name}>" --qoco-unavailable
            )
            set_tests_properties(
                device_scvx_production_test
                device_scvx_p1d_path_audit_test
                device_scvx_qoco_handback_test
                device_scvx_p1c_qoco_repeatability_test
                device_scvx_qoco_unavailable_test
                PROPERTIES TIMEOUT 300
            )
        endif()
    endforeach()
endif()

# User-facing planner executable on the validated persistent SCvx stack.
add_executable(spacepdhcg_plan tools/spacepdhcg_plan.cu)
target_link_libraries(spacepdhcg_plan PRIVATE spacepdhcg_cuda ${CMAKE_DL_LIBS})
target_compile_definitions(
    spacepdhcg_plan
    PRIVATE SPACEPDHCG_SOURCE_COMMIT="${SPACEPDHCG_SOURCE_COMMIT}"
)
target_compile_options(
    spacepdhcg_plan
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xcompiler=-Wall,-Wextra>
)
if(SPACEPDHCG_NATIVE_WARNINGS_AS_ERRORS)
    target_compile_options(
        spacepdhcg_plan
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>
    )
endif()
set_target_properties(
    spacepdhcg_plan
    PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/cuda-tools"
)
if(BUILD_TESTING)
    add_test(NAME spacepdhcg_plan_capabilities COMMAND spacepdhcg_plan --capabilities)
    set_tests_properties(spacepdhcg_plan_capabilities PROPERTIES TIMEOUT 60)
endif()

install(
    TARGETS spacepdhcg_cuda spacepdhcg_plan
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
install(FILES "${SPACEPDHCG_REPOSITORY_ROOT}/third_party/licenses/scipy-dop853.txt"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/spacepdhcg")
