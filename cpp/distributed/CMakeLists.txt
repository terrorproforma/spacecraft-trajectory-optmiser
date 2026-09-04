cmake_minimum_required(VERSION 3.24)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)
find_library(
    SPACEPDHCG_NCCL_LIBRARY
    NAMES nccl
    PATHS
        /usr/lib/x86_64-linux-gnu
        ${CUDAToolkit_LIBRARY_DIR}
    REQUIRED
)
find_path(
    SPACEPDHCG_NCCL_INCLUDE_DIR
    NAMES nccl.h
    PATHS
        /usr/include
        ${CUDAToolkit_INCLUDE_DIRS}
    REQUIRED
)

add_library(
    spacepdhcg_distributed
    STATIC
        src/runtime.cu
        src/workspace.cu
)
add_library(SpacePDHCG::distributed ALIAS spacepdhcg_distributed)
target_include_directories(
    spacepdhcg_distributed
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        ${SPACEPDHCG_NCCL_INCLUDE_DIR}
)
target_link_libraries(
    spacepdhcg_distributed
    PUBLIC
        spacepdhcg_cuda
        CUDA::cudart
        MPI::MPI_CXX
        ${SPACEPDHCG_NCCL_LIBRARY}
)
target_compile_features(spacepdhcg_distributed PUBLIC cxx_std_20)
target_compile_options(
    spacepdhcg_distributed
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xcompiler=-Wall,-Wextra>
)
if(SPACEPDHCG_NATIVE_WARNINGS_AS_ERRORS)
    target_compile_options(
        spacepdhcg_distributed
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Werror>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>
    )
endif()
set_target_properties(
    spacepdhcg_distributed
    PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
)

add_executable(
    g5_physical_validation_harness
    tools/physical_validation_harness.cu
)
target_link_libraries(
    g5_physical_validation_harness
    PRIVATE spacepdhcg_distributed
)
set_target_properties(
    g5_physical_validation_harness
    PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/distributed-tools"
)
target_compile_options(
    g5_physical_validation_harness
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xcompiler=-Wall,-Wextra>
)
if(SPACEPDHCG_NATIVE_WARNINGS_AS_ERRORS)
    target_compile_options(
        g5_physical_validation_harness
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>
    )
endif()

if(BUILD_TESTING)
    add_executable(g5_logical_rank_test tests/logical_rank_test.cpp)
    target_link_libraries(g5_logical_rank_test PRIVATE spacepdhcg_distributed)
    spacepdhcg_apply_native_options(g5_logical_rank_test)
    set_target_properties(
        g5_logical_rank_test
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/distributed-tests"
    )
    add_test(NAME g5_logical_rank_contracts COMMAND g5_logical_rank_test)
    set_tests_properties(g5_logical_rank_contracts PROPERTIES TIMEOUT 30)

    add_executable(g5_one_rank_runtime_test tests/one_rank_runtime_test.cu)
    target_include_directories(
        g5_one_rank_runtime_test
        PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../cuda/tests"
    )
    target_link_libraries(g5_one_rank_runtime_test PRIVATE spacepdhcg_distributed)
    set_target_properties(
        g5_one_rank_runtime_test
        PROPERTIES
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/distributed-tests"
    )
    add_test(
        NAME g5_one_rank_mpi_nccl_cuda
        COMMAND
            ${MPIEXEC_EXECUTABLE}
            ${MPIEXEC_NUMPROC_FLAG} 1
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:g5_one_rank_runtime_test>
            ${MPIEXEC_POSTFLAGS}
    )
    set_tests_properties(
        g5_one_rank_mpi_nccl_cuda
        PROPERTIES
            PROCESSORS 1
            RESOURCE_LOCK gpu0
            TIMEOUT 90
            LABELS "g5;one-rank;gpu"
    )

    add_custom_target(
        g5_logical_checks
        COMMAND
            ${CMAKE_CTEST_COMMAND}
            --test-dir "${CMAKE_BINARY_DIR}"
            --output-on-failure
            -R g5_logical_rank_contracts
        DEPENDS g5_logical_rank_test
        USES_TERMINAL
    )
    add_custom_target(
        g5_one_rank_checks
        COMMAND
            ${CMAKE_CTEST_COMMAND}
            --test-dir "${CMAKE_BINARY_DIR}"
            --output-on-failure
            -R g5_one_rank_mpi_nccl_cuda
        DEPENDS g5_one_rank_runtime_test
        USES_TERMINAL
    )
endif()

install(
    TARGETS spacepdhcg_distributed
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)
