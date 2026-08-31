cmake_minimum_required(VERSION 3.24)

enable_language(C)
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(Git REQUIRED)
find_package(ZLIB REQUIRED)
find_program(SPACEPDHCG_PATCH_EXECUTABLE patch REQUIRED)

set(SPACEPDHCG_REPOSITORY_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")
set(SPACEPDHCG_PDHCG_SOURCE_ROOT "${SPACEPDHCG_REPOSITORY_ROOT}/_upstream/pdhcg")
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

add_library(spacepdhcg_cuda SHARED src/persistent_pdhcg.cu)
add_library(SpacePDHCG::cuda ALIAS spacepdhcg_cuda)
target_include_directories(
    spacepdhcg_cuda
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(
    spacepdhcg_cuda
    PUBLIC
        CUDA::cudart
        CUDA::cusparse
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
    file(
        GLOB SPACEPDHCG_CUDA_TEST_SOURCES
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*_test.cu"
    )
    foreach(source IN LISTS SPACEPDHCG_CUDA_TEST_SOURCES)
        get_filename_component(test_name "${source}" NAME_WE)
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
        )
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
        else()
            add_test(NAME "${test_name}" COMMAND "${test_name}")
        endif()
        set_tests_properties("${test_name}" PROPERTIES TIMEOUT 180)
    endforeach()
endif()

install(
    TARGETS spacepdhcg_cuda
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
