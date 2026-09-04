cmake_minimum_required(VERSION 3.24)
project(spacepdhcg_native_core VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CTest)

add_library(
    spacepdhcg_native_core
    SHARED
        src/backend.cpp
        src/cw.cpp
        src/cw_problem.cpp
        src/native_c_api.cpp
        src/persistent_session.cpp
        src/powered_descent_3dof.cpp
        src/powered_descent_6dof.cpp
        src/powered_descent_cqp.cpp
        src/powered_descent_scvx.cpp
        src/robust_cqp.cpp
        src/scenario_tree.cpp
)
add_library(spacepdhcg::native_core ALIAS spacepdhcg_native_core)

target_include_directories(
    spacepdhcg_native_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(spacepdhcg_native_core PUBLIC cxx_std_20)
target_compile_definitions(spacepdhcg_native_core PRIVATE SPACEPDHCG_NATIVE_EXPORTS)
set_target_properties(
    spacepdhcg_native_core
    PROPERTIES
        CXX_EXTENSIONS OFF
        POSITION_INDEPENDENT_CODE ON
        VERSION ${PROJECT_VERSION}
        SOVERSION 0
)

option(
    SPACEPDHCG_NATIVE_ENABLE_SANITIZERS
    "Enable address and undefined-behaviour sanitizers"
    OFF
)

function(spacepdhcg_enable_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /WX)
    else()
        target_compile_options(
            ${target}
            PRIVATE
                -Wall
                -Wextra
                -Wpedantic
                -Wconversion
                -Wsign-conversion
                -Wshadow
                -Werror
        )
        if(SPACEPDHCG_NATIVE_ENABLE_SANITIZERS)
            target_compile_options(
                ${target}
                PRIVATE
                    -fsanitize=address,undefined
                    -fno-omit-frame-pointer
            )
            target_link_options(${target} PRIVATE -fsanitize=address,undefined)
        endif()
    endif()
endfunction()

spacepdhcg_enable_warnings(spacepdhcg_native_core)

option(SPACEPDHCG_NATIVE_BUILD_TESTS "Build native-core tests" ON)
function(spacepdhcg_add_native_test target test_name source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE spacepdhcg::native_core)
    target_compile_features(${target} PRIVATE cxx_std_20)
    set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)
    spacepdhcg_enable_warnings(${target})
    add_test(NAME ${test_name} COMMAND ${target})
endfunction()

if(SPACEPDHCG_NATIVE_BUILD_TESTS)
    spacepdhcg_add_native_test(
        spacepdhcg_native_core_smoke native-core-smoke tests/native_core_smoke.cpp
    )
    spacepdhcg_add_native_test(
        spacepdhcg_native_persistent_session_smoke
        native-persistent-session-smoke
        tests/persistent_session_smoke.cpp
    )
    spacepdhcg_add_native_test(
        spacepdhcg_native_powered_descent_smoke
        native-powered-descent-smoke
        tests/powered_descent_smoke.cpp
    )
    spacepdhcg_add_native_test(
        spacepdhcg_native_powered_descent_6dof_smoke
        native-powered-descent-6dof-smoke
        tests/powered_descent_6dof_smoke.cpp
    )
    spacepdhcg_add_native_test(
        spacepdhcg_native_powered_descent_cqp_smoke
        native-powered-descent-cqp-smoke
        tests/powered_descent_cqp_smoke.cpp
    )
    spacepdhcg_add_native_test(
        spacepdhcg_native_powered_descent_scvx_smoke
        native-powered-descent-scvx-smoke
        tests/powered_descent_scvx_smoke.cpp
    )
    spacepdhcg_add_native_test(
        spacepdhcg_native_robust_cqp_smoke native-robust-cqp-smoke tests/robust_cqp_smoke.cpp
    )
    spacepdhcg_add_native_test(
        spacepdhcg_native_scenario_tree_smoke
        native-scenario-tree-smoke
        tests/scenario_tree_smoke.cpp
    )
endif()

install(
    TARGETS spacepdhcg_native_core
    EXPORT spacepdhcgNativeTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(
    DIRECTORY ../include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
