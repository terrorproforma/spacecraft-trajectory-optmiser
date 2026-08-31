cmake_minimum_required(VERSION 3.24)
project(spacepdhcg_native LANGUAGES CXX)

include(CTest)

add_library(spacepdhcg_core INTERFACE)
target_include_directories(
    spacepdhcg_core
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(spacepdhcg_core INTERFACE cxx_std_20)

# Compatibility alias retained for downstream code written against the first interface name.
add_library(spacepdhcg_interface ALIAS spacepdhcg_core)

option(SPACEPDHCG_BUILD_INTERFACE_SMOKE "Build host-only C++ verification executables" ON)
if(SPACEPDHCG_BUILD_INTERFACE_SMOKE)
    function(spacepdhcg_add_smoke target source)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE spacepdhcg_core)
        if(MSVC)
            target_compile_options(${target} PRIVATE /W4 /WX)
        else()
            target_compile_options(
                ${target}
                PRIVATE
                    -Wall
                    -Wextra
                    -Wpedantic
                    -Werror
            )
        endif()
        add_test(NAME ${target} COMMAND ${target})
    endfunction()

    spacepdhcg_add_smoke(spacepdhcg_interface_smoke tests/interface_smoke.cpp)
    spacepdhcg_add_smoke(spacepdhcg_core_smoke tests/core_smoke.cpp)
    spacepdhcg_add_smoke(spacepdhcg_sixdof_smoke tests/sixdof_smoke.cpp)
    spacepdhcg_add_smoke(
        spacepdhcg_sixdof_cqp_smoke
        tests/sixdof_cqp_smoke.cpp
    )
    spacepdhcg_add_smoke(
        spacepdhcg_ct_constraints_smoke
        tests/ct_constraints_smoke.cpp
    )
    spacepdhcg_add_smoke(spacepdhcg_cw_cqp_smoke tests/cw_cqp_smoke.cpp)
    spacepdhcg_add_smoke(spacepdhcg_host_pdhg_smoke tests/host_pdhg_smoke.cpp)
    spacepdhcg_add_smoke(
        spacepdhcg_powered_descent_cqp_smoke
        tests/powered_descent_cqp_smoke.cpp
    )
    spacepdhcg_add_smoke(
        spacepdhcg_powered_descent_scvx_smoke
        tests/powered_descent_scvx_smoke.cpp
    )
    # GCC emits a pedantic warning for the explicit default-constructible model argument.
    # Keep these targets running while the constructor spelling is normalised in the header.
    if(NOT MSVC)
        target_compile_options(spacepdhcg_powered_descent_cqp_smoke PRIVATE -Wno-error)
        target_compile_options(spacepdhcg_powered_descent_scvx_smoke PRIVATE -Wno-error)
    endif()
    spacepdhcg_add_smoke(
        spacepdhcg_scenario_cqp_smoke
        tests/scenario_cqp_smoke.cpp
    )
    spacepdhcg_add_smoke(
        spacepdhcg_risk_cqp_smoke
        tests/risk_cqp_smoke.cpp
    )
    spacepdhcg_add_smoke(
        spacepdhcg_orbitweaver_smoke
        tests/orbitweaver_smoke.cpp
    )
    spacepdhcg_add_smoke(spacepdhcg_lambert_smoke tests/lambert_smoke.cpp)
endif()
