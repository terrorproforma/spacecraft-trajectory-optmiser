cmake_minimum_required(VERSION 3.24)
project(spacepdhcg_cpp_core LANGUAGES CXX)

include(CTest)

add_library(spacepdhcg_cpp_core INTERFACE)
target_include_directories(
    spacepdhcg_cpp_core
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_compile_features(spacepdhcg_cpp_core INTERFACE cxx_std_20)

option(SPACEPDHCG_CORE_WARNINGS_AS_ERRORS "Treat C++ core warnings as errors" ON)
option(SPACEPDHCG_CORE_ENABLE_SANITIZERS "Enable address and undefined-behaviour sanitizers" OFF)

function(spacepdhcg_add_core_test target source)
    add_executable(${target} ${CMAKE_CURRENT_SOURCE_DIR}/../tests/${source})
    target_link_libraries(${target} PRIVATE spacepdhcg_cpp_core)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
        if(SPACEPDHCG_CORE_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
        if(SPACEPDHCG_CORE_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
        if(SPACEPDHCG_CORE_ENABLE_SANITIZERS)
            target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
            target_link_options(${target} PRIVATE -fsanitize=address,undefined)
        endif()
    endif()
    add_test(NAME ${target} COMMAND ${target})
endfunction()

if(BUILD_TESTING)
    spacepdhcg_add_core_test(spacepdhcg_fixed_cqp_smoke fixed_cqp_smoke.cpp)
    spacepdhcg_add_core_test(spacepdhcg_powered_descent_smoke powered_descent_smoke.cpp)
    spacepdhcg_add_core_test(spacepdhcg_scenario_layout_smoke scenario_layout_smoke.cpp)
    spacepdhcg_add_core_test(spacepdhcg_scvx_policy_smoke scvx_policy_smoke.cpp)
    spacepdhcg_add_core_test(spacepdhcg_lambert_smoke lambert_smoke.cpp)
endif()
