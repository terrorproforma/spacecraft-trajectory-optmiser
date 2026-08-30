cmake_minimum_required(VERSION 3.24)
project(spacepdhcg_native LANGUAGES CXX)

add_library(spacepdhcg_interface INTERFACE)
target_include_directories(
    spacepdhcg_interface
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(spacepdhcg_interface INTERFACE cxx_std_20)

option(SPACEPDHCG_BUILD_INTERFACE_SMOKE "Build the host-only interface smoke test" ON)
if(SPACEPDHCG_BUILD_INTERFACE_SMOKE)
    add_executable(spacepdhcg_interface_smoke tests/interface_smoke.cpp)
    target_link_libraries(spacepdhcg_interface_smoke PRIVATE spacepdhcg_interface)
    if(MSVC)
        target_compile_options(spacepdhcg_interface_smoke PRIVATE /W4 /WX)
    else()
        target_compile_options(
            spacepdhcg_interface_smoke
            PRIVATE
                -Wall
                -Wextra
                -Wpedantic
                -Werror
        )
    endif()
endif()
