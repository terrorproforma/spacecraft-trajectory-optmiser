cmake_minimum_required(VERSION 3.24)
project(spacepdhcg_all_cpp LANGUAGES CXX)

include(CTest)

add_library(spacepdhcg_all_cpp INTERFACE)
target_include_directories(
    spacepdhcg_all_cpp
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_compile_features(spacepdhcg_all_cpp INTERFACE cxx_std_20)

option(SPACEPDHCG_ALL_CPP_WARNINGS_AS_ERRORS "Treat native warnings as errors" ON)
option(SPACEPDHCG_ALL_CPP_ENABLE_SANITIZERS "Enable ASan and UBSan" OFF)

file(GLOB SPACEPDHCG_SMOKE_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/*_smoke.cpp
)

foreach(source IN LISTS SPACEPDHCG_SMOKE_SOURCES)
    get_filename_component(test_name ${source} NAME_WE)
    set(target_name spacepdhcg_${test_name})
    add_executable(${target_name} ${source})
    target_link_libraries(${target_name} PRIVATE spacepdhcg_all_cpp)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W4)
        if(SPACEPDHCG_ALL_CPP_WARNINGS_AS_ERRORS)
            target_compile_options(${target_name} PRIVATE /WX)
        endif()
    else()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
        if(SPACEPDHCG_ALL_CPP_WARNINGS_AS_ERRORS)
            target_compile_options(${target_name} PRIVATE -Werror)
        endif()
        if(SPACEPDHCG_ALL_CPP_ENABLE_SANITIZERS)
            target_compile_options(
                ${target_name}
                PRIVATE
                    -fsanitize=address,undefined
                    -fno-omit-frame-pointer
            )
            target_link_options(${target_name} PRIVATE -fsanitize=address,undefined)
        endif()
    endif()
    add_test(NAME ${target_name} COMMAND ${target_name})
endforeach()
