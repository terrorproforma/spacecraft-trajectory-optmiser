cmake_minimum_required(VERSION 3.24)
project(spacepdhcg_native_core VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CTest)

add_library(
    spacepdhcg_native_core
    SHARED
        src/cw.cpp
        src/cw_problem.cpp
        src/native_c_api.cpp
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
    endif()
endfunction()

spacepdhcg_enable_warnings(spacepdhcg_native_core)

option(SPACEPDHCG_NATIVE_BUILD_TESTS "Build native-core tests" ON)
if(SPACEPDHCG_NATIVE_BUILD_TESTS)
    add_executable(spacepdhcg_native_core_smoke tests/native_core_smoke.cpp)
    target_link_libraries(spacepdhcg_native_core_smoke PRIVATE spacepdhcg::native_core)
    target_compile_features(spacepdhcg_native_core_smoke PRIVATE cxx_std_20)
    set_target_properties(spacepdhcg_native_core_smoke PROPERTIES CXX_EXTENSIONS OFF)
    spacepdhcg_enable_warnings(spacepdhcg_native_core_smoke)
    add_test(NAME native-core-smoke COMMAND spacepdhcg_native_core_smoke)
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
