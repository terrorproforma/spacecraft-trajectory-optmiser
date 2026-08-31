cmake_minimum_required(VERSION 3.24)
project(
    SpacePDHCG
    VERSION 0.1.0
    DESCRIPTION "Persistent conic optimisation infrastructure for spacecraft trajectories"
    LANGUAGES CXX
)

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(SPACEPDHCG_BUILD_C_API "Build the stable native C ABI library" ON)
option(SPACEPDHCG_BUILD_NATIVE_TESTS "Build every host-only native smoke test" ON)
option(SPACEPDHCG_NATIVE_WARNINGS_AS_ERRORS "Treat native warnings as errors" ON)
option(SPACEPDHCG_NATIVE_ENABLE_SANITIZERS "Enable ASan and UBSan" OFF)

add_library(spacepdhcg_interface INTERFACE)
add_library(SpacePDHCG::headers ALIAS spacepdhcg_interface)
target_include_directories(
    spacepdhcg_interface
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(spacepdhcg_interface INTERFACE cxx_std_20)
set_target_properties(spacepdhcg_interface PROPERTIES EXPORT_NAME headers)

function(spacepdhcg_apply_native_options target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
        if(SPACEPDHCG_NATIVE_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
        if(SPACEPDHCG_NATIVE_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
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

if(SPACEPDHCG_BUILD_C_API)
    add_library(spacepdhcg_c_api SHARED src/c_api.cpp)
    add_library(SpacePDHCG::c_api ALIAS spacepdhcg_c_api)
    target_link_libraries(spacepdhcg_c_api PUBLIC spacepdhcg_interface)
    set_target_properties(
        spacepdhcg_c_api
        PROPERTIES
            EXPORT_NAME c_api
            OUTPUT_NAME spacepdhcg
            VERSION ${PROJECT_VERSION}
            SOVERSION ${PROJECT_VERSION_MAJOR}
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN YES
    )
    spacepdhcg_apply_native_options(spacepdhcg_c_api)
endif()

if(BUILD_TESTING AND SPACEPDHCG_BUILD_NATIVE_TESTS)
    file(
        GLOB SPACEPDHCG_NATIVE_SMOKE_SOURCES
        CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/*_smoke.cpp
    )
    foreach(source IN LISTS SPACEPDHCG_NATIVE_SMOKE_SOURCES)
        get_filename_component(test_name ${source} NAME_WE)
        set(target_name spacepdhcg_${test_name})
        add_executable(${target_name} ${source})
        target_link_libraries(${target_name} PRIVATE spacepdhcg_interface)
        spacepdhcg_apply_native_options(${target_name})
        add_test(NAME ${target_name} COMMAND ${target_name})
    endforeach()
endif()

install(
    TARGETS spacepdhcg_interface
    EXPORT SpacePDHCGTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
if(TARGET spacepdhcg_c_api)
    install(
        TARGETS spacepdhcg_c_api
        EXPORT SpacePDHCGTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()
install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/SpacePDHCGConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/SpacePDHCGConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SpacePDHCG
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/SpacePDHCGConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/SpacePDHCGConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/SpacePDHCGConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SpacePDHCG
)
install(
    EXPORT SpacePDHCGTargets
    FILE SpacePDHCGTargets.cmake
    NAMESPACE SpacePDHCG::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SpacePDHCG
)

export(
    EXPORT SpacePDHCGTargets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/SpacePDHCGTargets.cmake
    NAMESPACE SpacePDHCG::
)
