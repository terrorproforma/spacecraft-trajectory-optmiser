cmake_minimum_required(VERSION 3.24)
project(spacepdhcg_package_consumer LANGUAGES CXX)

include(CTest)
find_package(SpacePDHCG CONFIG REQUIRED)

add_executable(spacepdhcg_package_consumer main.cpp)
target_link_libraries(
    spacepdhcg_package_consumer
    PRIVATE
        SpacePDHCG::headers
        SpacePDHCG::c_api
)
target_compile_features(spacepdhcg_package_consumer PRIVATE cxx_std_20)
add_test(NAME spacepdhcg_package_consumer COMMAND spacepdhcg_package_consumer)
