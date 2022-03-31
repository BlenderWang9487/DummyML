cmake_minimum_required(VERSION 3.4...3.18)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(dummyml)

add_subdirectory(extern/pybind11)
include_directories(${dummyml_SOURCE_DIR}/include/dummyml)
set(
    CMAKE_LIBRARY_OUTPUT_DIRECTORY
    ${dummyml_SOURCE_DIR}/dummyml
    )

pybind11_add_module(
    dummyml
    ${dummyml_SOURCE_DIR}/src/dummyml/main.cpp
    ${dummyml_SOURCE_FILES}
    )