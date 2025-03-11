cmake_minimum_required(VERSION 3.12)
project(tensordict)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the Python executable to the one from your virtual environment

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

file(GLOB SOURCES "*.cpp")

add_library(_C MODULE ${SOURCES})

set_target_properties(_C PROPERTIES
    OUTPUT_NAME "_C"
    PREFIX ""  # Remove 'lib' prefix
    SUFFIX ".so"  # Ensure correct suffix for macOS/Linux
)
set_target_properties(_C PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
)

target_include_directories(_C PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(_C PRIVATE Python3::Python pybind11::module)
