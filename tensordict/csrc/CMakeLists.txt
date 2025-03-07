cmake_minimum_required(VERSION 3.12)
project(tensordict)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the Python executable to the one from your virtual environment
# set(Python3_EXECUTABLE "/Users/vmoens/venv/rl2/bin/python3.10")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

file(GLOB SOURCES "*.cpp")

add_library(_C MODULE ${SOURCES})

set_target_properties(_C PROPERTIES
    OUTPUT_NAME "_C"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/../"
    PREFIX ""  # Remove 'lib' prefix
    SUFFIX ".so"  # Ensure correct suffix for macOS/Linux
)

target_include_directories(_C PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(_C PRIVATE Python3::Python pybind11::module)
