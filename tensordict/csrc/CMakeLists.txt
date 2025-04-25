cmake_minimum_required(VERSION 3.22)
project(tensordict)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the Python executable to the one from your virtual environment
if(APPLE) # Check if the target OS is OSX/macOS
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
    include(FindPythonPyEnv)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 2.13 REQUIRED)

file(GLOB SOURCES "*.cpp")

add_library(_C MODULE ${SOURCES})

if(WIN32)
    set_target_properties(_C PROPERTIES
        OUTPUT_NAME "_C"
        PREFIX ""  # Remove 'lib' prefix
        SUFFIX "${CMAKE_SHARED_LIBRARY_SUFFIX}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
    )
else()
    set_target_properties(_C PROPERTIES
        OUTPUT_NAME "_C"
        PREFIX ""  # Remove 'lib' prefix
        SUFFIX ".so"  # Ensure correct suffix for macOS/Linux (consider using CMAKE_SHARED_LIBRARY_SUFFIX instead for cross-platform compatibility)
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
    )
endif()

target_include_directories(_C PRIVATE ${PROJECT_SOURCE_DIR})

if(APPLE OR WIN32) # Check if the target OS is OSX/macOS
    target_link_libraries(_C PRIVATE pybind11::module)
else()
    target_link_libraries(_C PRIVATE Python3::Python pybind11::module)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -fsanitize=address")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()
set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "Minimum OS X deployment version")
set(CMAKE_VERBOSE_MAKEFILE ON)

if(WIN32)
    add_custom_command(TARGET _C POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_C> "${CMAKE_CURRENT_SOURCE_DIR}/tensordict/$<TARGET_FILE_NAME:_C>"
    )
endif()
