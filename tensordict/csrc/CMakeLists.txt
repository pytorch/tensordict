cmake_minimum_required(VERSION 3.18)
project(tensordict)

# Try C++20 first, fall back to C++17 if not available
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Check if C++20 is supported, fall back to C++17 if not
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++20" CXX20_SUPPORTED)
if(NOT CXX20_SUPPORTED)
    check_cxx_compiler_flag("-std=c++17" CXX17_SUPPORTED)
    if(CXX17_SUPPORTED)
        set(CMAKE_CXX_STANDARD 17)
        message(STATUS "C++20 not supported, falling back to C++17")
    else()
        message(FATAL_ERROR "Neither C++20 nor C++17 are supported by the compiler")
    endif()
endif()

# Set the Python executable to the one from your virtual environment
if(APPLE) # Check if the target OS is OSX/macOS
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
    include(FindPythonPyEnv)
    
    # Check for Apple Silicon (ARM64)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        message(STATUS "Detected Apple Silicon (ARM64) architecture")
        # Ensure we're building for the correct architecture
        set(CMAKE_OSX_ARCHITECTURES "arm64")
    endif()
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 2.13 REQUIRED)

file(GLOB SOURCES "*.cpp")

add_library(_C MODULE ${SOURCES})

if(WIN32)
    set_target_properties(_C PROPERTIES
        OUTPUT_NAME "_C"
        PREFIX ""  # Remove 'lib' prefix
        SUFFIX ".pyd"
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

find_package(Python COMPONENTS Development.Static)
target_link_libraries(_C ${Python_STATIC_LIBRARIES})

target_include_directories(_C PRIVATE ${PROJECT_SOURCE_DIR})

#if(APPLE OR WIN32) # Check if the target OS is OSX/macOS
target_link_libraries(_C PRIVATE pybind11::module)
#else()
#    target_link_libraries(_C PRIVATE Python3::Python pybind11::module)
#endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -fsanitize=address")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# More flexible macOS deployment target
if(APPLE)
    # Try to detect the minimum supported macOS version
    execute_process(
        COMMAND sw_vers -productVersion
        OUTPUT_VARIABLE MACOS_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(MACOS_VERSION)
        # Extract major version
        string(REGEX MATCH "^([0-9]+)" MACOS_MAJOR_VERSION "${MACOS_VERSION}")
        set(MACOS_MAJOR_VERSION "${CMAKE_MATCH_1}")
        # Use a reasonable minimum, but not too restrictive
        if(MACOS_MAJOR_VERSION GREATER_EQUAL 14)
            set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum OS X deployment version")
        else()
            set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum OS X deployment version")
        endif()
    else()
        # Fallback to a reasonable default
        set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum OS X deployment version")
    endif()
    message(STATUS "Setting macOS deployment target to ${CMAKE_OSX_DEPLOYMENT_TARGET}")
    
    # Additional macOS-specific compiler flags for better compatibility
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    endif()
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

if(WIN32)
    add_custom_command(TARGET _C POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_C> "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<TARGET_FILE_NAME:_C>"
    )
endif()
