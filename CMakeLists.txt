#
#  Copyright © 2024-Present, Arkin Terli. All rights reserved.
#
#  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
#  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
#  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
#  trade secret or copyright law. Dissemination of this information or reproduction of this
#  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

cmake_minimum_required(VERSION 3.24)

project(AIX)

# Set default build options.

option(AIX_BUILD_TESTS    "Build AIX test application"     OFF)
option(AIX_BUILD_EXAMPLES "Build AIX example applications" OFF)
option(AIX_BUILD_STATIC   "Build static AIX library"       OFF)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;Prof;CCov;ASan;TSan" CACHE STRING "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

# Set RPATH to look in the loader directory first to load libraries.
if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path")
elseif (UNIX)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

# Set default build type as Release.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)

if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-function")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
    set(CMAKE_CXX_FLAGS_PROF "${CMAKE_CXX_FLAGS_RELEASE} -O1 -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer")
    set(CMAKE_CXX_FLAGS_CCOV "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
    set(CMAKE_CXX_FLAGS_ASAN "${CMAKE_CXX_FLAGS} -g -O1 -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS_TSAN "${CMAKE_CXX_FLAGS} -g -O2 -fsanitize=thread -fPIE")
endif()

# Set external library versions.
set(EXTERNAL_DOCOPT_VERSION 400e6dd)
set(EXTERNAL_DOCTEST_VERSION v2.4.11)
set(EXTERNAL_YAML_VERSION 0.8.0)
set(EXTERNAL_METAL_CPP_URL "https://developer.apple.com/metal/cpp/files/metal-cpp_macOS14.2_iOS17.2.zip")

# Build external libraries.
include(Externals/Externals.cmake)

# Include folders.
include_directories(Targets/AIXLib)

# Target folders.
add_subdirectory(Targets/AIXLib)

if (AIX_BUILD_TESTS)
    add_subdirectory(Targets/AIXBenchmarks)
    add_subdirectory(Targets/AIXTests)
endif ()

if (AIX_BUILD_EXAMPLES)
    add_subdirectory(Targets/AIXExamples/XORApp)
    add_subdirectory(Targets/AIXExamples/XORLayerApp)
    add_subdirectory(Targets/AIXExamples/XORSequentialApp)

    # Build the following targets only on macOS with Apple Silicon.
    if (APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        add_subdirectory(Targets/AIXExamples/XORMetalApp)
    endif()
endif ()

# Install.
if (NOT AIX_BUILD_STATIC AND (AIX_BUILD_TESTS OR AIX_BUILD_EXAMPLES))
    install(TARGETS AIXLib DESTINATION ${CMAKE_INSTALL_PREFIX})
endif ()
