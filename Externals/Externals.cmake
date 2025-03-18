#
#  Copyright © 2024-Present, Arkin Terli. All rights reserved.
#
#  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
#  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
#  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
#  trade secret or copyright law. Dissemination of this information or reproduction of this
#  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

include(ExternalProject)

# ---------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------------

# Builds and installs external git projects.
function(add_external_git_project)
    set(options)
    set(oneValueArgs NAME GIT_REPOSITORY GIT_TAG EXTERNALS_BIN_DIR BUILD_TYPE)
    set(multiValueArgs CMAKE_ARGS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    message(STATUS "Configuring External Project: ${ARG_NAME}")
    set(lib_dir "${ARG_EXTERNALS_BIN_DIR}/${ARG_NAME}")

    ExternalProject_Add(
            ${ARG_NAME}
            GIT_REPOSITORY  ${ARG_GIT_REPOSITORY}
            GIT_TAG         ${ARG_GIT_TAG}
            PREFIX          "${lib_dir}/prefix"
            SOURCE_DIR      "${lib_dir}/src"
            STAMP_DIR       "${lib_dir}/stamp"
            BINARY_DIR      "${lib_dir}/build"
            INSTALL_DIR     "${lib_dir}/install"
            DOWNLOAD_DIR    "${lib_dir}/download"
            LOG_DIR         "${lib_dir}/log"
            CMAKE_ARGS      -DCMAKE_BUILD_TYPE=${ARG_BUILD_TYPE}
                            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                            ${ARG_CMAKE_ARGS}
            LOG_CONFIGURE ON
            LOG_BUILD ON
            LOG_INSTALL ON
            LOG_UPDATE ON
            LOG_PATCH ON
            LOG_TEST ON
            LOG_MERGED_STDOUTERR ON
            LOG_OUTPUT_ON_FAILURE ON
            GIT_SUBMODULES_RECURSE ON
            GIT_PROGRESS OFF
            GIT_SHALLOW  ON
            BUILD_ALWAYS ON
    )

    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES "${lib_dir}")

    include_directories(${lib_dir}/install/include)
    link_directories(${lib_dir}/install/lib)
endfunction()


# Installs header/source only project.
function(add_external_header_only_project)
    set(options)
    set(oneValueArgs NAME URL EXTERNALS_BIN_DIR)
    set(multiValueArgs)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    message(STATUS "Configuring External Project: ${ARG_NAME}")
    set(lib_dir "${ARG_EXTERNALS_BIN_DIR}/${ARG_NAME}")

    ExternalProject_Add(
            ${ARG_NAME}
            URL            ${ARG_URL}
            PREFIX         "${lib_dir}/prefix"
            SOURCE_DIR     "${lib_dir}/src"
            LOG_DIR        "${lib_dir}/log"
            CONFIGURE_COMMAND   ""
            BUILD_COMMAND       ""
            INSTALL_COMMAND     ""
            LOG_CONFIGURE ON
            LOG_BUILD ON
            LOG_INSTALL ON
            LOG_UPDATE ON
            LOG_PATCH ON
            LOG_TEST ON
            LOG_MERGED_STDOUTERR ON
            LOG_OUTPUT_ON_FAILURE ON
            DOWNLOAD_NO_PROGRESS ON
    )

    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES "${lib_dir}")

    include_directories(${lib_dir}/src)
endfunction()

# ---------------------------------------------------------------------------------
# COMMON SETTINGS
# ---------------------------------------------------------------------------------

# Externals build and install folder.
set(EXTERNALS_BINARY_DIR "${CMAKE_BINARY_DIR}/Externals")

# Common cmake project settings for the external projects.
set(EXTERNAL_COMMON_CMAKE_ARGS
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
        -DCMAKE_POSITION_INDEPENDENT_CODE=${CMAKE_POSITION_INDEPENDENT_CODE}
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=${CMAKE_BUILD_WITH_INSTALL_RPATH}
        -DCMAKE_INSTALL_RPATH=${CMAKE_INSTALL_RPATH}
)

# ---------------------------------------------------------------------------------
# DOCOPT CPP
# ---------------------------------------------------------------------------------
add_external_git_project(
        NAME                docopt_cpp
        GIT_REPOSITORY      https://github.com/docopt/docopt.cpp.git
        GIT_TAG             ${EXTERNAL_DOCOPT_VERSION}
        CMAKE_ARGS          ${EXTERNAL_COMMON_CMAKE_ARGS}
                            -DBUILD_SHARED_LIBS=OFF
        EXTERNALS_BIN_DIR   ${EXTERNALS_BINARY_DIR}
        BUILD_TYPE          Release
)

# ---------------------------------------------------------------------------------
# DOCTEST CPP
# ---------------------------------------------------------------------------------
add_external_git_project(
        NAME                doctest_cpp
        GIT_REPOSITORY      https://github.com/doctest/doctest.git
        GIT_TAG             ${EXTERNAL_DOCTEST_VERSION}
        CMAKE_ARGS          ${EXTERNAL_COMMON_CMAKE_ARGS}
                            -DDOCTEST_WITH_TESTS=OFF
        EXTERNALS_BIN_DIR   ${EXTERNALS_BINARY_DIR}
        BUILD_TYPE          Release
)

# ---------------------------------------------------------------------------------
# YAML CPP
# ---------------------------------------------------------------------------------
add_external_git_project(
        NAME                yaml_cpp
        GIT_REPOSITORY      https://github.com/jbeder/yaml-cpp.git
        GIT_TAG             ${EXTERNAL_YAML_VERSION}
        CMAKE_ARGS          ${EXTERNAL_COMMON_CMAKE_ARGS}
        EXTERNALS_BIN_DIR   ${EXTERNALS_BINARY_DIR}
        BUILD_TYPE          Release
)

# ---------------------------------------------------------------------------------
# METAL CPP
# ---------------------------------------------------------------------------------
if (APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    add_external_header_only_project(
            NAME                metal_cpp
            URL                 ${EXTERNAL_METAL_CPP_URL}
            EXTERNALS_BIN_DIR   ${EXTERNALS_BINARY_DIR}
    )
endif()
