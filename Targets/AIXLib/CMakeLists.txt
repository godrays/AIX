#
#  Copyright © 2024-Present, Arkin Terli. All rights reserved.
#
#  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
#  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
#  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
#  trade secret or copyright law. Dissemination of this information or reproduction of this
#  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

set(TARGET_NAME AIXLib)

set(AIX_LIB_TYPE SHARED)
if (AIX_BUILD_STATIC)
    set(AIX_LIB_TYPE STATIC)
endif()

set(SOURCE_FILES
       aixDevices.cpp
)

# Add metal device support for Apple Silicon
if (APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    set(SOURCE_FILES ${SOURCE_FILES}
            aixDeviceMetal.cpp
    )
endif()

# Build the following targets only on macOS with Apple Silicon.
add_library(AIXLib ${AIX_LIB_TYPE} ${SOURCE_FILES})

# Add metal device support for Apple Silicon
if (APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    add_dependencies(${TARGET_NAME} metal_cpp)
    target_link_libraries(${TARGET_NAME} PRIVATE
            "-framework Foundation"
            "-framework Metal"
    )
endif()

install(FILES aix.hpp aixDevices.hpp aixFloat16.hpp aixVersion.hpp DESTINATION include)
install(TARGETS ${TARGET_NAME} ARCHIVE DESTINATION lib)
