# convDirect CMakeLists.txt

cmake_minimum_required(VERSION 3.13)

project(convDirect VERSION 1.0 LANGUAGES C CXX)

find_package(OpenMP REQUIRED)

# ========================================================================
# Find the BLIS library
# ------------------------------------------------------------------------
# Convert $ENV{LD_LIBRARY_PATH} to a list of paths (using ; as separator)
if (DEFINED ENV{LD_LIBRARY_PATH})
    string(REPLACE ":" ";" LIBRARY_DIRS $ENV{LD_LIBRARY_PATH})
endif ()
# Do find the library
find_library(
        BLIS_LIBRARY
        NAMES libblis.so.4 libblis.so.3
        HINTS
        ${CMAKE_PREFIX_PATH}/lib
        ${CMAKE_INSTALL_PREFIX}/lib
        ${LIBRARY_DIRS})
if (BLIS_LIBRARY)
    message(STATUS "Linking against the BLIS library in '${BLIS_LIBRARY}'")
    # Set BLIS_LIBRARY_NAME, BLIS_LIBRARY_PATH and BLIS_ABI_VERSION
    set(BLIS_FOUND TRUE)
    get_filename_component(BLIS_LIBRARY_NAME ${BLIS_LIBRARY} NAME)
    get_filename_component(BLIS_LIBRARY_PATH ${BLIS_LIBRARY} DIRECTORY)
    get_filename_component(BLIS_PREFIX_PATH ${BLIS_LIBRARY_PATH} DIRECTORY)
    string(SUBSTRING ${BLIS_LIBRARY_NAME} 11 1 BLIS_ABI_VERSION)
    message(STATUS ${BLIS_PREFIX_PATH})
    include_directories(SYSTEM PUBLIC ${BLIS_PREFIX_PATH}/include)
    add_compile_options(-DBLIS_FOUND -DBLIS_ABI_VERSION=${BLIS_ABI_VERSION})
else ()
    set(BLIS_FOUND FALSE)
    message(WARNING
            "Could not find the BLIS library, algorithms that require it will be skipped.\n"
            "The BLIS library was searched in:\n"
            "  - CMAKE_PREFIX_PATH/lib/: ${CMAKE_PREFIX_PATH}/lib/\n"
            "  - CMAKE_INSTALL_PREFIX/lib/: ${CMAKE_INSTALL_PREFIX}/lib/\n"
            "  - \${LD_LIBRARY_PATH}: $ENV{LD_LIBRARY_PATH}"
            " Please read the README.md on how to help finding the BLIS library."
            )
endif ()
# ========================================================================


# ========================================================================
# Set ARCH compile option, i.e., one of:
#  + ARCH__x86_64__
#  + ARCH__aarch64__
#  + ARCH__riscv__
# ------------------------------------------------------------------------
add_compile_options(-DARCH__${CMAKE_SYSTEM_PROCESSOR}__)
# ========================================================================


# ========================================================================
# Common compilation options and include directories
# ------------------------------------------------------------------------
# Common compilation options to all objects
add_compile_options(
        -O3 -mtune=native -march=native
        -ftree-vectorize
        -fopenmp
        -fPIC
)
# Include SYSTEM directories (no compilation warnings due to these)
include_directories(
        SYSTEM PUBLIC
        "${CMAKE_INSTALL_PREFIX}/include"
)
if (CMAKE_PREFIX_PATH)
    include_directories(
            SYSTEM PUBLIC
            "${CMAKE_PREFIX_PATH}/include"
            "${CMAKE_PREFIX_PATH}"
    )
endif ()
# ========================================================================


# ========================================================================
# Add data type compilation option (force FP32 if no one is specified)
# ------------------------------------------------------------------------
if (INT8)
    add_compile_options(-DINT8)
    set(DTYPE int)
elseif (FP64)
    add_compile_options(-DFP64)
    set(DTYPE double)
else ()
    add_compile_options(-DFP32)
    set(DTYPE float)
endif ()
# ========================================================================


set(N 0)

add_subdirectory(src)

option(COMPILE_TESTS "Compile tests" OFF)

if (COMPILE_TESTS)
    add_subdirectory(tests)
endif ()
