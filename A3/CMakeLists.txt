cmake_minimum_required(VERSION 3.18)
project(A3 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set default build type to Debug if not specified
if(CMAKE_BUILD_TYPE)
    message("Debug build enabled")
    set(CMAKE_BUILD_TYPE Debug)
    # set(CMAKE_CXX_FLAGS_DEBUG "-g")
    # set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")
    add_definitions(-DDEBUG)
endif()

# set(CMAKE_CXX_FLAGS "-O2")

# Set CMake policies to handle CMP0146 and CMP0104 warnings
cmake_policy(SET CMP0146 NEW)
cmake_policy(SET CMP0104 NEW)

# CUDA architectures (80 and 89 for Ampere GPUs)
set(CMAKE_CUDA_ARCHITECTURES 80 89)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# Find and link CUDA toolkit (including cuBLAS)
find_package(CUDAToolkit REQUIRED)

# # FetchContent for matplotplusplus
# include(FetchContent)
# FetchContent_Declare(matplotplusplus
#     GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
#     GIT_TAG origin/master
# )
# FetchContent_MakeAvailable(matplotplusplus)

# Add the executable for the project
add_executable(assignment3 main.cu)

# Include directories
target_include_directories(assignment3 PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Link CUDA and cuBLAS libraries, and Matplot++ (use matplot target provided by FetchContent)
target_link_libraries(assignment3 PRIVATE 
    CUDA::cublas
    # matplot  # The target is typically `matplot` for matplotplusplus, but can vary
)

add_subdirectory(src)

# message(STATUS "CUDA Toolkit Library: ${CUDA_TOOLKIT_ROOT_DIR}")
# message(STATUS "CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
# message(STATUS "Linked libraries: ${CUDA_LIBRARIES}")
