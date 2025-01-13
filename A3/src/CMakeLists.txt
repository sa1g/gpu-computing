cmake_policy(SET CMP0076 NEW)

# List of source files
set(SRC_FILES
    definitions.h
    helper_cuda.h
    helper_string.h
    
    utils.cu
    kernels.cu

    # plot.cc
)

# Add the source files to the target
target_sources(assignment3 PRIVATE ${SRC_FILES})

# Ensure the target includes this directory for headers
target_include_directories(assignment3 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
