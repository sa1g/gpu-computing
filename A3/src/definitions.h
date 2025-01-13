#pragma once
#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#define EXPERIMENTS 10 // Number of experiments (how many kernel we have.)

#define RUNS 14 // Dimension of the matrix

#ifndef DTYPE
#define DTYPE float
#endif

#ifndef TILE_DIM
#define TILE_DIM 32
#endif

#ifndef BLOCK_ROWS
#define BLOCK_ROWS 8
#endif

#ifndef NUM_REPS
#define NUM_REPS 100
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#define INIT_CUDA_TIMER      \
    cudaEvent_t start, stop; \
    cudaEventCreate(&start); \
    cudaEventCreate(&stop);

#define START_CUDA_TIMER \
    cudaEventRecord(start);

#define STOP_CUDA_TIMER                       \
    cudaEventRecord(stop);                    \
    checkCudaErrors(cudaDeviceSynchronize()); \
    cudaEventSynchronize(stop);

#define GET_CUDA_ELAPSED_TIME                        \
    ({                                               \
        float runtime = 0.0f;                        \
        cudaEventElapsedTime(&runtime, start, stop); \
        runtime;                                     \
    })

#define DELETE_CUDA_TIMER        \
    if (start)                   \
        cudaEventDestroy(start); \
    if (stop)                    \
        cudaEventDestroy(stop);

#endif
