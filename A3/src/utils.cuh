/**
 * @file utils.cuh
 *
 * @brief Utility functions for the matrix transpose assignment
 *
 * I know this file is mostly spaghetti code, but I'm trying to keep it simple
 *
 */
#pragma once
#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <array>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifndef KERNELS_CUH
#include "kernels.cuh"
#endif

#include "helper_cuda.h"

#include "definitions.h"

typedef void (*KernelFunction)(const float *, float *, size_t);

/**
 * dim: one side size of a square matrix
 * time_s: execution time in seconds
 */
double effectiveBandWidthSquaredMatrixTranspose(const unsigned int dim, double time_s);

void check_device();

void get_info_and_device(bool *coopLaunch);

void rand_init_matrix(DTYPE *matrix, size_t DIM);

void matrix_T(DTYPE *source, DTYPE *destination, size_t DIM);

DTYPE mError(unsigned int DIM, const DTYPE *A, const DTYPE *B);

void mem_copy(float *data_gpu, float *result_gpu, size_t mem_size, size_t dim, float *runtime, double *bandwidth);

void checkCudaError(const char *msg = "");

// void runInterBlock(dim3 grid_dim, dim3 block_dim, void *kernelArgs[], size_t N);

struct KernelData
{
    // float avg_runtime;
    // double avg_bandwidth;
    // double min_bandwidth;
    // double max_bandwidth;
    // float std_runtime;

    std::array<float, NUM_REPS> runtime;
    std::array<float, NUM_REPS> bandwidth;

    float error;
    std::string name;
};

void kernel_experiment(
    KernelFunction kernel,
    dim3 dimGrid,
    dim3 dimBlock,
    float *data_gpu,
    float *result_gpu,
    float *result_cpu,
    size_t mem_size,
    size_t dim,
    std::array<float, NUM_REPS> &runs_buffer,
    KernelData &data,
    DTYPE *reference);

void copy_experiment(
    KernelFunction kernel,
    dim3 dimGrid,
    dim3 dimBlock,
    float *data_gpu,
    float *result_gpu,
    float *result_cpu,
    size_t mem_size,
    size_t dim,
    std::array<float, NUM_REPS> &runtime_buffer,
    KernelData &data,
    DTYPE *reference);

void cuBLAS_experiment(
    float *data_gpu,
    float *result_cpu,
    float *result_gpu,
    size_t mem_size,
    size_t dim,
    std::array<float, NUM_REPS> &runs_buffer,
    KernelData &data,
    DTYPE *reference);


#endif
