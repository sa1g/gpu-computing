#include <iostream>
#include <chrono>
#include <sys/time.h>
#include <cuda_runtime.h>

#include "helper_cuda.h"
#include <string>

#define EXPERIMENTS 4

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

// #define PRETTY_PRINT

#define INIT_CUDA_TIMER      \
    cudaEvent_t start, stop; \
    cudaEventCreate(&start); \
    cudaEventCreate(&stop);

#define START_CUDA_TIMER \
    cudaEventRecord(start);

#define STOP_CUDA_TIMER                       \
    checkCudaErrors(cudaDeviceSynchronize()); \
    cudaEventRecord(stop);                    \
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

/**
 * Simple copy reference kernel
 *
 * Arguments:
 * - `idata`: source matrix defined in a single pointer.
 * - `odata`: destination matrix, can be pre-filled, defined in a single pointer.
 */
__global__ void copy(const DTYPE *idata, DTYPE *odata)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        odata[(y + i) * width + x] = idata[(y + i) * width + x];
    }
}

/**
 * Naive implementation of matrix transpose with memory coalescing.
 * Global memory reads are coalesced but writes are not.
 *
 * @param idata: input matrix (single pointer)
 * @param odata: output matrix (single pointer)
 * @param width: matrix width
 * @param height: matrix height
 *
 * @note Source: [NVidia Paper](https://www.cs.colostate.edu/~cs675/MatrixTranspose.pdf)
 * @note Source: [NVidia cuda-samples](https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/transpose/transpose.cu#L98)
 */
__global__ void transposeNaive(const DTYPE *idata, DTYPE *odata)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        odata[width * x + (y + i)] = idata[(y + i) * width + x];
    }
}

/**
 * Use shared memory to achieve coalesing in both
 * reads and writes.
 *
 * @param idata:    
 * @param odata:
 *
 * @note Source: [NVidia cuda-samples](https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/transpose/transpose.cu#L111)
 */
__global__ void transposeCoalesced(const float *idata, float *odata)
{
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        tile[threadIdx.y + i][threadIdx.x] = idata[(y + i) * width + x];
    }

    __syncthreads();
    // block offset
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        odata[(y + i) * width + x] = tile[threadIdx.x][threadIdx.y + i];
    }
}

__global__ void transposeCoalescedNoBankConflicts(const float *idata, float *odata)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        tile[threadIdx.y + i][threadIdx.x] = idata[(y + i) * width + x];
    }

    __syncthreads();
    // block offset
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        odata[(y + i) * width + x] = tile[threadIdx.x][threadIdx.y + i];
    }
}

double effectiveBandWidthSquaredMatrixTranspose(const unsigned int dim, double time)
{
    return 2 * (dim * dim) * sizeof(float) * 1e-6 * NUM_REPS / (time);
}

DTYPE mError(unsigned int DIM, const DTYPE *A, const DTYPE *B)
{
    int i, j;

    DTYPE error = static_cast<DTYPE>(0);
    for (i = 0; i < DIM; i++)
    {
        for (j = 0; j < DIM; j++)
        {
            error += fabs(A[i * DIM + j] - B[i * DIM + j]);
        }
    }

    return error;
}

int main(int argc, char *argv[])
{

    // INITIALIZE MATRIX SHAPE
    size_t DIM = 2;
    if (argc == 1)
    {
#ifdef PRETTY_PRINT
        std::cout
            << "No arguments" << std::endl
            << "\tNote that this works only with squared matrices." << std::endl;
#endif
    }

    if (argc > 1)
    {
        int exponent = std::stoi(argv[1]);
        if (exponent < 1)
        {
#ifdef PRETTY_PRINT
            std::cerr << "ERROR: exponent must be > 1!" << std::endl;
#endif
            return EXIT_FAILURE;
        }
        if (exponent > 14)
        {
#ifdef PRETTY_PRINT
            std::cerr << "ERROR: dude, you seriusly want to allocate more than 4GB of data?" << std::endl;
#endif
            return EXIT_FAILURE;
        }
        DIM = DIM << exponent;
    }
#ifdef PRETTY_PRINT
    printf("Matrix has size: [%zu x %zu]\n", DIM, DIM);
#endif
    // VERIFY DEFINED VARIABLES
    if (DIM % TILE_DIM)
    {
#ifdef PRETTY_PRINT
        std::cerr << "Matrix shape: [" << DIM << ", " << DIM << "] must be a multiple of TILE_DIM: " << TILE_DIM << std::endl;
#endif
        return EXIT_FAILURE;
    }

    // DEFINE MATRIX
    size_t mem_size = DIM * DIM * sizeof(DTYPE);

    DTYPE *data_cpu = (DTYPE *)malloc(mem_size);
    DTYPE *reference = (DTYPE *)malloc(mem_size);
    DTYPE *result_cpu = (DTYPE *)malloc(mem_size);
    DTYPE *data_gpu, *result_gpu;

    unsigned int repetitions = 0;
    double time = 0.0;

    const char *experimentsNames[EXPERIMENTS];
    DTYPE errors[EXPERIMENTS];
    double bandwidths[EXPERIMENTS];

    cudaMalloc(&data_gpu, mem_size);
    cudaMalloc(&result_gpu, mem_size);

    // INITIALIZE CPU MATRIX
    for (size_t i = 0; i < DIM; ++i)
    {
        for (size_t j = 0; j < DIM; ++j)
        {
            data_cpu[i * DIM + j] = static_cast<DTYPE>(rand()) / static_cast<DTYPE>(RAND_MAX);
        }
    }

    for (size_t i = 0; i < DIM; i++)
    {
        for (size_t j = 0; j < DIM; j++)
        {
            reference[j * DIM + i] = data_cpu[i * DIM + j];
        }
    }

    // COPY DATA TO GPU
    cudaMemcpy(data_gpu, data_cpu, mem_size, cudaMemcpyHostToDevice);
    cudaMemset(result_gpu, 0, mem_size);
#ifdef PRETTY_PRINT
    printf("======================================= Device properties ========================================\n");
#endif
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
#ifdef PRETTY_PRINT
        printf("Result = FAIL\n");
#endif
        exit(EXIT_FAILURE);
    }

    if (deviceCount == 0)
    {
#ifdef PRETTY_PRINT
        printf("No CUDA enabled devices available. Exiting");
#endif
        exit(EXIT_FAILURE);
    }
    else
    {
#ifdef PRETTY_PRINT
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
        printf("Using device 0\n");
#endif
    }

    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    // Calculate memory bandwidth
    int mem_clock_rate = deviceProp.memoryClockRate; // kHz
    int mem_width = deviceProp.memoryBusWidth;       // bits
    // 2 * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth/8)/ 1.0e6;
    double mem_bandwidth = (double)mem_clock_rate * 1e3 * (mem_width / 8) * 2 / 1e9;

#ifdef PRETTY_PRINT
    printf("Memory clock rate: %d MHz\n", mem_clock_rate / 1000);
    printf("Memory width: %d bits\n", mem_width);
    printf("Theoretical memory bandwidth: %f GBps\n", mem_bandwidth);
#endif
    // printf("====================================== Problem computations ======================================\n");
    // SET GRID AND BLOCK DIMENSIONS
    dim3 dim_grid(DIM / TILE_DIM, DIM / TILE_DIM, 1);
    dim3 dim_block(TILE_DIM, BLOCK_ROWS, 1);

    // ############# COPY (reference)
    // warm up
    copy<<<dim_grid, dim_block>>>(data_gpu, result_gpu);
    INIT_CUDA_TIMER;
    cudaMemset(result_gpu, 0, mem_size);
    START_CUDA_TIMER;
    for (repetitions = 0; repetitions < NUM_REPS; ++repetitions)
    {
        copy<<<dim_grid, dim_block>>>(data_gpu, result_gpu);
    }
    STOP_CUDA_TIMER;

    time = GET_CUDA_ELAPSED_TIME;
    experimentsNames[0] = "Copy - reference";
    bandwidths[0] = effectiveBandWidthSquaredMatrixTranspose(DIM, time);
    errors[0] = 0.0f;

    // ############# NAIVE 2
    transposeNaive<<<dim_grid, dim_block>>>(data_gpu, result_gpu);
    cudaMemset(result_gpu, 0, mem_size);
    START_CUDA_TIMER;
    for (repetitions = 0; repetitions < NUM_REPS; ++repetitions)
    {
        transposeNaive<<<dim_grid, dim_block>>>(data_gpu, result_gpu);
    }
    STOP_CUDA_TIMER;

    checkCudaErrors(cudaMemcpy(result_cpu, result_gpu, mem_size, cudaMemcpyDeviceToHost));

    time = GET_CUDA_ELAPSED_TIME;
    experimentsNames[1] = "Transpose Naive";
    bandwidths[1] = effectiveBandWidthSquaredMatrixTranspose(DIM, time);
    errors[1] = mError(DIM, result_cpu, reference);

    // ############# SHARED 1 COALESCED
    transposeCoalesced<<<dim_grid, dim_block>>>(data_gpu, result_gpu);
    cudaMemset(result_gpu, 0, mem_size);
    START_CUDA_TIMER;
    for (repetitions = 0; repetitions < NUM_REPS; ++repetitions)
    {
        transposeCoalesced<<<dim_grid, dim_block>>>(data_gpu, result_gpu);
    }
    STOP_CUDA_TIMER;

    checkCudaErrors(cudaMemcpy(result_cpu, result_gpu, mem_size, cudaMemcpyDeviceToHost));

    time = GET_CUDA_ELAPSED_TIME;
    experimentsNames[2] = "Transpose Shared Coalesced";
    bandwidths[2] = effectiveBandWidthSquaredMatrixTranspose(DIM, time);
    errors[2] = mError(DIM, result_cpu, reference);

    // ############# SHARED 2 COALESCED
    transposeCoalescedNoBankConflicts<<<dim_grid, dim_block>>>(data_gpu, result_gpu);
    cudaMemset(result_gpu, 0, mem_size);
    START_CUDA_TIMER;
    for (repetitions = 0; repetitions < NUM_REPS; ++repetitions)
    {
        transposeCoalescedNoBankConflicts<<<dim_grid, dim_block>>>(data_gpu, result_gpu);
    }
    STOP_CUDA_TIMER;

    checkCudaErrors(cudaMemcpy(result_cpu, result_gpu, mem_size, cudaMemcpyDeviceToHost));

    time = GET_CUDA_ELAPSED_TIME;
    experimentsNames[3] = "Transpose Naive Coalesced No Bank Conflicts";
    bandwidths[3] = effectiveBandWidthSquaredMatrixTranspose(DIM, time);
    errors[3] = mError(DIM, result_cpu, reference);

// ############# Final Print
#ifdef PRETTY_PRINT
    printf("====================================== Results ======================================\n");
    printf("%40s\t%25s\t%25s\n", "experiments", "Bandwidth [GBps]", "Errors #");

    for (int i = 0; i < EXPERIMENTS; i++)
    {
        printf("%40s\t%20.5f\t%20.5f\n", experimentsNames[i], bandwidths[i], errors[i]);
    }

#else
    printf("%zu,%i,%i,%i,%i,%f,%f,%f,%f\n", DIM, dim_grid.x, dim_grid.y, dim_block.x, dim_block.y, bandwidths[0], bandwidths[1], bandwidths[2], bandwidths[3]);
#endif

    // CLEANING
    DELETE_CUDA_TIMER;
    free(data_cpu);
    free(reference);
    cudaFree(data_gpu);
    cudaFree(result_gpu);

    return EXIT_SUCCESS;
}
