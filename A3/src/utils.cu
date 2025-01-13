#include "utils.cuh"

double effectiveBandWidthSquaredMatrixTranspose(const unsigned int dim, double time_s)
{
    // Source: https://github.com/NVIDIA/cuda-samples/blob/master/Samples/1_Utilities/bandwidthTest/bandwidthTest.cu#L655
    size_t num_elements{dim * dim};
    size_t num_bytes{num_elements * sizeof(DTYPE) * 2};
    double time_ms = time_s * 1e3;
    double bandwidth{(num_bytes * 1e-6f) / time_ms};

    return bandwidth;
}

void check_device()
{
#ifdef DEBUG
    printf("======================================= Device properties ========================================\n");
#endif
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
#ifdef DEBUG
        printf("Result = FAIL\n");
#endif
        exit(EXIT_FAILURE);
    }

    if (deviceCount == 0)
    {
#ifdef DEBUG
        printf("No CUDA enabled devices available. Exiting");
#endif
        exit(EXIT_FAILURE);
    }
    else
    {
#ifdef DEBUG
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
        printf("Using device 0\n");
#endif
    }
}

void get_info_and_device(bool *coopLaunch)
{
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int mem_clock_rate = deviceProp.memoryClockRate; // kHz
    int mem_width = deviceProp.memoryBusWidth;       // bits
    // 2 * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth/8)/ 1.0e6;
    double mem_bandwidth = (double)mem_clock_rate * 1e3 * (mem_width / 8) * 2 / 1e9;

    // deviceProp.cooperativeLaunch
    *coopLaunch = deviceProp.cooperativeLaunch;

#ifdef DEBUG
    printf("Memory clock rate: %d MHz\n", mem_clock_rate / 1000);
    printf("Memory width: %d bits\n", mem_width);
    // printf("Theoretical memory bandwidth: %f GBps\n", mem_bandwidth);
    float const peak_bandwidth{
        static_cast<float>(2.0f * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8) / 1.0e6)};
    printf(" Theoretical memory bandwidth: %f GB/s\n", peak_bandwidth);

#endif
}

void rand_init_matrix(DTYPE *matrix, size_t DIM)
{
    for (size_t i = 0; i < DIM; ++i)
    {
        for (size_t j = 0; j < DIM; ++j)
        {
            matrix[i * DIM + j] = static_cast<DTYPE>(rand()) / static_cast<DTYPE>(RAND_MAX);
        }
    }
}

void matrix_T(DTYPE *source, DTYPE *destination, size_t DIM)
{
    for (size_t i = 0; i < DIM; i++)
    {
        for (size_t j = 0; j < DIM; j++)
        {
            destination[j * DIM + i] = source[i * DIM + j];
        }
    }
}

DTYPE mError(unsigned int DIM, const DTYPE *A, const DTYPE *B)
{
    int i, j;

    // DTYPE error = static_cast<DTYPE>(0);
    int error = 0;
    for (i = 0; i < DIM; i++)
    {
        for (j = 0; j < DIM; j++)
        {
            if (fabs(A[i * DIM + j] - B[i * DIM + j]) != 0.0)
            {
                error++;
            }
        }
    }

    return error;
}

void mem_copy(float *data_gpu, float *result_gpu, size_t mem_size, size_t dim, float *runtime, double *bandwidth)
{
    cudaMemset(result_gpu, 0, mem_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    unsigned int repetitions = 0;

    cudaEventRecord(start);

    for (repetitions = 0; repetitions < NUM_REPS; ++repetitions)
    {
        cudaMemcpy(result_gpu, data_gpu, mem_size, cudaMemcpyDeviceToDevice);
    }

    // Get the time taken by the kernel
    cudaEventRecord(stop);
    checkCudaErrors(cudaDeviceSynchronize());
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(runtime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Calculate the bandwidth
    *bandwidth = effectiveBandWidthSquaredMatrixTranspose(dim, *runtime / 1e3);
}

void checkCudaError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
    }
}

void kernel_experiment(
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
    DTYPE *reference)
{
    // Warm up the kernel
    kernel<<<dimGrid, dimBlock>>>(data_gpu, result_gpu, dim);
    checkCudaError("Warm up kernel failed - KE");

    cudaMemset(result_gpu, 0, mem_size); // Reset result buffer

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    unsigned int index = 0;
    float runtime;

    cudaError_t last_error;
    cudaError_t sync_error;
    for (index = 0; index < NUM_REPS; ++index)
    {
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        kernel<<<dimGrid, dimBlock>>>(data_gpu, result_gpu, dim);

        last_error = cudaGetLastError();
        sync_error = cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&runtime, start, stop);
        runtime_buffer[index] = runtime;

        if (last_error != cudaSuccess)
        {
            std::cerr << "CUDA error: " << cudaGetErrorString(last_error) << std::endl;
            return; 
        }

        if (sync_error != cudaSuccess)
        {
            std::cerr << "CUDA error during synchronization: " << cudaGetErrorString(sync_error) << std::endl;
            return; 
        }
    }

    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Save statistics
    data.runtime = runtime_buffer;

    std::array<float, NUM_REPS> bandwidth_buffer;
    for (size_t i = 0; i < NUM_REPS; ++i)
    {
        bandwidth_buffer[i] = effectiveBandWidthSquaredMatrixTranspose(dim, runtime_buffer[i] / 1e3);
    }

    data.bandwidth = bandwidth_buffer;

    checkCudaErrors(cudaMemcpy(result_cpu, result_gpu, mem_size, cudaMemcpyDeviceToHost));

    // Calculate error
    data.error = mError(dim, result_cpu, reference);
}

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
    DTYPE *reference)
{
    // Warm up the kernel
    kernel<<<dimGrid, dimBlock>>>(data_gpu, result_gpu, dim);
    checkCudaError("Warm up kernel failed - CE");

    cudaMemset(result_gpu, 0, mem_size); // Reset result buffer

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    unsigned int index = 0;
    float runtime;

    cudaError_t last_error;
    cudaError_t sync_error;
    for (index = 0; index < NUM_REPS; ++index)
    {
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        kernel<<<dimGrid, dimBlock>>>(data_gpu, result_gpu, dim);

        last_error = cudaGetLastError();
        sync_error = cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&runtime, start, stop);
        runtime_buffer[index] = runtime;

        if (last_error != cudaSuccess)
        {
            std::cerr << "CUDA error: " << cudaGetErrorString(last_error) << std::endl;
            return;
        }

        if (sync_error != cudaSuccess)
        {
            std::cerr << "CUDA error during synchronization: " << cudaGetErrorString(sync_error) << std::endl;
            return;
        }
    }

    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Save statistics
    data.runtime = runtime_buffer;

    std::array<float, NUM_REPS> bandwidth_buffer;
    for (size_t i = 0; i < NUM_REPS; ++i)
    {
        bandwidth_buffer[i] = effectiveBandWidthSquaredMatrixTranspose(dim, runtime_buffer[i] / 1e3);
    }

    data.bandwidth = bandwidth_buffer;

    checkCudaErrors(cudaMemcpy(result_cpu, result_gpu, mem_size, cudaMemcpyDeviceToHost));

    // Calculate error
    data.error = 0;
}

void cuBLAS_experiment(
    float *data_gpu,
    float *result_cpu,
    float *result_gpu,
    size_t mem_size,
    size_t dim,
    std::array<float, NUM_REPS> &runtime_buffer,
    KernelData &data,
    DTYPE *reference)

{

    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS initialization failed!" << std::endl;
        return;
    }
    checkCudaError();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, dim, dim, &alpha, data_gpu, dim, &beta, data_gpu, dim, result_gpu, dim);
    checkCudaError("Cublas warmup failed");

    cudaMemset(result_gpu, 0, mem_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (unsigned int repetitions = 0; repetitions < NUM_REPS; ++repetitions)
    {
        cudaDeviceSynchronize();
        cudaEventRecord(start);

        // Perform the cuBLAS operation
        cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, dim, dim, &alpha, data_gpu, dim, &beta, data_gpu, dim, result_gpu, dim);

        // Wait for the kernel to complete
        cudaDeviceSynchronize();

        // Check for any errors during the kernel launch
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float runtime;
        cudaEventElapsedTime(&runtime, start, stop);
        runtime_buffer[repetitions] = runtime;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Save statistics
    data.runtime = runtime_buffer;

    std::array<float, NUM_REPS> bandwidth_buffer;
    for (size_t i = 0; i < NUM_REPS; ++i)
    {
        bandwidth_buffer[i] = effectiveBandWidthSquaredMatrixTranspose(dim, runtime_buffer[i] / 1e3);
    }

    data.bandwidth = bandwidth_buffer;

    checkCudaErrors(cudaMemcpy(result_cpu, result_gpu, mem_size, cudaMemcpyDeviceToHost));

    // Calculate error
    data.error = mError(dim, result_cpu, reference);
}
