#include "helper_cuda.h"
#include <iomanip>

#include "definitions.h"
#include "utils.cuh"
#include "kernels.cuh"

int main(int argc, char *argv[])
{
    size_t DIM = 2;

    // VARIABLES
    int experiment_counter{0};

    KernelData experiment_data[RUNS][EXPERIMENTS];
    std::array<float, NUM_REPS> runs_buffer;

    // CHECK CUDA AVAILABLE DEVICES
    bool coopLaunch(false);
    check_device();
    get_info_and_device(&coopLaunch);

    for (int isize = 0; isize < RUNS; isize++)
    {
        experiment_counter = 0;

        DIM = DIM << 1;

        std::cout << "DIM " << DIM << std::endl;

        // DEFINE MATRIX (CPU)
        size_t mem_size = DIM * DIM * sizeof(float);

        float *data_cpu = (float *)malloc(mem_size);
        float *reference = (float *)malloc(mem_size);
        float *result_cpu = (float *)malloc(mem_size);
        float *data_gpu, *result_gpu;

        // DEFINE MATRIX (GPU)
        cudaMalloc(&data_gpu, mem_size);
        cudaMalloc(&result_gpu, mem_size);

        // INITIALIZE CPU MATRIX
        rand_init_matrix(data_cpu, DIM);
        matrix_T(data_cpu, reference, DIM);

        // COPY FROM CPU TO GPU
        cudaMemcpy(data_gpu, data_cpu, mem_size, cudaMemcpyHostToDevice);
        cudaMemset(result_gpu, 0, mem_size);

        // CALCULATE GRID AND BLOCK DIMENSIONS
        dim3 dim_grid((DIM + TILE_DIM - 1) / TILE_DIM, (DIM + TILE_DIM - 1) / TILE_DIM, 1);
        dim3 dim_block(TILE_DIM, BLOCK_ROWS, 1);

        // Copy
        {
            copy_experiment(simpleCopy, dim_grid, dim_block, data_gpu, result_gpu, result_cpu, mem_size, DIM, runs_buffer, experiment_data[isize][experiment_counter], reference);
            experiment_data[isize][experiment_counter++].name = "C_Naive";
        }

        {
            copy_experiment(copyShared, dim_grid, dim_block, data_gpu, result_gpu, result_cpu, mem_size, DIM, runs_buffer, experiment_data[isize][experiment_counter], reference);
            experiment_data[isize][experiment_counter++].name = "C_shared";
        }

        {
            copy_experiment(copySharedCoop, dim_grid, dim_block, data_gpu, result_gpu, result_cpu, mem_size, DIM, runs_buffer, experiment_data[isize][experiment_counter], reference);
            experiment_data[isize][experiment_counter++].name = "C_coop";
        }

        // Transpose
        {
            kernel_experiment(transposeNaive, dim_grid, dim_block, data_gpu, result_gpu, result_cpu, mem_size, DIM, runs_buffer, experiment_data[isize][experiment_counter], reference);
            experiment_data[isize][experiment_counter++].name = "T_Naive";
        }

        {
            kernel_experiment(transposeCoalesced, dim_grid, dim_block, data_gpu, result_gpu, result_cpu, mem_size, DIM, runs_buffer, experiment_data[isize][experiment_counter], reference);
            experiment_data[isize][experiment_counter++].name = "T_Coalesced";
        }

        {
            kernel_experiment(transposeCoalescedCoop, dim_grid, dim_block, data_gpu, result_gpu, result_cpu, mem_size, DIM, runs_buffer, experiment_data[isize][experiment_counter], reference);
            experiment_data[isize][experiment_counter++].name = "T_CoalescedCoop";
        }

        {
            kernel_experiment(transposeCoalescedNoBankConflicts, dim_grid, dim_block, data_gpu, result_gpu, result_cpu, mem_size, DIM, runs_buffer, experiment_data[isize][experiment_counter], reference);
            experiment_data[isize][experiment_counter++].name = "T_CNBC";
        }

        {
            kernel_experiment(transposeNoBankConflictsCoop, dim_grid, dim_block, data_gpu, result_gpu, result_cpu, mem_size, DIM, runs_buffer, experiment_data[isize][experiment_counter], reference);
            experiment_data[isize][experiment_counter++].name = "T_CNBC Coop";
        }

        {
            kernel_experiment(transposeCoopGroupsSimple, dim_grid, dim_block, data_gpu, result_gpu, result_cpu, mem_size, DIM, runs_buffer, experiment_data[isize][experiment_counter], reference);
            experiment_data[isize][experiment_counter++].name = "T_Coop_Simple";
        }

        {
            if (DIM > 32)
            {
                // Horrible hack to avoid the kernel launch
                // Done so that we have one less error in the results.
                for (int index = 0; index < NUM_REPS; ++index)
                {
                    runs_buffer[index] = -1;
                }

                experiment_data[isize][experiment_counter].runtime = runs_buffer;
                std::array<float, NUM_REPS> bandwidth_buffer;
                for (size_t i = 0; i < NUM_REPS; ++i)
                {
                    bandwidth_buffer[i] = -1;
                }

                experiment_data[isize][experiment_counter].bandwidth = bandwidth_buffer;
                experiment_data[isize][experiment_counter].error = -1;
            }
            else
            {

                if (coopLaunch && DIM <= 32)
                {
                    dim3 block_dim(32, 32);                          // Tile size (threads per block)
                    dim3 grid_dim((DIM + 31) / 32, (DIM + 31) / 32); // Number of tiles

                    void *kernelArgs[] = {(void *)&data_gpu, (void *)&result_gpu, (void *)&DIM};

                    // Warmup
                    cudaLaunchCooperativeKernel(
                        (void *)transposeInterBlock,
                        grid_dim,
                        block_dim,
                        kernelArgs);

                    cudaMemset(result_gpu, 0, mem_size);

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

                        cudaLaunchCooperativeKernel(
                            (void *)transposeInterBlock,
                            grid_dim,
                            block_dim,
                            kernelArgs);

                        last_error = cudaGetLastError();
                        sync_error = cudaDeviceSynchronize();

                        cudaEventRecord(stop);
                        cudaEventSynchronize(stop);

                        cudaEventElapsedTime(&runtime, start, stop);
                        runs_buffer[index] = runtime;

                        if (last_error != cudaSuccess)
                        {
                            std::cerr << "CUDA error: " << cudaGetErrorString(last_error) << std::endl;
                        }

                        if (sync_error != cudaSuccess)
                        {
                            std::cerr << "CUDA error during synchronization: " << cudaGetErrorString(sync_error) << std::endl;
                        }
                    }

                    // Clean up events
                    cudaEventDestroy(start);
                    cudaEventDestroy(stop);

                    // Save statistics
                    experiment_data[isize][experiment_counter].runtime = runs_buffer;

                    std::array<float, NUM_REPS> bandwidth_buffer;
                    for (size_t i = 0; i < NUM_REPS; ++i)
                    {
                        bandwidth_buffer[i] = effectiveBandWidthSquaredMatrixTranspose(DIM, runs_buffer[i] / 1e3);
                    }

                    experiment_data[isize][experiment_counter].bandwidth = bandwidth_buffer;

                    checkCudaErrors(cudaMemcpy(result_cpu, result_gpu, mem_size, cudaMemcpyDeviceToHost));

                    // Calculate error
                    experiment_data[isize][experiment_counter].error = mError(DIM, result_cpu, reference);
                }
            }

            experiment_data[isize][experiment_counter++].name = "T_InterBlock";
        }

        {
            cuBLAS_experiment(data_gpu, result_cpu, result_gpu, mem_size, DIM, runs_buffer, experiment_data[isize][experiment_counter], reference);
            experiment_data[isize][experiment_counter++].name = "T_cuBLAS";
        }

        free(data_cpu);
        free(reference);
        free(result_cpu);
        cudaFree(data_gpu);
        cudaFree(result_gpu);
    }

#ifndef DEBUG
    for (int i = 0; i < experiment_counter; i++)
    {
        std::string file_name = "logs/results_" + experiment_data[0][i].name + ".csv";

        // Create file if not exists, otherwise append
        std::ofstream file(file_name, std::ios::app);

        // Write header
        if (file.tellp() == 0)
        {
            for (int j = 0; j < RUNS; j++)
            {
                // int dim = 2 << j;
                file << "bandwidth_" << j << ",time_" << j << ",";
            }
            file << std::endl;
        }

        // Write data
        for (int k = 0; k < NUM_REPS; k++)
        {

            for (int j = 0; j < RUNS; j++)
            {
                if (experiment_data[j][i].error != 0)
                {
                    file << ",,";
                }
                else
                {
                    file << experiment_data[j][i].bandwidth[k] << "," << experiment_data[j][i].runtime[k] << ",";
                }
            }
            file << "\n";
        }

        file.close();
    }

#else
    // Print header
    for (int i = 0; i < EXPERIMENTS; i++)
    {
        std::cout << std::setw(20) << experiment_data[0][i].name << " - ";
    }

    std::cout << std::endl;

    // Print data
    for (int i = 0; i < RUNS; i++)
    {
        for (int j = 0; j < EXPERIMENTS; j++)
        {
            if (experiment_data[i][j].error != 0)
            {
                std::cout << std::setw(20) << "ERROR";
                continue;
            }
            std::cout << std::setw(20) << experiment_data[i][j].bandwidth[0] << " ";
        }
        std::cout << std::endl;
    }

#endif

    std::cout << "Done!" << std::endl;

    return EXIT_SUCCESS;
}
