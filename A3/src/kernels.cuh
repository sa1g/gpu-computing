#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cooperative_groups.h>

#include "definitions.h"

/**
 * Simple copy reference kernel.
 *
 * Arguments:
 * @param idata: source matrix defined in a single pointer.
 * @param odata: destination matrix, can be pre-filled, defined in a single pointer.
 */
__global__ void simpleCopy(const DTYPE *idata, DTYPE *odata, size_t _);

/**
 * Copy kernel using shared memory. It doesn't have bank conflicts as the shared memory is accessed in a coalesced way.
 *
 * Arguments:
 * - `idata`: source matrix defined in a single pointer.
 * - `odata`: destination matrix, can be pre-filled, defined in a single pointer.
 */
__global__ void copyShared(const float *idata, float *odata, size_t _);

/**
 * Copy kernel using shared memory. It uses coop groups to synchronize threads.
 *
 * Arguments:
 * - `idata`: source matrix defined in a single pointer.
 * - `odata`: destination matrix, can be pre-filled, defined in a single pointer.
 * 
 * @note Source: [NVidia cuda-samples](https://github.com/NVIDIA/cuda-samples/blob/master/Samples/6_Performance/transpose/transpose.cu#L93)
 */
__global__ void copySharedCoop(const float *idata, float *odata, size_t n);



/**
 * Naive implementation of matrix transpose.
 *
 * @param idata: input matrix (single pointer)
 * @param odata: output matrix (single pointer)
 * @param width: matrix width
 * @param height: matrix height
 *
 * @note Source: [NVidia Paper](https://www.cs.colostate.edu/~cs675/MatrixTranspose.pdf)
 * @note Source: [NVidia cuda-samples](https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/transpose/transpose.cu#L98)
 */
__global__ void transposeNaive(const DTYPE *idata, DTYPE *odata, size_t n);

/**
 * Transpose kernel with coalesced global memory reads and writes.
 * 
 * @param idata: input matrix (single pointer)
 * @param odata: output matrix (single pointer)
 */
__global__ void transposeCoalesced(const float *idata, float *odata, size_t n);

/**
 * Transpose kernel with shared memory. Coop groups are used to synchronize threads.
 *
 * @param idata: input matrix (single pointer)
 * @param odata: output matrix (single pointer)
 *
 * @note Source: [NVidia cuda-samples](https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/transpose/transpose.cu#L139)
 */
__global__ void transposeCoalescedCoop(const float *idata, float *odata, size_t n);


/**
 * Transpose kernel with shared memory and coalesced global memory reads and writes.
 * This version avoids bank conflicts in shared memory.
 *
 * @param idata: input matrix (single pointer)
 * @param odata: output matrix (single pointer)
 *
 * @note Source: [NVidia cuda-samples](https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/transpose/transpose.cu#L111)
 */
__global__ void transposeCoalescedNoBankConflicts(const float *idata, float *odata, size_t n);

/**
 * Transpose kernel with shared memory. Coop groups are used to synchronize threads.
 *
 * @param idata: input matrix (single pointer)
 * @param odata: output matrix (single pointer)
 *
 * @note Source: [NVidia cuda-samples](https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/transpose/transpose.cu#L139)
 */
__global__ void transposeNoBankConflictsCoop(const float *idata, float *odata, size_t n);


__global__ void transposeCoopGroupsSimple(const float *idata, float *odata, size_t N);

/**
 * This kernel works only with "small" matrices: 2 <= N <= 32.
 * Implementation with TODO: add docs
 *
 * @param idata: input matrix (single pointer)
 * @param odata: output matrix (single pointer)
 * @param N: matrix size
 */
__global__ void transposeInterBlock(const float *idata, float *odata, size_t N);

__global__ void transposeCoopGroupsWarpSubgroups(const float *idata, float *odata, size_t N);


#endif




