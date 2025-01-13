#include "kernels.cuh"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void simpleCopy(const DTYPE *idata, DTYPE *odata, size_t _)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        odata[(y + i) * width + x] = idata[(y + i) * width + x];
    }
}

__global__ void copyShared(const float *idata, float *odata, size_t _)
{
    __shared__ float tile[TILE_DIM][TILE_DIM+1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    // Load data from global memory to shared memory
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        if (y + i < width && x < width) // Boundary check
        {
            tile[threadIdx.y + i][threadIdx.x] = idata[(y + i) * width + x];
        }
    }

    __syncthreads();

    // Store data from shared memory to global memory
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        if (y + i < width && x < width) // Boundary check
        {
            odata[(y + i) * width + x] = tile[threadIdx.y + i][threadIdx.x];
        }
    }
}

__global__ void copySharedCoop(const float *idata, float *odata, size_t n)
{
    cg::thread_block cta = cg::this_thread_block();
    __shared__ float tile[TILE_DIM][TILE_DIM+1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    // Load data from global memory to shared memory
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        if (y + i < width && x < width) // Boundary check
        {
            tile[threadIdx.y + i][threadIdx.x] = idata[(y + i) * width + x];
        }
    }

    cg::sync(cta);

    // Store data from shared memory to global memory
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        if (y + i < width && x < width) // Boundary check
        {
            odata[(y + i) * width + x] = tile[threadIdx.y + i][threadIdx.x];
        }
    }
}

__global__ void transposeNaive(const DTYPE *idata, DTYPE *odata, size_t n)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = n;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        // Ensure we are within bounds before accessing memory
        if ((x < width) && ((y + i) < width))
        {
            odata[(y + i) * width + x] = idata[x * width + (y + i)];
        }
    }
}

__global__ void transposeCoalesced(const float *idata, float *odata, size_t N)
{
    // Dynamically determine effective tile size
    const int effective_tile_dim = (N < TILE_DIM) ? N : TILE_DIM;

    // Declare shared memory with the fixed maximum size
    __shared__ float tile[TILE_DIM][TILE_DIM];

    // Global thread coordinates
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Handle boundary conditions when reading into shared memory
    if (threadIdx.x < effective_tile_dim && threadIdx.y < effective_tile_dim)
    {
        for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        {
            int global_y = y + i;
            if (global_y < N && x < N)
            {
                tile[threadIdx.y + i][threadIdx.x] = idata[global_y * N + x];
            }
            else
            {
                tile[threadIdx.y + i][threadIdx.x] = 0.0f; // Boundary padding
            }
        }
    }

    __syncthreads();

    // Transpose the tile in shared memory
    x = blockIdx.y * TILE_DIM + threadIdx.x; // Transposed block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Write transposed data back to global memory
    if (threadIdx.x < effective_tile_dim && threadIdx.y < effective_tile_dim)
    {
        for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        {
            int global_y = y + i;
            if (global_y < N && x < N)
            {
                odata[global_y * N + x] = tile[threadIdx.x][threadIdx.y + i];
            }
        }
    }
}

__global__ void transposeCoalescedCoop(const float *idata, float *odata, size_t n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ float tile[TILE_DIM][TILE_DIM];

  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
  int index_in = xIndex + (yIndex)*n;

  xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*n;

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * n];
  }

  cg::sync(cta);

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    odata[index_out + i * n] = tile[threadIdx.x][threadIdx.y + i];
  }
}

__global__ void transposeCoalescedNoBankConflicts(const float *idata, float *odata, size_t N)
{
    // Dynamically determine effective tile size
    const int effective_tile_dim = (N < TILE_DIM) ? N : TILE_DIM;

    // Declare shared memory with the fixed maximum size
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // Global thread coordinates
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Handle boundary conditions when reading into shared memory
    if (threadIdx.x < effective_tile_dim && threadIdx.y < effective_tile_dim)
    {
        for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        {
            int global_y = y + i;
            if (global_y < N && x < N)
            {
                tile[threadIdx.y + i][threadIdx.x] = idata[global_y * N + x];
            }
            else
            {
                tile[threadIdx.y + i][threadIdx.x] = 0.0f; // Boundary padding
            }
        }
    }

    __syncthreads();

    // Transpose the tile in shared memory
    x = blockIdx.y * TILE_DIM + threadIdx.x; // Transposed block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Write transposed data back to global memory
    if (threadIdx.x < effective_tile_dim && threadIdx.y < effective_tile_dim)
    {
        for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        {
            int global_y = y + i;
            if (global_y < N && x < N)
            {
                odata[global_y * N + x] = tile[threadIdx.x][threadIdx.y + i];
            }
        }
    }
}

__global__ void transposeNoBankConflictsCoop(const float *idata, float *odata, size_t n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
  int index_in = xIndex + (yIndex)*n;

  xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*n;

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * n];
  }

  cg::sync(cta);

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    odata[index_out + i * n] = tile[threadIdx.x][threadIdx.y + i];
  }
}

__global__ void transposeCoopGroupsSimple(const float *idata, float *odata, size_t N)
{
    cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();
    const int effective_tile_dim = (N < TILE_DIM) ? N : TILE_DIM;

    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (threadIdx.x < effective_tile_dim && threadIdx.y < effective_tile_dim)
    {
        for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        {
            int global_y = y + i;
            if (global_y < N && x < N)
            {
                tile[threadIdx.y + i][threadIdx.x] = idata[global_y * N + x];
            }
            else
            {
                tile[threadIdx.y + i][threadIdx.x] = 0.0f;
            }
        }
    }

    cta.sync();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (threadIdx.x < effective_tile_dim && threadIdx.y < effective_tile_dim)
    {
        for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        {
            int global_y = y + i;
            if (global_y < N && x < N)
            {
                odata[global_y * N + x] = tile[threadIdx.x][threadIdx.y + i];
            }
        }
    }
}


__global__ void transposeInterBlock(const float *idata, float *odata, size_t N)
{

    // Grid group for inter-block synchronization
    cg::grid_group grid = cg::this_grid();

    // Define shared memory for tile
    __shared__ float tile[TILE_DIM][TILE_DIM+1];

    // Thread and block indices
    int tile_size = TILE_DIM;                     // Tile dimensions (32x32)
    int x = blockIdx.x * tile_size + threadIdx.x; // Global column index
    int y = blockIdx.y * tile_size + threadIdx.y; // Global row index

    // Load matrix tile into shared memory
    if (x < N && y < N)
    {
        tile[threadIdx.y][threadIdx.x] = idata[y * N + x];
    }

    // Inter-block synchronization
    grid.sync();

    // Transpose the tile and write to output matrix
    x = blockIdx.y * tile_size + threadIdx.x; // Transposed column index
    y = blockIdx.x * tile_size + threadIdx.y; // Transposed row index
    if (x < N && y < N)
    {
        odata[y * N + x] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void transposeCoopGroupsWarpSubgroups(const float *idata, float *odata, size_t N)
{
    // Cooperative groups block and warp
    cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();
    cooperative_groups::thread_block_tile<16> warp_group1 = cooperative_groups::tiled_partition<16>(cta);
    cooperative_groups::thread_block_tile<16> warp_group2 = cooperative_groups::tiled_partition<16>(cta);

    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // Calculate matrix indices
    int global_x = blockIdx.x * TILE_DIM + threadIdx.x;
    int global_y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    // Subgroup indices
    int warp_group_idx = warp_group1.thread_rank();

    // First subgroup processes the top half of the tile
    if (warp_group1.meta_group_rank() == 0)
    {
        for (int i = 0; i < TILE_DIM / 2; i += BLOCK_ROWS)
        {
            if ((global_y + i) < N && global_x < N)
            {
                tile[threadIdx.y + i][threadIdx.x] = idata[(global_y + i) * width + global_x];
            }
        }
    }

    // Second subgroup processes the bottom half of the tile
    if (warp_group2.meta_group_rank() == 1)
    {
        for (int i = TILE_DIM / 2; i < TILE_DIM; i += BLOCK_ROWS)
        {
            if ((global_y + i) < N && global_x < N)
            {
                tile[threadIdx.y + i][threadIdx.x] = idata[(global_y + i) * width + global_x];
            }
        }
    }

    cta.sync();

    // Recalculate indices for transposed write
    global_x = blockIdx.y * TILE_DIM + threadIdx.x;
    global_y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Write back transposed elements
    if (warp_group1.meta_group_rank() == 0)
    {
        for (int i = 0; i < TILE_DIM / 2; i += BLOCK_ROWS)
        {
            if ((global_y + i) < N && global_x < N)
            {
                odata[(global_y + i) * width + global_x] = tile[threadIdx.x][threadIdx.y + i];
            }
        }
    }

    if (warp_group2.meta_group_rank() == 1)
    {
        for (int i = TILE_DIM / 2; i < TILE_DIM; i += BLOCK_ROWS)
        {
            if ((global_y + i) < N && global_x < N)
            {
                odata[(global_y + i) * width + global_x] = tile[threadIdx.x][threadIdx.y + i];
            }
        }
    }
}
