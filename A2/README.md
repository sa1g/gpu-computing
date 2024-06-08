## How to run
You can build it in two different ways:
- `make` // It defaults to a 32x8 block size
- `make TILE_DIM=<tiledim> BLOCK_ROWS=<blockrows>` 

## Instructions
Implement a simple algorithm in CUDA that transposes a no-symmetric matrix of size NxN.
- The algorithm takes the dimension of the matrix as an input (use a power of 2).
- For example, `./transpose 10` tarnsposes a 2^10 x 2^10 matrix.
- Meature the Effective bandwidth ofyour implementation.
- The implementation should be based on the techniques discussed during the lectures
