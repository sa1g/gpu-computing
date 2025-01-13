# About
This report evaluates three methods for transposing square, non-symmetric, dense matrices on CUDA-capable devices using NVIDIA’s [Cooperative Groups](https://developer.nvidia.com/blog/cooperative-groups/) , compared against the cuBLAS library. Experiments reveal that Cooperative Groups provide no significant performance benefits for this task. While cuBLAS is not always the fastest, it demonstrates greater stability in throughput, highlighting its robustness for dense matrix transposition.

The _main novelty_ is `transposeInterBlock` which is a ~~dumb~~ cooperative impelementation of a matrix transpose that needs the whole grid synchronization to work.

Finally it's pretty obvious that cooperative groups are not a good idea for matrix transposes as they are useful when you need granularity, anyway I tried.

# How to run
You need a CUDA capable machine, probably CUDA > 8.0 and about 8GB of RAM and 8 of VRAM (if you don't have so much reduce `RUNS` in `/src/definitions.h` to a decent value. Consider that the matrix is float based, with side from $2^2$ up to $2^{RUNS+1}$).

If you have CMake just create a directory, cd into it, then `cmake ..`, `cd ..` and `./<your_dir>/assignment3`. Otherwise `make` and `./bin/main_executable`.
