#include <cstdio>
#include <iostream>
#include <cstring>
#include <chrono>
#include <tuple>

#include "utils.h"
#include "matrix.h"

class Timer
{
public:
    Timer(MatrixShape _shape) : shape(_shape) // Initialize shape in the constructor initializer list
    {
        start = std::chrono::high_resolution_clock::now();
    }

    ~Timer()
    {
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_double = end - start;
        std::cout << ms_double.count() << ", " << effective_bandwidth_T(shape, ms_double.count()) << std::endl;
    }

private:
    MatrixShape shape; // Declare shape as a member variable
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
};

int main(int argc, char *argv[])
{
    std::tuple<int, int, int> args = arg_parser(argc, argv);

    // Extract the values from the tuple
    int DIM = std::get<0>(args);
    int N_RUN = std::get<1>(args);
    int BLOCK_SIZE = std::get<2>(args);

    Matrix source{DIM, DIM, initRand};

    Matrix destination{DIM, DIM};
    for (int i = 0; i < N_RUN; i++)
    {
        {
            Timer t{source.shape};
#ifndef BLOCK
            destination = source.T0();
#else
            destination = source.T2(BLOCK_SIZE);
#endif
        }
    }

    return 0;
}