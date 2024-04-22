#ifndef DTYPE
#define DTYPE float
#endif

#include <stdlib.h>
#include <math.h>
#include <cstdio>
#include <stdexcept>

#include "matrix.h"


#include <tuple>
#include <iostream>
#include <cstdlib> // for std::exit

std::tuple<int, int, int> arg_parser(int argc, char *argv[]) {
    int DIM = 2;
    int N_RUN = 1;
    int BLOCK_SIZE = 16;

    if (argc > 1) {
        int exponent = std::stoi(argv[1]);

        if (exponent > 30 || exponent < 1) {
            std::cerr << "Error: exponent must be > 1 and <30" << std::endl;
            std::exit(1); // Exit with error
        }

        DIM = 1 << exponent;
    }

    if (argc > 2) {
        N_RUN = std::stoi(argv[2]);
    }

    if (argc > 3) {
        BLOCK_SIZE = std::stoi(argv[3]);
    }

    return {DIM, N_RUN, BLOCK_SIZE};
}

double effective_bandwidth_T(const MatrixShape shape, double time)
{
    // dim matr letta 1024x1024 br
    // dim matr scrta 1024x1024 dw
    // 32bit -> 4 byte dtb
    // time

    // (br+bw)*dtb/time [s]

    int br = shape.x * shape.y;
    int bw = shape.x * shape.y;
    int dtype_bytes = sizeof(DTYPE);
    time = time / 1000; // Convert time to milliseconds if it's in seconds

    return static_cast<double>((br + bw) * dtype_bytes) / (time * std::pow(10.0, 9));
}
