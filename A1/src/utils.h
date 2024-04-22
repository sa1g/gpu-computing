#include "matrix.h"

std::tuple<int, int, int> arg_parser(int argc, char *argv[]);

double effective_bandwidth_T(const MatrixShape shape, const double time);