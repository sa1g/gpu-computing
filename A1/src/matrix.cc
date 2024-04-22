#ifndef DTYPE
#define DTYPE float
#endif

#include <sstream>
#include <iostream>
#include <immintrin.h> // Include SIMD intrinsics header

#include "matrix.h"

void initZero(Matrix &matrix)
{
    for (int i = 0; i < matrix.shape.x; ++i)
    {
        for (int j = 0; j < matrix.shape.y; ++j)
        {
            matrix[i * matrix.shape.y + j] = 0;
        }
    }
}

void initRand(Matrix &matrix)
{
    for (int i = 0; i < matrix.shape.x; ++i)
    {
        for (int j = 0; j < matrix.shape.y; ++j)
        {
            matrix[i * matrix.shape.y + j] = static_cast<DTYPE>(rand()) / static_cast<DTYPE>(RAND_MAX);
        }
    }
}

void initSequential(Matrix &matrix)
{
    int counter = 0;
    for (int i = 0; i < matrix.shape.x; ++i)
    {
        for (int j = 0; j < matrix.shape.y; ++j)
        {
            matrix[i * matrix.shape.y + j] = counter++;
        }
    }
}

Matrix::Matrix(int x, int y) : Matrix::Matrix{x, y, initZero}
{
}

Matrix::Matrix(int x, int y, InitFunction initializer) : shape{x, y}
{
    data = new DTYPE[x * y]; // Allocate memory for data

    // Call the initializer function
    initializer(*this);
}

Matrix::~Matrix()
{
    delete[] data;
}

void Matrix::print() const
{
    std::stringstream ss;
    for (int i = 0; i < shape.x; ++i)
    {
        for (int j = 0; j < shape.y; ++j)
        {
            ss << data[i * shape.y + j] << " ";
        }
        ss << "\n";
    }
    std::cout << ss.str();
}

Matrix Matrix::T0()
{
    Matrix result(shape.y, shape.x);

    for (int i = 0; i < shape.x; ++i)
    {
        for (int j = 0; j < shape.y; ++j)
        {
            // __builtin_prefetch(&data[i * shape.y + j], 0, 3); 

            result[j * shape.x + i] = data[i * shape.y + j];
        }
    }

    return result;
}

Matrix Matrix::T1()
{
    Matrix result(shape.y, shape.x);
    int ts = shape.x * shape.y;

    for (int n = 0; n < ts; ++n)
    {
        int i = n / shape.x;
        int j = n % shape.x;

        result[n] = data[shape.y * j + i];
    }

    return result;
}

Matrix Matrix::T2(int block = 16)
{
    Matrix result(shape.y, shape.x);

    const int N = shape.x;
    const int M = shape.y;

    // Iterate over blocks
    for (int ii = 0; ii < N; ii += block)
    {
        for (int jj = 0; jj < M; jj += block)
        {
            // Transpose the current block
            for (int i = ii; i < ii + block && i < N; ++i)
            {
                for (int j = jj; j < jj + block && j < M; ++j)
                {
                    //__builtin_prefetch(&data[i * shape.y + j], 0, 3);

                    result[j * shape.x + i] = data[i * shape.y + j];
                }
            }
        }
    }

    return result;
}

bool Matrix::same_shape(const Matrix &other) const{
    if (shape.x != other.shape.x || shape.y != other.shape.y)
    {
        return false;
    }
    return true;
}

bool Matrix::operator==(const Matrix &other) const
{
    if (!same_shape(other))
    {
        return false;
    }

    // This may be parallelized/optimized
    for (int i = 0; i < shape.x; ++i)
    {
        for (int j = 0; j < shape.y; ++j)
        {
            if (data[i * shape.y + j] != other.data[i * shape.y + j])
            {
                return false;
            }
        }
    }

    return true;
}

bool Matrix::operator!=(Matrix const &other)
{
    return !(*this == other);
}

DTYPE &Matrix::operator[](int index)
{
    if (index < 0 || index >= shape.x * shape.y)
    {
        throw std::out_of_range("Index out of bonds");
    }
    return data[index];
}

Matrix &Matrix::operator=(const Matrix &other)
{
    if (this != &other)
    {
        if (!same_shape(other))
        {
            abort();
        }
        std::copy(other.data, other.data + (shape.x * shape.y), data);
    }
    return *this;
}
