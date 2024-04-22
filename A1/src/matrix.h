#include <stdexcept>

#ifndef MATRIX_SIZE
#define MATRIX_SIZE
struct MatrixShape
{
    const int x;
    const int y;
};
#endif

#ifndef MATRIX_H
#define MATRIX_H

#ifndef DTYPE
#define DTYPE float
#endif

typedef void (*InitFunction)(class Matrix &);

class Matrix
{
private:
    DTYPE *data;//__attribute__((aligned(64)));

    bool same_shape(const Matrix &other) const;

public:
    Matrix(int x, int y);
    Matrix(int x, int y, InitFunction initializer);

    MatrixShape shape;

    Matrix T0();
    Matrix T1();
    Matrix T2(int block);
    ~Matrix();

    void print() const;

    bool operator==(const Matrix &other) const;
    bool operator!=(const Matrix &other);
    DTYPE &operator[](int index);
    Matrix &operator=(const Matrix &other);
};

void initZero(Matrix &matrix);
void initRand(Matrix &matrix);
void initSequential(Matrix &matrix);
#endif