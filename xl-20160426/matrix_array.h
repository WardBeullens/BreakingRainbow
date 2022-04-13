#ifndef _MATRIX_ARRAY_H
#define _MATRIX_ARRAY_H

#include <stdint.h>

#include "util.h"
#include "matrix.h"

template <unsigned M, unsigned N, unsigned DEG>
class matrix_array
{
    public:

    static const unsigned n = N;
    static const unsigned m = M;

    static const unsigned deg = DEG;

    matrix<M, N> coef[deg];

    const matrix<m, n>& operator [](int i) const
    {
       return this->coef[i];
    }

    matrix<m, n>& operator [](int i)
    {
       return this->coef[i];
    }

    operator matrix<m, n>* ()
    {
       return this->coef;
    }
};

#endif // def MATRIX_ARRAY_H

