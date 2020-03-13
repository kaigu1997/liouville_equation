#ifndef PES_H
#define PES_H

// this header file contains the
// parameter used in DVR calculation
// and gives the Potential Energy Surface
// (PES) of different representations:
// diabatic, adiabatic and force-basis.
// Transformation is done by unitary
// matrices calculate by LAPACK in MKL

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <mkl.h>
#include <utility>
#include "matrix.h"
using namespace std;

// definition of different basis
enum Representation
{
    Diabatic, Adiabatic, ForceBasis
};
// the number of basis
const int NoBasis = 3;
// the number of potential energy surfaces
const int NumPES = 2;
// the number of potential matrix elements, =numpes^2
const int NoMatrixElement = 4;
// function objects: potential(V), force(F=-dV/dR), NAC, and basis trans matrix
extern const function<RealMatrix(const double)> potential[NoBasis];
extern const function<RealMatrix(const double)> force[NoBasis];
extern const function<RealMatrix(const double)> coupling[NoBasis];

extern const function<void(ComplexMatrixMatrix&, int, const double* const)> basis_transform[NoBasis][NoBasis];

#endif // !PES_H
