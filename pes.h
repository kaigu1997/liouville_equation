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
#include <mkl.h>
#include <utility>
#include "general.h"
#include "matrix.h"
using namespace std;

// the number of potential energy surfaces
const int NumPES = 2;
// the number of potential matrix elements, =numpes^2
const int NoMatrixElement = 4;
// diabatic PES matrix: analytical
RealMatrix DiaPotential(const double x);
RealMatrix DiaForce(const double x);

// transformation matrix from diabatic density matrix to adiabatic one
// i.e. C*rho(dia)=rho(adia), which diagonalize PES only (instead of diagonal L)
ComplexMatrix DiaToAdia(const int NGrids, const double* GridPosition);

#endif // !PES_H
