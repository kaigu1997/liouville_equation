/// @file pes.h
/// @brief Declaration of potential energy surfaces and their derivatives
///
/// This header file contains the
/// parameter used in DVR calculation
/// and gives the Potential Energy Surface
/// (PES) of different representations:
/// diabatic, adiabatic and force-basis.
/// Transformation is done by unitary
/// matrices calculate by LAPACK in MKL

#ifndef PES_H
#define PES_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <mkl.h>
#include <utility>
#include "matrix.h"
using namespace std;

/// @enum different basis
enum Representation
{
    Diabatic, ///< Diabatic basis, NAC=0
    Adiabatic, ///< Adiabatic basis, subsystem Hamiltonian is diagonal
    ForceBasis ///< Force basis, the -dH/dR operator is diagonal
};

/// @enum different models
enum Model
{
    SAC = 1, ///< Simple Avoided Crossing, tully's first model
    DAC, ///< Dual Avoided Crossing, tully's second model
    ECR ///< Extended Coupling with Reflection, tully's third model
};

const int NoBasis = 3; ///< the number of basis
const int NumPES = 2; ///< the number of potential energy surfaces
const int NoMatrixElement = 4; ///< the number of potential matrix elements, =numpes^2
const Model TestModel = DAC; ///< the model to use

// function objects: potential(V), force(F=-dV/dR), NAC, and basis trans matrix
extern const function<RealMatrix(const double)> potential[NoBasis]; ///< function object: potential (V of environment, H of subsystem)
extern const function<RealMatrix(const double)> force[NoBasis]; ///< function object: force (F=-dV/dR)
extern const function<RealMatrix(const double)> coupling[NoBasis]; ///< function object: non-adiabatic coupling (dij=<i|d/dR|j>)

extern const function<void(ComplexMatrixMatrix&, int, const double* const)> basis_transform[NoBasis][NoBasis]; ///< function object: basis transformation matrices

#endif // !PES_H
