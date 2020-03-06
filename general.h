#ifndef GENERAL_H
#define GENERAL_H

// This header file contains some functions that will be used 
// in main function, but they are often call once or twice.
// However, putting the function codes directly in main make the
// main function and main file too long. In that case, these 
// functions are moved to an individual file (general.cpp)
// and the interfaces are declared here.

#include <iostream>
#include <memory>
#include <tuple>
#include "pes.h"
#include "matrix.h"

// mathematical and physical constants
const double pi = acos(-1.0), hbar = 1.0, PlanckH = 2 * pi * hbar;
// Alpha and Beta are for cblas complex matrix multiplication functions
// as they behave as A=alpha*B*C+beta*A
const Complex Alpha(1.0, 0.0), Beta(0.0, 0.0);
// AbsorbLim means if the absorbed population is over this
// then the program should stop. Used only when Absorb is on
const double AbsorbLim = 1.0e-2;
// ChangeLim is that, if the population change on each PES is smaller 
// than ChangeLim, then it is stable and could stop simulation
const double ChangeLim = 1e-5;

// two kinds of unique_ptr array, no need to free manually
typedef unique_ptr<double[]> doubleVector;
typedef unique_ptr<Complex[]> ComplexVector;


// utility functions

// do the cutoff, e.g. 0.2493 -> 0.2, 1.5364 -> 1
double cutoff(const double val);

// sign function; working for all type that have '<' and '>'
// return -1 for negative, 1 for positive, and 0 for 0
template <typename valtype>
inline int sgn(const valtype& val)
{
    return (val > valtype(0.0)) - (val < valtype(0.0));
}

// returns (-1)^n
int pow_minus_one(const int n);


// I/O functions

// read a double: mass, x0, etc
double read_double(istream& is);

// to print current time
ostream& show_time(ostream& os);


// evolution related functions

// initialize the PWTDM (Partial Wigner-Transformed Density Matrix), and normalize it
void density_matrix_initialization(const int NGrids, const double* GridPosition, const double* GridMomentum, const double dx, const double dp, const double x0, const double p0, const double SigmaX, const double SigmaP, ComplexMatrixMatrix& rho_adia);

// calculate the population on each PES
void calculate_popultion(const int NGrids, const double dx, const double dp, const ComplexMatrixMatrix& rho_adia, double* Population);

// evolve the quantum liouville: exp(-iLQt)rho
// -iLQrho=-i/hbar[V-i*hbar*P/M*D,rho]
// input t should be dt/2
void quantum_liouville_propagation(ComplexMatrixMatrix& rho, const int NGrids, const double* GridPosition, const double* GridMomentum, const double mass, const double dt, const Representation BasisOfRho);

// evolve the classical position liouville: exp(-iLRt)rho
// -iLRrho=-P/M*drho/dR=-i*P/M*(-i*DR)*rho
// input t should be dt/2 as well
void classical_position_liouville_propagator(ComplexMatrixMatrix& rho, const int NGrids, const double* GridMomentum, const double mass, const double dx, const double dt);

// evolve the classical position liouville: exp(-iLRt)rho
// -iLPrho=-1/2(F*drho/dP+drho/dP*F)=-i(Faa+Fbb)/2*(-i*DP)*rho
// if under force basis. input t should be dt
void classical_momentum_liouville_propagator(ComplexMatrixMatrix& rho, const int NGrids, const double* GridPosition, const double dp, const double dt);

#endif // !GENERAL_H