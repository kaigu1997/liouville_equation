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

// indexing
int indexing(const int a, const int b, const int i, const int j, const int NGrids);


// I/O functions

// read a double: mass, x0, etc
double read_double(istream& is);

// to print current time
ostream& show_time(ostream& os);


// evolution related functions

// initialize the 2d gaussian wavepacket, and normalize it
void density_matrix_initialization(const int NGrids, const double* GridPosition, const double* GridMomentum, const double dx, const double dp, const double x0, const double p0, const double SigmaX, const double SigmaP, Complex* rho_adia);

// construct the Liouville superoperator
ComplexMatrix Liouville_construction(const int NGrids, const double* GridPosition, const double* GridMomentum, const double dx, const double dp, const double mass);

// calculate the population on each PES
void calculate_popultion(const int NGrids, const double dx, const double dp, const Complex* rho_adia, double* Population);

#endif // !GENERAL_H