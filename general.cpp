// implementation of general.h

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <iostream>
#include <memory>
#include <mkl.h>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include "general.h"
#include "matrix.h"
#include "pes.h"
using namespace std;

// utility functions

// do the cutoff, e.g. 0.2493 -> 0.2, 1.5364 -> 1
double cutoff(const double val)
{
    double pownum = pow(10, static_cast<int>(floor(log10(val))));
    return static_cast<int>(val / pownum) * pownum;
}

// returns (-1)^n
int pow_minus_one(const int n)
{
    return n % 2 == 0 ? 1 : -1;
}

// indexing
int indexing(const int a, const int b, const int i, const int j, const int NGrids)
{
    return ((a * NumPES + b) * NGrids + i) * NGrids + j;
}

// I/O functions

// read a double: mass, x0, etc
double read_double(istream& is)
{
    static string buffer;
    static double temp;
    getline(is, buffer);
    is >> temp;
    getline(is, buffer);
    return temp;
}

// to print current time
ostream& show_time(ostream& os)
{
    auto time = chrono::system_clock::to_time_t(chrono::system_clock::now());
    os << ctime(&time);
    return os;
}


// evolution related function

// initialize the 2d gaussian wavepacket, and normalize it
void density_matrix_initialization(const int NGrids, const double* GridPosition, const double* GridMomentum, const double dx, const double dp, const double x0, const double p0, const double SigmaX, const double SigmaP, Complex* rho_adia)
{
    // constant: NGrids^2
    const int SquareNGrids = NGrids * NGrids;
    // for non-[0][0] elements, the initial density matrix is zero
    memset(rho_adia + SquareNGrids, 0, SquareNGrids * (NoMatrixElement - 1) * sizeof(Complex));
    // for ground state, it is a gaussian. rho[0][0](x,p,0)=exp(-(x-x0)^2/2sigma_x^2-(p-p0)^2/2sigma_p^2)/(2*pi*sigma_x*sigma_p)
    for (int i = 0; i < NGrids; i++)
    {
        for (int j = 0; j < NGrids; j++)
        {
            rho_adia[i * NGrids + j] = exp(-(pow((GridPosition[i] - x0) / SigmaX, 2) - pow((GridMomentum[j] - p0) / SigmaP, 2)) / 2.0) / (2.0 * pi * SigmaX * SigmaP);
        }
    }
    // normalization, because of numerical error
    double NormFactor = sqrt(accumulate(rho_adia, rho_adia + SquareNGrids, Complex(0.0)).real() * dx * dp);
    for (int i = 0; i < NGrids; i++)
    {
        rho_adia[i] /= NormFactor;
    }
}

// construct the Liouville superoperator
ComplexMatrix Liouville_construction(const int NGrids, const double* GridPosition, const double* GridMomentum, const double dx, const double dp, const double mass)
{
    // index: alpha*numpes*ngrids^2+beta*ngrids^2+i*ngrids+j
    ComplexMatrix Liouville(NGrids * NGrids * NoMatrixElement);
    // in the following part, a=alpha, b=beta, double letter means \prime
    // 1. F*drho/dP, i*(-1)^{j'-j}/2(j'-j)dp *(faa'(b=b')+fb'b(a=a'))(j!=j',i=i')
    for (int i = 0; i < NGrids; i++)
    {
        const RealMatrix F = DiaForce(GridPosition[i]);
        for (int j = 0; j < NGrids; j++)
        {
            for (int jj = 0; jj < NGrids; jj++)
            {
                if (jj != j)
                {
                    // here, delta_{ii'}(1-delta_{jj'}) is used
                    const Complex Coefficient(0.0, pow_minus_one(jj - j) / 2.0 / (jj - j) / dp);
                    for (int a = 0; a < NumPES; a++)
                    {
                        for (int b = 0; b < NumPES; b++)
                        {
                            for (int c = 0; c < NumPES; c++)
                            {
                                // c=a' here, b=b'
                                Liouville(indexing(a, b, i, j, NGrids), indexing(c, b, i, jj, NGrids)) += Coefficient * F(a, c);
                                // c=b' here, a=a'
                                Liouville(indexing(a, b, i, j, NGrids), indexing(a, c, i, jj, NGrids)) += Coefficient * F(c, b);
                            }
                        }
                    }
                }
            }
        }
    }
    // 2. P/M*drho/dx, i*(-1)^{i'-i}Pj/(i'-i)Mdx (i'!=i, j'=j, a'=a, b'=b)
    for (int i = 0; i < NGrids; i++)
    {
        for (int ii = 0; ii < NGrids; ii++)
        {
            if (ii != i)
            {
                for (int j = 0; j < NGrids; j++)
                {
                    const Complex Coefficient(0.0, pow_minus_one(ii - i) * GridMomentum[j] / (ii - i) / mass / dx);
                    for (int a = 0; a < NumPES; a++)
                    {
                        for (int b = 0; b < NumPES; b++)
                        {
                            Liouville(indexing(a, b, i, j, NGrids), indexing(a, b, ii, j, NGrids)) += Coefficient;
                        }
                    }
                }
            }
        }
    }
    // 3. (Haa'(b'=b)-Hb'b(a'=a))/hbar(i'=i,j'=j)
    for (int i = 0; i < NGrids; i++)
    {
        for (int j = 0; j < NGrids; j++)
        {
            const RealMatrix H = DiaPotential(GridPosition[i]) + pow(GridMomentum[j], 2) / 2.0 / mass;
            for (int a = 0; a < NumPES; a++)
            {
                for (int b = 0; b < NumPES; b++)
                {
                    for (int c = 0; c < NumPES; c++)
                    {
                        // c=a' here
                        Liouville(indexing(a, b, i, j, NGrids), indexing(c, b, i, j, NGrids)) += H(a, c);
                        // c=b' here
                        Liouville(indexing(a, b, i, j, NGrids), indexing(a, c, i, j, NGrids)) += H(c, b);
                    }
                }
            }
        }
    }
    return Liouville;
}

// calculate the population on each PES
void calculate_popultion(const int NGrids, const double dx, const double dp, const Complex* rho_adia, double* Population)
{
    Complex InnerProduct;
    const int NoPSGrids = NGrids * NGrids;
    // calculate the inner product of each PES
    for (int i = 0; i < NumPES; i++)
    {
        cblas_zdotc_sub(NoPSGrids, rho_adia + i * i * NoPSGrids, 1, rho_adia + i * i * NoPSGrids, 1, &InnerProduct);
        Population[i] = InnerProduct.real() * dx * dp;
    }
}
