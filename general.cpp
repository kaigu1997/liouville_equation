/// @file general.cpp
/// @brief Implementation of general.h

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

/// Do the cutoff, e.g. 0.2493 -> 0.2, 1.5364 -> 1
double cutoff(const double val)
{
    double pownum = pow(10, static_cast<int>(floor(log10(val))));
    return static_cast<int>(val / pownum) * pownum;
}

int pow_minus_one(const int n)
{
    return n % 2 == 0 ? 1 : -1;
}

// I/O functions

/// The file is one line declarator and one line the value,
/// so using a buffer string to read the declarator
/// and the rest of the second line.
double read_double(istream& is)
{
    static string buffer;
    static double temp;
    getline(is, buffer);
    is >> temp;
    getline(is, buffer);
    return temp;
}

/// Print current time in RFC 2822 format
ostream& show_time(ostream& os)
{
    auto time = chrono::system_clock::to_time_t(chrono::system_clock::now());
    os << ctime(&time);
    return os;
}


// evolution related function

/// Initialize the PWTDM (Partial Wigner-Transformed Density Matrix) with a gaussian wavepacket
/// 
/// i.e. the ground state is exp(-(x-x0)^2/2sigma_x^2-(p-p0)^2/2sigma_p^2)/(2*pi*sigma_x*sigma_p)
/// and all the other PES and off-diagonal elements are 0, and then normalize it
void density_matrix_initialization(const int NGrids, const double* const GridPosition, const double* const GridMomentum, const double dx, const double dp, const double x0, const double p0, const double SigmaX, const double SigmaP, ComplexMatrixMatrix& rho_adia)
{
    // NormFactor is for normalization
    double NormFactor = 0;
    for (int i = 0; i < NGrids; i++)
    {
        const double& xi = GridPosition[i];
        for (int j = 0; j < NGrids; j++)
        {
            const double& pj = GridMomentum[j];
            // for non-[0][0] elements, the initial density matrix is zero
            rho_adia[i][j] *= 0.0;
            // for ground state, it is a gaussian. rho[0][0](x,p,0)=exp(-(x-x0)^2/2sigma_x^2-(p-p0)^2/2sigma_p^2)/(2*pi*sigma_x*sigma_p)
            rho_adia[i][j][0][0] = exp(-(pow((xi - x0) / SigmaX, 2) + pow((pj - p0) / SigmaP, 2)) / 2.0) / (2.0 * pi * SigmaX * SigmaP);
            NormFactor += rho_adia[i][j][0][0].real();
        }
    }
    NormFactor *= dx * dp;
    // normalization, because of numerical error
    for (int i = 0; i < NGrids; i++)
    {
        for (int j = 0; j < NGrids; j++)
        {
            rho_adia[i][j][0][0] /= NormFactor;
        }
    }
}

void calculate_popultion(const int NGrids, const double dx, const double dp, const ComplexMatrixMatrix& rho_adia, double* const Population)
{
    // calculate the inner product of each PES
    for (int a = 0; a < NumPES; a++)
    {
        Population[a] = 0;
        for (int i = 0; i < NGrids; i++)
        {
            for (int j = 0; j < NGrids; j++)
            {
                Population[a] += rho_adia[i][j][a][a].real();
            }
        }
        Population[a] *= dx * dp;
    }
}

tuple<double, double, double> calculate_average(const ComplexMatrixMatrix& rho, const int NGrids, const double* const GridPosition, const double* const GridMomentum, const double mass, const double dx, const double dp, const Representation BasisOfRho)
{
    double energy = 0, x = 0, p = 0;
    // calculate: <B>=sum_{i,j}(B(xi,pj)*P(xi,pj))*dx*dp
    for (int i = 0; i < NGrids; i++)
    {
        const double& xi = GridPosition[i];
        for (int j = 0; j < NGrids; j++)
        {
            const double& pj = GridMomentum[j];
            // loop over diagonal elements only
            for (int a = 0; a < NumPES; a++)
            {
                const double& ppl = rho[i][j][a][a].real();
                energy += ppl * (potential[BasisOfRho](xi)[a][a] + pj * pj / 2.0 / mass);
                x += ppl * xi;
                p += ppl * pj;
            }
        }
    }
    return make_tuple(energy * dx * dp, x * dx * dp, p * dx * dp);
}

/// Evolve the quantum liouville: exp(-iLQt)rho
///
/// -iLQrho=-i/hbar[V-i*hbar*P/Mass*D,rho]
///
/// Input t should be dt/2
void quantum_liouville_propagation(ComplexMatrixMatrix& rho, const int NGrids, const double* const GridPosition, const double* const GridMomentum, const double mass, const double dt, const Representation BasisOfRho)
{
    if (BasisOfRho == Diabatic)
    {
        // for diabatic basis, D=0, simple transform to adiabatic representation
        // exp(-iLQt)rho_dia=to_dia(exp(-iVd t/hbar)*rho_adia*exp(iVd t/hbar))
        basis_transform[Diabatic][Adiabatic](rho, NGrids, GridPosition);
        for (int i = 0; i < NGrids; i++)
        {
            const RealMatrix& H = potential[Adiabatic](GridPosition[i]);
            for (int j = 0; j < NGrids; j++)
            {
                for (int a = 0; a < NumPES; a++)
                {
                    // no operation needed for diagonal elements
                    for (int b = a + 1; b < NumPES; b++)
                    {
                        rho[i][j][a][b] *= exp(Complex(0.0, (H[b][b] - H[a][a]) * dt / hbar));
                        rho[i][j][b][a] *= exp(Complex(0.0, (H[a][a] - H[b][b]) * dt / hbar));
                    }
                }
            }
        }
        basis_transform[Adiabatic][Diabatic](rho, NGrids, GridPosition);
    }
    else
    {
        // for other basis, no tricks to play
        // exp(-iLQt)rho=exp(-iH't/hbar)*rho*exp(iH't/hbar)
        // =C^T*exp(-iH'd t/hbar)*C*rho*C^T*exp(iH'd t/hbar)*C
        for (int i = 0; i < NGrids; i++)
        {
            const double& x = GridPosition[i];
            const ComplexMatrix& H = potential[BasisOfRho](x);
            const ComplexMatrix& D = coupling[BasisOfRho](x);
            for (int j = 0; j < NGrids; j++)
            {
                const double& p = GridMomentum[j];
                ComplexMatrix EigVec = H - Complex(0.0, hbar * p / mass) * D;
                double EigVal[NumPES];
                if (LAPACKE_zheev(LAPACK_ROW_MAJOR, 'V', 'U', NumPES, reinterpret_cast<MKL_Complex16*>(EigVec.data()), NumPES, EigVal) != 0)
                {
                    cerr << "UNABLE TO DIAGONALIZE THE QUANTUM LIOUVILLE AT X=" << x << " AND P=" << p << endl;
                    exit(300);
                }
                Complex EigDenEig[NumPES * NumPES];
                // first, C*rho*C^T
                cblas_zhemm(CblasRowMajor, CblasRight, CblasUpper, NumPES, NumPES, &Alpha, rho[i][j].data(), NumPES, EigVec.data(), NumPES, &Beta, EigDenEig, NumPES);
                cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, NumPES, NumPES, NumPES, &Alpha, EigDenEig, NumPES, EigVec.data(), NumPES, &Beta, rho[i][j].data(), NumPES);
                // make it hermitian
                rho[i][j].hermitize();
                // second, exp(-iH'd t/hbar)*C*rho*C^T*exp(iH'd t/hbar)
                for (int a = 0; a < NumPES; a++)
                {
                    // no operation needed for diagonal elements
                    for (int b = a + 1; b < NumPES; b++)
                    {
                        rho[i][j][a][b] *= exp(Complex(0.0, (EigVal[b] - EigVal[a]) * dt / hbar));
                        rho[i][j][b][a] *= exp(Complex(0.0, (EigVal[a] - EigVal[b]) * dt / hbar));
                    }
                }
                // make itself hermitian for next basis transformation
                rho[i][j].hermitize();
                // last, C^T*exp(-iH'd t/hbar)*C*rho*C^T*exp(iH'd t/hbar)*C
                cblas_zhemm(CblasRowMajor, CblasLeft, CblasUpper, NumPES, NumPES, &Alpha, rho[i][j].data(), NumPES, EigVec.data(), NumPES, &Beta, EigDenEig, NumPES);
                cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, NumPES, NumPES, NumPES, &Alpha, EigVec.data(), NumPES, EigDenEig, NumPES, &Beta, rho[i][j].data(), NumPES);
                // make it hermitian again
                rho[i][j].hermitize();
            }
        }
    }
}

/// Generate the derivative matrix
///
/// Dij=-(-1)^{j-i}/(j-i)/dx(or dp)*(1-delta_ij)
/// 
/// Derived from infinite order finite difference of DVR
///
/// @param NGrids the number of grids in density matrix, overall NGrids^2 sub-density matrices
/// @param delta dx or dp, the spacing between adjacent grids
/// @return the 1st-order derivative matrix, which is real anti-hermitian
static RealMatrix derivative(const int NGrids, const double delta)
{
    RealMatrix result(NGrids);
    for (int i = 0; i < NGrids; i++)
    {
        for (int j = i + 1; j < NGrids; j++)
        {
            result[i][j] = -pow_minus_one(j - i) / delta / (j - i);
            result[j][i] = -result[i][j];
        }
    }
    return result;
}

/// Calculate the eigenvalue/vector of -i*position_derivative matrix
///
/// i.e., to diagonalize the hermitian matrix -i*DR
///
/// C^T*exp(-iDR)*C is diagonal, C is the return matrix
///
/// @param NGrids the number of grids in density matrix, overall NGrids^2 sub-density matrices
/// @param dx the grid spacing of position coordinate
/// @param eigenvalue the array to save the eigenvalues of the -i*DR matrix
/// @return the transformation matrix to diagonalize the -i*DR matrix
/// @see derivative(), diagonalize_derivative_p(), classical_position_liouville_propagator()
static ComplexMatrix diagonalize_derivative_x(const int NGrids, const double dx, double* const eigenvalue)
{
    static ComplexMatrix TransformationMatrix = -1.0i * ComplexMatrix(derivative(NGrids, dx));
    static double* EigVal = nullptr;
    static int LastNGrids = NGrids;
    static double LastDx = dx;
    static bool FirstRun = true;
    if (FirstRun == true)
    {
        // for the first run, diagonalization
        EigVal = new double[NGrids];
        if (LAPACKE_zheev(LAPACK_ROW_MAJOR, 'V', 'U', NGrids, reinterpret_cast<MKL_Complex16*>(TransformationMatrix.data()), NGrids, EigVal) != 0)
        {
            cerr << "UNABLE TO DIAGONALIZE THE POSITION LIOUVILLE" << endl;
            exit(301);
        }
        FirstRun == false;
    }
    else if (NGrids != LastNGrids || abs (dx - LastDx) > 1e-2 * dx)
    {
        // not the first time, and use different parameter, redo everything
        if (NGrids != LastNGrids)
        {
            LastNGrids = NGrids;
            delete[] EigVal;
            EigVal = new double[NGrids];
        }
        LastDx = dx;
        TransformationMatrix = -1.0i * ComplexMatrix(derivative(NGrids, dx));
        if (LAPACKE_zheev(LAPACK_ROW_MAJOR, 'V', 'U', NGrids, reinterpret_cast<MKL_Complex16*>(TransformationMatrix.data()), NGrids, EigVal) != 0)
        {
            cerr << "UNABLE TO DIAGONALIZE THE POSITION LIOUVILLE" << endl;
            exit(301);
        }
    } // for not the first time with same parameter, nothing to do
    copy(EigVal, EigVal + NGrids, eigenvalue);
    return TransformationMatrix;
}

/// evolve the classical position liouville: exp(-iLRt)rho
///
/// -iLRrho=-P/M*drho/dR=-i*P/M*(-i*DR)*rho
///
/// input t should be dt/2 as well
void classical_position_liouville_propagator(ComplexMatrixMatrix& rho, const int NGrids, const double* const GridMomentum, const double mass, const double dx, const double dt)
{
    double* EigVal = new double[NGrids];
    const ComplexMatrix& EigVec = diagonalize_derivative_x(NGrids, dx, EigVal);
    for (int j = 0; j < NGrids; j++)
    {
        // the evolving matrix is the same for a and b, so construct and diagonalize here
        const double& p = GridMomentum[j];
        for (int a = 0; a < NumPES; a++)
        {
            for (int b = 0; b < NumPES; b++)
            {
                // construct the vector: rho_W^{ab}(.,P_j)
                Complex* DifferentPosition = new Complex[NGrids];
                Complex* Transformed = new Complex[NGrids];
                // assign the value
                for (int i = 0; i < NGrids; i++)
                {
                    DifferentPosition[i] = rho[i][j][a][b];
                }
                // evolve
                // basis transformation
                cblas_zgemv(CblasRowMajor, CblasConjTrans, NGrids, NGrids, &Alpha, EigVec.data(), NGrids, DifferentPosition, 1, &Beta, Transformed, 1);
                // evolve
                for (int i = 0; i < NGrids; i++)
                {
                    Transformed[i] *= exp(Complex(0.0, - p / mass * EigVal[i] * dt));
                }
                // transform back
                cblas_zgemv(CblasRowMajor, CblasNoTrans, NGrids, NGrids, &Alpha, EigVec.data(), NGrids, Transformed, 1, &Beta, DifferentPosition, 1);
                // assign value back
                for (int i = 0; i < NGrids; i++)
                {
                    rho[i][j][a][b] = DifferentPosition[i];
                }
                // free the memory
                delete[] Transformed;
                delete[] DifferentPosition;
            }
        }
        // to make density matrix on each grid hermitian
        for (int i = 0; i < NGrids; i++)
        {
            rho[i][j].hermitize();
        }
    }
    // free the memory
    delete[] EigVal;
}

/// Calculate the eigenvalue/vector of -i*momentum_derivative matrix
///
/// i.e., to diagonalize the hermitian matrix -i*DP
///
/// C^T*exp(-iDP)*C is diagonal, C is the return matrix
///
/// @param NGrids the number of grids in density matrix, overall NGrids^2 sub-density matrices
/// @param dp the grid spacing of momentum coordinate
/// @param eigenvalue the array to save the eigenvalues of the -i*DP matrix
/// @return the transformation matrix to diagonalize the -i*DP matrix
/// @see derivative(), diagonalize_derivative_x(), classical_momentum_liouville_propagator()
static ComplexMatrix diagonalize_derivative_p(const int NGrids, const double dp, double* const eigenvalue)
{
    static ComplexMatrix TransformationMatrix = -1.0i * ComplexMatrix(derivative(NGrids, dp));
    static double* EigVal = nullptr;
    static int LastNGrids = NGrids;
    static double LastDp = dp;
    static bool FirstRun = true;
    if (FirstRun == true)
    {
        // for the first run, diagonalization
        EigVal = new double[NGrids];
        if (LAPACKE_zheev(LAPACK_ROW_MAJOR, 'V', 'U', NGrids, reinterpret_cast<MKL_Complex16*>(TransformationMatrix.data()), NGrids, EigVal) != 0)
        {
            cerr << "UNABLE TO DIAGONALIZE THE POSITION LIOUVILLE" << endl;
        }
        FirstRun == false;
    }
    else if (NGrids != LastNGrids || abs(dp - LastDp) > 1e-2 * dp)
    {
        // not the first time, and use different parameter, redo everything
        LastNGrids = NGrids;
        LastDp = dp;
        delete[] EigVal;
        EigVal = new double[NGrids];
        TransformationMatrix = -1.0i * ComplexMatrix(derivative(NGrids, dp));
        if (LAPACKE_zheev(LAPACK_ROW_MAJOR, 'V', 'U', NGrids, reinterpret_cast<MKL_Complex16*>(TransformationMatrix.data()), NGrids, EigVal) != 0)
        {
            cerr << "UNABLE TO DIAGONALIZE THE MOMENTUM LIOUVILLE" << endl;
        }
    } // for not the first time with same parameter, nothing to do
    copy(EigVal, EigVal + NGrids, eigenvalue);
    return TransformationMatrix;
}

/// evolve the classical position liouville: exp(-iLRt)rho
///
/// -iLPrho=-1/2(F*drho/dP+drho/dP*F)=-i(Faa+Fbb)/2*(-i*DP)*rho
/// if under force basis.
///
/// input t should be dt
void classical_momentum_liouville_propagator(ComplexMatrixMatrix& rho, const int NGrids, const double* const GridPosition, const double dp, const double dt)
{
    // transform to force basis first
    basis_transform[Diabatic][ForceBasis](rho, NGrids, GridPosition);
    double* EigVal = new double[NGrids];
    const ComplexMatrix& EigVec = diagonalize_derivative_p(NGrids, dp, EigVal);
    for (int i = 0; i < NGrids; i++)
    {
        // the eigen forces
        const double& x = GridPosition[i];
        const RealMatrix& F = force[ForceBasis](x);
        for (int a = 0; a < NumPES; a++)
        {
            for (int b = 0; b < NumPES; b++)
            {
                // construct the vector: rho_W^{ab}(R_i,.)
                Complex* DifferentMomentum = new Complex[NGrids];
                Complex* Transformed = new Complex[NGrids];
                // assign the value
                for (int j = 0; j < NGrids; j++)
                {
                    DifferentMomentum[j] = rho[i][j][a][b];
                }
                // evolve
                // basis transformation
                cblas_zgemv(CblasRowMajor, CblasConjTrans, NGrids, NGrids, &Alpha, EigVec.data(), NGrids, DifferentMomentum, 1, &Beta, Transformed, 1);
                // evolve
                for (int j = 0; j < NGrids; j++)
                {
                    Transformed[j] *= exp(Complex(0.0, -(F[a][a] + F[b][b]) / 2.0 * EigVal[j] * dt));
                }
                // transform back
                cblas_zgemv(CblasRowMajor, CblasNoTrans, NGrids, NGrids, &Alpha, EigVec.data(), NGrids, Transformed, 1, &Beta, DifferentMomentum, 1);
                // assign value back
                for (int j = 0; j < NGrids; j++)
                {
                    rho[i][j][a][b] = DifferentMomentum[j];
                }
                // free the memory
                delete[] Transformed;
                delete[] DifferentMomentum;

            }
        }
        for (int j = 0; j < NGrids; j++)
        {
            rho[i][j].hermitize();
        }
    }
    // free the memory
    delete[] EigVal;
    // finally transform back to diabatic basis
    basis_transform[ForceBasis][Diabatic](rho, NGrids, GridPosition);
}