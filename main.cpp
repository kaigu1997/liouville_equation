// The purpose of this program is to give
// an exact solution of quantum mechanic problem
// using Discrete Variable Representation (DVR)
// in [1]J. Chem. Phys., 1992, 96(3): 1982-1991,
// with Absorbing Boundary Condition in
// [2]J. Chem. Phys., 2002, 117(21): 9552-9559
// and [3]J. Chem. Phys., 2004, 120(5): 2247-2254.
// This program could be used to solve
// exact solution under diabatic basis ONLY.
// It requires C++17 or newer C++ standards when compiling
// and needs connection to Intel(R) Math Kernel Library
// (MKL) by whatever methods: icpc/msvc/gcc -I.
// Error code criteria: 1XX for matrix, 
// 2XX for general, 3XX for pes, and 4XX for main.

#include <chrono>
#include <ctime>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mkl.h>
#include <numeric>
#include "general.h"
#include "pes.h"
#include "matrix.h"
using namespace std;

int main(void)
{
    // initialize: read input and calculate cooresponding constants
    // including the number of grids, their coordinates, etc
    // in: the input file
    ifstream in("input");
    in.sync_with_stdio(false);
    // read mass: the mass of the bath
    const double mass = read_double(in);
    // read initial wavepacket info
    // the center/width of the wavepacket
    // calculate SigmaX by SigmaP using minimum uncertainty rule
    const double x0 = read_double(in);
    const double p0 = read_double(in);
    const double SigmaP = read_double(in);
    const double SigmaX = hbar / 2.0 / SigmaP;
    // 99.7% initial momentum in this region: p0+-3SigmaP
    // calculate the region of momentum by p0 and SigmaP
    const double p0min = p0 - 3.0 * SigmaP;
    const double p0max = p0 + 3.0 * SigmaP;
    clog << "The particle weighes " << mass << " a.u.,\n"
        << "starting from " << x0 << " with initial momentum " << p0 << ".\n"
        << "Initial width of x and p are " << SigmaX << " and " << SigmaP << ", respectively.\n";
    
    // read interaction region
    const double xmin = read_double(in);
    const double xmax = read_double(in);
    const double TotalPositionLength = xmax - xmin;
    // read grid spacing, should be "~ 4 to 5 grids per de Broglie wavelength"
    // and then do the cut off, e.g. 0.2493 -> 0.2, 1.5364 -> 1
    // and the number of grids are thus determined
    const double dx = cutoff(min(read_double(in), PlanckH / p0max / 5.0));
    // NGrids: number of grids in [xmin, xmax], also in [pmin, pmax]
    const int NGrids = static_cast<int>(TotalPositionLength / dx) + 1;
    // momentum region is determined by fourier transformation:
    // p in pi*hbar/dx*[-1,1), dp=2*pi*hbar/(xmax-xmin)
    const double pmin = -pi * hbar / dx;
    const double pmax = -pmin;
    const double TotalMomentumLength = pmax - pmin;
    const double dp = TotalMomentumLength / static_cast<double>(NGrids - 1);
    // NoPSGrids: Number of Phase Space Grids
    const int NoPSGrids = NGrids * NGrids;
    // dim: total number of elements (dimension) in L/rho
    const int dim = NoPSGrids * NoMatrixElement;

    // Position/Momentum contains each grid coordinate, one in a line
    ofstream Position("x.txt"), Momentum("p.txt");
    Position.sync_with_stdio(false);
    Momentum.sync_with_stdio(false);
    // the coordinates of the grids, i.e. value of xi/pj
    double* GridPosition = new double[NGrids];
    double* GridMomentum = new double[NGrids];
    // calculate the grid coordinates, and print them
    for (int i = 0; i < NGrids; i++)
    {
        GridPosition[i] = xmin + i * dx;
        Position << GridPosition[i] << '\n';
        GridMomentum[i] = pmin + i * dp;
        Momentum << GridMomentum[i] << '\n';
    }
    clog << "dx = " << dx << ", dp = " << dp << ",\n"
        << "and there is overall " << NGrids << " grids\n"
        << "in [" << xmin << ", " << xmax <<"] and [" << pmin << ", " << pmax << "].\n";
    Position.close();
    Momentum.close();

    // read evolving time and output time, in unit of a.u.
    const double TotalTime = read_double(in);
    const double OutputTime = read_double(in);
    const double dt = OutputTime;
    // finish reading
    in.close();
    // calculate corresponding dt of the above (how many dt they have)
    const int TotalStep = static_cast<int>(TotalTime / dt);
    clog << "dt = " << dt << ", and there is overall " << TotalStep << " time steps.\n";

    // construct the Liouville superoperator.
    // diabatic Liouville used for propagator
    // drho/dt=-iLrho => rho(t)=e^(-iLt)rho(0)
    // rho is the density matrix, \rho_W^{ab}(R_i,P_j)
    // =rho[((a*numpes+b)*ngrids+i)*ngrids+j],
    // meaning the density matrix element
    // rho[a][b] at the phase space grid (Ri,Pj)
    const ComplexMatrix Liouville = Liouville_construction(NGrids, GridPosition, GridMomentum, dx, dp, mass);
    // TransformationMatrix makes dia to adia
    const ComplexMatrix TransformationMatrix = DiaToAdia(NGrids, GridPosition);

    // diagonalize the Liouville superoperator and then evolve
    // diagonalize L
    ComplexMatrix EigVec = Liouville;
    double* EigVal = new double[dim];
    if (LAPACKE_zheev(LAPACK_ROW_MAJOR, 'V', 'U', dim, reinterpret_cast<MKL_Complex16*>(EigVec.data()), dim, EigVal) > 0)
    {
        cerr << "FAILING DIAGONALIZE HAMILTONIAN WITHOUT ABSORBING POTENTIAL" << endl;
        exit(400);
    }

    // memory allocation:
    // rho(t)_adia=C2*rho(t)_dia=C2*C1*exp(-i*Hd*t)*C1T*rho(0)_dia
    // so we need: C2, C1(=EigVal), rho(t)_adia, rho(t)_dia, rho(t)_diag, rho(0)_diag
    // besides, we need to save the population on each PES
    // rho_t: diabatic/adiabatic/diagonal representation
    Complex* rho_t_dia = new Complex[dim];
    Complex* rho_t_adia = new Complex[dim];
    Complex* rho_t_diag = new Complex[dim];
    // construct the initial adiabatic wavepacket: gaussian on the ground state PES
    // rho[0][0](x,p,0)=exp(-(x-x0)^2/2sigma_x-(p-p0)^2/2sigma_p)/(pi*hbar)
    density_matrix_initialization(NGrids, GridPosition, GridMomentum, dx, dp, x0, p0, SigmaX, SigmaP, rho_t_adia);
    // calculate the initial diagonal density matrix
    cblas_zgemv(CblasRowMajor, CblasConjTrans, dim, dim, &Alpha, TransformationMatrix.data(), dim, rho_t_adia, 1, &Beta, rho_t_dia, 1);
    Complex* rho_0 = new Complex[dim];
    cblas_zgemv(CblasRowMajor, CblasConjTrans, dim, dim, &Alpha, EigVec.data(), dim, rho_t_dia, 1, &Beta, rho_0, 1);
    const ComplexVector rho_0_diag(rho_0);

    // population on each PES, and the population on each PES at last output moment
    double Population[NumPES] = {1.0}, OldPopulation[NumPES] = {0};
    // before calculating population, check if wavepacket have passed through center(=0.0)
    bool PassedCenter = false;
    // Steps contains when is each step, also one in a line
    ofstream Steps("t.txt");
    Steps.sync_with_stdio(false);
    // Output gives the partial wigner-transformed density matrix
    // In Phase space, each line is the PS-distribution at a moment:
    // rho[0][0](x0,p0,t0), rho[0][0](x0,p1,t0), ... rho[0][0](x1,p0,t0), ...
    // (continue) rho[0][1](x0,p0,t0), ... rho[1][0](x0,p0,t0), ... rho[n][n](xN,pN,t0)
    // (new line) rho[0][0](x0,p0,t1), ...
    ofstream Output("phase.txt");
    Output.sync_with_stdio(false);
    clog << "Finish diagonalization and memory allocation.\n" << show_time;


    // evolution:
    for (int iStep = 0; iStep <= TotalStep; iStep++)
    {
        const double Time = iStep * dt;
        Steps << Time << '\n';

        // calculate rho_t_diag = exp(-iH(diag)*t) * rho_0_diag,
        // each diagonal element be the eigenvalue
        for (int i = 0; i < dim; i++)
        {
            rho_t_diag[i] = exp(Complex(0.0, -EigVal[i] * Time)) * rho_0_diag[i];
        }
        // calculate rho_t_dia=C1*rho_t_diag
        cblas_zgemv(CblasRowMajor, CblasNoTrans, dim, dim, &Alpha, EigVec.data(), dim, rho_t_diag, 1, &Beta, rho_t_dia, 1);
        // calculate rho_t_adia=C2*rho_t_dia
        cblas_zgemv(CblasRowMajor, CblasNoTrans, dim, dim, &Alpha, TransformationMatrix.data(), dim, rho_t_dia, 1, &Beta, rho_t_adia, 1);
        
        // output the whole density matrix; for diagonal only real part (imag==0)
        for (int a = 0; a < NumPES; a++)
        {
            for (int b = 0; b < NumPES; b++)
            {
                for (int i = 0; i < NGrids; i++)
                {
                    for (int j = 0; j < NGrids; j++)
                    {
                        Output << ' ' << rho_t_adia[indexing(a, b, i, j, NGrids)].real();
                    }
                }
                for (int i = 0; i < NGrids; i++)
                {
                    for (int j = 0; j < NGrids; j++)
                    {
                        Output << ' ' << rho_t_adia[indexing(a, b, i, j, NGrids)].real()
                            << ' ' << rho_t_adia[indexing(a, b, i, j, NGrids)].imag();
                    }
                }
            }
        }
        Output << '\n';        

        
    }
    // after evolution, print time and frees the resources
    clog << "Finish evolution.\n" << show_time << endl;
    Steps.close();
    Output.close();

    // print the final info
    /*/ model 1 and 3
    cout << p0; // */
    // model 2
    cout << log(p0 * p0 / 2.0 / mass);// */
    calculate_popultion(NGrids, dx, dp, rho_t_adia, Population);
    for (int i = 0; i < NumPES; i++)
    {
        cout << ' ' << Population[i];
    }
    cout << '\n';

    // end. free the memory.
    delete[] rho_t_adia;
    delete[] rho_t_dia;
    delete[] rho_t_diag;
    delete[] EigVal;    
    delete[] GridPosition;
    delete[] GridMomentum;
	return 0;
}
