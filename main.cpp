// The purpose of this program is to give
// an exact solution of quantum mechanic problem
// using Mixed Quantum-Classical Liouville Equation
// (MQCLE) by Discrete Variable Representation (DVR).
// This program could be used to solve the MQCLE
// under diabatic/adiabatic/force basis, etc.
// It requires C++17 or newer C++ standards when compiling
// and needs connection to Intel(R) Math Kernel Library
// (MKL) by whatever methods: icpc/msvc/gcc -I.
// Error code criteria: 1XX for matrix, 
// 2XX for pes, 3XX for general, and 4XX for main.

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
    cout.sync_with_stdio(false);
    clog.sync_with_stdio(false);
    cerr.sync_with_stdio(false);
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
        << "Initial width of x and p are " << SigmaX << " and " << SigmaP << ", respectively." << endl;
    
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
    const double dt = cutoff(read_double(in));
    // finish reading
    in.close();
    // calculate corresponding dt of the above (how many dt they have)
    const int TotalStep = static_cast<int>(TotalTime / dt);
    const int OutputStep = static_cast<int>(OutputTime / dt);
    clog << "dt = " << dt << ", and there is overall " << TotalStep << " time steps." << endl;

    // memory allocation: density matrix
    ComplexMatrixMatrix rho(NGrids, NumPES);
    // construct the initial adiabatic PWTDM: gaussian on the ground state PES
    // rho[0][0](x,p,0)=exp(-(x-x0)^2/2sigma_x-(p-p0)^2/2sigma_p)/(pi*hbar)
    // initially in the adiabatic basis
    density_matrix_initialization(NGrids, GridPosition, GridMomentum, dx, dp, x0, p0, SigmaX, SigmaP, rho);
    // then transform to diabatic basis
    basis_transform[Adiabatic][Diabatic](rho, NGrids, GridPosition);

    // population on each PES, and the population on each PES at last output moment
    double Population[NumPES] = {1.0};
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
    clog << "Finish diagonalization and memory allocation.\n" << show_time << endl;


    // evolve: Trotter expansion
    // rho(t+dt)=exp(-iLQdt/2)exp(-iLRdt/2)exp(-iLPdt)exp(-iLRdt/2)exp(-iLQdt/2)rho(t)
    // -iLQrho=-i/hbar[H-ihbarP/M*D,rho], -iLRrho=-P/M*drho/dR, -iLPrho=-(F*drho/dP+drho/dP*F)/2
    // derivatives are calculated by infinite order finite difference
    for (int iStep = 0; iStep <= TotalStep; iStep++)
    {
        const double Time = iStep * dt;

        // for the output case
        if (iStep % OutputStep == 0)
        {
            basis_transform[Diabatic][Adiabatic](rho, NGrids, GridPosition);
            // Steps << Time << '\n';            
            // output the whole density matrix
            Output << rho << endl;
            basis_transform[Adiabatic][Diabatic](rho, NGrids, GridPosition);
        }

        // evolve
        // 1. Quantum Liouville, -iLQ*rho=-i/hbar[V-i*hbar*P/m*D, rho]
        // for diabatic basis, D=0, so simply trans to adia basis
        // exp(-iLQt)rho_dia=exp(-iVd t/hbar)*rho_adia*exp(iVd t/hbar), t=dt/2
        quantum_liouville_propagation(rho, NGrids, GridPosition, GridMomentum, mass, dt / 2.0, Diabatic);
        // 2. Classical position Liouville, -iLRrho=-P/M*drho/dR = -P/M*D_R*rho
        // so exp(-iLRt)=exp(-i*(-iPD_R/M)*t), t=dt/2
        classical_position_liouville_propagator(rho, NGrids, GridMomentum, mass, dx, dt / 2.0);
        // 3. Classical Momentum Liouville, under force basis,
        // -iLQrho=-(Fd*drho/dP+drho/dP*Fd)/2
        // so exp(-iLQt)=exp(-i(-i*(Fdaa+Fdbb)*D_P/2)t)
        // transform the density matrix to force basis
        classical_momentum_liouville_propagator(rho, NGrids, GridPosition, dp, dt);
        // 4. Classical position Liouville again
        classical_position_liouville_propagator(rho, NGrids, GridMomentum, mass, dx, dt / 2.0);
        // 5. Quantum Liouville again
        quantum_liouville_propagation(rho, NGrids, GridPosition, GridMomentum, mass, dt / 2.0, Diabatic);
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
    calculate_popultion(NGrids, dx, dp, rho, Population);
    for (int i = 0; i < NumPES; i++)
    {
        cout << ' ' << Population[i];
    }
    cout << '\n';

    // end. free the memory.    
    delete[] GridPosition;
    delete[] GridMomentum;
	return 0;
}
