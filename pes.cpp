// implementation of pes.h:
// diabatic PES and absorbing potential

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mkl.h>
#include "general.h"
#include "matrix.h"
#include "pes.h"

// Diabatic reprersentation
/*Model 1
// constants in the diabatic Potential
static const double A = 0.01, B = 1.6, C = 0.005, D = 1.0;
// subsystem diabatic Potential, the force, and hessian
// the "forces" are negative derivative over x
// the hessian matrix is the second derivative over x
static RealMatrix diabatic_potential(const double x)
{
    RealMatrix Potential(NumPES);
    Potential[0][1] = Potential[1][0] = C * exp(-D * x * x);
    Potential[0][0] = sgn(x) * A * (1.0 - exp(-sgn(x) * B * x));
    Potential[1][1] = -Potential[0][0];
    return Potential;
}
static RealMatrix diabatic_force(const double x)
{
    RealMatrix Force(NumPES);
    Force[0][1] = Force[1][0] = 2.0 * C * D * x * exp(-D * x * x);
    Force[0][0] = -A * B * exp(-sgn(x) * B * x);
    Force[1][1] = -Force[0][0];
    return Force;
}
static RealMatrix diabatic_hesse(const double x)
{
    RealMatrix Hesse(NumPES);
    Hesse[0][1] = Hesse[1][0] = 2 * C * D * (2 * D * x * x - 1) * exp(-D * x * x);
    Hesse[0][0] = -sgn(x) * A * B * B * exp(-sgn(x) * B * x);
    Hesse[1][1] = -Hesse[0][0];
    return Hesse;
}// */
//*Model 2
// constants in the diabatic Potential
static const double A = 0.10, B = 0.28, C = 0.015, D = 0.06, E = 0.05;
// subsystem diabatic Potential, the force, and hessian
// the "forces" are negative derivative over x
// the hessian matrix is the second derivative over x
static RealMatrix diabatic_potential(const double x)
{
    RealMatrix Potential(NumPES);
    Potential[0][1] = Potential[1][0] = C * exp(-D * x * x);
    Potential[1][1] = E - A * exp(-B * x * x);
    return Potential;
}
static RealMatrix diabatic_force(const double x)
{
    RealMatrix Force(NumPES);
    Force[0][1] = Force[1][0] = 2 * C * D * x * exp(-D * x * x);
    Force[1][1] = -2 * A * B * x * exp(-B * x * x);
    return Force;
}
static RealMatrix diabatic_hesse(const double x)
{
    RealMatrix Hesse(NumPES);
    Hesse[0][1] = Hesse[1][0] = 2 * C * D * (2 * D * x * x - 1) * exp(-D * x * x);
    Hesse[1][1] = -2 * A * B * (2 * B * x * x - 1) * exp(-B * x * x);
    return Hesse;
}// */
/*Model 3
// constants in the diabatic Potential
static const double A = 6e-4, B = 0.10, C = 0.90;
// subsystem diabatic Potential, the force, and hessian
// the "forces" are negative derivative over x
// the hessian matrix is the second derivative over x
static RealMatrix diabatic_potential(const double x)
{
    RealMatrix Potential(NumPES);
    Potential[0][0] = A;
    Potential[1][1] = -A;
    Potential[0][1] = Potential[1][0] = B * (1 - sgn(x) * (exp(-sgn(x) * C * x) - 1));
    return Potential;
}
static RealMatrix diabatic_force(const double x)
{
    RealMatrix Force(NumPES);
    Force[0][1] = Force[1][0] = -B * C * exp(-sgn(x) * C * x);
    return Force;
}
static RealMatrix diabatic_hesse(const double x)
{
    RealMatrix Hesse(NumPES);
    Hesse[0][1] = Hesse[1][0] = -sgn(x) * B * C * C * exp(-sgn(x) * C * x);
    return Hesse;
}// */
static RealMatrix diabatic_coupling(const double x)
{
    return RealMatrix(2);
}


// Adiabatic representation
// transformation matrix from diabatic density matrix to adiabatic one
// i.e. C^T*rho(dia)*C=rho(adia), which diagonalize PES only
static RealMatrix diabatic_to_adiabatic_at_one_point(const double x)
{
    RealMatrix EigVec = diabatic_potential(x);
    double EigVal[NumPES];
    if (LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'N', 'U', NumPES, EigVec.data(), NumPES, EigVal) != 0)
    {
        cerr << "UNABLE TO CALCULATE ADIABATIC REPRESENTATION AT " << x << endl;
        exit(200);
    }
    return EigVec;
}
// calculate adiabatic eigenvalues
static RealMatrix adiabatic_potential(const double x)
{
    RealMatrix EigVec = diabatic_potential(x);
    double EigVal[NumPES];
    if (LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', NumPES, EigVec.data(), NumPES, EigVal) != 0)
    {
        cerr << "UNABLE TO CALCULATE ADIABATIC HAMILTONIAN AT " << x << endl;
        exit(201);
    }
    RealMatrix result(NumPES);
    for (int i = 0; i < NumPES; i++)
    {
        result[i][i] = EigVal[i];
    }
    return result;    
}
// calculate from diaforce
static RealMatrix adiabatic_force(const double x)
{
    const RealMatrix TransformMatrix = diabatic_to_adiabatic_at_one_point(x);
    double FoEig[NumPES * NumPES];
    RealMatrix result(NumPES);
    cblas_dsymm(CblasRowMajor, CblasLeft, CblasUpper, NumPES, NumPES, 1.0, diabatic_force(x).data(), NumPES, TransformMatrix.data(), NumPES, 0.0, FoEig, NumPES);
    cblas_dgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, NumPES, NumPES, NumPES, 1.0, TransformMatrix.data(), NumPES, FoEig, NumPES, 0.0, result.data(), NumPES);
    result.symmetrize();
    return result;
}
// NAC under adiabatic representation is dij=Fij/(Ei-Ej)
static RealMatrix adiabatic_coupling(const double x)
{
    RealMatrix result(NumPES);
    const RealMatrix Force = adiabatic_force(x), Energy = adiabatic_potential(x);
    for (int i = 0; i < NumPES; i++)
    {
        for (int j = 0; j < NumPES; j++)
        {
            if (j < i)
            {
                result[i][j] = -result[j][i];
            }
            else if (j == i)
            {
                result[i][j] = 0;
            }
            else
            {
                result[i][j] = Force[i][j] / (Energy[i][i] - Energy[j][j]);
            }
            
        }
    }
    return result;
}

// Force basis
// basis transformation, C^T*M(dia)*C=M(force)
static RealMatrix diabatic_to_force_basis_at_one_point(const double x)
{
    RealMatrix EigVec = diabatic_force(x);
    double EigVal[NumPES];
    if (LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'N', 'U', NumPES, EigVec.data(), NumPES, EigVal) != 0)
    {
        cerr << "UNABLE TO CALCULATE ADIABATIC REPRESENTATION AT " << x << endl;
        exit(202);
    }
    return EigVec;
}
// V under force basis
static RealMatrix force_basis_potential(const double x)
{
    const RealMatrix TransformMatrix = diabatic_to_force_basis_at_one_point(x);
    double VEig[NumPES * NumPES];
    RealMatrix result(NumPES);
    cblas_dsymm(CblasRowMajor, CblasLeft, CblasUpper, NumPES, NumPES, 1.0, diabatic_potential(x).data(), NumPES, TransformMatrix.data(), NumPES, 0.0, VEig, NumPES);
    cblas_dgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, NumPES, NumPES, NumPES, 1.0, TransformMatrix.data(), NumPES, VEig, NumPES, 0.0, result.data(), NumPES);
    result.symmetrize();
    return result;
}
// diagonal matrix
static RealMatrix force_basis_force(const double x)
{
    RealMatrix EigVec = diabatic_force(x);
    double EigVal[NumPES];
    if (LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', NumPES, EigVec.data(), NumPES, EigVal) != 0)
    {
        cerr << "UNABLE TO CALCULATE ADIABATIC REPRESENTATION AT " << x << endl;
        exit(203);
    }
    RealMatrix result(NumPES);
    for (int i = 0; i < NumPES; i++)
    {
        result[i][i] = EigVal[i];
    }
    return result;
}
// d2V/dx2, for calculating NAC under force basis
static RealMatrix force_basis_hesse(const double x)
{
    const RealMatrix TransformMatrix = diabatic_to_force_basis_at_one_point(x);
    double HeEig[NumPES * NumPES];
    RealMatrix result(NumPES);
    cblas_dsymm(CblasRowMajor, CblasLeft, CblasUpper, NumPES, NumPES, 1.0, diabatic_hesse(x).data(), NumPES, TransformMatrix.data(), NumPES, 0.0, HeEig, NumPES);
    cblas_dgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, NumPES, NumPES, NumPES, 1.0, TransformMatrix.data(), NumPES, HeEig, NumPES, 0.0, result.data(), NumPES);
    result.symmetrize();
    return result;
}
// NAC, dij = Hesse[i][j]/(F[i] - F[j])
static RealMatrix force_basis_coupling(const double x)
{
    RealMatrix result(NumPES);
    const RealMatrix Hesse = force_basis_hesse(x), Force = force_basis_force(x);
    for (int i = 0; i < NumPES; i++)
    {
        for (int j = 0; j < NumPES; j++)
        {
            if (j < i)
            {
                result[i][j] = -result[j][i];
            }
            else if (j == i)
            {
                result[i][j] = 0;
            }
            else
            {
                result[i][j] = Hesse[i][j] / (Force[i][i] - Force[j][j]);
            }
            
        }
    }
    return result;
}

// save in array
extern const function<RealMatrix(const double)> potential[3] = { diabatic_potential, adiabatic_potential, force_basis_potential };
extern const function<RealMatrix(const double)> force[3] = { diabatic_force, adiabatic_force, force_basis_force };
extern const function<RealMatrix(const double)> coupling[3] = { diabatic_coupling, adiabatic_coupling, force_basis_coupling };



// transform the whole matrix
static void do_nothing(ComplexMatrixMatrix& rho, const int NGrids, const double* GridPosition)
{
}
static void diabatic_to_adiabatic(ComplexMatrixMatrix& rho_dia, const int NGrids, const double* GridPosition)
{
    Complex MatTrans[NumPES * NumPES];
    for (int i = 0; i < NGrids; i++)
    {
        const ComplexMatrix TransformationMatrix = diabatic_to_adiabatic_at_one_point(GridPosition[i]);
        for (int j = 0; j < NGrids; j++)
        {
            cblas_zhemm(CblasRowMajor, CblasLeft, CblasUpper, NumPES, NumPES, &Alpha, rho_dia[i][j].data(), NumPES, TransformationMatrix.data(), NumPES, &Beta, MatTrans, NumPES);
            cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, NumPES, NumPES, NumPES, &Alpha, TransformationMatrix.data(), NumPES, MatTrans, NumPES, &Beta, rho_dia[i][j].data(), NumPES);
            rho_dia[i][j].hermitize();
        }
    }
}
static void diabatic_to_force_basis(ComplexMatrixMatrix& rho_dia, const int NGrids, const double* GridPosition)
{
    Complex MatTrans[NumPES * NumPES];
    for (int i = 0; i < NGrids; i++)
    {
        const ComplexMatrix TransformationMatrix = diabatic_to_force_basis_at_one_point(GridPosition[i]);
        for (int j = 0; j < NGrids; j++)
        {
            cblas_zhemm(CblasRowMajor, CblasLeft, CblasUpper, NumPES, NumPES, &Alpha, rho_dia[i][j].data(), NumPES, TransformationMatrix.data(), NumPES, &Beta, MatTrans, NumPES);
            cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, NumPES, NumPES, NumPES, &Alpha, TransformationMatrix.data(), NumPES, MatTrans, NumPES, &Beta, rho_dia[i][j].data(), NumPES);
            rho_dia[i][j].hermitize();
        }
    }
}
static void adiabatic_to_diabatic(ComplexMatrixMatrix& rho_adia, const int NGrids, const double* GridPosition)
{
    Complex TransMat[NumPES * NumPES];
    for (int i = 0; i < NGrids; i++)
    {
        const ComplexMatrix TransformationMatrix = diabatic_to_adiabatic_at_one_point(GridPosition[i]);
        for (int j = 0; j < NGrids; j++)
        {
            cblas_zhemm(CblasRowMajor, CblasRight, CblasUpper, NumPES, NumPES, &Alpha, rho_adia[i][j].data(), NumPES, TransformationMatrix.data(), NumPES, &Beta, TransMat, NumPES);
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, NumPES, NumPES, NumPES, &Alpha, TransMat, NumPES, TransformationMatrix.data(), NumPES, &Beta, rho_adia[i][j].data(), NumPES);
            rho_adia[i][j].hermitize();
        }
    }
}
static void adiabatic_to_force_basis(ComplexMatrixMatrix& rho_adia, const int NGrids, const double* GridPosition)
{
    adiabatic_to_diabatic(rho_adia, NGrids, GridPosition);
    diabatic_to_force_basis(rho_adia, NGrids, GridPosition);
}
static void force_basis_to_diabatic(ComplexMatrixMatrix& rho_force, const int NGrids, const double* GridPosition)
{
    Complex TransMat[NumPES * NumPES];
    for (int i = 0; i < NGrids; i++)
    {
        const ComplexMatrix TransformationMatrix = diabatic_to_force_basis_at_one_point(GridPosition[i]);
        for (int j = 0; j < NGrids; j++)
        {
            cblas_zhemm(CblasRowMajor, CblasRight, CblasUpper, NumPES, NumPES, &Alpha, rho_force[i][j].data(), NumPES, TransformationMatrix.data(), NumPES, &Beta, TransMat, NumPES);
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, NumPES, NumPES, NumPES, &Alpha, TransMat, NumPES, TransformationMatrix.data(), NumPES, &Beta, rho_force[i][j].data(), NumPES);
            rho_force[i][j].hermitize();
        }
    }
}
static void force_basis_to_adiabatic(ComplexMatrixMatrix& rho_force, const int NGrids, const double* GridPosition)
{
    force_basis_to_diabatic(rho_force, NGrids, GridPosition);
    diabatic_to_adiabatic(rho_force, NGrids, GridPosition);
}
extern const function<void(ComplexMatrixMatrix&, int, const double*)> basis_transform[3][3] = 
{
    { do_nothing, diabatic_to_adiabatic, diabatic_to_force_basis },
    { adiabatic_to_diabatic, do_nothing, adiabatic_to_force_basis },
    { force_basis_to_diabatic, force_basis_to_adiabatic, do_nothing }
};