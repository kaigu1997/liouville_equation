#ifndef MATRIX_H
#define MATRIX_H

// the header file to deal with rowwise 
// real(double) or complex (double) matrices
// stored in 1D array, calculated by
// Vector Mathematical Functions in MKL
// multiply with vector/matrix could be done via BLAS
// eigenvalue/eigenvector could be gotten from LAPACK
// Error code: 100 for different size

#include <complex>
#include <memory>
#include <mkl.h>
#include <utility>
using namespace std;

// equivalent to MKL_Complex16
typedef complex<double> Complex;
// to access to matrix elements
typedef pair<int, int> Index;

// the mode constant for VMF functions in MKL
const MKL_INT64 mode = VML_HA;

// declaration of all kinds of matrix
class RealMatrix;
class ComplexMatrix;
class ComplexMatrixMatrix;

// transfrom a double array to a complex array
void real_to_complex(const double* da, Complex* ca, const int length);


// the functions of real matrix
class RealMatrix
{
private:
    int length;
    int nelements;
    double** content;
public:
    // default constructor with all zero
    RealMatrix(const int size);
    // copy constructor
    RealMatrix(const RealMatrix& matrix);
    // quasi copy constructor
    RealMatrix(const int size, const double* array);
    // one element is give number and the other are all zero
    RealMatrix(const int size, const Index& idx, const double& val);
    // destructor
    ~RealMatrix(void);
    // the size of the matrix
    int length_of_matrix(void) const;
    // direct access to internal data
    double* data(void);
    const double* data(void) const;
    // copy to an array
    void transform_to_1d(double* array) const;
    // make it symmetry
    void symmetrize(void);
    // overload operator[]
    double* operator[](const int idx);
    const double* operator[](const int idx) const;
    // overload numerical calculation by VMF
    friend RealMatrix operator+(const RealMatrix& lhs, const double& rhs);
    friend RealMatrix operator+(const double& lhs, const RealMatrix& rhs);
    friend RealMatrix operator+(const RealMatrix& lhs, const RealMatrix& rhs);
    RealMatrix& operator+=(const double& rhs);
    RealMatrix& operator+=(const RealMatrix& rhs);
    friend RealMatrix operator-(const RealMatrix& lhs, const double& rhs);
    friend RealMatrix operator-(const double& lhs, const RealMatrix& rhs);
    friend RealMatrix operator-(const RealMatrix& lhs, const RealMatrix& rhs);
    RealMatrix& operator-=(const double& rhs);
    RealMatrix& operator-=(const RealMatrix& rhs);
    friend RealMatrix operator*(const RealMatrix& lhs, const double& rhs);
    friend RealMatrix operator*(const double& lhs, const RealMatrix& rhs);
    RealMatrix& operator*=(const double& rhs);
    friend RealMatrix operator/(const RealMatrix& lhs, const double& rhs);
    RealMatrix& operator/=(const double& rhs);
    // assignment operator
    RealMatrix& operator=(const RealMatrix& rhs);
    RealMatrix& operator=(const double* array);
    // all kinds of matrix could access to each other
    friend class ComplexMatrix;
    friend class ComplexMatrixMatrix;
};

// the functions of complex matrix, similar to above
class ComplexMatrix
{
private:
    int length;
    int nelements;
    Complex** content;
public:
    // default constructor with all zero
    ComplexMatrix(const int size);
    // copy constructor
    ComplexMatrix(const ComplexMatrix& matrix);
    // quasi copy constructor
    ComplexMatrix(const int size, const Complex* array);
    // copy constructor from real matrix
    ComplexMatrix(const RealMatrix& matrix);
    // quasi copy constructor from real matrix
    ComplexMatrix(const int size, const double* array);
    // one element is give number and the other are all zero
    ComplexMatrix(const int size, const Index& idx, const Complex& val);
    // destructor
    ~ComplexMatrix(void);
    // the size of the matrix
    int length_of_matrix(void) const;
    // direct access to internal data
    Complex* data(void);
    const Complex* data(void) const;
    // copy to an array
    void transform_to_1d(Complex* array) const;
    // to make the matrix hermitian
    void hermitize(void);
    // overload operator[]
    Complex* operator[](const int idx);
    const Complex* operator[](const int idx) const;
    // overload numerical calculation
    friend ComplexMatrix operator+(const ComplexMatrix& lhs, const Complex& rhs);
    friend ComplexMatrix operator+(const Complex& lhs, const ComplexMatrix& rhs);
    friend ComplexMatrix operator+(const ComplexMatrix& lhs, const ComplexMatrix& rhs);
    ComplexMatrix& operator+=(const Complex& rhs);
    ComplexMatrix& operator+=(const ComplexMatrix& rhs);
    friend ComplexMatrix operator-(const ComplexMatrix& lhs, const Complex& rhs);
    friend ComplexMatrix operator-(const Complex& lhs, const ComplexMatrix& rhs);
    friend ComplexMatrix operator-(const ComplexMatrix& lhs, const ComplexMatrix& rhs);
    ComplexMatrix& operator-=(const Complex& rhs);
    ComplexMatrix& operator-=(const ComplexMatrix& rhs);
    friend ComplexMatrix operator*(const ComplexMatrix& lhs, const Complex& rhs);
    friend ComplexMatrix operator*(const Complex& lhs, const ComplexMatrix& rhs);
    ComplexMatrix& operator*=(const Complex& rhs);
    friend ComplexMatrix operator/(const ComplexMatrix& lhs, const Complex& rhs);
    ComplexMatrix& operator/=(const Complex& rhs);
    // assignment operator
    ComplexMatrix& operator=(const ComplexMatrix& rhs);
    ComplexMatrix& operator=(const Complex* array);
    // all kinds of matrix could access to each other
    friend class RealMatrix;
    friend class ComplexMatrixMatrix;
};

// class for density matrix;
// outer matrix is grid point (Ri, Pj)
// inner matrix is density matrix (a pes, b pes)
// using dynamic memory management in C++17
class ComplexMatrixMatrix
{
private:
    const int length;
    const int nelements;
    ComplexMatrix** content;
    allocator<ComplexMatrix> MatrixAllocator;
public:
    // constructor; out for this, inner for ComplexMatrix
    ComplexMatrixMatrix(const int OuterLength, const int InnerLength);
    // destructor;
    ~ComplexMatrixMatrix(void);
    // operator[]
    ComplexMatrix* operator[](const int idx);
    const ComplexMatrix* operator[](const int idx) const;
    // output
    friend ostream& operator<<(ostream& os, const ComplexMatrixMatrix& rho);
};

#endif // !MATRIX_H
