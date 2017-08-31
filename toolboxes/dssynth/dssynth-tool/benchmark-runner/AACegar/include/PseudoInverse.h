//Authors: Dario Cattaruzza, Alessandro Abate, Peter Schrammel, Daniel Kroening
//University of Oxford 2016
//This code is supplied under the BSD license agreement (see license.txt)

#ifndef PSEUDOINVERSE_H
#define PSEUDOINVERSE_H

#include <limits>
#include <stdlib.h>

#include "JordanMatrix.h"

namespace abstract
{

template <class scalar> class PseudoInverse : public JordanMatrix<scalar>
{
public:
    using typename JordanMatrix<scalar>::refScalar;
    using typename JordanMatrix<scalar>::func;
    using typename JordanMatrix<scalar>::MatrixS;

    using JordanMatrix<scalar>::ms_logger;

    /// Constructs an empty buffer
    /// @param dimension dimension of the space
    PseudoInverse(int dimension=0);

    /// Calculates the pseudoinverse of a matrix
    MatrixS pseudoInverseSvd(const MatrixS &matrix,bool &hasInverse);

    /// Calculates the pseudoinverse of a matrix
    MatrixS pseudoInverseEigen(const MatrixS &matrix,bool &hasInverse);

    /// Calculates the pseudoinverse of a matrix
    MatrixS pseudoInverseJordan(const MatrixS &matrix,bool &hasInverse);

protected:
    static Eigen::JacobiSVD<MatrixS>        ms_svd;
public:
    static bool                             ms_trace_dynamics;
};

}

#endif
