//Authors: Dario Cattaruzza, Alessandro Abate, Peter Schrammel, Daniel Kroening
//University of Oxford 2016
//This code is supplied under the BSD license agreement (see license.txt)

#include <streambuf>
#include <iostream>
#include <fstream>
#include <math.h>
#include "PseudoInverse.h"

namespace abstract{

  using std::max;

  template <class scalar>
  Eigen::JacobiSVD<typename JordanMatrix<scalar>::MatrixS> PseudoInverse<scalar>::ms_svd;

  template <class scalar>
  bool PseudoInverse<scalar>::ms_trace_dynamics=false;

  /// Constructs an empty buffer
  template <class scalar>
  PseudoInverse<scalar>::PseudoInverse(int dimension) :
    JordanMatrix<scalar>(dimension)
  {}

  /// Calculates the pseudoinverse of a matrix
  template <class scalar>
  typename JordanMatrix<scalar>::MatrixS PseudoInverse<scalar>::pseudoInverseSvd(const MatrixS &matrix,bool &hasInverse)
  {
    ms_svd.compute(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
    MatrixS diag=ms_svd.singularValues().asDiagonal();
    if (ms_trace_dynamics) {
      ms_logger.logData(matrix,"Inverse:");
      MatrixS u=ms_svd.matrixU();
      ms_logger.logData(u);
      ms_logger.logData(diag);
      MatrixS v=ms_svd.matrixV();
      ms_logger.logData(v);
    }
    for (int i=0;i<matrix.rows();i++) {
      if (func::isZero(diag.coeff(i,i))) hasInverse=false;
      else diag.coeffRef(i,i)=scalar(1)/diag.coeff(i,i);
    }
    if (hasInverse) return matrix.inverse();
    return ms_svd.matrixV()*diag*ms_svd.matrixU().adjoint();
  }

  /// Calculates the pseudoinverse of a matrix
  template <class scalar>
  typename JordanMatrix<scalar>::MatrixS PseudoInverse<scalar>::pseudoInverseJordan(const MatrixS &matrix,bool &hasInverse)
  {
    return JordanMatrix<scalar>::getPseudoInverse(matrix,hasInverse);
  }

#ifdef USE_LDOUBLE
  #ifdef USE_SINGLES
    template class PseudoInverse<long double>;
  #endif
  #ifdef USE_INTERVALS
    template class PseudoInverse<ldinterval>;
  #endif
#endif
#ifdef USE_MPREAL
  #ifdef USE_SINGLES
    template class PseudoInverse<mpfr::mpreal>;
  #endif
  #ifdef USE_INTERVALS
    template class PseudoInverse<mpinterval>;
  #endif
#endif

}
