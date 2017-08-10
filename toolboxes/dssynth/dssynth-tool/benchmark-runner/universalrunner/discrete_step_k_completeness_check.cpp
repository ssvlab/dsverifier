#include <Eigen/Eigenvalues>
#include <eigen3/unsupported/Eigen/MPRealSupport>
#include <mpfr.h>
#include <iostream>

typedef mpfr::mpreal realt;
typedef std::complex<realt> complext;
typedef realt __plant_typet;
#define CONTROL_TYPES_H_
#define interval(x) x
#include "benchmark.h"

using namespace Eigen;
using namespace std;
using namespace mpfr;

#define EXIT_INCREASE_K 2
#define EXIT_INCREASE_SAMPLE_RATE 3
#define PRECISION 256
#define NUM_PROG_ARGS 2 + NSTATES
#define K_SIZE_ARG_INDEX 1u
#define K_ARG_OFFSET 2u

typedef Matrix<realt, NSTATES, NSTATES> matrixt;

//const realt _controller_K[] = { 0.0234375,-0.1328125, 0.00390625 };
//const realt _controller_K[] = { 0.0234375,0.1328125, 0.00390625 };
realt _controller_K[NSTATES];

bool is_imaginary(const realt &imaginary_offset, const complext &complex) {
  const realt imag_value = std::imag(complex);
  cout << "imag_value: " << imag_value << endl;
  cout << "imaginary_offset: " << imaginary_offset << endl;
  return abs(imag_value) > imaginary_offset;
}

int main(const int argc, const char * const argv[]) {
  const realt two = "2.0";
  const realt two_pi = const_pi() * two;
  const realt imaginary_offset = pow(two, -PRECISION/4);
  if (argc != NUM_PROG_ARGS) return EXIT_FAILURE;
  mpreal::set_default_prec(PRECISION);
  const realt K_SIZE = argv[K_SIZE_ARG_INDEX];
  for (size_t i=0; i < NSTATES; ++i)
    _controller_K[i]=argv[i + K_ARG_OFFSET];

  Matrix<realt, NSTATES, NSTATES> A;
  for (size_t i = 0; i < NSTATES; ++i) {
    for (size_t j = 0; j < NSTATES; ++j) {
      A(i, j) = _controller_A[i][j];
    }
  }
  Matrix<realt, NSTATES, 1> B;
  for (size_t i = 0; i < NSTATES; ++i) {
    B(i) = _controller_B[i];
  }
  Matrix<realt, 1, NSTATES> K;
  for (size_t i = 0; i < NSTATES; ++i) {
    K[i] = _controller_K[i];
  }

  // Check K_SIZE
  matrixt result = A - B * K;
  EigenSolver<matrixt> eigenSpace(result);
  if (Success != eigenSpace.info())
    return EXIT_FAILURE;
  const EigenSolver<matrixt>::EigenvalueType eigenvalues = eigenSpace.eigenvalues();
  cout << "num_eigenvalues: " << eigenvalues.size() << endl;
  for (size_t i=0; i < eigenvalues.size(); ++i) {
    cout << "eigenvalue: " << eigenvalues[i] << endl;
    if (!is_imaginary(imaginary_offset, eigenvalues[i])) continue;
    const realt angle = std::arg(eigenvalues[i]);
    const realt expected = abs(two_pi / angle);
    cout << "expected_k_size=" << expected << endl;
    cout << "actual_k_size=" << K_SIZE << endl;
    if (expected > K_SIZE)
      return EXIT_INCREASE_K;
  }

  // Check sample rate
  /*Matrix<realt, Dynamic, Dynamic> vertices(Matrix<realt, Dynamic, Dynamic>::Ones(NSTATES, ::pow(2, NSTATES)));
  int step = 1;
  for (size_t row = 0; row < vertices.rows(); ++row) {
    for (size_t col = 1; col < vertices.cols(); ++col) {
      vertices.coeffRef(row, col) = vertices.coeffRef(row, col - 1);
      if (col % step == 0) vertices.coeffRef(row, col) = -vertices.coeffRef(row, col);
    }
    step <<= 1;
  }
  const Matrix<complext, NSTATES, NSTATES> eigenvectors(eigenSpace.eigenvectors());
  cout << "eigenvectors: " << endl << eigenvectors << endl;
  matrixt pseudo_eigenvectors(eigenvectors.real());
  for (size_t i=0; i < NSTATES - 1; ++i) {
    if (!is_imaginary(imaginary_offset, eigenvalues[i])) continue;
    pseudo_eigenvectors.col(i + 1) = eigenvectors.col(i).imag();
    ++i;
  }
  cout << "pseudo_eigenvectors: " << endl << pseudo_eigenvectors << endl;
  vertices *= pseudo_eigenvectors.transpose();*/

  return EXIT_SUCCESS;
}
