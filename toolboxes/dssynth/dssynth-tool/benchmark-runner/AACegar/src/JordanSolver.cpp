//Authors: Dario Cattaruzza, Alessandro Abate, Peter Schrammel, Daniel Kroening
//University of Oxford 2016
//This code is supplied under the BSD license agreement (see license.txt)

#include <math.h>
#include "JordanSolver.h"
#include "MatrixToString.h"
#include <Eigen/Eigenvalues>

namespace abstract{

template <class scalar>
typename JordanSolver<scalar>::complexType JordanSolver<scalar>::ms_complexOne(1,0);

template <class scalar>
MatToStr<scalar> JordanSolver<scalar>::ms_logger(true);

template <class scalar>
MatToStr<scalar> JordanSolver<scalar>::ms_decoder(false);

template <class scalar>
traceDynamics_t JordanSolver<scalar>::ms_trace_dynamics=eTraceNoDynamics;

/// Constructs an empty matrix
template <class scalar>
JordanSolver<scalar>::JordanSolver(const int dimension) :
    m_dimension(dimension),
    m_zero(func::ms_weakZero),
    m_largeZero(dimension*dimension*func::ms_weakZero),
    m_dynamics(dimension,dimension)
{
}

/// Changes the default dimension of the system
template <class scalar>
void JordanSolver<scalar>::changeDimensions(const int dimension)
{
  if (dimension!=m_dimension) {
    m_dimension=dimension;
    m_dynamics.resize(dimension,dimension);
  }
}

template <class scalar>
void JordanSolver<scalar>::computeJordan(const MatrixType &matrix)
{
  this->setMaxIterations(1000);
  changeDimensions(matrix.rows());
  m_dynamics=matrix;
  calculateJordanForm();
}

/// Transforms the matrix to Row Echelon Form
template <class scalar>
int JordanSolver<scalar>::toREF(ComplexMatrixType &matrix)
{
  int rank=m_dimension;
  int col=0;
  for (int row=0;col<matrix.rows();row++,col++) {
    while (func::isZero(func::norm2(matrix.coeff(row,col)),m_zero)) {
      for (int row2=row+1;row2<matrix.rows();row2++) {
        if (!func::isZero(func::norm2(matrix.coeff(row2,col)),m_zero)) {
          matrix.row(row)+=matrix.row(row2);
          break;
        }
      }
      if (func::isZero(func::norm2(matrix.coeff(row,col)),m_zero)) {
        col++;
        if (col==matrix.rows()) break;
      }
    }
    if (col==matrix.rows()) break;
    complexType multiplier=matrix.coeff(row,col);
    if (!func::isZero(func::norm2(multiplier),m_zero)) {
      matrix.row(row)/=multiplier;
      rank--;
    }
    for (int row2=row+1;row2<matrix.rows();row2++) {
      complexType multiplier=matrix.coeff(row2,col);
      matrix.row(row2)-=multiplier*matrix.row(row);
    }
  }
  if (ms_trace_dynamics>=eTraceREF) ms_logger.logData(matrix,"REF:");
  return rank;
}

/// Transforms the matrix to Row Echelon Form
template <class scalar>
int JordanSolver<scalar>::toRREF(ComplexMatrixType &matrix)
{
  int rank=toREF(matrix);
  int col=0;
  for (int row=1;(row<matrix.rows()) && (col<matrix.cols());row++) {
    while ((col<matrix.cols()) && func::isZero(func::norm2(matrix.coeff(row,col)),m_zero)) col++;
    if (col<matrix.cols()) {
      for (int row2=0;row2<row;row2++) {
        if (!func::isZero(func::norm2(matrix.coeff(row2,col)),m_zero)) {
          complexType multiplier=matrix.coeff(row2,col);
          matrix.row(row2)-=multiplier*matrix.row(row);
        }
      }
    }
  }
  if (ms_trace_dynamics>=eTraceREF) ms_logger.logData(matrix,"RREF:");
  return rank;
}

/// Checks if the given row pair belongs to the same jordan block
template <class scalar>
bool JordanSolver<scalar>::isJordanBlock(const int row1,const int row2)
{
  if (func::norm2(m_eigenValues.coeff(row1,row1)-m_eigenValues.coeff(row2,row2))>m_largeZero) return false;
  if (m_conjugatePair[row1]>=0) {
    MatrixType dotReal=m_eigenVectors.row(row1).real().transpose()*m_eigenVectors.row(row2).real();
    MatrixType dotImag=m_eigenVectors.row(row1).imag().transpose()*m_eigenVectors.row(row2).imag();
    scalar vNorm1real=m_eigenVectors.row(row1).real().norm();
    scalar vNorm2real=m_eigenVectors.row(row2).real().norm();
    scalar vNorm1imag=m_eigenVectors.row(row1).imag().norm();
    scalar vNorm2imag=m_eigenVectors.row(row2).imag().norm();
    scalar realAngle=func::norm2(dotReal.coeff(0,0))/(vNorm1real*vNorm2real);
    scalar imagAngle=func::norm2(dotImag.coeff(0,0))/(vNorm1imag*vNorm2imag);
    realAngle=acos(realAngle);
    imagAngle=acos(imagAngle);
    return func::isZero(func::toUpper(realAngle),0.01/*m_zero*/) && func::isZero(func::toUpper(imagAngle),0.01/*m_zero*/);
  }
  else {
    ComplexMatrixType dotProd=m_eigenVectors.row(row1).transpose()*m_eigenVectors.row(row2);
    scalar vNorm1=m_eigenVectors.row(row1).norm();
    scalar vNorm2=m_eigenVectors.row(row2).norm();
    scalar angle=func::norm2(dotProd.coeff(0,0))/(vNorm1*vNorm2);
    angle=acos(angle);
    return func::isZero(func::toUpper(angle),0.01/*m_zero*/);
  }
  return false;
}

/// Calculates the Jordan block and generalised eigenvector for the row pair
template <class scalar> bool JordanSolver<scalar>::makeJordanBlock(const int row1,const int row2)
{
  scalar radius=func::norm2(m_eigenValues.coeff(row1,row1)-m_eigenValues.coeff(row2,row2));
  if (!func::isZero(radius)) return false;
  m_hasMultiplicities=true;
  ComplexMatrixType matrixBase=ComplexMatrixType::Zero(m_dimension,m_dimension);
  matrixBase.real()=m_dynamics;
  for (int i=0;i<m_dimension;i++) {
    matrixBase.coeffRef(i,i)-=m_eigenValues.coeff(row1,row1);
  }
  ComplexMatrixType matrix=matrixBase;
  int order=1;
  int rank=toREF(matrix);
  if (rank==0) return false;
  if (rank==m_dimension) return false;
  m_jordanIndex[row1]=m_jordanIndex[row2]+1;
  while (rank<=m_jordanIndex[row1]) {
    order++;
    matrix=matrixBase;
    for (int i=1;i<order;i++) matrix*=matrixBase;
    rank=toREF(matrix);
  }

  int row=m_dimension-rank-1;
  if (row<0) row=0;//TODO: What happens when rank is m_dim? Is this right?
  int col=row;
  while ((col<m_dimension) && (func::isZero(func::norm2(matrix.coeff(row,col)),m_zero))) col++;
  if (++col>=m_dimension) col=m_dimension-1;//TODO: check for col out of range and 0 coeffs (is this right?)
  ComplexMatrixType vector=ComplexMatrixType::Zero(m_dimension,1);
  vector.coeffRef(col,0)=1;

  while (row>=0) {
    vector.coeffRef(row,0)=-(matrix.row(row)*vector).sum();
    row--;
  }
  for (int i=1;i<rank-m_jordanIndex[row1];i++) vector=matrixBase*vector;
  refScalar vectorNorm=func::toUpper(vector.norm());
  refScalar vectorEpsilon=vectorNorm*func::ms_weakEpsilon*func::ms_weakEpsilon;
  for (int i=0;i<vector.rows();i++) {
    if (func::norm2(vector.coeff(i,0))<vectorEpsilon) vector.coeffRef(i,0)=0;
  }
  vector/=scalar(vectorNorm);
  m_eigenVectors.col(row1)=vector;
  m_eigenValues.coeffRef(row2,row1)=ms_complexOne;

  if (m_conjugatePair[row1]>=0) {
    m_eigenValues.coeffRef(row2+1,row1+1)=ms_complexOne;
    for (int row=0;row<m_dimension;row++) m_eigenVectors.coeffRef(row,row1+1)=conj(vector.coeff(row,0));
  }
  for (int i=1;i<=m_jordanIndex[row1];i++) {
    vector=matrixBase*vector;
    if (m_conjugatePair[row1]>=0) {
      m_eigenVectors.col(row1-2*i)=vector;
      for (int row=0;row<m_dimension;row++) m_eigenVectors.coeffRef(row,row1-2*i+1)=conj(vector.coeff(row,0));
    }
    else {
      m_eigenVectors.col(row1-i)=vector;
    }
  }

  if (ms_trace_dynamics>=eTraceREF) {
    ms_logger.logData(m_eigenVectors,"Intermediate EigenVectors:");
    matrix=matrixBase;
    for (int i=1;i<=m_jordanIndex[row1];i++) matrix*=matrixBase;
    ms_logger.logData(matrix,"Matrix Base:");
    ComplexMatrixType nullSpace=getNullSpace(matrix);
    ms_logger.logData(nullSpace,"nullSpace:");
  }
  return true;
}

/// calculates the estimated roundoff error of a matrix operation
template <class scalar>
inline typename JordanSolver<scalar>::refScalar JordanSolver<scalar>::calculateEpsilon(const MatrixType &matrix)
{
  if (matrix.rows()>0) {
    refScalar max=func::toUpper(func::norm2(matrix.coeff(0,0)));
    refScalar min=func::toLower(func::norm2(matrix.coeff(0,0)));
    for (int row=0;row<matrix.rows();row++) {
      for (int col=0;col<matrix.cols();col++) {
        refScalar upper=func::toUpper(func::norm2(matrix.coeff(row,col)));
        refScalar lower=func::toLower(func::norm2(matrix.coeff(row,col)));
        if (upper>max) max=upper;
        if (lower<min) min=lower;
      }
    }
    //scalar max=matrix.maxCoeff();
    //scalar min=matrix.minCoeff();
    if (-min>max) max=-min;
    return max*func::ms_weakEpsilon;
  }
  return 0;
}

/// calculates the estimated roundoff error of a matrix operation
template <class scalar>
inline typename JordanSolver<scalar>::refScalar JordanSolver<scalar>::calculateEpsilon(const ComplexMatrixType &matrix)
{
  if (matrix.rows()>0) {
    refScalar max=func::toUpper(func::norm2(matrix.coeff(0,0)));
    refScalar min=func::toLower(func::norm2(matrix.coeff(0,0)));
    for (int row=0;row<matrix.rows();row++) {
      for (int col=0;col<matrix.cols();col++) {
        refScalar upper=func::toUpper(func::norm2(matrix.coeff(row,col)));
        refScalar lower=func::toLower(func::norm2(matrix.coeff(row,col)));
        if (upper>max) max=upper;
        if (lower<min) min=lower;
      }
    }
    //scalar max=matrix.maxCoeff();
    //scalar min=matrix.minCoeff();
    if (-min>max) max=-min;
    return max*func::ms_weakEpsilon;
  }
  return 0;
}


/// Loads the transformation matrix for the state space
template <class scalar>
bool JordanSolver<scalar>::calculateJordanForm()
{
    m_zero=calculateEpsilon(m_dynamics);
    m_inverse.conservativeResize(0,0);
    m_largeZero=m_zero*m_dimension*m_dimension;
    this->setMaxIterations(1000);
    this->compute(m_dynamics);

    if (this->info()!=Eigen::Success) return false;
    m_eigenValues=this->eigenvalues().asDiagonal();
    m_eigenVectors=this->eigenvectors();

    if (ms_trace_dynamics>=eTraceAll) {
      ms_logger.logData(m_dynamics,"Dynamics:");
      ms_logger.logData(m_eigenValues,"EigenValues:");
      ms_logger.logData(m_eigenVectors,"Initial EigenVectors:");
    }
    m_hasOnes=false;
    m_hasZeros=false;
    m_hasMultiplicities=false;
    m_isOne.resize(2*m_dimension);
    m_conjugatePair.resize(2*m_dimension);
    m_jordanIndex.resize(2*m_dimension);
    for (int i=0;i<m_dimension;i++) {
      if (func::isZero(func::norm2(m_eigenValues.coeff(i,i)))) m_hasZeros=true;
    }
    for (int i=0;i<m_dimension;i++) {
      m_conjugatePair[i]=-1;
      m_jordanIndex[i]=0;
      m_isOne[i]=false;
      if ((i<(m_dimension-1)) && !func::isZero(m_eigenValues.coeff(i,i).imag(),m_zero)) {
        m_conjugatePair[i]=i+1;
        if (i>=2) makeJordanBlock(i,i-2);
        m_jordanIndex[i+1]=m_jordanIndex[i];
        m_conjugatePair[i+1]=i;
        i++;
      }
      else {
        m_isOne[i]=func::isZero(func::norm2(m_eigenValues.coeff(i,i)-ms_complexOne),m_zero);
        m_hasOnes|=m_isOne[i];
        if (i>0) makeJordanBlock(i,i-1);
      }
    }
    refScalar eigenVectorEpsilon=calculateEpsilon(m_eigenVectors);
    for (int row=0;row<m_eigenVectors.rows();row++) {
      for (int col=0;col<m_eigenVectors.cols();col++) {
        if (func::norm2(m_eigenVectors.coeff(row,col))<eigenVectorEpsilon) m_eigenVectors.coeffRef(row,col)=func::ms_hardZero;
      }
    }

    for (int i=m_dimension;i<2*m_dimension;i++) {
      m_conjugatePair[i]=-1;
      m_jordanIndex[i]=0;
      m_isOne[i]=m_isOne[i-m_dimension];
    }
    if (ms_trace_dynamics>=eTraceDynamics) ms_logger.logData(m_eigenVectors,"Generalised EigenVectors:");
    return true;
}

/// Returns the nullSpace vectors of M
template<class scalar>
typename JordanSolver<scalar>::ComplexMatrixType JordanSolver<scalar>::getNullSpace(const ComplexMatrixType &matrixBase,bool normalized)
{
  ComplexMatrixType result;
  ComplexMatrixType nullSpace=matrixBase;
  toRREF(nullSpace);
  std::vector<bool> vars(nullSpace.cols());
  int row=0;
  int freeVars=nullSpace.cols();
  for (int col=0;col<nullSpace.cols();col++) {
    vars[col]=true;
    if (!func::isZero(norm(nullSpace.coeff(row,col)))) {
      vars[col]=false;
      row++;
      freeVars--;
    }
  }
  result.resize(m_dimension,freeVars);
  int col=0;
  for (int j=0;j<nullSpace.cols();j++) {
    if (vars[j]) {
      result.row(j)=ComplexMatrixType::Zero(1,freeVars);
      result.coeffRef(j,col)=func::ms_c_1;
    }
    else {
      int pos=0;
      for (int k=0;k<nullSpace.cols();k++) {
        if (vars[k]) result.coeffRef(j,pos++)=-nullSpace.coeff(j,k);
      }
    }
  }
  if (normalized) {
    for (int col=0;col<result.cols();col++) {
      scalar scale=result.col(col).norm();
      result.col(col)/=scale;
    }
  }
  return result;
}

#ifdef USE_LDOUBLE
  template class JordanSolver<long double>;
#endif
#ifdef USE_MPREAL
  template class JordanSolver<mpfr::mpreal>;
#endif
}
