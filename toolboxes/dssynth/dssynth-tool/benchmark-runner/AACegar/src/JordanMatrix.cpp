//Authors: Dario Cattaruzza, Alessandro Abate, Peter Schrammel, Daniel Kroening
//University of Oxford 2016
//This code is supplied under the BSD license agreement (see license.txt)

#include <math.h>

#include <Eigen/Eigenvalues>

#include <boost/timer.hpp>

#include "JordanMatrix.h"
#include "MatrixToString.h"

namespace abstract{

template <class scalar>
scalar  JordanMatrix<scalar>::ms_half(0.5);

template <class scalar>
scalar  JordanMatrix<scalar>::ms_one(1);

template <class scalar>
scalar  JordanMatrix<scalar>::ms_two(2);

template <class scalar>
typename JordanMatrix<scalar>::complexS JordanMatrix<scalar>::ms_complexOne(1,0);

template <class scalar>
MatToStr<scalar>  JordanMatrix<scalar>::ms_logger(true);

template <class scalar>
MatToStr<scalar>  JordanMatrix<scalar>::ms_decoder(false);

template <class scalar>
typename JordanMatrix<scalar>::MatrixS JordanMatrix<scalar>::ms_emptyMatrix(0,0);

template <class scalar>
traceDynamics_t JordanMatrix<scalar>::ms_trace_dynamics=eTraceNoDynamics;

template <class scalar>
bool JordanMatrix<scalar>::ms_trace_time=false;

/// Constructs an empty matrix
template <class scalar>
JordanMatrix<scalar>::JordanMatrix(int dimension) :
  m_dimension(dimension),
  m_zero(func::ms_weakZero),
  m_largeZero(dimension*dimension*func::ms_weakZero),
  m_dynamics(dimension,dimension),
  m_refDynamics(dimension,dimension),
  m_eigenSpace(dimension),
  m_minSeparation(dimension,1),
  m_jordanTime(0)
{
  m_pseudoEigenVectors=MatrixS(m_dimension,m_dimension);
  m_invPseudoEigenVectors=MatrixS(m_dimension,m_dimension);
}

/// Changes the default dimension of the system
template <class scalar>
void JordanMatrix<scalar>::changeDimensions(const int dimension)
{
  if (dimension!=m_dimension) {
    m_dimension=dimension;
    m_dynamics.conservativeResize(dimension,dimension);
    m_minSeparation.conservativeResize(dimension,1);
    m_refDynamics.resize(dimension,dimension);
    m_pseudoEigenVectors.resize(dimension,dimension);
    m_invPseudoEigenVectors.resize(dimension,dimension);
  }
}

/// Loads a matrix from a given description
template <class scalar>
int JordanMatrix<scalar>::load(const std::string &data,size_t pos)
{
  int result=ms_logger.StringToMat(m_dynamics,data,pos);
  //changeDimensions(m_dynamics.rows());
  if ((result>0) && !calculateJordanForm()) return -1;
  return result;
}

template <class scalar>
bool JordanMatrix<scalar>::loadFromRef(const MatrixR &matrix)
{
  changeDimensions(matrix.rows());
  for (int row=0;row<matrix.rows();row++) {
    for (int col=0;col<matrix.cols();col++) {
      m_dynamics.coeffRef(row,col)=matrix.coeff(row,col);
    }
  }
  return calculateJordanForm();
}

template <class scalar>
bool JordanMatrix<scalar>::load(const MatrixS &dynamics)
{
  changeDimensions(dynamics.rows());
  m_dynamics=dynamics;
  return calculateJordanForm();
}

template <class scalar>
bool JordanMatrix<scalar>::loadJordan(const MatrixS &matrix)
{
  boost::timer timer;
  changeDimensions(matrix.rows());
  m_dynamics=matrix;
  m_zero=calculateEpsilon(m_dynamics);
  func::setZero(m_zero);
  m_largeZero=m_zero*m_dimension*m_dimension;
  interToRef(m_refDynamics,m_dynamics);
  m_pseudoEigenValues=m_dynamics;
  m_pseudoEigenVectors=MatrixS::Identity(m_dimension,m_dimension);
  m_invPseudoEigenVectors=m_pseudoEigenVectors;
  m_hasOnes=false;
  m_hasMultiplicities=false;
  m_isOne.resize(2*m_dimension);
  m_conjugatePair.resize(2*m_dimension);
  m_jordanIndex.resize(2*m_dimension);
  for (int i=0;i<m_dimension;i++) {
    m_conjugatePair[i]=-1;
    m_conjugatePair[i+m_dimension]=-1;
    if ((i<m_dimension-1) && !func::isZero(m_dynamics.coeff(i,i+1))) m_conjugatePair[i]=i+1;
    else if ((i>0) && !func::isZero(m_dynamics.coeff(i-1,i)))        m_conjugatePair[i]=i-1;
    int mult=(m_conjugatePair[i]<0) ? 1 : 2;
    m_jordanIndex[i]=0;
    if ((i>=mult) && !func::isZero(m_dynamics.coeff(i,i-mult))) m_jordanIndex[i-mult]+1;
    m_jordanIndex[i+m_dimension]=0;
    if (m_jordanIndex[i]>0) m_hasMultiplicities=true;
    m_isOne[i]=func::isZero(ms_one-m_dynamics.coeff(i,i)) && (m_conjugatePair[i]<0);
    m_hasOnes|=m_isOne[i];
  }
  m_eigenValues=pseudoToJordan(m_pseudoEigenValues,eToEigenValues);
  m_eigenNorms.resize(m_eigenValues.rows(),1);
  for (int i=0;i<m_eigenValues.rows();i++) m_eigenNorms.coeffRef(i,0)=func::norm2(m_eigenValues.coeff(i,i));
  m_eigenVectors=MatrixC::Identity(m_dimension,m_dimension);
  m_invEigenVectors=m_eigenVectors;
  m_error=func::ms_hardZero;
  m_jordanTime=timer.elapsed()*1000;
  if (ms_trace_time) ms_logger.logData(m_jordanTime,"Pole Extraction time:",true);
  return true;
}

/// calculates the estimated roundoff error of a matrix operation
template <class scalar>
template <class MatrixType> inline typename JordanMatrix<scalar>::refScalar JordanMatrix<scalar>::calculateEpsilon(const MatrixType &matrix)
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
bool JordanMatrix<scalar>::calculateJordanForm(bool includeSvd)
{
  boost::timer timer;
  m_zero=calculateEpsilon(m_dynamics);
  func::setZero(m_zero);
  m_largeZero=m_zero*m_dimension*m_dimension;
  interToRef(m_refDynamics,m_dynamics);

  m_eigenSpace.computeJordan(m_refDynamics);
  if (ms_trace_time) ms_logger.logData(timer.elapsed()*1000,"Jordan Form:",true);
  if (m_eigenSpace.info()!=Eigen::Success) {
    if (ms_trace_dynamics>=eTraceDynamics) ms_logger.logData("Failed to find Jordan Form");
    return false;
  }
  refToInter(m_eigenValues,m_eigenSpace.getEigenValues());
  refToInter(m_eigenVectors,m_eigenSpace.getEigenVectors());

  m_jordanIndex=m_eigenSpace.getJordanIndeces();
  m_conjugatePair=m_eigenSpace.getConjugatePairs();
  m_isOne=m_eigenSpace.getOnes();
  m_hasOnes=m_eigenSpace.hasOnes();
  m_hasMultiplicities=m_eigenSpace.hasMultiplicities();
  m_eigenNorms.resize(m_eigenValues.rows(),1);
  for (int i=0;i<m_eigenValues.rows();i++) m_eigenNorms.coeffRef(i,0)=func::norm2(m_eigenValues.coeff(i,i));
  try {
    m_invEigenVectors=m_eigenVectors.inverse();
  }
  catch(...) {
    refToInter(m_invEigenVectors,m_eigenSpace.getEigenVectors().inverse());
  }
  if (ms_trace_dynamics>=eTraceDynamics) {
    ms_logger.logData(m_dynamics,"Dynamics:");
    ms_logger.logData(m_eigenValues,"EigenValues:");
    ms_logger.logData(m_eigenVectors,"EigenVectors:");
    ms_logger.logData(m_invEigenVectors,"InvEigenVectors:");
  }

  m_pseudoEigenValues=jordanToPseudoJordan(m_eigenValues,eToEigenValues);
  m_pseudoEigenVectors=jordanToPseudoJordan(m_eigenVectors,eToEigenVectors);
  m_invPseudoEigenVectors=m_pseudoEigenVectors.inverse();
  if (ms_trace_dynamics>=eTraceDynamics) {
    MatrixS pseudoCalculated=m_pseudoEigenVectors*m_pseudoEigenValues*m_invPseudoEigenVectors;
    ms_logger.logData(m_pseudoEigenValues,"PseudoEigenValues");
    ms_logger.logData(m_pseudoEigenVectors,"PseudoEigenVectors");
    ms_logger.logData(m_invPseudoEigenVectors,"InvPseudoEigenVectors");
    ms_logger.logData(pseudoCalculated,"PseudoCalc");
  }
  if (includeSvd) {
    calculateBlockSVD();
    m_minSigma=func::toLower(m_blockSingularValues.coeff(0,0));
    m_maxSigma=func::toUpper(m_blockSingularValues.coeff(0,0));
    for (int row=1;row<m_blockSingularValues.rows();row++) {
      if (func::toUpper(m_blockSingularValues.coeff(row,0))>m_maxSigma) {
        m_maxSigma=func::toUpper(m_blockSingularValues.coeff(row,0));
      }
      if (func::toLower(m_blockSingularValues.coeff(row,0))<m_minSigma) {
        m_minSigma=func::toLower(m_blockSingularValues.coeff(row,0));
      }
    }
    m_jordanTime=timer.elapsed()*1000;
    if (ms_trace_time) ms_logger.logData(m_jordanTime,"SVD time:",true);
  }
  calculateEigenError();
  m_jordanTime=timer.elapsed()*1000;
  if (ms_trace_time) ms_logger.logData(m_jordanTime,"Jordan Error time:",true);
  for (int row=0;row<m_invPseudoEigenVectors.rows();row++) {
    for (int col=0;col<m_invPseudoEigenVectors.cols();col++) {
      if (func::isNan(m_invPseudoEigenVectors.coeff(row,col))) {
        return false;
      }
    }
  }
  return true;
}

/// Retrieves an equivalent real Jordan from a complex representation
template <class scalar>
typename JordanMatrix<scalar>::MatrixS JordanMatrix<scalar>::jordanToPseudoJordan(const MatrixC &source,const pseudoType_t conversionType)
{
  MatrixS result=source.real();
  if (conversionType==eToEigenValues) {
    for (int col=0;col<source.rows();col++) {
      if (m_conjugatePair[col]>col) {
        result.coeffRef(col+1,col)=-source.coeff(col,col).imag();
        result.coeffRef(col,col+1)=source.coeff(col,col).imag();
        for (int offset=1;offset<=m_jordanIndex[col];offset++) {
          int row=col-2*offset;
          result.coeffRef(row+1,col)=-source.coeff(row,col).imag();
          result.coeffRef(row,col+1)=source.coeff(row,col).imag();
        }
        col++;
      }
    }
  }
  else if (conversionType==eToEigenVectors) {
    for (int col=0;col<source.cols();col++) {
      if (m_conjugatePair[col]>col) {
        result.col(col+1)=source.col(col).imag();
        col++;
      }
    }
  }
  else {
      for (int row=0;row<source.rows();row++) {
        if (m_conjugatePair[row]>row) {
          result.row(row+1)=source.row(row).imag();
          row++;
        }
      }
  }
  return result;
}

/// Retrieves an equivalent complex Jordan from a real representation
template <class scalar>
typename JordanMatrix<scalar>::MatrixC JordanMatrix<scalar>::pseudoToJordan(const MatrixS &source,const pseudoType_t conversionType)
{
  MatrixC result=MatrixC::Zero(source.rows(),source.cols());
  if (conversionType==eToEigenValues) {
    int mult=1;
    for (int row=0;row<source.rows();row+=mult) {
      mult=(m_conjugatePair[row]<0) ? 1 : 2;
      if (m_jordanIndex[row+mult]>0) {
        if (m_conjugatePair[row]<0) {
          result.coeffRef(row,row)=complexS(source.coeff(row,row),func::ms_hardZero);
        }
        else {
          result.coeffRef(row,row)=complexS(source.coeff(row,row),source.coeff(row,row+1));
          result.coeffRef(row+1,row+mult+1)=ms_complexOne;
        }
        result.coeffRef(row,row+mult)=ms_complexOne;
      }
      else {
        result.coeffRef(row,row)=complexS(source.coeff(row,row),source.coeff(row,row+1));
      }
    }
  }
  else if (conversionType==eToEigenVectors) {
    for (int col=0;col<source.cols();col++) {
      result.col(col).real()=source.col(col);
      if (m_conjugatePair[col]>col) {
        result.col(col).imag()=source.col(col+1);
        col++;
        result.col(col).real()=source.col(col-1);
        result.col(col).imag()=-source.col(col);
      }
    }
  }
  else {
    for (int row=0;row<source.rows();row++) {
      result.row(row).real()=source.row(row);
      if (m_conjugatePair[row]>row) {
        result.row(row).imag()=source.row(row+1);
        row++;
        result.row(row).real()=source.row(row-1);
        result.row(row).imag()=-source.row(row);
      }
    }
  }
  return result;
}

/// Retrieves a scalar matrix from a refScalar one
template <class scalar>
void JordanMatrix<scalar>::interToRef(SolverMatrixType &dest,const MatrixS &source)
{
  dest.conservativeResize(source.rows(),source.cols());
  for (int row=0;row<source.rows();row++) {
    for (int col=0;col<source.cols();col++) {
      refScalar coef=func::toCentre(source.coeff(row,col));
      dest.coeffRef(row,col)=coef;
    }
  }
}

/// Retrieves a scalar matrix from a refScalar one
template <class scalar>
void JordanMatrix<scalar>::refToInter(MatrixC &dest,const SolverComplexMatrixType &source)
{
  dest.conservativeResize(source.rows(),source.cols());
  for (int row=0;row<source.rows();row++) {
    for (int col=0;col<source.cols();col++) {
      dest.coeffRef(row,col)=func::toScalar(source.coeff(row,col));
    }
  }
}

/// Retrieves a scalar matrix from a refScalar one
template <class scalar>
void JordanMatrix<scalar>::refToInter(MatrixS &dest,const SolverMatrixType &source)
{
  dest.conservativeResize(source.rows(),source.cols());
  for (int row=0;row<source.rows();row++) {
    for (int col=0;col<source.cols();col++) {
      dest.coeffRef(row,col)=source.coeff(row,col);
    }
  }
}

/// Transforms the matrix to Reduced Row Echelon Form
template <class scalar>
int JordanMatrix<scalar>::toRREF(MatrixC &matrix)
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
    complexR multiplier=func::toCentre(matrix.coeff(row,col));
    if (!func::isZero(func::norm2(multiplier),m_zero)) {
      for (int col2=0;col2<matrix.cols();col2++) matrix.coeffRef(row,col2)/=multiplier;
      rank--;
    }
    for (int row2=row+1;row2<matrix.rows();row2++) {
      complexS multiplier=matrix.coeff(row2,col);
      matrix.row(row2)-=multiplier*matrix.row(row);
    }
  }
  col=0;
  for (int row=1;(row<matrix.rows()) && (col<matrix.cols());row++) {
    while ((col<matrix.cols()) && func::isZero(func::norm2(matrix.coeff(row,col)),m_zero)) col++;
    if (col<matrix.cols()) {
      for (int row2=0;row2<row;row2++) {
        if (!func::isZero(func::norm2(matrix.coeff(row2,col)),m_zero)) {
          complexS multiplier=matrix.coeff(row2,col);
          matrix.row(row2)-=multiplier*matrix.row(row);
        }
      }
    }
  }
  return rank;
}

/// Returns the description of a complex matrix
template <class scalar>
std::string JordanMatrix<scalar>::getMatrix(const MatrixC &matrix,bool brackets)
{
  if (brackets) return ms_logger.MatToString(matrix);
  return ms_decoder.MatToString(matrix);
}

/// Returns the description of a matrix
template <class scalar>
std::string JordanMatrix<scalar>::getMatrix(const MatrixS &matrix,bool brackets)
{
  if (brackets) return ms_logger.MatToString(matrix);
  return ms_decoder.MatToString(matrix);
}

/// Returns the complex eigenvector matrix (S)
template <class scalar>
std::string JordanMatrix<scalar>::getEigenVectorsDesc(bool pseudo)
{
  if (pseudo) return getMatrix(m_pseudoEigenVectors,false);
  return getMatrix(m_eigenVectors,false);
}

/// Returns the singular values of the matrix
template <class scalar>
std::string JordanMatrix<scalar>::getSingularValuesDesc()
{
  return getMatrix(m_blockSingularValues,false);
}

/// Returns the inverse complex eigenvector matrix (S^-1)
template <class scalar>
std::string JordanMatrix<scalar>::getInvEigenVectorsDesc(bool pseudo)
{
  if (pseudo) return getMatrix(m_invPseudoEigenVectors,false);
  return getMatrix(m_invEigenVectors,false);
}

/// Returns the schur decomposition of the dynamics
template <class scalar>
std::string JordanMatrix<scalar>::getSJinvS()
{
  MatrixC matrix=m_eigenVectors;
  std::string result=getMatrix(matrix);
  result+=getMatrix(m_eigenValues);
  matrix=m_eigenVectors.inverse();
  result+=getMatrix(matrix);
  return result;
}

/// Retrieves the inverse of the dynamics
template <class scalar>
typename JordanMatrix<scalar>::MatrixS JordanMatrix<scalar>::getPseudoInverse(const MatrixS &matrix,bool &hasInverse)
{
  load(matrix);
  MatrixC diag=m_eigenValues;
  hasInverse=true;
  for (int i=0;i<m_dimension;i++) {
    char sign=func::hardSign(func::norm2(diag.coeff(i,i)));
    if (sign!=0) {
      diag.coeffRef(i,i)=ms_complexOne/diag.coeff(i,i);
      for (int j=1;j<=m_jordanIndex[i];j++) {
        diag.coeffRef(i-j,i)=-func::c_pow(-diag.coeffRef(i,i),j+1);
      }
    }
    else hasInverse=false;
  }
  if (hasInverse) return m_dynamics.inverse();
  diag=m_invEigenVectors*diag*m_eigenVectors;
  return diag.real();
}

/// Calculates the pseudoinverse of a matrix
template <class scalar>
typename JordanMatrix<scalar>::MatrixS JordanMatrix<scalar>::getSVDpseudoInverse(const MatrixS &matrix,bool &hasInverse)
{
  interToRef(m_refDynamics,matrix);
  m_svdSpace.compute(m_refDynamics, Eigen::ComputeFullU | Eigen::ComputeFullV);
  SolverMatrixType d=m_svdSpace.singularValues().asDiagonal();
  SolverMatrixType u=m_svdSpace.matrixU();
  SolverMatrixType v=m_svdSpace.matrixV();
  MatrixS diag;
  MatrixS matrixU;
  MatrixS matrixV;
  refToInter(diag,d);
  refToInter(matrixU,u);
  refToInter(matrixV,v);
  if (ms_trace_dynamics>=eTraceDynamics) {
    ms_logger.logData(matrix,"Inverse:");
    ms_logger.logData(matrixU);
    ms_logger.logData(diag);
    ms_logger.logData(matrixV);
  }
  for (int i=0;i<matrix.rows();i++) {
    if (func::isZero(diag.coeff(i,i))) hasInverse=false;
    else diag.coeffRef(i,i)=scalar(1)/diag.coeff(i,i);
  }
  if (matrix.rows()!=matrix.cols()) {
    diag.conservativeResize(matrixV.cols(),matrixU.rows());
    if (matrixV.cols()>matrixU.rows()) {
      diag.block(matrixU.rows(),0,matrixV.cols()-matrixU.rows(),matrixU.rows())=MatrixS::Zero(matrixV.cols()-matrixU.rows(),matrixU.rows());
    }
    else {
      diag.block(0,matrixV.cols(),matrixV.cols(),matrixU.rows()-matrixV.cols())=MatrixS::Zero(matrixV.cols(),matrixU.rows()-matrixV.cols());
    }
    hasInverse=false;
  }
  if (hasInverse) return matrix.inverse();
  return matrixV*diag*matrixU.adjoint();
}

/// Calculates a lower bound for the minimum separation between any two jordan blocks
template <class scalar>
typename JordanMatrix<scalar>::refScalar JordanMatrix<scalar>::calculateMinSeparation()
{
  MatrixS base;
  refToInter(base,m_eigenSpace.getSchur());
  scalar nonDiagNorm2=0;
  for (int row=0;row<m_dimension-1;row++) {
    if (m_conjugatePair[row]<row) nonDiagNorm2+=func::squared(base.coeff(row,row+1));//*base.coeff(row,row+1);
    for (int col=row+2;col<m_dimension;col++) {
      nonDiagNorm2+=func::squared(base.coeff(row,col));//*base.coeff(row,col);
    }
  }
  scalar dimension=m_dimension;
  scalar factor=sqrt(func::pow((dimension-ms_one)/dimension,m_dimension-1));
  factor*=sqrt(func::pow(ms_two,(m_dimension-1)*(m_dimension+1)));//The smallest col/row is ignored

  for (int i=0;i<m_dimension;i++) {
    scalar diagNorm2=0;
    scalar det=ms_one;
    int mult=(m_conjugatePair[i]<0) ? 1 : 2;
    m_minSeparation.coeffRef(i,0)=func::toLower(func::norm2(m_eigenValues.coeff(i,i)));
    for (int j=0;j<m_dimension;j++) {
      if (i+mult*m_jordanIndex[j]==j) continue;
      scalar sep=func::norm2(m_eigenValues.coeff(i,i)-m_eigenValues.coeff(j,j));
      det*=sep;
      diagNorm2+=func::squared(sep);//*sep;
    }
    if (func::toLower(det)==0) func::imprecise(det,func::ms_hardZero);
    diagNorm2+=nonDiagNorm2;
    scalar maxColOrRowProd=func::pow(sqrt(diagNorm2),m_dimension-1);
    while (m_jordanIndex[i+mult]>0) i+=mult;
    if (m_jordanIndex[i]>0) {
      maxColOrRowProd*=func::pow(dimension,m_dimension);
      m_minSeparation.coeffRef(i,0)=func::toLower(factor*func::pow(det,m_jordanIndex[i])/maxColOrRowProd);
      for (int j=0;j<m_jordanIndex[i]*mult;j++) m_minSeparation.coeffRef(i-j,0)=m_minSeparation.coeffRef(i,0);
    }
    else m_minSeparation.coeffRef(i,0)=func::toLower(factor*det/maxColOrRowProd);
    if (m_conjugatePair[i]>i) {
      m_minSeparation.coeffRef(i+1,0)=m_minSeparation.coeffRef(i,0);
      i++;
    }
  }
  return m_minSeparation.minCoeff();
}

/// Calculates the maximum error for the numerical approximation of the eigencvalues
template <class scalar>
scalar JordanMatrix<scalar>::calculateEigenError()
{
  scalar kP=m_pseudoEigenVectors.norm()*m_invPseudoEigenVectors.norm();
  MatrixS calculated=m_pseudoEigenVectors*m_pseudoEigenValues*m_invPseudoEigenVectors;
  if (ms_trace_dynamics>=eTraceErrors) ms_logger.logData(calculated,"Calculated:");
  calculated-=m_dynamics;
  scalar errorNorm=calculated.norm();
  m_error=errorNorm*kP;
  if (func::toUpper(m_error)>m_zero) func::imprecise(m_error,m_zero);

  m_boundForError=0;
  if (m_hasMultiplicities) return m_error;

  if (ms_trace_dynamics>=eTraceErrors) ms_logger.logData(m_error,"Error:",true);
  m_error=func::setpm(m_error);
  complexS complexError=m_error;
  for (int i=0;i<m_dimension;i++) {
    m_eigenValues.coeffRef(i,i)+=complexError;
//    m_eigenValues.coeffRef(i,i).real()+=m_error;
//    if (m_conjugatePair[i]>=0) m_eigenValues.coeffRef(i,i).imag()+=m_error;
  }
  m_pseudoEigenValues=jordanToPseudoJordan(m_eigenValues,eToEigenValues);

  calculateMinSeparation();
  for (int i=0;i<m_dimension;i++) {
    if (m_minSeparation.coeff(i,0)>0) {
      scalar angleError=errorNorm/m_minSeparation.coeff(i,0);
      scalar cosTheta=func::toLower(func::cosine(angleError));
      scalar invCosTheta=ms_one/cosTheta;
      scalar vError=func::getHull(cosTheta,invCosTheta);
      m_eigenVectors.col(i)*=vError;
    }
  }
  m_pseudoEigenVectors=jordanToPseudoJordan(m_eigenVectors,eToEigenVectors);
  m_invPseudoEigenVectors=m_pseudoEigenVectors.inverse();//jordanToPseudoJordan(m_invEigenVectors,eToInvEigenVectors);
  if (ms_trace_dynamics>=eTraceDynamics) {
    ms_logger.logData(m_pseudoEigenValues,"PseudoEigenValues");
    ms_logger.logData(m_pseudoEigenVectors,"PseudoEigenVectors");
    ms_logger.logData(m_invPseudoEigenVectors,"InvPseudoEigenVectors");
  }
  return m_error;
}

/// Calculates the maximum error for the numerical approximation of the matrix to the nth power
/// @return the maximum variation of the eigenvalues
template <class scalar>
scalar JordanMatrix<scalar>::calculateBoundedEigenError(scalar iteration)
{
  if (func::isNegative(iteration)) iteration=-iteration;
  if (func::isZero(iteration-m_boundForError)) return m_error;
  m_pseudoEigenVectors=jordanToPseudoJordan(m_eigenVectors,eToEigenVectors);
  m_invPseudoEigenVectors=m_pseudoEigenVectors.inverse();//jordanToPseudoJordan(m_invEigenVectors,eToInvEigenVectors);
  if (ms_trace_dynamics>=eTraceDynamics) {
      ms_logger.logData(m_pseudoEigenVectors,"S");
      ms_logger.logData(m_pseudoEigenValues,"J");
      ms_logger.logData(m_invPseudoEigenVectors,"invS");
  }
  MatrixS calculated=m_pseudoEigenVectors*m_pseudoEigenValues*m_invPseudoEigenVectors;
  if (ms_trace_dynamics>=eTraceErrors) ms_logger.logData(calculated,"Calculated");
  MatrixS jordanError=m_invPseudoEigenVectors*m_dynamics*m_pseudoEigenVectors;
  if (ms_trace_dynamics>=eTraceErrors) ms_logger.logData(jordanError,"Calculated Jordan");
  jordanError-=m_pseudoEigenValues;
  if (ms_trace_dynamics>=eTraceErrors) ms_logger.logData(jordanError,"Jordan Error");
  m_error=jordanError.norm();
  scalar theta=acos((ms_one-m_error)/(ms_one+m_error));
  scalar nTheta=iteration*theta;
  if (func::toUpper(nTheta)>m_zero) func::imprecise(nTheta,m_zero);
  scalar cosn=func::toLower(func::cosine(nTheta));
  scalar invCosN=ms_one/cosn;
  scalar vError=func::getHull(cosn,invCosN);
  m_pseudoEigenVectors*=vError;
  m_invPseudoEigenVectors=m_pseudoEigenVectors.inverse();
  return m_error;
}

/// Calculates the singular values
template <class scalar>
bool JordanMatrix<scalar>::calculateSVD()
{
  boost::timer timer;
  MatrixS dynamicsSq=m_dynamics*m_dynamics.transpose();
  interToRef(m_refDynamics,dynamicsSq);
  m_eigenSpace.computeJordan(m_refDynamics);
  if (ms_trace_time) ms_logger.logData(timer.elapsed()*1000,"Full Svd:",true);
  if (m_eigenSpace.info()!=Eigen::Success) return false;
  MatrixS singularValues,singularVectors,inverseVectors;
  refToInter(singularValues,m_eigenSpace.getEigenValues().real());
  refToInter(singularVectors,m_eigenSpace.getEigenVectors().real());
  inverseVectors=singularVectors.inverse();
  scalar kP=singularVectors.norm()*inverseVectors.norm();
  dynamicsSq-=singularVectors*singularValues*singularVectors.inverse();
  scalar error=dynamicsSq.norm()*kP;
  if (func::toUpper(m_error)>m_zero) func::imprecise(m_error,m_zero);
  error=func::setpm(error);
  for (int i=0;i<m_dimension;i++) {
    singularValues.coeffRef(i,i)+=error;
    singularValues.coeffRef(i,i)=sqrt(singularValues.coeff(i,i));
  }
  return true;
}

/// Calculates the singular values for each Jordan Block
template <class scalar>
void JordanMatrix<scalar>::calculateBlockSVD()
{
  m_blockSingularValues.resize(m_dimension,2);
  for (int row=m_dimension-1;row>=0;row--) {
    if (m_jordanIndex[row]>0) {
      int blockSize=(m_jordanIndex[row]+1);
      MatrixR jordanBlock=MatrixR::Zero(m_dimension,m_dimension);
      jordanBlock.coeffRef(0,0)=func::toUpper(m_eigenNorms.coeff(row,0));
      for (int i=1;i<blockSize;i++) {
        jordanBlock.coeffRef(i,i)=jordanBlock.coeff(0,0);
        jordanBlock.coeffRef(i,i-1)=1;
      }
      m_svdSpace.compute(jordanBlock);
      scalar norm=m_svdSpace.singularValues().coeff(0)*(ms_one+m_dimension*m_dimension*Eigen::NumTraits<refScalar>::epsilon());
      if (m_conjugatePair[row]>=0) blockSize*=2;
      for (int j=0;j<blockSize;j++) m_blockSingularValues.coeffRef(row--,0)=norm;
      row++;
    }
    else {
      m_blockSingularValues.coeffRef(row,0)=m_eigenNorms.coeff(row,0);
    }
  }
  m_blockSingularValues.col(1)=MatrixS::Ones(m_dimension,1);
  for (int row=1;row<m_dimension;row++) {
    if (m_jordanIndex[row]>0)
    {
      int mult=(m_conjugatePair[row]>=0) ? 2 : 1;
      m_blockSingularValues.coeffRef(row,1)=m_blockSingularValues.coeff(row-mult,1)/(this->ms_one-norm(m_eigenValues.coeff(row,row)));
      m_blockSingularValues.coeffRef(row-mult*m_jordanIndex[row],1)+=m_blockSingularValues.coeff(row,1);
    }
  }
}

#ifdef USE_LDOUBLE
  #ifdef USE_SINGLES
    template class JordanMatrix<long double>;
  #endif
  #ifdef USE_INTERVALS
    template class JordanMatrix<ldinterval>;
  #endif
#endif
#ifdef USE_MPREAL
  #ifdef USE_SINGLES
    template class JordanMatrix<mpfr::mpreal>;
  #endif
  #ifdef USE_INTERVALS
    template class JordanMatrix<mpinterval>;
  #endif
#endif
}
