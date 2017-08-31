//Authors: Dario Cattaruzza, Alessandro Abate, Peter Schrammel, Daniel Kroening
//University of Oxford 2016
//This code is supplied under the BSD license agreement (see license.txt)

#include "AccelMatrix.h"

namespace abstract{

/// Constructs an empty buffer
template <class scalar>
AccelMatrix<scalar>::AccelMatrix(int dimension) : JordanMatrix<scalar>(dimension)
{
}

template <class scalar>
scalar AccelMatrix<scalar>::binomial(powerS n,int k)
{
  scalar num=n;
  for (powerS i=n-k+1;i<n;i++) num*=scalar(i);//TODO: should be direct to reduce cost
  scalar den=k;
  for (int i=2;i<k;i++) den*=scalar(i);
  return num/den;
}

/// retrieves J^n
template <class scalar>
typename AccelMatrix<scalar>::MatrixC AccelMatrix<scalar>::getPoweredEigenValues(const powerS iteration)
{
  typename JordanMatrix<scalar>::MatrixC result=m_eigenValues;
  if (iteration==1) return result;
  for (int row=0;row<m_eigenValues.rows();row++)
  {
    result.coeffRef(row,row)=func::c_pow(m_eigenValues.coeff(row,row),iteration);
    int mult=(m_conjugatePair[row]<0) ? 1 : 2;
    for (int offset=1;offset<=m_jordanIndex[row];offset++) {
      result.coeffRef(row-mult*offset,row)=binomial(iteration,offset)*func::c_pow(m_eigenValues.coeff(row,row),iteration-offset);
    }
  }
  return result;
}

/// retrieves F^n
template <class scalar>
typename AccelMatrix<scalar>::MatrixS AccelMatrix<scalar>::getPoweredSingularValues(const powerS iteration)
{
  MatrixS result=m_foldedEigenValues;
  int power=iteration;//TODO: change for scalar
  for (int row=0;row<m_foldedEigenValues.rows();row++) {
    result.coeffRef(row,row)=func::pow(m_foldedEigenValues.coeff(row,row),power);
  }
  return result;
}

/// retrieves a real pseudoDiagonal of J^n
template <class scalar> typename AccelMatrix<scalar>::MatrixS AccelMatrix<scalar>::getPoweredPseudoEigenValues(powerS iteration)
{
  MatrixC matrix=getPoweredEigenValues(iteration);
  return jordanToPseudoJordan(matrix,eToEigenValues);
}

/// retrieves the dynamics matrix for a given iteration (n)
template <class scalar>
typename JordanMatrix<scalar>::MatrixC AccelMatrix<scalar>::getDynamics(const powerS iteration,const bool input,const space_t space)
{
  if (iteration>func::ms_infPower) return getDynamics(func::ms_infPower,input,space);
  MatrixC nthEigenValues=input ? (calculateIminJn(iteration)*m_invIminJ) : getPoweredEigenValues(iteration);
  MatrixC eigenCalculated=nthEigenValues*m_invEigenVectors;
  if (space>=eEigenSpace) return eigenCalculated;
  MatrixC calculated=m_eigenVectors*eigenCalculated;
  if (iteration<0) calculated=m_invEigenVectors*nthEigenValues*m_eigenVectors;
  return calculated;
}

/// retrieves the dynamics matrix for a given iteration (n) using pseudo eigenvalues
template <class scalar> typename JordanMatrix<scalar>::MatrixS AccelMatrix<scalar>::getPseudoDynamics(powerS iteration,bool input,space_t space)
{
  if (iteration>func::ms_infPower) iteration=func::ms_infPower;
  if (space==eSingularSpace) {
    typename JordanMatrix<scalar>::MatrixS nthEigenValues=getPoweredSingularValues(iteration)*m_invIminF;
    return nthEigenValues;
  }
  MatrixC cEigenValues=input ? (calculateIminJn(iteration)*m_invIminJ) : getPoweredEigenValues(iteration);
  MatrixS nthEigenValues=jordanToPseudoJordan(cEigenValues,eToEigenValues);
  if (space==eEigenSpace) return nthEigenValues;
  if (iteration<0) return m_invPseudoEigenVectors*nthEigenValues*m_invPseudoEigenVectors;
  return m_pseudoEigenVectors*nthEigenValues*m_invPseudoEigenVectors;
}

/// Calculates (I-J^n)
template <class scalar>
typename JordanMatrix<scalar>::MatrixC AccelMatrix<scalar>::calculateIminJn(const powerS iteration)
{
  MatrixC result=MatrixC::Identity(m_dimension,m_dimension)-getPoweredEigenValues(iteration);
  for (int row=0;row<m_dimension;row++) {
    if (m_isOne[row]) {
      result.coeffRef(row,row)=iteration;
      for (int offset=1;offset<=m_jordanIndex[row];offset++) {
        result.coeffRef(row-offset,row)=binomial(iteration,offset+1);
      }
    }
  }
  return result;
}

/// Calculates (I-F^n)
template <class scalar>
typename JordanMatrix<scalar>::MatrixS AccelMatrix<scalar>::calculateIminFn(const powerS iteration)
{
  MatrixS result=MatrixS::Identity(m_dimension,m_dimension)-getPoweredSingularValues(iteration);
  scalar sIteration=iteration;
  for (int row=0;row<m_dimension;row++) {
    if (abs(m_invIminF.coeff(row,row))<this->m_zero) result.coeffRef(row,row)=sIteration;
  }
  return result;
}

/// Calculates (I-J)^-1
template <class scalar> void AccelMatrix<scalar>::calculateInvIminJ()
{
  m_pseudoIminJ=MatrixS::Identity(m_dimension,m_dimension)-this->m_pseudoEigenValues;
  m_invIminJ=MatrixC::Identity(m_dimension,m_dimension)-m_eigenValues;

  for (int row=0;row<m_dimension;row++) {
    if (m_isOne[row]) {
      m_invIminJ.coeffRef(row,row)=this->ms_complexOne;
      int mult=(m_conjugatePair[row]<0) ? 1 : 2;
      for (int offset=1;offset<=m_jordanIndex[row];offset++) {
        m_invIminJ.coeffRef(row-mult*offset,row)=0;
      }
    }
    else {
      m_invIminJ.coeffRef(row,row)=this->ms_complexOne/m_invIminJ.coeff(row,row);
      if (m_conjugatePair[row]>=0) {
        if (m_jordanIndex[row+2]>0) {
          for (int col=row+2;col<m_dimension;col+=2){
            int power=(col-row)>>1;
            m_invIminJ.coeffRef(row,col)=func::c_pow(m_invIminJ.coeffRef(row,row),power+1);
            if ((power+1)&1) m_invIminJ.coeffRef(row,col)=-m_invIminJ.coeffRef(row,col);
          }
        }
      }
      else if (m_jordanIndex[row+1]>0) {
        for (int col=row+1;col<m_dimension;col++){
          m_invIminJ.coeffRef(row,col)=func::c_pow(m_invIminJ.coeffRef(row,row),col+1-row);
          if ((col+1-row)&1) m_invIminJ.coeffRef(row,col)=-m_invIminJ.coeffRef(row,col);
        }
      }
    }
  }
  if (m_hasOnes) m_pseudoInvIminJ=this->jordanToPseudoJordan(m_invIminJ,eToEigenValues);
  else           m_pseudoInvIminJ=m_pseudoIminJ.inverse();
  m_IminA=m_pseudoEigenVectors*m_pseudoIminJ*m_invPseudoEigenVectors;
  this->m_invIminA=m_pseudoEigenVectors*m_pseudoInvIminJ*m_invPseudoEigenVectors;
}

/// Calculates the folded matrix over-approximation on the rotation and dilation axes
template <class scalar> void AccelMatrix<scalar>::calculateF()
{
  this->m_foldedEigenValues=MatrixS::Identity(m_dimension,m_dimension);
  m_binomialMultipliers=MatrixS::Ones(m_dimension,1);
  for (int row=0;row<m_dimension;row++) {
    if (m_jordanIndex[row]>0)
    {
      int mult=(m_conjugatePair[row]>=0) ? 2 : 1;
      m_binomialMultipliers.coeffRef(row,0)=m_binomialMultipliers.coeff(row-mult,0)/(this->ms_one-norm(m_eigenValues.coeff(row,row)));
      m_binomialMultipliers.coeffRef(row-mult*m_jordanIndex[row],0)+=m_binomialMultipliers.coeff(row,0);
    }
    //else m_binomialMultipliers.coeffRef(row,0)/=(this->ms_one-norm(m_eigenValues.coeff(row,row)));//This is the original acceleration
  }
  int j=1;
  m_foldedBinomialMultipliers=m_binomialMultipliers;
  m_foldedEigenValues.coeffRef(0,0)=this->m_blockSingularValues.coeff(0,0);
  m_svnIndex.resize(m_dimension);
  m_svnIndex[0]=0;
  for (int row=1;row<m_dimension;row++) {
    m_svnIndex[row]=j;
    if ((m_jordanIndex[row]>0) || (m_conjugatePair[row]==row-1)) {
      m_svnIndex[row]--;
      continue;
    }
    int offset=m_jordanIndex[row]*((m_conjugatePair[row]<0) ? 1 : 2);
    m_foldedEigenValues.coeffRef(j,j)=this->m_blockSingularValues.coeff(row,0);
    m_foldedBinomialMultipliers.coeffRef(j,0)=this->m_binomialMultipliers.coeff(row-offset,0);
    j++;
  }
  m_foldedEigenValues.conservativeResize(j,j);
  m_foldedBinomialMultipliers.conservativeResize(j,1);
}

/// Calculates (I-F)^-1
template <class scalar> void AccelMatrix<scalar>::calculateInvIminF()
{
  int dimension=m_foldedEigenValues.rows();
  m_IminF=MatrixS::Identity(dimension,dimension)-m_foldedEigenValues;
  m_invIminF=m_IminF;
  for (int row=0;row<dimension;row++) {
    if (abs(m_foldedEigenValues.coeff(row,row))<this->m_zero) m_invIminF.coeffRef(row,row)=this->ms_one;
    else m_invIminF.coeffRef(row,row)=this->ms_one/m_IminF.coeff(row,row);
  }
}

template <class scalar> bool AccelMatrix<scalar>::calculateJordanForm(bool includeSvd)
{
  if (!JordanMatrix<scalar>::calculateJordanForm(includeSvd)) return false;
  calculateInvIminJ();
  calculateF();
  calculateInvIminF();
  if (this->ms_trace_dynamics>=eTraceDynamics) {
    ms_logger.logData(m_pseudoInvIminJ,"InvIminJ:");
    ms_logger.logData(m_invIminF,"InvIminF:");
  }
  return true;
}

// Loads a matrix assumed to be a canonical Jordan form
template <class scalar> bool AccelMatrix<scalar>::loadJordan(const MatrixS &matrix)
{
  if (!JordanMatrix<scalar>::loadJordan(matrix)) return false;
  calculateBlockSVD();
  calculateInvIminJ();
  calculateF();
  calculateInvIminF();
  if (this->ms_trace_dynamics>=eTraceDynamics) {
    ms_logger.logData(m_pseudoInvIminJ,"InvIminJ:");
    ms_logger.logData(m_invIminF,"InvIminF:");
  }
  return true;
}

#ifdef USE_LDOUBLE
  #ifdef USE_SINGLES
    template class AccelMatrix<long double>;
  #endif
  #ifdef USE_INTERVALS
    template class AccelMatrix<ldinterval>;
  #endif
#endif
#ifdef USE_MPREAL
#ifdef USE_SINGLES
    template class AccelMatrix<mpfr::mpreal>;
  #endif
  #ifdef USE_INTERVALS
    template class AccelMatrix<mpinterval>;
  #endif
#endif

}
