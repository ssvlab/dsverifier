#include "RefinedDynamicalSystem.h"
#include <boost/timer.hpp>
#include <set>
#include <Eigen/QR>
#include <Eigen/Eigenvalues>

namespace abstract{

/// Constructs an empty buffer
template<class scalar>
CegarSystem<scalar>::CegarSystem(int dimension,int idimension) :
  DynamicalSystem<scalar>(dimension,idimension)
{}

/// Solves the Sylvester equation AX+XB=C for X
template<class scalar>
typename CegarSystem<scalar>::MatrixS CegarSystem<scalar>::solveSylvester(const MatrixS &A,const MatrixS &B,const MatrixS &C,bool BisDiagonal)
{
  MatrixS U,V;
  SolverMatrixType refA,refB;
  interToRef(refA,A);
  interToRef(refB,B);
  Eigen::RealSchur<SolverMatrixType> realSchur(refA.transpose());
  refToInter(U,realSchur.matrixU());//We want transposed so keep it this way
  refA=realSchur.matrixT().transpose();//Lower triangular
  if (BisDiagonal) {
    V=MatrixS::Identity(C.rows(),refB.rows());
  }
  else {
    interToRef(refB,B);
    realSchur.compute(refB);
    refToInter(V,realSchur.matrixU());
    refB=realSchur.matrixT();
  }
  MatrixS result=U*C*V;
  for (int row=0;row<result.rows();row++) {
    for (int col=0;col<result.cols();col++) {
      for (int i=0;i<row;i++) {
        result.coeffRef(row,col)-=refA.coeff(row,i)*result.coeff(i,col);
      }
      for (int i=0;i<col;i++) {
        result.coeffRef(row,col)-=result.coeff(row,i)*refB.coeff(i,col);
      }
      result.coeffRef(row,col)/=refA.coeff(row,row)+refB.coeff(col,col);
    }
  }
  return result;
}

/// Retrieves the characteristic polynomial coefficients of the dynamics
template <class scalar>
typename CegarSystem<scalar>::MatrixS CegarSystem<scalar>::getDynamicPolynomialCoefficients()
{
  MatrixC result=MatrixC::Zero(2,m_dimension+1);
  result.coeffRef(0,0)=ms_one;
  result.coeffRef(0,1)=-m_eigenValues.coeff(0,0);
  for (int i=1;i<m_dimension;i++) {
    result.row(1)=-result.row(0)*m_eigenValues.coeff(i,i);
    result.block(0,1,1,i+1)+=result.block(1,0,1,i+1);
  }
  return result.real();
}

/// Retrieves the reachability matrix [B AB A^2B ...A^{n-1}B]
template <class scalar>
typename CegarSystem<scalar>::MatrixS CegarSystem<scalar>::getReachabilityMatrix()
{
  MatrixS result(m_sensitivity.rows(),m_dimension*m_sensitivity.cols());
  result.block(0,0,m_sensitivity.rows(),m_sensitivity.cols())=m_sensitivity;
  MatrixS multiplier=m_dynamics;
  for (int i=1;i<m_dimension;i++)
  {
    result.block(0,i*m_sensitivity.cols(),m_sensitivity.rows(),m_sensitivity.cols())=multiplier*m_sensitivity;
    multiplier*=m_dynamics;
  }
  if (ms_trace_dynamics) ms_logger.logData(result,"Reachability Matrix");
  return result;
}

/// Retrieves the reachability matrix [1 a1 a2...a_{n-1};0 1 a1 ... a_{n-2}...]
template <class scalar>
typename CegarSystem<scalar>::MatrixS CegarSystem<scalar>::getCanonicalReachabilityMatrix()
{
  //TODO: not sure how this works on MIMO
  MatrixS result=MatrixS::Identity(m_sensitivity.rows(),m_dimension*m_sensitivity.cols());
  MatrixS coefficients=getDynamicPolynomialCoefficients();
  if (ms_trace_dynamics>=eTraceDynamics) {
    MatrixS matrix=coefficients.row(0);
    ms_logger.logData(matrix,"Coefficients");
  }
  for (int i=1;i<m_dimension;i++) result.block(i-1,i,1,m_dimension-i)=coefficients.block(0,1,1,m_dimension-i);
  if (ms_trace_dynamics>=eTraceDynamics) ms_logger.logData(result,"Canonical Reachability Matrix");
  return result;
}

/// Retrieves the transform matrix T : z=T^{-1}x turns A,B,C into controllable canonical form
template <class scalar>
typename CegarSystem<scalar>::MatrixS CegarSystem<scalar>::getReachableCanonicalTransformMatrix()
{
  MatrixS result=getReachabilityMatrix()*getCanonicalReachabilityMatrix();
  return result.inverse();
}

/// Retrieves the observability matrix [C CA CA^2 ...CA^{n-1}]^T
template <class scalar>
typename CegarSystem<scalar>::MatrixS CegarSystem<scalar>::getObservabilityMatrix()
{
  MatrixS result(m_dimension*m_outputSensitivity.rows(),m_outputSensitivity.cols());
  result.block(0,0,m_outputSensitivity.rows(),m_outputSensitivity.cols())=m_outputSensitivity;
  MatrixS multiplier=m_dynamics;
  for (int i=1;i<m_dimension;i++)
  {
    result.block(i*m_outputSensitivity.rows(),0,m_outputSensitivity.rows(),m_outputSensitivity.cols())=m_outputSensitivity*multiplier;
    multiplier*=m_dynamics;
  }
  return result;
}

/// Retrieves the observability matrix [1 0 ...;a1 1 0 ...;...;a_{n-1}...a1 1]^-1
template <class scalar>
typename CegarSystem<scalar>::MatrixS CegarSystem<scalar>::getInverseCanonicalObservabilityMatrix()
{
  //TODO: not sure how this works on MIMO
  MatrixS result=MatrixS::Identity(m_dimension*m_outputSensitivity.rows(),m_outputSensitivity.cols());
  MatrixS coefficients=getDynamicPolynomialCoefficients();
  for (int i=0;i<m_dimension;i++) {
    for (int j=0;j<i;j++) {
      result.coeffRef(i,j)=coefficients.coeff(0,m_dimension-i-j-1);
    }
  }
  return result;
}

/// Retrieves the transform matrix T : z=T^{-1}x turns A,B,C into observable canonical form
template <class scalar>
typename CegarSystem<scalar>::MatrixS CegarSystem<scalar>::getObservableCanonicalTransformMatrix()
{
  return getObservabilityMatrix().inverse()*getInverseCanonicalObservabilityMatrix();
}

/// Retrieves the reference gain
template <class scalar>
typename CegarSystem<scalar>::MatrixS CegarSystem<scalar>::getReferenceGain()
{
  return m_outputSensitivity*((m_dynamics-(m_sensitivity*m_feedback)).inverse()*m_sensitivity);
}

/// Retrieves the reach tube at the given iteration
template <class scalar>
/// Retrieves refined dynamics given a safety specification
AbstractPolyhedra<scalar>& CegarSystem<scalar>::getRefinedDynamics(int refinements,powerS iteration,int directions,inputType_t inputType)
{
  AbstractPolyhedra<scalar>&reachTube=getAbstractReachTube(iteration,2,directions,inputType);
  if (ms_trace_dynamics>=eTraceDynamics) {
    reachTube.logTableau("PreCegar:",true);
    reachTube.logVertices(true);
  }
  //if (m_safeReachTube.getPolyhedra().contains(reachTube)) return getAbstractDynamics(inputType);
  if (!m_safeReachTube.isEmpty()) {
    AbstractPolyhedra<scalar> bounds=synthesiseDynamicBounds(m_inputType,m_safeReachTube.getPolyhedra(eEigenSpace));
    if (ms_trace_dynamics>=eTraceDynamics) {
      bounds.logTableau("Projected Bounds: ",true);
    }
    for(int i=0;(i<refinements) && refineAbstractDynamics(bounds);i++);
    if (ms_trace_dynamics>=eTraceDynamics) {
      AbstractPolyhedra<scalar>& result=getRefinedAbstractReachTube(eNormalSpace);
      result.logTableau("PosCegar:",true);
      result.logVertices(true);
    }
  }
  return getAbstractDynamics(inputType);
}

/// Synthesises a bound on the dynamics given a known guard and eigenvectors.
template<class scalar>
AbstractPolyhedra<scalar> CegarSystem<scalar>::synthesiseDynamicBounds(inputType_t inputType,AbstractPolyhedra<scalar> &end)
{
  m_inputType=inputType;
  MatrixS vectors;
  int numVertices;
  getAbstractVertices(end.getDirections(),vectors,numVertices);
  vectors.transposeInPlace();
  MatrixS supports(vectors.rows(),1);
  MatrixS endSupports=end.getSupports();
  int perTemplate=supports.rows()/endSupports.rows();
  for (int i=0;i<endSupports.rows();i++)
  {
    for (int j=0;j<perTemplate;j++) {
      supports.coeffRef(i*perTemplate+j,0)=endSupports.coeff(i,0);
    }
  }
  if (m_inputType>eNoInputs) {
    MatrixS inSupports=m_accelVertices*end.getDirections();
    demergeAccelInSupports(supports,inSupports,endSupports.rows());
  }
  AbstractPolyhedra<scalar> bounds;
  bounds.load(vectors,supports);
  if (ms_trace_dynamics>=eTraceAll) {
    bounds.logTableau("Full Bounds",true);
  }
  bounds.removeRedundancies();
  return bounds;
}

/// Creates a model for the quantization noise as an input specification
template<class scalar>
AbstractPolyhedra<scalar> CegarSystem<scalar>::generateNoiseInput()
{
  int odimension=(m_odimension>0) ? m_odimension : m_dimension;
  int ndimension=1;
  AbstractPolyhedra<scalar> result(ndimension);
  int fbits=m_paramValues.coeff(eNumBits,1)-m_paramValues.coeff(eNumBits,2);
  if (fbits<=0) fbits=m_paramValues.coeff(eNumBits,0);
  if (fbits<=0) fbits=func::getDefaultPrec();
  scalar lsb=func::pow(this->ms_two,-fbits);
  MatrixS faces(2*ndimension,ndimension);
  MatrixS supports(2*ndimension,1);
  faces.block(0,0,ndimension,ndimension)=MatrixS::Identity(ndimension,ndimension);
  faces.block(ndimension,0,ndimension,ndimension)=-MatrixS::Identity(ndimension,ndimension);
  scalar fb_noise=m_feedback.cwiseAbs().sum()/*lpNorm<1>()*/+this->ms_one;//|Noise|<(|K|_1+1)lsb assuming q1,q2=1lsb,(q3:=q1->K) = (|K|_1q1+q2)
  supports.coeffRef(0,0)=lsb*(fb_noise);
  supports.coeffRef(1,0)=supports.coeff(0,0);
  result.load(faces,supports);
  return result;
}

/// Creates a model for the input of the closed loop
template<class scalar>
AbstractPolyhedra<scalar> CegarSystem<scalar>::generateFeedbackInput(int fdimension,bool makeNoise,MatrixS &sensitivity)
{
  AbstractPolyhedra<scalar> inputs(0);
  if (m_reference.getDimension()>0) {
    inputs.copy(m_reference);
    fdimension=m_reference.getDimension();
  }
  if (m_idimension>fdimension) {
    MatrixS inputFaces=m_inputs.getFaceDirections().rightCols(m_idimension-fdimension);
    MatrixS inSupports=m_inputs.getSupports();
    inputs.concatenate(inputFaces,inSupports);
  }
  if (makeNoise) {
    //m_closedLoop.setInputType(eVariableInputs);
    AbstractPolyhedra<scalar> noiseInputs=generateNoiseInput();
    inputs.concatenate(noiseInputs);
    sensitivity.conservativeResize(m_dimension,inputs.getDimension());
    sensitivity.block(0,inputs.getDimension()-noiseInputs.getDimension(),m_dimension,noiseInputs.getDimension())=MatrixS::Ones(m_dimension,noiseInputs.getDimension());
  }
  return inputs;
}

/// Retrieves a list of iterations whose reach set fails the specification
template<class scalar>
bool CegarSystem<scalar>::findCounterExampleIterations(powerList &iterations,AbstractPolyhedra<scalar> &bounds)
{
  AbstractPolyhedra<scalar>& dynamics=getAbstractDynamics(m_inputType);
  MatrixS &vertices=dynamics.getVertices(true);
  bool found=false;
  for (int i=0;i<vertices.rows();i++) {
    MatrixS point=vertices.row(i);
    if (!bounds.isInside(point)) {
      findIterations(point,iterations);
      found=true;
    }
  }
  return found;
}

/// Refines the abstraction in order to meet the safety specification
template<class scalar>
bool CegarSystem<scalar>::refineAbstractDynamics(AbstractPolyhedra<scalar> &bounds,powerList &iterations)
{
  AbstractPolyhedra<scalar>& dynamics=getAbstractDynamics(m_inputType);
  if (findCounterExampleIterations(iterations,bounds)) {
    typename powerList::iterator it;
    for (it=iterations.begin();it!=iterations.end();it++) {
      addSupportsAtIteration(dynamics,it->first,m_maxIterations);
    }
    return true;
  }
  return false;
}

/// Corrects the support set by the input offset
template <class scalar>
void CegarSystem<scalar>::demergeAccelInSupports(MatrixS &supports,MatrixS &inSupports,int numTemplates)
{
  if (!m_hasOnes || (m_inputType==eVariableInputs))  {
    for (int row=0;row<numTemplates;row++) {
      int pos=row*m_numVertices;
      supports.coeffRef(pos,0)-=inSupports.coeff(0,row);
      for (int point=1;point<m_numVertices;point++) {
        supports.coeffRef(pos+point,0)-=inSupports.coeff(point%inSupports.rows(),row);
      }
    }
  }
}


/// Retrieves the support set for the inputs
template <class scalar>
typename JordanMatrix<scalar>::MatrixS& CegarSystem<scalar>::getRefinedAccelInSupports()
{
  if (m_hasOnes && (m_inputType==eVariableInputs)) {
    AbstractPolyhedra<scalar>& inputDynamics=getAbstractDynamics(eParametricInputs);
    MatrixS supports;
    inputDynamics.maximiseAll(m_abstractInputVertices,supports);
    m_accelInSupports=supports.transpose();
    if (ms_trace_dynamics>=eTraceAbstraction) ms_logger.logData(m_accelInSupports,"Input Supports",true);
  }
  return m_accelInSupports;
}

/// Retrieves the reach tube at the given iteration
template <class scalar>
AbstractPolyhedra<scalar>& CegarSystem<scalar>::getRefinedAbstractReachTube(space_t space,bool guarded)
{
  boost::timer timer;
  AbstractPolyhedra<scalar>& init=m_initialState.getPolyhedra(eEigenSpace);
  AbstractPolyhedra<scalar>& dynamics=getAbstractDynamics(m_inputType);

  MatrixS& templates=getTemplates(eEigenSpace);
  if (ms_trace_time) ms_logger.logData(timer.elapsed()*1000,"Abstract Vertices: ",true);
  MatrixS supports;
  if (!dynamics.maximiseAll(m_abstractVertices,supports)) processError(dynamics.getName());

  if (m_inputType>eNoInputs) getRefinedAccelInSupports();
  if (ms_trace_dynamics>=eTraceAll) {
    traceSupports(templates,supports,dynamics,m_abstractVertices);
  }
  if (m_inputType>eNoInputs) {
    mergeAccelInSupports(supports,templates.cols());
    if (ms_trace_dynamics>=eTraceAll) {
      ms_logger.logData(m_abstractVertices,supports,"Combined",true);
    }
  }
  mergeAbstractSupports(templates,supports);
  MatrixS faces=templates.transpose();
  m_pAbstractReachTube->mergeLoad(init,faces,supports,eEigenSpace);
  AbstractPolyhedra<scalar>& result=m_pAbstractReachTube->getPolyhedra(space);
  if (guarded) getGuardedReachTube(result,space);
  if (ms_trace_dynamics>=eTraceAbstraction) result.logTableau();
  m_reachTime=timer.elapsed()*1000;
  result.setCalculationTime(m_reachTime);
  if (ms_trace_time) ms_logger.logData(m_reachTime,"Abstract Reach Time: ",true);
  return result;
}

#ifdef USE_LDOUBLE
  #ifdef USE_SINGLES
    template class CegarSystem<long double>;
  #endif
  #ifdef USE_INTERVALS
    template class CegarSystem<ldinterval>;
  #endif
#endif
#ifdef USE_MPREAL
  #ifdef USE_SINGLES
    template class CegarSystem<mpfr::mpreal>;
  #endif
  #ifdef USE_INTERVALS
    template class CegarSystem<mpinterval>;
  #endif
#endif

}
