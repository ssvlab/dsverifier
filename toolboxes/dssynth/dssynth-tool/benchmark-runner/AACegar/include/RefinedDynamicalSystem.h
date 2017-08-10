#ifndef CEGARSYSTEM_H
#define CEGARSYSTEM_H

#include <map>
#include "DynamicalSystem.h"

namespace abstract
{
template <class scalar> class CegarSystem : public DynamicalSystem<scalar>
{
public:
    using typename JordanMatrix<scalar>::refScalar;
    using typename JordanMatrix<scalar>::func;
    using typename JordanMatrix<scalar>::complexS;
    using typename JordanMatrix<scalar>::MatrixS;
    using typename JordanMatrix<scalar>::MatrixC;
    using typename JordanMatrix<scalar>::SolverMatrixType;
    using typename JordanMatrix<scalar>::SolverComplexMatrixType;
    using typename AccelMatrix<scalar>::powerS;
    using typename AbstractMatrix<scalar>::powerList;

    using JordanMatrix<scalar>::m_dimension;
    using JordanMatrix<scalar>::m_isOne;
    using JordanMatrix<scalar>::m_hasOnes;
    using JordanMatrix<scalar>::m_hasMultiplicities;
    using JordanMatrix<scalar>::m_jordanIndex;
    using JordanMatrix<scalar>::m_conjugatePair;
    using DynamicalSystem<scalar>::m_idimension;
    using DynamicalSystem<scalar>::m_fdimension;
    using DynamicalSystem<scalar>::m_odimension;
    using DynamicalSystem<scalar>::m_reachTime;
    using DynamicalSystem<scalar>::m_inputType;
    using DynamicalSystem<scalar>::m_initialState;
    using DynamicalSystem<scalar>::m_transformedInputs;
    using DynamicalSystem<scalar>::m_eigenValues;
    using DynamicalSystem<scalar>::m_eigenVectors;
    using DynamicalSystem<scalar>::m_invEigenVectors;
    using DynamicalSystem<scalar>::m_pseudoEigenVectors;
    using DynamicalSystem<scalar>::m_invPseudoEigenVectors;
    using DynamicalSystem<scalar>::m_source;
    using DynamicalSystem<scalar>::m_guard;
    using DynamicalSystem<scalar>::m_dynamics;
    using DynamicalSystem<scalar>::m_sensitivity;
    using DynamicalSystem<scalar>::m_inputs;
    using DynamicalSystem<scalar>::m_reference;
    using DynamicalSystem<scalar>::m_outputSensitivity;
    using DynamicalSystem<scalar>::m_outputGuard;
    using DynamicalSystem<scalar>::m_safeReachTube;
    using DynamicalSystem<scalar>::m_feedback;
    using DynamicalSystem<scalar>::m_numVertices;
    using DynamicalSystem<scalar>::m_accelVertices;
    using DynamicalSystem<scalar>::m_abstractVertices;
    using DynamicalSystem<scalar>::m_abstractInputVertices;
    using DynamicalSystem<scalar>::m_accelInSupports;
    using DynamicalSystem<scalar>::m_paramValues;
    using DynamicalSystem<scalar>::m_dynamicParams;
    using DynamicalSystem<scalar>::m_pAbstractReachTube;
    using DynamicalSystem<scalar>::m_maxIterations;
    using DynamicalSystem<scalar>::m_loadTime;
    using DynamicalSystem<scalar>::ms_one;
    using DynamicalSystem<scalar>::ms_logger;
    using DynamicalSystem<scalar>::ms_trace_time;
    using DynamicalSystem<scalar>::ms_trace_dynamics;
    using DynamicalSystem<scalar>::ms_emptyMatrix;


    using JordanMatrix<scalar>::interToRef;
    using JordanMatrix<scalar>::refToInter;
    using AbstractMatrix<scalar>::findIterations;
    using AbstractMatrix<scalar>::addSupportsAtIteration;
    using DynamicalSystem<scalar>::getTemplates;
    using DynamicalSystem<scalar>::getGuardPoly;
    using DynamicalSystem<scalar>::getInitPoly;
    using DynamicalSystem<scalar>::getInputPoly;
    using DynamicalSystem<scalar>::getAccelInputPoly;
    using DynamicalSystem<scalar>::getReachPoly;
    using DynamicalSystem<scalar>::getTubePoly;
    using DynamicalSystem<scalar>::getAbsTubePoly;
    using DynamicalSystem<scalar>::getRoundedDirections;
    using DynamicalSystem<scalar>::makeLogahedralTemplates;
    using DynamicalSystem<scalar>::getAbstractReachTube;
    using DynamicalSystem<scalar>::getAbstractDynamics;
    using DynamicalSystem<scalar>::getAbstractVertices;
    using DynamicalSystem<scalar>::addSupportsAtIteration;
    using DynamicalSystem<scalar>::getAccelInSupports;
    using DynamicalSystem<scalar>::mergeAccelInSupports;
    using DynamicalSystem<scalar>::mergeAbstractSupports;
    using DynamicalSystem<scalar>::traceSupports;
    using DynamicalSystem<scalar>::getGuardedReachTube;
    using DynamicalSystem<scalar>::loadFromFile;
    using DynamicalSystem<scalar>::load;
    using DynamicalSystem<scalar>::save;
    using DynamicalSystem<scalar>::kronecker;
    using DynamicalSystem<scalar>::traceDynamics;
    using DynamicalSystem<scalar>::traceSimplex;
    using DynamicalSystem<scalar>::changeDimensions;
    using DynamicalSystem<scalar>::loadGuard;
    using DynamicalSystem<scalar>::loadOutputGuard;
    using DynamicalSystem<scalar>::loadDynamics;
    using DynamicalSystem<scalar>::loadInitialState;
    using DynamicalSystem<scalar>::loadSensitivities;
    using DynamicalSystem<scalar>::loadInputs;
    using DynamicalSystem<scalar>::loadSafeReachTube;
    using DynamicalSystem<scalar>::loadTemplates;
    using DynamicalSystem<scalar>::loadARMAXModel;
    using DynamicalSystem<scalar>::processError;

    /// Constructs an empty buffer
    /// @param dimension dimension of the space
    CegarSystem(int dimension=0,int idimension=0);

    /// Retrieves the characteristic polynomial coefficients of the dynamics
    MatrixS getDynamicPolynomialCoefficients();

    /// Retrieves the reachability matrix [B AB A^2B ...A^{n-1}B]
    MatrixS getReachabilityMatrix();

    /// Retrieves the reachability matrix [1 a1 a2...a_{n-1};0 1 a1 ... a_{n-2}...]
    MatrixS getCanonicalReachabilityMatrix();

    /// Retrieves the transform matrix T : z=T^{-1}x turns A,B,C into controllable canonical form
    MatrixS getReachableCanonicalTransformMatrix();

    /// Retrieves the observability matrix [C CA CA^2 ...CA^{n-1}]^T
    MatrixS getObservabilityMatrix();

    /// Retrieves the observability matrix [1 0 ...;a1 1 0 ...;...;a_{n-1}...a1 1]^-1
    MatrixS getInverseCanonicalObservabilityMatrix();

    /// Retrieves the transform matrix T : z=T^{-1}x turns A,B,C into observable canonical form
    MatrixS getObservableCanonicalTransformMatrix();

    /// Retrieves the reference gain
    MatrixS getReferenceGain();

    /// Synthesises a bound on the dynamics given a known guard and eigenvectors.
    AbstractPolyhedra<scalar> synthesiseDynamicBounds(inputType_t inputType,AbstractPolyhedra<scalar> &end);

    /// Creates a model for the input of the closed loop
    AbstractPolyhedra<scalar> generateFeedbackInput(int fdimension=0,bool makeNoise=false,MatrixS &sensitivity=ms_emptyMatrix);

    /// Creates a model for the quantization noise as an input specification
    AbstractPolyhedra<scalar> generateNoiseInput();

    /// Retrieves a list of iterations whose reach set fails the specification
    bool findCounterExampleIterations(powerList &iterations,AbstractPolyhedra<scalar> &bounds);

    /// Refines the abstraction in order to meet the safety specification
    bool refineAbstractDynamics(AbstractPolyhedra<scalar> &bounds,powerList &iterations);

    /// Refines the abstraction in order to meet the safety specification
    bool refineAbstractDynamics(AbstractPolyhedra<scalar> &bounds)
    {
      powerList iterations;
      refineAbstractDynamics(bounds,iterations);
      return !iterations.empty();
    }

    /// Retrieves the support set for the inputs
    MatrixS& getRefinedAccelInSupports();

    /// Corrects the support set by the input offset
    void demergeAccelInSupports(MatrixS &supports,MatrixS &inSupports,int numTemplates);

    /// Retrieves the reach tube given the refined dnamics
    AbstractPolyhedra<scalar>& getRefinedAbstractReachTube(space_t space,bool guarded=false);

    /// Retrieves refined dynamics given a safety specification
    AbstractPolyhedra<scalar>& getRefinedDynamics(int refinements,powerS iteration=0,int directions=0,inputType_t inputType=eParametricInputs);

    /// Solves the Sylvester equation AX+XB=C for X
    /// @return X
    MatrixS solveSylvester(const MatrixS &A,const MatrixS &B,const MatrixS &C,bool BisDiagonal=false);
};

}

#endif //CEGARSYSTEM_H
