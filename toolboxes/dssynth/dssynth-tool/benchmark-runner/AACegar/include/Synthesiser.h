#ifndef SYNTHESISER_H
#define SYNTHESISER_H

#include <map>
#include "RefinedDynamicalSystem.h"

namespace abstract
{
template <class scalar> class Synthesiser : public CegarSystem<scalar>
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
    using DynamicalSystem<scalar>::getInputVertices;
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
    using DynamicalSystem<scalar>::setInputType;
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
    using DynamicalSystem<scalar>::combineAB;
    using CegarSystem<scalar>::getReachableCanonicalTransformMatrix;
    using CegarSystem<scalar>::getRefinedDynamics;
    using CegarSystem<scalar>::generateNoiseInput;
    using CegarSystem<scalar>::generateFeedbackInput;
    using CegarSystem<scalar>::refineAbstractDynamics;
    using CegarSystem<scalar>::findCounterExampleIterations;

    /// Constructs an empty buffer
    /// @param dimension dimension of the space
    Synthesiser(int dimension=0,int idimension=0);

    /// Synthetises an input vector
    AbstractPolyhedra<scalar> synthesiseInputs(inputType_t inputType,int precision,AbstractPolyhedra<scalar> &init,AbstractPolyhedra<scalar> &end,AbstractPolyhedra<scalar> &dynamics,MatrixS& templates,refScalar tightness);

    /// Maximises the vector interval
    AbstractPolyhedra<scalar> synthesiseInitialState(inputType_t inputType,AbstractPolyhedra<scalar> &input,AbstractPolyhedra<scalar> &end,AbstractPolyhedra<scalar> &dynamics);

    /// Synthetises the eigenstructure of a pole location
    AbstractPolyhedra<scalar> synthesiseEigenStructure(inputType_t inputType,int precision,int directions,AbstractPolyhedra<scalar> &end,AbstractPolyhedra<scalar> &dynamics);

    /// Synthesises a bound on the dynamics given a known guard and eigenvectors.
    AbstractPolyhedra<scalar> synthesiseDynamicBounds(inputType_t inputType,AbstractPolyhedra<scalar> &end);

    /// Retrieves the support set for the inputs
    MatrixS& getRefinedAccelInSupports();

    /// Corrects the support set by the input offset
    void demergeAccelInSupports(MatrixS &supports,MatrixS &inSupports,int numTemplates);

    /// Retrieves the reach tube given the refined dnamics
    AbstractPolyhedra<scalar>& getRefinedAbstractReachTube(space_t space,bool guarded=false);

    /// Loads a controller candidate for the system
    int loadController(const std::string &data,size_t pos=0);

    /// Loads a reference input set for the system
    int loadReference(const std::string &data,size_t pos=0,bool vertices=false);

    /// Loads and cleans up a state space synthesis result into the corresponding set
    bool loadSynthesisedResult(synthesisType_t type, AbstractPolyhedra<scalar> &result,MatrixS& templates,refScalar tightness,int time);

    /// Synthetises an input/state polyhedra given a set of conditions
    bool synthesiseAll(synthesisType_t type,powerS iteration,int precision,int directions,inputType_t inputType,space_t space,refScalar tightness);

    /// Processes a set of problems
    void processFiles(stringList &files,displayType_t displayType=eInequalities,space_t space=eNormalSpace,bool interval=false,optionList_t &options=optionList_t());

    /// Processes/modifies a problem stated by the input options
    int processOptions(optionList_t &options,displayType_t displayType=eInequalities,space_t space=eNormalSpace,bool interval=false,bool run=false);

    /// Processes a problems with varying parameters
    bool process(const displayType_t displayType=eInequalities,const space_t space=eNormalSpace,const bool interval=false,const bool append=false);

    /// Loads a pole placement description
    int loadPoles(const std::string &data,size_t pos=0);

    /// Calculates the closed loop dynamics given a plant and a controller
    bool makeClosedLoop(bool useObserver=false,bool makeReference=false,bool makeNoise=false);

    /// Retrieves closed loop the dynamics matrix (A_c)
    MatrixS& getClosedLoopDynamics()        { return m_closedLoop.getBasicDynamics(); }

    /// Returns the complex poles matrix (J_c)
    MatrixC& getComplexPoles()              { return m_closedLoop.getEigenValues(); }

    /// Returns the pole matrix (J_c)
    MatrixS& getPoles()                     { return m_closedLoop.getPseudoEigenValues(); }

    /// Returns the nullSpace vectors of U_1^T(A-\lambda_iI) where B=[U_0 U_1][Z,0]
    std::vector<MatrixS> getNullSpace(const MatrixC &eigenValues);

    /// Returns a tranformation of the eigenvector inequalities into their corresponding nullSpace coefficients
    MatrixS getNullSpaceFaces(MatrixS &faces,std::vector<MatrixS> &nullSpace);

    /// Solves the Sylvester equation AX+XB=C for X
    /// @return X
    MatrixS solveSylvester(const MatrixS &A,const MatrixS &B,const MatrixS &C,bool BisDiagonal=false);

    /// Retrieves a set of viable Left Eigenvectors closest to a desired set
    MatrixS getValidLeftEigenVectors(const MatrixS &pseudoEigenValues,const MatrixS &desired);

    /// Retrieves a set of viable Left Eigenvectors for given eigenvalues
    MatrixS getLeftEigenVectors(const MatrixS &pseudoEigenValues,AbstractPolyhedra<scalar> &eigenVectorSpace);

    /// Retrieves constraints on the controller coefficients based on the Input constraints
    AbstractPolyhedra<scalar> getControllerInBounds(AbstractPolyhedra<scalar>& reachTube);

    /// Retrieves constraints on the controller coefficients based on the Dynamic constraints
    AbstractPolyhedra<scalar> getControllerDynBounds(AbstractPolyhedra<scalar>& reachTube,int &orBlockSize);

    AbstractPolyhedra<scalar>& getAbstractClosedLoopDynamics(const inputType_t inputType)
                                            { return m_closedLoop.getAbstractDynamics(inputType); }

    ///Creates a c header (defines and data types) for CEGIS
    std::string makeCEGISHeader(bool intervals=true);

    ///Creates a c specification for CEGIS
    std::string makeCEGISDescription(bool intervals=true);

    ///Creates a list of iterations for CEGIS
    std::string makeCEGISIterations(std::string &existing);

    ///Creates a c header file for CEGIS
    bool makeCEGISFiles();

protected:
    /// Finds the statespace guard given an output guard
    AbstractPolyhedra<scalar>& calculateGuardFromOutput();
public:
    CegarSystem<scalar>     m_closedLoop;
    synthesisType_t         m_synthType;
};

}

#endif //SYNTHESISER_H
