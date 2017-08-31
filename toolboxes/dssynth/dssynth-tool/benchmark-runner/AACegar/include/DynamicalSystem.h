//Authors: Dario Cattaruzza, Alessandro Abate, Peter Schrammel, Daniel Kroening
//University of Oxford 2016
//This code is supplied under the BSD license agreement (see license.txt)

#ifndef DYNAMICAL_SYSTEM_H
#define DYNAMICAL_SYSTEM_H

#include "AbstractMatrix.h"
#include "EigenPolyhedra.h"
#include <memory>
#include <map>
#include <stdlib.h>

namespace abstract
{

typedef enum
{
    eStateReachTube         = 0x01,
    eInputReachTube         = 0x02,
    eAbstractReachTube      = 0x04,
    eCombinedReachTube      = 0x08,
    eAbstractInputReachTube = 0x06,
    eAbstractStateReachTube = 0x05,
    eFullReachTube          = 0x0F,
    eEigenReach             = 0x10,
    eNormalisedReachsets    = 0x20,
    eNonDeterministic       = 0x40,
    eEigenTemplates         = 0x80
} AbstractionFlags;

typedef enum{eParamStr,eDynamicsStr,iSenseStr,oSenseStr,ioSenseStr,eGuardStr,iGuardStr,sGuardStr,oGuardStr,eInitStr,eInputStr,eTemplateStr,eARMAXStr,eControlStr,eRefStr,eMaxStr} optionStr_t;
typedef std::list<std::string> stringList;
typedef std::map<optionStr_t,std::string> optionList_t;

/// Implements the interpolation algorithms for a given profile
template <class scalar> class DynamicalSystem: public AbstractMatrix<scalar>
{
public:
    using typename JordanMatrix<scalar>::refScalar;
    using typename JordanMatrix<scalar>::func;
    using typename JordanMatrix<scalar>::complexS;
    using typename JordanMatrix<scalar>::MatrixS;
    using typename JordanMatrix<scalar>::MatrixC;
    using typename AccelMatrix<scalar>::powerS;

    using JordanMatrix<scalar>::m_dimension;
    using JordanMatrix<scalar>::m_isOne;
    using JordanMatrix<scalar>::m_hasOnes;
    using JordanMatrix<scalar>::m_hasMultiplicities;
    using JordanMatrix<scalar>::m_jordanIndex;
    using JordanMatrix<scalar>::m_conjugatePair;

    using JordanMatrix<scalar>::m_dynamics;
    using JordanMatrix<scalar>::m_eigenValues;
    using JordanMatrix<scalar>::m_pseudoEigenValues;
    using JordanMatrix<scalar>::m_pseudoEigenVectors;
    using JordanMatrix<scalar>::m_invPseudoEigenVectors;

    using JordanMatrix<scalar>::ms_one;
    using JordanMatrix<scalar>::ms_logger;
    using JordanMatrix<scalar>::ms_decoder;
    using JordanMatrix<scalar>::ms_trace_dynamics;
    using JordanMatrix<scalar>::ms_trace_time;
    using JordanMatrix<scalar>::ms_emptyMatrix;

    using AccelMatrix<scalar>::binomial;
    using AbstractMatrix<scalar>::m_inputType;
    using AbstractMatrix<scalar>::m_zeniths;
    using AbstractMatrix<scalar>::m_freq;
    using AbstractMatrix<scalar>::findZeniths;
    using AbstractMatrix<scalar>::findFrequencies;
    using AbstractMatrix<scalar>::isDivergent;
    using AbstractMatrix<scalar>::getPseudoDynamics;
    using AbstractMatrix<scalar>::getAbstractDynamics;
    using AbstractMatrix<scalar>::getSJinvS;
    using AbstractMatrix<scalar>::getMatrix;
    using AbstractMatrix<scalar>::jordanToPseudoJordan;
    using AbstractMatrix<scalar>::calculateJordanForm;
    using AbstractMatrix<scalar>::calculateMaxIterations;

    /// Constructs an empty buffer
    /// @param pParent owner of the system
    /// @param dimension dimension of the statespace
    /// @param idimension dimension of the input space
    DynamicalSystem(const int dimension=0,const int idimension=0,const int odimension=0);

    /// deletes the array of CalibrationSamples that represent the file
    /// thread safe
    ~DynamicalSystem(void);

    /// Changes the default dimension of the system
    /// @param dimension dimension of the statespace
    /// @param idimension dimension of the input space
    /// @param odimension dimension of the output space
    /// @param fdimension dimension of the feedback space
    void changeDimensions(int dimension,int idimension,int odimension,int fdimension);

    /// Loads a full dynamic description
    /// @param data buffer containing the description in the form
    /// @param pos position to start parsing on the data buffer
    /// @return position after parsing. Negative if failed
    /**
     [ g11 ... g1p h1
       gr1 ... grp hr ]
     ->
     [ a11 ... a1p
       ap1 ... app ]
     [ c11 ... c1p d1
       cm1 ... cmp dr ]
     +
     [ b11 ... b1q
       bp1 ... bpq ]
     [ e11 ... e1q f1
       en1 ... enq fn ]
     =
     [ t11 ... t1p
       tr1 ... trp]
    **/
    int load(std::string &data,size_t pos=0);

    /// Loads a full dynamic description
    /// @param guard string containing the description in the guard (G)
    /// @param dynamics string containing the description in the dynamics (A)
    /// @param init string containing the description in the initil state (X_0)
    /// @param sensitivity string containing the description in the input sensitivity matrix (B)
    /// @param inputs string containing the description in the input (U)
    /// @param templates string containing the description in the templates
    /// @param vertices indicates whether polyhedral descriptions are vertices [or inequalities]
    int load(std::string &guard,std::string &dynamics,std::string &init,std::string &sensitivity,std::string &inputs,std::string &templates,bool vertices);

    /// Loads a system from known structures
    bool load(const MatrixS &dynamics,const MatrixS &sensitivity,AbstractPolyhedra<scalar> &guard,AbstractPolyhedra<scalar> &init,AbstractPolyhedra<scalar> &inputs,AbstractPolyhedra<scalar> &safeSpace,AbstractPolyhedra<scalar> *pOutputs=NULL);

    /// Loads a system description from file (file should contain data as described above)
    /// @param fileName name of the file to load the data from
    int loadFromFile(std::string &fileName);

    /// returns a description of the processed system
    /// @return a string describing the analysis in the following format
    /**
     [ g_11 ... g_1p h_1
       g_r1 ... g_rp h_r ]
     =
     [ g'_11 ... g'_1p h'_1
       g'_r1 ... g'_rp h'_r ]
     [ r_11 ... r_1p i_11 ... i_1p
       r_p1 ... r_pp i_p1 ... i_pp]
     ->
     [ a_11 ... a_1p
       a_p1 ... a_pp ]
     =
     [ s_11 ... s_1p
       s_p1 ... s_pp ]
     [ j_11 ... j_1p
       j_p1 ... j_pp ]
     [ s^-1_11 ... s^-1_1p
       s^-1_p1 ... s^-1_pp ]
     =
     [ a'11 ... a'1p
       a'p1 ... a'pp ]

     [ c11 ... c1p d1
       cm1 ... cmp dr ]
     +
     [ u'_11 ... u'_1q
       u'_p1 ... u'_pq ]
     =
     [ I-A^-1_11 ... I-A^-1_1p
       I-A^-1_p1 ... I-A^-1_pp ]
     [ b_11 ... b_1q
       b_p1 ... b_pq ]
     [ e_11 ... e_1q f_1
       e_n1 ... e_nq fn ]
    **/
    std::string getDescription(displayType_t displayType=eInequalities,space_t space=eNormalSpace,bool interval=false,bool useBrackets=true);

    /// Saves a description of the analysis onto a file
    bool save(displayType_t displayType=eInequalities,space_t space=eNormalSpace,bool interval=false,bool append=false);

    /// Retrieves the number of non-correlated dimensions in the eigenspace
    int getRoundedDimension();

    /// Retrieves the number of radii dimensions in the eigenspace (only one per conjugate pair)
    int getNormedDimension();

    /// Retrieves the corresponding circular vectors for a given set of vectors
    /// @param result container for the resulting directions
    /// @param vectors origianl directions to transform
    /// @param transposed indicates if the supplied vectors are row oriented
    MatrixS& getRoundedDirections(MatrixS &result,const MatrixS &vectors,bool transposed=false);

    /// Retrieves the corresponding radial vectors for a given set of vectors
    /// @param result container for the resulting directions
    /// @param vectors origianl directions to transform
    /// @param transposed indicates if the supplied vectors are row oriented
    MatrixS& getNormedDirections(MatrixS &result,const MatrixS &vectors,bool transposed=false);

    /// Creates a template matrix for the given logahedral dimension
    /// @param precision order of the logahedral templates
    /// @param space space in which the templates are orthogonal
    /// @param dimension subdimension of the templates
    MatrixS& makeLogahedralTemplates(int precision,space_t space,int dimension,MatrixS &logTemplates);
    MatrixS& makeLogahedralTemplates(int precision,space_t space)
    { return makeLogahedralTemplates(precision,space,m_dimension,m_logTemplates); }

    /// Creates a template matrix for the reach set directions at the given iteration
    /// @param iteration i
    MatrixS& makeInverseTemplates(int iteration,space_t space);

    /// Creates a template matrix for the given logahedral dimension
    /// @param precision order of the logahedral templates
    /// @param dimension subdimension of the templates
    MatrixS& makeSphericalTemplates(int precision,int dimension,MatrixS &logTemplates,bool normalized);

    /// Calculates the number of iterations necessary to supersede the guard
    /// @param pInit initial polyhedra to apply the dynamics on
    /// @param inputType type of inputs to use
    powerS calculateIterations(AbstractPolyhedra<scalar> &init,inputType_t inputType);

    /// Calculates the number of iterations necessary for a point to supersede a specific guard
    powerS calculateDivergentIterations(const MatrixS& direction,const MatrixS& point,const MatrixS& inpoint,const scalar &guard,const scalar &inguard,const MatrixS &normedLambdas,const refScalar logSigma,const inputType_t inputType);

    /// Calculates the number of iterations necessary to supersede the guard
    /// @param pInit initial polyhedra to apply the dynamics on
    /// @param inputType type of inputs to use
    powerS calculateDivergentIterations(AbstractPolyhedra<scalar> &init,inputType_t inputType);

    /// Calculates the number of iterations necessary for a point to supersede a specific guard
    powerS calculateNormedIterations(const MatrixS& point,const scalar &guard,const scalar &inguard,const MatrixS &normedLambdas,const refScalar logSigma,const inputType_t inputType);

    /// Calculates the number of iterations necessary to supersede the guard
    /// @param pInit initial polyhedra to apply the dynamics on
    /// @param inputType type of inputs to use
    powerS calculateNormedIterations(AbstractPolyhedra<scalar> &init,inputType_t inputType);

    /// Retrieves the reach set at the given iteration
    /// @param iteration the n power of A for the iteration
    /// @param init start state for the reach set
    /// @param inputType indicates what type of input analysis to do
    /// @param inEigenSpace indicates if the result should be calculated in the eigenspace
    /// @param direction logahedral power of the template directions for the result
    AbstractPolyhedra<scalar>& getReachSet(powerS iteration,AbstractPolyhedra<scalar> &init,inputType_t inputType=eParametricInputs,bool accelerated=false,bool inEigenSpace=false,int directions=0,bool retemplate=true,bool keep=true);

    /// Retrieves an abstractreachtube projected across a guard
    void getGuardedReachTube(AbstractPolyhedra<scalar>& reachTube,space_t space);

    /// Retrieves the abstract reach tube at the given iteration
    /// @param iteration the n power of A for the iteration
    /// @param precision order of faces for the abstract matrix (faces=p^(precision+1)
    /// @param direction logahedral power of the template directions for the result
    /// @param inputType indicates what type of input analysis to do
    /// @param space indicates if the result should be given in eigenspace or normalspace
    AbstractPolyhedra<scalar>& getAbstractReachTube(powerS iteration,int precision=2,int directions=0,inputType_t inputType=eParametricInputs,space_t space=eNormalSpace,bool guarded=false);

    /// Retrieves the reach tube at the given iteration
    /// @param iteration the n power of A for the iteration
    /// @param inputType indicates what type of input analysis to do
    /// @param space indicates if the result should be given in eigenspace or normalspace
    /// @param direction logahedral power of the template directions for the result
    AbstractPolyhedra<scalar>& getIterativeReachTube(powerS iteration,inputType_t inputType=eParametricInputs,space_t space=eNormalSpace,int directions=0);

    /// Retrieves the output reach tube given a state-space reach tube
    /// @param reachTube state-space reach tube
    /// @param precision order of faces for the abstract matrix (faces=p^(precision+1)
    AbstractPolyhedra<scalar>& getOutputReachTube(AbstractPolyhedra<scalar>& reachTube, int precision);

    /// Indicates if there is a face of the abstract reach tube that is not an overapproximation of the reach tube
    /// @return the line number of the support breaking the overapproximation. -1 if none.
    int findBrokenOverapproximation();

    /// Sets a number of template directions for polyhedra representation
    /// @param data matrix description of the column vector directions
    /// @return position after parsing, negative if failed
    int loadTemplates(const std::string &data,size_t pos=0);

    /// Loads a polyhedral description for the safety region
    /// (if no supports are provided loads the template directions for the reach tube)
    /// @param data buffer containing the polyhedra description in the form
    /**
     g_11,g_12, ... ,g_1p, h_1
     g_21,g_22, ... ,g_2p, h_2
     ...
     g_r1,g_r2, ... ,g_rp, h_r
     **/
    /// @param pos position to start parsing from
    /// @param vertices indicates if the description is vertex or inequality oriented
    /// @return position after parsing, negative if failed
    int loadSafeReachTube(const std::string &data,size_t pos=0,bool vertices=false);

    /// Loads a polyhedral description for the guard
    /// @param data buffer containing the polyhedra description in the form
    /**
     g_11,g_12, ... ,g_1p, h_1
     g_21,g_22, ... ,g_2p, h_2
     ...
     g_r1,g_r2, ... ,g_rp, h_r
     **/
    /// @param pos position to start parsing from
    /// @param vertices indicates if the description is vertex or inequality oriented
    /// @return position after parsing, negative if failed
    int loadGuard(const std::string &data,size_t pos=0,bool vertices=false);

    /// Loads a polyhedral description for the output guard
    /// @param data buffer containing the polyhedra description in the form
    /**
     g_11,g_12, ... ,g_1p, h_1
     g_21,g_22, ... ,g_2p, h_2
     ...
     g_r1,g_r2, ... ,g_rp, h_r
     **/
    /// @param pos position to start parsing from
    /// @param vertices indicates if the description is vertex or inequality oriented
    /// @return position after parsing, negative if failed
    int loadOutputGuard(const std::string &data,size_t pos=0,bool vertices=false);

    /// Loads a polyhedral description for the initial state
    /// @param data buffer containing the polyhedra description
    /// @param pos position to start parsing from
    /// @param vertices indicates if the description is vertex or inequality oriented
    /// @return position after parsing, negative if failed 
    int loadInitialState(const std::string &data,size_t pos=0,bool vertices=false);

    /// Loads a polyhedral description for the inputs
    /// @param data buffer containing the polyhedra description
    /// @param pos position to start parsing from
    /// @param vertices indicates if the description is vertex or inequality oriented
    /// @return position after parsing, negative if failed
    int loadInputs(const std::string &data,size_t pos=0,bool vertices=false);

    /// Processes the transformed inputs
    void processInputs();

    /// Loads the sensitivity matrix for the inputs
    /// @param data buffer containing the matrix description in the form
    /**
     a_11,a_12, ... ,a_1q
     a_21,a_22, ... ,a_2q
     ...
     a_p1,a_p2, ... ,a_pq
     **/
    /// @return position after parsing, negative if failed
    int loadSensitivities(std::string &data,size_t pos=0);

    /// Loads the sensitivity matrix for the outputs
    /// @param data buffer containing the matrix description in the form
    /**
     a_11,a_12, ... ,a_1q
     a_21,a_22, ... ,a_2q
     ...
     a_p1,a_p2, ... ,a_pq
     **/
    /// @return position after parsing, negative if failed
    int loadOutputSensitivities(std::string &data,size_t pos=0);

    /// Retrieve the template directions, if available, in the corresponding space
    /// @param space space in which the templates are given
    /// @param precision if greater than -2 indicates to supply logahedral faces when no templates are available
    MatrixS& getTemplates(const space_t space=eNormalSpace,int precision=-2);

    /// Retrieves the parametric input vertices for the current problem configuration
    const MatrixS& getInputVertices(space_t space=eEigenSpace,bool fromSource=false);

    long long int getLoadTime()         { return m_loadTime; }
    long long int getReachTime()        { return m_reachTime; }
protected:

    /// Creates a combinatorial matrix A-B
    MatrixS combineAminB(const MatrixS& A,const MatrixS& B,bool isVector=true);

    /// Creates a combinatorial matrix with rows/cols A_i B_j ordered by templates
    MatrixS combineAB(const MatrixS& A,const MatrixS& B,const int templateSz,const int AvertexSz,const int BvertexSz,const bool isVector);

    /// Creates a combinatorial matrix with rows A_i \cdot B_i
    MatrixS combineAB(const MatrixS& A,const MatrixS& B);

    /// retrieve the kronecker product of A and B
    MatrixS kronecker(const MatrixS& A,const MatrixS& B,const bool transA=false,const bool transB=false);

    /// Retrieves the abstract vector set for a given template set
    MatrixS getCombinedVectors(MatrixS &vectors,const MatrixS& templates,AbstractPolyhedra<scalar> &inputs,const MatrixS &inputVertices);

    /// Retrieves the abstract vector set for a given template set
    MatrixS& getAbstractVertices(const MatrixS& templates,MatrixS &vectors,int &numVertices);
    MatrixS& getAbstractVertices(const MatrixS& templates)
    { return getAbstractVertices(templates,m_abstractVertices,m_numVertices); }

    /// Retrieves the parametric accelerated input vertices for the current problem configuration
    MatrixS& getAccelVertices();

    /// Calculates the parametric input supports for the given abstraction
    MatrixS& getAccelInSupports(powerS iteration, int precision,const MatrixS& templates);

    /// Merges the supports from the abstract vertices into the corresponding output template direction
    void mergeAbstractSupports(const MatrixS &templates,MatrixS &supports);

    /// Retrieves the support set for the inputs
    void mergeAccelInSupports(MatrixS &supports,int numTemplates);

    /// Records a full description of the supports and input supports
    void traceSupports(const MatrixS &templates,MatrixS &supports,AbstractPolyhedra<scalar>& dynamics,const MatrixS &vectors);

    /// Sets the eigenspace of the polyhedral representations
    void setEigenSpace();

    /// Processes a calculation error either executing a throw or tagging an imprecision
    void processError(std::string source);
public:
    /// Loads the transformation matrix for the state space
    /// @param data buffer containing the matrix description in the form
    /**
     a_11,a_12, ... ,a_1p
     a_21,a_22, ... ,a_2p
     ...
     a_p1,a_p2, ... ,a_pp
     **/
    /// @return true if succesful
    virtual int loadDynamics(const std::string &data,size_t pos=0);

    /// Loads the LTI dynamics of an armax model
    /// @param data buffer containing the armax description in the form
    /// a_1,a_2, ... ,a_p;b_1,b_2, ... ,b_q
    /// @return position after parsing, negative if failed
    int loadARMAXModel(std::string &data,size_t pos=0);

    /// Centralizes the inputs in eigenspace
    void centralizeInputs()                                                 { m_transformedInputs.centralize(true); }

    /// Returns a pointer to the guard polyhedra (G)
    AbstractPolyhedra<scalar>& getGuard(space_t space=eNormalSpace)         { return m_guard.getPolyhedra(space,m_conjugatePair,m_jordanIndex); }

    /// Returns a pointer to the initial state polyhedra (X_0)
    AbstractPolyhedra<scalar>& getInitialState(space_t space=eNormalSpace)  { return m_initialState.getPolyhedra(space,m_conjugatePair,m_jordanIndex); }

    /// Returns a pointer to the input polyhedra (U)
    AbstractPolyhedra<scalar> &getInputs()          { return m_inputs; }

    /// Returns a pointer to the input polyhedra (U)
    AbstractPolyhedra<scalar> &getOutputs()          { return m_outputs; }

    /// Retrieves the input multiplier matrix (B)
    MatrixS &getInputDynamics()                     { return m_sensitivity; }

    /// Retrieves the input multiplier matrix (B)
    MatrixS &getOutputDynamics()                    { return m_outputSensitivity; }

    /// Retrieves the input multiplier matrix (B)
    std::string getInputDynamicsDesc()              { return getMatrix(m_sensitivity,false); }

    /// Retrieves the input multiplier matrix (B)
    std::string getOutputDynamicsDesc()             { return getMatrix(m_outputSensitivity,false); }

    EigenPolyhedra<scalar> &getGuardPoly()          { return m_guard; }
    EigenPolyhedra<scalar> &getInitPoly()           { return m_initialState; }
    EigenPolyhedra<scalar> &getInputPoly()          { return m_transformedInputs; }
    EigenPolyhedra<scalar> &getAccelInputPoly()     { return m_transformedStateInputs; }
    EigenPolyhedra<scalar> &getReachPoly()          { return *m_pReachSet; }
    EigenPolyhedra<scalar> &getTubePoly()           { return *m_pReachTube; }
    EigenPolyhedra<scalar> &getAbsTubePoly()        { return *m_pAbstractReachTube; }

    /// Processes a set of problems
    virtual void processFiles(stringList &files,displayType_t displayType=eInequalities,space_t space=eNormalSpace,bool interval=false,optionList_t &options=optionList_t());

    /// Processes/modifies a problem stated by the input options
    virtual int processOptions(optionList_t &options,displayType_t displayType=eInequalities,space_t space=eNormalSpace,bool interval=false,bool run=false);

    // Processes a problems with varying parameters
    virtual bool process(const displayType_t displayType=eInequalities,const space_t space=eNormalSpace,const bool interval=false,const bool append=false);

    // Defines a name for the system
    void setName(const std::string name);

    // Sets the input type for reachability analysis
    void setInputType(inputType_t inputType)        { m_inputType=inputType; }

    // Retrieves the parmeter matrix
    ParamMatrix& getParams()                        { return m_paramValues; }

    // Sets the system parameters
    void setParams(ParamMatrix& params);

    static void traceDynamics(const traceDynamics_t traceType);
    static void traceSimplex(const traceTableau_t traceTableau,const traceVertices_t traceVertices);

protected:
    int                         m_loadTime;
    int                         m_reachTime;
    int                         m_idimension;   // dimension of the input space
    int                         m_fdimension;   // dimension of the feedback space
    int                         m_odimension;   // dimension of the output space
    int                         m_numVertices;  // combined number of vertices (init x param input)
    AbstractPolyhedra<scalar>   m_inputs;
    AbstractPolyhedra<scalar>   m_reference;
    AbstractPolyhedra<scalar>   m_outputs;
    AbstractPolyhedra<scalar>   m_reach;
    AbstractPolyhedra<scalar>   m_subReach;
    EigenPolyhedra<scalar>      m_initialState;
    EigenPolyhedra<scalar>      m_transformedInputs;
    EigenPolyhedra<scalar>      m_transformedStateInputs;
    AbstractPolyhedra<scalar>*  m_pTransformedRoundInputs;

    EigenPolyhedra<scalar>      m_guard;
    EigenPolyhedra<scalar>      m_safeReachTube;
    AbstractPolyhedra<scalar>   m_outputGuard;

    EigenPolyhedra<scalar>     *m_pReachSet;
    EigenPolyhedra<scalar>     *m_pReachTube;
    EigenPolyhedra<scalar>     *m_pAbstractReachTube;


    MatrixS                     m_templates;
    MatrixS                     m_eigenTemplates;
    MatrixS                     m_logTemplates;
    MatrixS                     m_sensitivity;
    MatrixS                     m_outputSensitivity;
    MatrixS                     m_feedback;
    MatrixS                     m_abstractVertices;
    MatrixS                     m_abstractInputVertices;
    MatrixS                     m_accelVertices;
    MatrixS                     m_accelInSupports;
    std::string                 m_name;
    std::string                 m_source;
    ParamMatrix                 m_paramValues;
    MatrixS                     m_dynamicParams;
    std::vector<int>            m_normedJordanIndex;
    std::vector<bool>           m_normedOnes;
public:
    static bool                 ms_fullAnswers;
};

}

#endif //DYNAMICAL_SYSTEM_H
