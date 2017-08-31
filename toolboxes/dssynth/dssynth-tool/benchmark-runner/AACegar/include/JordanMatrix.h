//Authors: Dario Cattaruzza, Alessandro Abate, Peter Schrammel, Daniel Kroening
//University of Oxford 2016
//This code is supplied under the BSD license agreement (see license.txt)

#ifndef JORDAN_MATRIX_H
#define JORDAN_MATRIX_H

#include <limits>

#include "JordanSolver.h"
#include "MatrixToString.h"

namespace abstract
{

/// Calculates the Jordan and SVD decomposition of a matrix
template <class scalar> class  JordanMatrix
{
public:
    typedef enum {eToEigenValues,eToEigenVectors,eToInvEigenVectors} pseudoType_t;
    typedef interval_def<scalar> type;
    typedef typename type::scalar_type refScalar;
    typedef typename type::complexS complexS;
    typedef typename type::MatrixS  MatrixS;
    typedef typename type::MatrixC  MatrixC;
    typedef typename type::complexR complexR;
    typedef typename type::MatrixR  MatrixR;
    typedef typename type::MatrixRC MatrixRC;
    typedef functions<refScalar> func;
    typedef typename interval_def<scalar>::power_type powerS;

    typedef refScalar /*mpfr::mpreal*/ solverScalar;
    typedef typename interval_def<solverScalar>::MatrixS SolverMatrixType;
    typedef typename interval_def<solverScalar>::MatrixC SolverComplexMatrixType;

    /// Constructs an empty buffer
    /// @param dimension dimension of the matrix
    JordanMatrix(const int dimension);

    /// Changes the default dimension of the system
    /// @param dimension dimension of the statespace
    virtual void changeDimensions(const int dimension);

    /// Loads the t matrix from a string
    /// @param data buffer containing the matrix description in the form
    /**
     a_11,a_12, ... ,a_1p
     a_21,a_22, ... ,a_2p
     ...
     a_p1,a_p2, ... ,a_pp
     **/
    /// @return true if succesful
    virtual int load(const std::string &data,size_t pos=0);

    bool load(const MatrixS &matrix);

    // Loads a matrix assumed to be a canonical Jordan form
    virtual bool loadJordan(const MatrixS &matrix);

    bool loadFromRef(const MatrixR &matrix);
protected:
    /// calculates the estimated roundoff error of a matrix operation
    template <class MatrixType> refScalar calculateEpsilon(const MatrixType &matrix);

    /// Calculates the singular values for each Jordan Block
    void calculateBlockSVD();

    /// Calculates the singular values
    bool calculateSVD();

    /// Returns the schur decomposition of the dynamics
    std::string getSJinvS();

    virtual bool calculateJordanForm(bool includeSvd=true);

    /// Calculates a lower bound for the minimum separation between any two jordan blocks
    refScalar calculateMinSeparation();

    /// Calculates the maximum error for the numerical approximation of the eigenvalues
    /// @return the maximum variation of the eigenvalues
    scalar calculateEigenError();

    /// Calculates the maximum error for the numerical approximation of the matrix to the nth power
    /// @return the maximum variation of the eigenvalues
    scalar calculateBoundedEigenError(scalar iteration);

    /// Retrieves an equivalent real Jordan from a complex representation
    /// @param source matrix to be converted
    /// @param rot indicates if the complex conjugates are a diagonal rotation or a complex vector column
    MatrixS jordanToPseudoJordan(const MatrixC &source,const pseudoType_t conversionType);

    /// Retrieves an equivalent complex Jordan from a real representation
    MatrixC pseudoToJordan(const MatrixS &source,const pseudoType_t conversionType);

    /// Retrieves a scalar matrix from a refScalar one
    /// @param dest reference matrix to fill
    /// @param source interval matrix containing the data
    void interToRef(SolverMatrixType &dest,const MatrixS &source);

    /// Retrieves a scalar matrix from a refScalar one
    /// @param dest interval matrix to fill
    /// @param source matrix containing the data
    void refToInter(MatrixC &dest,const SolverComplexMatrixType &source);

    /// Retrieves a scalar matrix from a refScalar one
    /// @param dest interval matrix to fill
    /// @param source matrix containing the data
    void refToInter(MatrixS &dest,const SolverMatrixType &source);

    /// Transforms the matrix to Reduced Row Echelon Form
    int toRREF(MatrixC &matrix);

public:
    /// Retrieves the pseudoinverse of the dynamics
    MatrixS getPseudoInverse(const MatrixS &matrix,bool &hasInverse);

    /// Retrieves the pseudoinverse of the dynamics using svd decomposition
    ///
    MatrixS getSVDpseudoInverse(const MatrixS &matrix,bool &hasInverse);

    /// Returns the description of a complex matrix
    std::string getMatrix(const MatrixC &matrix,const bool brackets=true);

    /// Returns the description of a matrix
    std::string getMatrix(const MatrixS &matrix,const bool brackets=true);

    /// Retrieves the dynamics matrix (A)
    MatrixS& getBasicDynamics()                             { return m_dynamics; }

    /// Retrieves the dynamics matrix (A)
    std::string getBasicDynamicsDesc()                      { return getMatrix(m_dynamics,false); }

    /// Returns the complex eigenvector matrix (S)
    MatrixC& getEigenVectors()                              { return m_eigenVectors; }

    /// Returns the real pseudoeigenvector matrix (S)
    MatrixS& getPseudoEigenVectors()                        { return m_pseudoEigenVectors; }

    /// Returns the complex eigenvector matrix (S)
    MatrixC& getInvEigenVectors()                           { return m_invEigenVectors; }

    /// Returns the real pseudoeigenvector matrix (S)
    MatrixS& getInvPseudoEigenVectors()                     { return m_invPseudoEigenVectors; }

    /// Returns the complex eigenvector matrix (S)
    std::string getEigenVectorsDesc(bool pseudo=false);

    /// Returns the complex eigenvalue matrix (J)
    MatrixC& getEigenValues()                               { return m_eigenValues; }

    /// Returns the complex eigenvalue matrix (J)
    MatrixS& getPseudoEigenValues()                         { return m_pseudoEigenValues; }

    /// Returns the complex singular value matrix (F)
    MatrixS& getSingularValues()                            { return m_blockSingularValues; }

    /// Returns system dynamics in the given space
    MatrixS& getBaseDynamics(const space_t space)           { return (space==eNormalSpace) ? m_dynamics : m_pseudoEigenValues; }

    /// Returns the singular values of the matrix
    std::string getSingularValuesDesc();

    /// Returns the inverse complex eigenvector matrix (S^-1)
    std::string getInvEigenVectorsDesc(bool pseudo=false);

    double getError()                                       { return func::toDouble(func::toUpper(m_error)); }

protected:
    int                            m_dimension;
    refScalar                      m_zero;
    refScalar                      m_largeZero;
    MatrixS                        m_dynamics;
    SolverMatrixType               m_refDynamics;
    JordanSolver<solverScalar>     m_eigenSpace;
    Eigen::JacobiSVD<MatrixR>      m_svdSpace;
    MatrixC                        m_eigenValues;
    MatrixC                        m_eigenVectors;
    MatrixC                        m_invEigenVectors;
    MatrixS                        m_pseudoEigenValues;
    MatrixS                        m_eigenNorms;
    MatrixS                        m_blockSingularValues;
    MatrixS                        m_pseudoEigenVectors;              // column list of pseudo eigenvectors
    MatrixS                        m_invPseudoEigenVectors;           // column list of inverse pseudo eigenvectors
    MatrixS                        m_transposeInvPseudoEigenVectors;  // column list of pseudo eigenvectors of A^-T
    std::vector<int>               m_jordanIndex;
    std::vector<int>               m_conjugatePair;
    std::vector<bool>              m_isOne;
    bool                           m_hasOnes;
    bool                           m_hasMultiplicities;
    scalar                         m_error;
    refScalar                      m_boundForError;
    refScalar                      m_maxSigma;
    refScalar                      m_minSigma;
    MatrixR                        m_minSeparation;
    static scalar                  ms_half;
    static scalar                  ms_one;
    static scalar                  ms_two;
    static complexS                ms_complexOne;
    static MatToStr<scalar>        ms_logger;
    static MatToStr<scalar>        ms_decoder;
    static MatrixS                 ms_emptyMatrix;
public:
    int                            m_jordanTime;
public:
    static traceDynamics_t         ms_trace_dynamics;
    static bool                    ms_trace_time;
};

}

#endif
