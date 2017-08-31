//Authors: Dario Cattaruzza, Alessandro Abate, Peter Schrammel, Daniel Kroening
//University of Oxford 2016
//This code is supplied under the BSD license agreement (see license.txt)

#ifndef ACCEL_MATRIX_H
#define ACCEL_MATRIX_H

#include "JordanMatrix.h"

namespace abstract
{

/// Implements the interpolation algorithms for a given profile
template <class scalar> class AccelMatrix : public JordanMatrix<scalar>
{
public:
    using typename JordanMatrix<scalar>::refScalar;
    using typename JordanMatrix<scalar>::func;
    using typename JordanMatrix<scalar>::complexS;
    using typename JordanMatrix<scalar>::MatrixS;
    using typename JordanMatrix<scalar>::MatrixC;
    typedef typename interval_def<scalar>::power_type powerS;
    using JordanMatrix<scalar>::eToEigenValues;
    using JordanMatrix<scalar>::eToEigenVectors;
    using JordanMatrix<scalar>::eToInvEigenVectors;

    using JordanMatrix<scalar>::m_dimension;
    using JordanMatrix<scalar>::m_jordanIndex;
    using JordanMatrix<scalar>::m_conjugatePair;
    using JordanMatrix<scalar>::m_eigenValues;
    using JordanMatrix<scalar>::m_eigenVectors;
    using JordanMatrix<scalar>::m_invEigenVectors;
    using JordanMatrix<scalar>::m_pseudoEigenVectors;
    using JordanMatrix<scalar>::m_invPseudoEigenVectors;
    using JordanMatrix<scalar>::m_isOne;
    using JordanMatrix<scalar>::m_hasOnes;

    using JordanMatrix<scalar>::ms_one;
    using JordanMatrix<scalar>::ms_logger;
    using JordanMatrix<scalar>::ms_decoder;
    using JordanMatrix<scalar>::ms_trace_dynamics;
    using JordanMatrix<scalar>::ms_trace_time;

    using JordanMatrix<scalar>::calculateBlockSVD;
    using JordanMatrix<scalar>::jordanToPseudoJordan;

    /// Constructs an empty buffer
    /// @param dimension dimension of the statespace
    AccelMatrix(int dimension);

    /// retrieves J^n
    MatrixC getPoweredEigenValues(const powerS iteration);

    /// retrieves F^n
    MatrixS getPoweredSingularValues(const powerS iteration);

    /// retrieves a real pseudoDiagonal of J^n
    MatrixS getPoweredPseudoEigenValues(const powerS iteration);

    /// retrieves (I-J)^-1
    MatrixC& getInvIminJ()         { return m_invIminJ; }

    /// retrieves a real pseudoDiagonal of (I-J)^-1
    MatrixS& getPseudoInvIminJ()   { return m_pseudoInvIminJ; }

    /// retrieves the acceleration factor (I-A)^-1
    MatrixS& getInvIminA()         { return m_invIminA; }

    /// retrieves the acceleration factor (I-F)^-1
    MatrixS& getInvIminF()         { return m_invIminF; }

    /// retrieves the dynamics matrix for a given iteration (n)
    /// @param iteration number of iterations since the initial state
    /// @param input indicates whether we want state (A^n) or input (I-A^n/I-A) dynamics
    /// @param inEigenSpace retrieves (J^nS^-1) or (I-J^n/I-J)S^-1 rather than their A^n equivalents
    MatrixC getDynamics(const powerS iteration,const bool input=false,const space_t space=eNormalSpace);

    /// retrieves the dynamics matrix for a given iteration (n) using pseudo eigenvalues
    /// @param iteration number of iterations since the initial state
    /// @param input indicates whether we want state (A^n) or input (I-A^n/I-A) dynamics
    /// @param space Indicates whther to use A(normal), J(eigen) or E(singular) as the dynamics
    MatrixS getPseudoDynamics(const powerS iteration,const bool input=false,const space_t space=eNormalSpace);

    /// retrieves the dynamics matrix for a given iteration (n) using pseudo eigenvalues
    /// @param iteration number of iterations since the initial state
    /// @param input indicates whether we want state (A^n) or input (I-A^n/I-A) dynamics
    /// @param inEigenSpace retrieves (J^nS^-1) or (I-J^n/I-J)S^-1 rather than their A^n equivalents
    MatrixS getRoundDynamics(const powerS iteration,const bool input=false,const bool inEigenSpace=false);

    /// Loads a matrix assumed to be a canonical Jordan form
    bool loadJordan(const MatrixS &matrix);
protected:
    scalar binomial(powerS n,int k);

    /// Calculates (I-J)^-1
    void calculateInvIminJ();

    /// Calculates (I-F)^-1
    void calculateInvIminF();

    /// Calculates the folded matrix over-approximation on the rotation and dilation axes
    void calculateF();

    /// Calculates (I-J^n)
    MatrixC calculateIminJn(const powerS iteration);

    /// Calculates (I-F^n)
    MatrixS calculateIminFn(const powerS iteration);

    virtual bool calculateJordanForm(bool includeSvd=true);

public:
    /// Returns the symbolic description of the series convergence
    std::string getIminAnInvIMinA(const bool full);

protected:
    MatrixS             m_foldedEigenValues;
    MatrixS             m_binomialMultipliers;
    MatrixS             m_foldedBinomialMultipliers;
    MatrixC             m_invIminJ;
    MatrixS             m_pseudoIminJ;
    MatrixS             m_pseudoInvIminJ;
    MatrixS             m_IminA;
    MatrixS             m_invIminA;
    MatrixS             m_IminF;
    MatrixS             m_invIminF;
    std::vector<int>    m_svnIndex;

};

}

#endif
