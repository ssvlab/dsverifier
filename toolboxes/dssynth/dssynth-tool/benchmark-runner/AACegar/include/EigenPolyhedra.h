//Authors: Dario Cattaruzza, Alessandro Abate, Peter Schrammel, Daniel Kroening
//University of Oxford 2016
//This code is supplied under the BSD license agreement (see license.txt)

#ifndef EIGEN_POLYHEDRA_H
#define EIGEN_POLYHEDRA_H

#include <memory>
#include "AbstractPolyhedra.h"

namespace abstract
{

template <class scalar> class EigenPolyhedra
{
public:
    typedef typename Tableau<scalar>::refScalar refScalar;
    typedef typename Tableau<scalar>::MatrixS MatrixS;

    /// Constructs transformed polyhedra
    /// @param name id of the polyhedra
    /// @param dimension dimension of the polyhedra
    /// @param pSource template to copy to create this polyhedra
    /// @param pTransform transformation matrix to obtain this polyhedra
    /// @param pInverse pTransform->inverse()
    EigenPolyhedra(const std::string name="",const int dimension=0);
    EigenPolyhedra(const std::string name,const int dimension,AbstractPolyhedra<scalar>& source,MatrixS& transform=Polyhedra<scalar>::ms_emptyMatrix,MatrixS& inverse=Polyhedra<scalar>::ms_emptyMatrix);

    /// Destructor
    virtual ~EigenPolyhedra();

    /// Changes the dimensionality of the polyhedra
    /// @param dimension new dimensionality
    void changeDimension(const int dimension);

    /// Loads a polyhedral description from a string
    /// @param data buffer containing the polyhedra description in the form
    /// @param pos position in the string to start parsing
    /// @param vertices indicates if the data loaded are vertices as opposed to faces
    /**
     c_11,c_12, ... ,c_1p, d_1
     c_21,c_22, ... ,c_2p, d_2
     ...
     c_m1,c_m2, ... ,c_mp, d_m
     **/
    /// @return true if succesful
    int load(const std::string &data,size_t pos,const bool vertices=false);

    /// Loads a known polyhedra
    /// @param source template to copy
    /// @param transform transformation matrix for the polyhedra
    /// @param inverse inverse of transform (if known, saves processing time)
    /// @param space indicates whether the loaded source is in eigenSpace or normalSpace
    AbstractPolyhedra<scalar>& load(AbstractPolyhedra<scalar> &source,const MatrixS &transform=Polyhedra<scalar>::ms_emptyMatrix,const MatrixS &inverse=Polyhedra<scalar>::ms_emptyMatrix,space_t space=eNormalSpace);

    /// Loads a polyhedral description from file
    /// @param fileName name of the file to load the data from
    /// @return true if succesful
    bool loadFromFile(const std::string &fileName);

    /// Sets the eigenvector transformation matrix
    /// @param transform matrix that transforms the polyhedra into eigenspace
    /// @param inverse matrix that transforms the polyhedra from eigenspace
    void setEigenSpace(MatrixS &transform,MatrixS &inverse);

    /// Loads a polyhedral description from a set of matrices
    /// @param faces directions of the face normal
    /// @param supports support values for each face direction
    /// @param space space to load the description on
    /// @return pointer to the loaded polyhedra
    AbstractPolyhedra<scalar>& load(MatrixS &faces,MatrixS &supports,space_t space);

    /// Loads a polyhedral description from a set of matrices
    /// @param source original polyhedra to merge the faces with
    /// @param faces directions of the face normal
    /// @param supports support values for each face direction
    /// @param space space to load the description on
    /// @return pointer to the loaded polyhedra
    AbstractPolyhedra<scalar>& mergeLoad(AbstractPolyhedra<scalar> &source,MatrixS &faces,MatrixS &supports,space_t space);

    /// Removes any existing faces
    void clear();

    /// Indicates if the objects contains a null polyhedra
    bool isEmpty()      { return m_polyhedra.isEmpty(); }

    /// Gets the polyhedral representation in the requested space
    AbstractPolyhedra<scalar>& getPolyhedra(space_t space=eNormalSpace,bool force=false);
    AbstractPolyhedra<scalar>& getPolyhedra(space_t space,const std::vector<int> &rotations,const std::vector<int> &dilations);

    /// Retrieves a copy of this polyhedra transformed by the given matrix
    /// @param polyhedra container for the result (if NULL, a new one is created)
    /// @param transform transform matrix
    /// @param inverse inverse of transform matrix
    AbstractPolyhedra<scalar>& getTransformedPolyhedra(Polyhedra<scalar>& polyhedra,space_t space,const MatrixS& transform,const MatrixS& inverse=Polyhedra<scalar>::ms_emptyMatrix,const MatrixS &templates=Polyhedra<scalar>::ms_emptyMatrix);

    /// Centralizes the polyhedra and reflects the change in the converse space
    void centralize(const bool inEigenSpace=false);

    /// Retrieve the centre of the polyhedra
    MatrixS getCentre(const bool fromEigenSpace=false,const bool inEigenSpace=false);

    void setName(const std::string name);

    /// Returns a reference to the round polyhedra object for external modification
    /// @return pointer to the rounded eigenspace polyhedra m_pRoundPolyhedra
    AbstractPolyhedra<scalar>& getSingularPolyhedraRef()        { return m_singularPolyhedra; }

protected:
    /// Creates m_pEigenPolyhedra if not existent
    /// @return pointer to the eigenspace polyhedra m_pEigenPolyhedra
    AbstractPolyhedra<scalar>& getEigenPolyhedra(bool force=false);

    /// Creates m_pRoundPolyhedra if not existent
    /// @return pointer to the rounded eigenspace polyhedra m_pRoundPolyhedra
    AbstractPolyhedra<scalar>& getSingularPolyhedra(const std::vector<int> &rotations,const std::vector<int> &dilations);

protected:
    std::string                   m_name;
    int                           m_dimension;
    AbstractPolyhedra<scalar>     m_polyhedra;
    AbstractPolyhedra<scalar>     m_eigenPolyhedra;
    AbstractPolyhedra<scalar>     m_singularPolyhedra;
    MatrixS                      *m_pEigenVectors;
    MatrixS                      *m_pInverseEigenVectors;
    MatrixS                       m_centre;
    MatrixS                       m_eigenCentre;
};

}

#endif
