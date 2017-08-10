//Authors: Dario Cattaruzza, Alessandro Abate, Peter Schrammel, Daniel Kroening
//University of Oxford 2016
//This code is supplied under the BSD license agreement (see license.txt)

#ifndef ABSTRACT_POLYHEDRA_H
#define ABSTRACT_POLYHEDRA_H

#include <limits>
#include <stdlib.h>

#include "Polyhedra.h"

namespace abstract
{
template <class scalar> class AbstractPolyhedra : public Polyhedra<scalar>
{
public:
    using typename Tableau<scalar>::refScalar;
    using typename Tableau<scalar>::func;
    using typename Tableau<scalar>::MatrixS;

    using Tableau<scalar>::m_faces;
    using Tableau<scalar>::m_supports;
    using Tableau<scalar>::ms_logger;
    using Polyhedra<scalar>::ms_emptyMatrix;

    using Tableau<scalar>::getDimension;
    using Polyhedra<scalar>::getVertices;
    using Polyhedra<scalar>::getName;

    /// Constructs an empty buffer
    /// @param dimension dimension of the space
    AbstractPolyhedra(int dimension=0);

    /// Constructs transformed polyhedra
    /// @param source template to copy to create this polyhedra
    /// @param transform transformation matrix to obtain this polyhedra
    /// @param inverse precalculated inverse of transform
    AbstractPolyhedra(const AbstractPolyhedra& source,const MatrixS &transform=ms_emptyMatrix,const MatrixS &inverse=ms_emptyMatrix);

    /// Retrieves the corresponding circular vectors for a given set of vectors
    /// @param rotations rotating axis to unfold
    /// @param dilation dilating axis to unfold (Jordan blocks)
    MatrixS getRoundedDirections(const MatrixS &vectors,const std::vector<int> &rotations,const std::vector<int> &dilations);

    /// Retrieves an abstraction that has circular faces along the rotating axis and jordan blocks
    /// @param rotations rotating axis to unfold
    /// @param dilation dilating axis to unfold (Jordan blocks)
    AbstractPolyhedra& getRounded(const std::vector<int> &rotations,const std::vector<int> &dilations,AbstractPolyhedra &roundAbstraction);

    /// Performs the Minkowski sum of this polyhedra to another
    /// @param polyhedra polyhedra to merge
    /// @param rotations rotating axis represented by a single vector in the added polyhedra
    /// @param dilations dilating axis represented by a single vector in the added polyhedra
    /// @return true if successful
    bool addRounded(Polyhedra<scalar> &polyhedra,const std::vector<int> &rotations,const std::vector<int> &dilations);

    /// Retrives the vertices mapped into an abstract domain of a linear matrix
    /// @param rotations rotating axis represented by a single vector in the added polyhedra
    /// @param dilations dilating axis represented by a single vector in the added polyhedra
    MatrixS getAbstractVertices(MatrixS &vector,const int row,const std::vector<int> &rotations,const std::vector<int> &dilations,const MatrixS &vertices);
    MatrixS getAbstractVertices(MatrixS &vector,const int row,const std::vector<int> &rotations,const std::vector<int> &dilations)
    { return getAbstractVertices(vector,row,rotations,dilations,getVertices()); }

    /// Retrives the vertices mapped into an abstract domain of a linear matrix
    /// @param vectors directions to merge the vertices with
    /// @param rotations rotating axis represented by a single vector in the added polyhedra
    /// @param dilations dilating axis represented by a single vector in the added polyhedra
    MatrixS getAbstractVertices(const MatrixS &vectors,const std::vector<int> &rotations,const std::vector<int> &dilations,const MatrixS &vertices);
    MatrixS getAbstractVertices(const MatrixS &vectors,const std::vector<int> &rotations,const std::vector<int> &dilations)
    { return getAbstractVertices(vectors,rotations,dilations,getVertices()); }

    MatrixS getAbstractVertices(MatrixS &vectors);

    /// Retrives the vertices mapped into an abstract domain of a linear matrix
    /// @param vectors directions to merge the vertices with
    /// @param rotations rotating axis represented by a single vector in the added polyhedra
    /// @param dilations dilating axis represented by a single vector in the added polyhedra
    MatrixS getSynthVertices(const MatrixS &vectors,const std::vector<int> &rotations,const std::vector<int> &dilations,const MatrixS &vertices);
    MatrixS getSynthVertices(const MatrixS &vectors,const std::vector<int> &rotations,const std::vector<int> &dilations)
    { return getSynthVertices(vectors,rotations,dilations,getVertices()); }

    /// Retrieves a copy of this polyhedra transformed by the given matrix
    /// @param pTransform transform matrix
    /// @param pInverse inverse of transform matrix
    /// @param pPolyhedra container for the result (if NULL, a new one is created)
    AbstractPolyhedra& getTransformedPolyhedra(Polyhedra<scalar>& polyhedra,const MatrixS& transform,const MatrixS& inverse=Polyhedra<scalar>::ms_emptyMatrix,const MatrixS &templates=Polyhedra<scalar>::ms_emptyMatrix);
    AbstractPolyhedra& getTransformedPolyhedra(const MatrixS& transform,const MatrixS& inverse=Polyhedra<scalar>::ms_emptyMatrix,const MatrixS &templates=Polyhedra<scalar>::ms_emptyMatrix)
    { return getTransformedPolyhedra(*(new AbstractPolyhedra(this->getDimension())),transform,inverse,templates); }

public:
    static bool                         ms_trace_abstraction;

};

}

#endif //ABSTRACT_POLYHEDRA_H
