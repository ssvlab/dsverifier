//Authors: Dario Cattaruzza, Alessandro Abate, Peter Schrammel, Daniel Kroening
//University of Oxford 2016
//This code is supplied under the BSD license agreement (see license.txt)

#ifndef POLYHEDRA_H
#define POLYHEDRA_H

#include <limits>
#include <stdlib.h>

#include <stdlib.h>
#include <limits>
#include "VertexEnumerator.h"
#include "JordanMatrix.h"

namespace abstract
{

template <class scalar> class Polyhedra : public DualSimplex<scalar>
{
public:
    using typename Tableau<scalar>::refScalar;
    using typename Tableau<scalar>::func;
    using typename Tableau<scalar>::MatrixS;
    using typename Tableau<scalar>::MatrixC;
    using typename Tableau<scalar>::MatrixR;
    using typename Tableau<scalar>::MatrixRC;

    using Tableau<scalar>::m_faces;
    using Tableau<scalar>::m_supports;
    using Tableau<scalar>::m_dimension;
    using Tableau<scalar>::ms_logger;
    using Tableau<scalar>::ms_decoder;
    using Tableau<scalar>::ms_trace_tableau;
    using Tableau<scalar>::ms_trace_errors;

    using Tableau<scalar>::order;
    using Tableau<scalar>::getDimension;
    using Tableau<scalar>::logTableau;
    using DualSimplex<scalar>::maximiseAll;

    /// Constructs an empty buffer
    /// @param dimension dimension of the space
    Polyhedra(int dimension=0);

    /// Constructs transformed polyhedra
    /// @param source template to copy to create this polyhedra
    /// @param pTransform transformation matrix to obtain this polyhedra
    /// @param pInverse precalculated inverse of pTransform
    Polyhedra(const Polyhedra &source,const MatrixS &transform=ms_emptyMatrix,const MatrixS &inverse=ms_emptyMatrix);

    /// Destructor
    ~Polyhedra() {}

    /// Loads a polyhedral description from file
    /// @param data buffer containing the polyhedra description in the form
    /// @param vertices indicates if the data loaded are vertices as opposed to faces
    /**
     c_11,c_12, ... ,c_1p, d_1
     c_21,c_22, ... ,c_2p, d_2
     ...
     c_m1,c_m2, ... ,c_mp, d_m
     **/
    /// @return position after decoding. Negative (or zero) if failed.
    int loadData(const std::string &data,size_t pos=0,const bool vertices=false);

    /// Calculates the convex hull of a point cloud
    /// @param points point cloud
    /// @param vectors template directions
    bool convexHull(const MatrixS &points,const MatrixS &vectors);

    /// Changes the dimensionality of the polyhedra
    /// @param dimension new dimensionality
    /// @param keep indicates if the polyhedral description should be kept (flat at 0 on new dimensions)
    void changeDimension(const int dimension,const bool keep=false);

    /// Loads a polyhedral description from file
    /// @param fileName name of the file to load the data from
    /// @return true if succesful
    bool loadFromFile(const std::string &fileName);

    /// Returns a description of the polyhedra
    /// @param normalised normalises the direction vectors
    std::string getDescription(displayType_t displayType,bool interval,bool useBrackets=false,MatrixS &templates=ms_emptyMatrix);

    /// Returns a description of the polyhedra as a list of halfspaces Ax<b
    /// @param normalised normalises the direction vectors
    /// @param interval indicates if the description should include the interval or just the centre of each value
    /// @param pTemplates if not NULL, retemplates the polyhedra in the given directions
    /// @return string describing the faces of the polyhedra (normal vector and corresponding support function)
    std::string getFaces(bool normalised=false,bool interval=true,bool useBrackets=false,MatrixS &templates=ms_emptyMatrix);

    /// Returns a description of the vertices of the polyhedra
    /// @param separator string to use to separate the elements of each vertex
    /// @param interval indicates if the description should include the interval or just the centre of each value
    /// @return string describing the vertices of the polyhedra (one per row)
    virtual std::string getVertices(std::string separator,bool interval=true,bool useBrackets=false);

    /// Retrieves the template directions of the polyhedra
    /// @return column vectors normal to the faces of the polyhedra
    MatrixS getFaceDirections() { return m_faces; }

    /// Retrieves the template directions of the polyhedra
    /// @return column vectors normal to the faces of the polyhedra
    MatrixS getDirections()     { return m_faces.transpose(); }

    /// Retrieves the support functions for the faces of the polyhedra
    /// @return column vector holding the support functions of each face of the polyhedra
    MatrixS getSupports()       { return m_supports; }

    /// Copies the polyhedra from another source
    /// @param pSource polyhedra to copy
    /// @return true if successful
    virtual bool copy(const Polyhedra &source);

    /// Loads a polyhedral description
    /// @param faces directions normal to the faces of the polyhedra (each row is a vector)
    /// @param edged support function for the corresponding direction (row)
    /// @param transpose if true, the faces are given as column vectors as opposed to row vectors
    virtual bool load(const MatrixS &faces,const MatrixS &supports,const bool transpose=false);

    /// Loads a polyhedral description
    /// @param faces vertices points indicating the vertices of the polyhedra(each row is a vertex)
    virtual bool loadVertices(const MatrixS &vertices);

    /// Intersects the polyhedra with another polyhedra
    /// @param polyhedra polyhedra to intersect
    /// @param over indicates if small overapproximations are allowed
    /// @return true if successful
    bool intersect(const Polyhedra &polyhedra,const bool over=true);

    /// Performs the union of another polyhedra with this one
    /// @param polyhedra polyhedra to merge
    /// @param extend indicates if the new polygon should have te combined faces (true or just the original (false)
    /// @return true if successful
    bool merge(Polyhedra &polyhedra,const bool extend=true);

    /// Increases the dimension of the polyhedra by orthogonally multiplying it with another
    /// @param polyhedra polyhedra to concatenate
    /// @return true if successful
    bool concatenate(Polyhedra &polyhedra);

    /// Increases the dimension of the polyhedra by orthogonally multiplying it with a set of inequalities
    /// @param faces directions of the inequalities
    /// @param supports support functions of the inequalities
    /// @return true if successful
    bool concatenate(MatrixS &faces,MatrixS &supports);

    /// Performs the Minkowski sum of this polyhedra to another
    /// @param polyhedra polyhedra to merge
    /// @param extended indicates whether to use both template directions or only those of this polyhedra
    /// @return true if successful
    bool add(Polyhedra &polyhedra,const bool extended=true);

    /// Performs the Minkowski difference of this polyhedra with another
    /// @param polyhedra polyhedra to substract
    /// @return true if successful
    bool erode(Polyhedra &polyhedra);

    /// Adds a number of directions to the template of the polhedra
    /// @param directions column vectors to add
    /// @param supports support vectors of the polyhedra in the given directions (when known)
    /// @return true if successful
    bool addDirection(const MatrixS &directions);
    bool addDirection(const MatrixS &directions,MatrixS &supports);

    /// templates the polyhedra in the given directions
    /// @param templates column vectors to use as directions
    /// @param aprox acceptable percentage compromise
    bool retemplate(const MatrixS &templates,refScalar aprox=functions<refScalar>::ms_infinity);

    /// Retrieves a copy of this polyhedra transformed by the given matrix
    /// @param transform transform matrix
    /// @param inverse inverse of transform matrix
    /// @param pPolyhedra container for the result (if NULL, a new one is created)
    virtual Polyhedra& getTransformedPolyhedra(Polyhedra& polyhedra,const MatrixS& transform,const MatrixS& inverse=ms_emptyMatrix,const MatrixS &templates=ms_emptyMatrix);
    virtual Polyhedra& getTransformedPolyhedra(const MatrixS& transform,const MatrixS& inverse=ms_emptyMatrix,const MatrixS &templates=ms_emptyMatrix)
    { return getTransformedPolyhedra(*(new Polyhedra<scalar>(this->getDimension())),transform,inverse,templates);}

    /// linearly transofrm the polyhedra (rotate, translate, stretch)
    /// @param pTransform transform matrix
    /// @param pInverse precalculate inverso of the transform matrix (if null, the pseudoinverse will be calculated)
    bool transform(const MatrixS &transform,const MatrixS& inverse=ms_emptyMatrix);

    /// linearly transofrm the polyhedra (rotate, translate, stretch)
    /// @param coefficient expansion/compression value
    void transform(const scalar &coefficient);

    /// linearly transofrm the polyhedra through vertex enumeration
    /// @param pTransform transform matrix
    bool vertexTransform(const MatrixS &transform,const MatrixS &templates);

    /// Translates the polyhedra in the opposite direction of vector, making it its centre (ie translating by -vector)
    /// @param vector cartesian translation coordinates (colwise)
    /// @param storeOffset indicates if the translation is reversible
    void translate(const MatrixS &vector,const bool storeOffset=true);

    /// Indicates if the referenced polyhedra is contained inside this one
    /// @param polyhedra polyhedra to check
    /// @return true if polyhedra is entirely inside this one
    bool contains(Polyhedra<scalar>& polyhedra);

    /// finds the inequalities of the polyhedra and stores them in a matrix
    /// @return true if successful
    bool makeFaces();

    /// finds the vertices of the polyhedra and stores them in a matrix
    /// @return true if successful
    bool makeVertices(bool force=false);

    /// Retrieves the (hyper-rectangular) center of the polyhedra
    /// @return vector containing the centre of the polyhedra
    MatrixS& getCentre();

    /// Finds the (hyper-rectangular) center of the polyhedra, and moves it to the origin
    void centralize();

    /// Moves the polyhedra by the vector indicated in centre.
    void decentralize();

    /// Indicates if there is a selected non-zero central point
    bool isCentralized();

    /// Indicates if a set of points is inside the Polyhedra
    /// @param points, points in the state space to evaluate
    /// @return true if all given points are inside the polyhedra
    bool isInside(const MatrixS &points);

    /// Removes any existing faces
    void clear();

    /// Retrieves the list of vertices of the polyhedra
    /// @return pointer to the vertex matrix of the polyhedra. Null if not found
    MatrixS& getVertices(bool force=false)
    {
      if (!makeVertices() && force) {
        m_vertices.resize(1,getDimension());
        m_vertices.row(0)=MatrixS::Zero(1,getDimension());
      }
      return m_vertices;
    }

    void ComputeVertexOrderVector();

    ///
    MatrixS vertexMaximize(const MatrixS &vectors,const bool all=false);

    ///Retrieves a vertex description of the smallest axis-oriented hyperbox containing the polyhedra
    MatrixS boundingHyperBox();

    /// Retrieves the user-defined name of the polyhedra
    std::string getName()                   { return m_name; }

    /// Sets the name of the polyhedra
    void setName(const std::string name)    { m_name=name; }

    /// Retrieves the user-defined tag of the polyhedra (demangles homonimous polyhedra)
    int getTag()                            { return m_tag; }

    /// Sets the user-defined tag of the polyhedra (to demangle homonimous polyhedra)
    void setTag(int tag)                    { m_tag=tag; }

    /// Retrieves the total execution time used to load and recalculate the polyhedra
    int getTime()                           { return (m_loadTime+m_transformTime+m_enumerationTime+m_calculationTime); }

    /// Sets external execution times for calculating polyhedral transformations
    void setCalculationTime(int time)       { m_calculationTime=time; }

    /// Sets external execution times for calculating polyhedral transformations
    void addCalculationTime(int time)       { m_calculationTime+=time; }

    ///Retrieves the ordered element at the given position
    /// @param row position to search in the ordered set
    /// @return corresponding row of the tableau at the given position
    inline short vertexOrder(const int row) { return m_vertices.order(row); }

    void logVertices(bool force=false);
    virtual void logPolyhedra(const std::string parameters="");
protected:
    /// Intersects the polyhedra with another polyhedra without removing redundancies
    /// @param polyhedra polyhedra to intersect
    /// @return true if successful
    bool pseudoIntersect(const Polyhedra &polyhedra);

    /// Calculates the pseudoinverse of a matrix
    MatrixS pseudoInverseEigen(const MatrixS &matrix,bool &hasInverse);

    /// Calculates the pseudoinverse of a matrix
    MatrixS pseudoInverseSVD(const MatrixS &matrix,bool &hasInverse);

    /// Calculates the pseudoinverse of a matrix
    MatrixS pseudoInverseJordan(const MatrixS &matrix,bool &hasInverse);

protected:
    bool                                    m_isCentralised;
    MatrixS                                 m_centre;
    SortedMatrix<scalar>                    m_vertices;
    std::string                             m_name;
    int                                     m_tag;
    int                                     m_loadTime;
    int                                     m_transformTime;
    int                                     m_enumerationTime;
    int                                     m_calculationTime;
    static Eigen::JacobiSVD<MatrixS>        ms_svd;
    static JordanMatrix<scalar>*            ms_pJordan;
    static JordanMatrix<scalar>*            getJordanSolver();
public:
    static MatrixS                          ms_emptyMatrix;
    static traceVertices_t                  ms_trace_vertices;
    static traceDynamics_t                  ms_trace_dynamics;
    static bool                             ms_auto_make_vertices;
};

}

#endif
