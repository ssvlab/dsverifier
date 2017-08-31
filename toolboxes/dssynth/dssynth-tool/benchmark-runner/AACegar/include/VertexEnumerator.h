//Authors: Dario Cattaruzza, Alessandro Abate, Peter Schrammel, Daniel Kroening
//University of Oxford 2016
//This code is based on Komei Fukuda's ccd implementation and as such supplied under the GPL license agreement (see license.txt)

#ifndef VERTEX_ENUMERATORH
#define VERTEX_ENUMERATORH
#include "DualSimplex.h"

namespace abstract {
template <class scalar> class Ray
{
public:
  Ray(int size,int dimension) : data(1,dimension),ZeroSet(size) {}
  typename interval_def<scalar>::MatrixS data;
  Set ZeroSet;          //Set of half-spaces that intersect the ray
  long FirstInfeasIndex;//The first inequality the ray violates
  bool feasible;        //Flag to store the feasibility of the ray
  scalar ARay;          //Temporary area to store some row of A*Ray
};

template <class scalar> class AdjacentRays {
public:
  AdjacentRays(const Ray<scalar> &ray1,const Ray<scalar> &ray2) : Ray1(ray1),Ray2(ray2),Next(NULL) {}
  const Ray<scalar> &Ray1;
  const Ray<scalar> &Ray2;
  AdjacentRays<scalar> *Next;
};

template <class scalar> class VertexEnumerator : public DualSimplex<scalar> {
public:
    using typename Tableau<scalar>::refScalar;
    using typename Tableau<scalar>::func;
    using typename Tableau<scalar>::MatrixS;

    using Tableau<scalar>::m_dimension;
    using Tableau<scalar>::m_size;
    using Tableau<scalar>::m_tableau;
    using Tableau<scalar>::ms_logger;

    typedef std::list<Ray<scalar> > RayList;
    VertexEnumerator(int size,int dimension);
    ~VertexEnumerator();

    /// Clears the state of the enumerator
    void clear();

    /// Stores the given vector into the given ray structure
    /// @param vector value of the ray
    /// @param ray structure to store the data in
    /// @result true if feasible
    bool StoreRay(const MatrixS &vector, Ray<scalar> &ray);

    /// Adds a new ray to the list
    /// @param vector data to store in the new ray
    void AddRay(const MatrixS &vector);

    /// Creates a new ray derived from the given Rays
    /// @param Ray1 first ray to use to derive the data
    /// @param Ray2 second ray to use to derive the data
    /// @param row inequality use to evaluate the weights of each ray
    void CreateNewRay(const MatrixS &Ray1, const MatrixS &Ray2, long row);

    /// Orders the list of rays w.r.t row into -,+,0
    /// @param row inequality used to set the ordering
    /// @return true if there are positive or zero rays in the list
    bool EvaluateRay(const long row);

    /// Removes all negative rays (w.r.t last ordering) from the list
    void DeleteNegativeRays();

    /// Obtains a list of rays to start evalution from
    bool FindInitialRays(Set &initHyperplanes, MatrixS &InitRays, std::vector<int> &pivotRows);

    void validateRays(const bool over=true);

    /// Loads a description into the tableau (Ax<b)
    /// @param faces (A-Matrix) normal to the half-spaces describing the problem
    /// @param supports (b-vector) boundaries of the half-spaces (given as support functions)
    /// @param transpose, directions are col vectors when true and row vectors when false
    /// @return false if failed
    virtual bool load(const MatrixS &faces,const MatrixS &supports,const bool transpose=false);

    /// Logs the current list of rays into trace.txt
    void logRays();
protected:
    int createHyperplaneList();
    int SelectNextHyperplane(const Set &excluded);
    void ConditionalAddEdge(const Ray<scalar> &Ray1, const Ray<scalar> &Ray2, const typename RayList::iterator &first, long iter);
    void CreateInitialEdges();
    void UpdateEdges(const typename RayList::iterator &first, long iter);

    /// Checks whether two rays are adjacent to each other
    bool CheckAdjacency(const Ray<scalar> &RMin, const Ray<scalar> &RMax,const typename RayList::iterator &first,long iter);
    bool AddNewHyperplane(long, long iter);
    void initialize(const MatrixS &InitialRays, std::vector<int> &pivotRows);
public:
    void enumerate();
    RayList& findVertices(MatrixS &faces,MatrixS &supports,char reverse);
public:
    RayList                     m_posRays;  //Temporary list used to store the positive rays
    RayList                     m_negRays;  //Temporary list used to store the negative rays
    RayList                     m_zeroRays; //Temporary list used to store the zero rays
    RayList                     m_rayList;  //List of currently available rays
    typename RayList::iterator  m_posHead;  //Iterator pointing to the first positive ray in m_rayList
    typename RayList::iterator  m_zeroHead; //Iterator pointing to the first zero ray in m_rayList

    long         FeasibleRayCount;          //Number of feasible rays in m_rayList
    long         TotalRayCount;             //Number of rays originally added to m_rayList
    long         VertexCount;               //Number of found vertices

    long         EdgeCount;                 //Number of evaluating edges
    long         TotalEdgeCount;            //Total number of evaluated edges
    Set          AddedHyperplanes;          //Set of hyperplanes already evaluated
    Set          WeaklyAddedHyperplanes;
    Set          InitialHyperplanes;        //Set of hyperplanes used in initial evaluation
    Set          Face;
    Set          Face1;
    std::vector<AdjacentRays<scalar>*> Edges;  // adjacency relation storage for iteration k
    static bool  ms_normalised_rays;
    static traceVertices_t      ms_trace_vertices;
};
}
#endif//VERTEX_ENUMERATORH
