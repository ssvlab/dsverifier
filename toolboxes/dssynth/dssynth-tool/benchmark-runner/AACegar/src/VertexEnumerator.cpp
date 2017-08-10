//Authors: Dario Cattaruzza, Alessandro Abate, Peter Schrammel, Daniel Kroening
//University of Oxford 2016
//This code is based on Komei Fukuda's ccd implementation and as such supplied under the GPL license agreement (see license.txt)

#include "VertexEnumerator.h"

namespace abstract {

template <class scalar>
bool  VertexEnumerator<scalar>::ms_normalised_rays=false;

template <class scalar>
traceVertices_t VertexEnumerator<scalar>::ms_trace_vertices=eTraceNoVertex;

template <class scalar>
VertexEnumerator<scalar>::VertexEnumerator(int size,int dimension) :
  DualSimplex<scalar>(size,dimension),
  FeasibleRayCount(0),
  TotalRayCount(0),
  VertexCount(0),
  EdgeCount(0),
  TotalEdgeCount(0),
  AddedHyperplanes(m_size),
  WeaklyAddedHyperplanes(m_size),
  InitialHyperplanes(m_size),
  Face(m_size),
  Face1(m_size)
{
  clear();
}

template <class scalar>
int VertexEnumerator<scalar>::SelectNextHyperplane(const Set &excluded)
{
  for (int i=1; i<=m_size; i++){
    if (!excluded.member(m_tableau.zeroOrder(i))) return (m_tableau.zeroOrder(i));//TODO: zero index
  }
  return -1;
}

template <class scalar>
void VertexEnumerator<scalar>::clear()
{
  m_rayList.clear();
  m_posHead=m_rayList.begin();
  m_zeroHead=m_rayList.begin();
  TotalRayCount = 0;
  FeasibleRayCount = 0;
  VertexCount = 0;
  EdgeCount=0;
  TotalEdgeCount=0;

  Edges.assign(m_size+1,(AdjacentRays<scalar>*)NULL);
  AddedHyperplanes.initializeEmpty(m_size);
  WeaklyAddedHyperplanes.initializeEmpty(m_size);
  InitialHyperplanes.initializeEmpty(m_size);
  Face.initializeEmpty(m_size);
  Face1.initializeEmpty(m_size);
}


template <class scalar>
VertexEnumerator<scalar>::~VertexEnumerator()
{
  clear();
}

template <class scalar>
bool VertexEnumerator<scalar>::load(const MatrixS &faces,const MatrixS &supports,const bool transpose)
{
  clear();
  this->Error=this->None;
  return Tableau<scalar>::load(faces,supports,transpose);
}

template <class scalar>
bool VertexEnumerator<scalar>::StoreRay(const MatrixS &vector, Ray<scalar> &ray)
{  // Original ray storing routine
  ray.feasible = true;
  ray.ZeroSet.initializeEmpty(m_size);
  ray.ARay = 0.0;
  ray.FirstInfeasIndex=m_size+1;
  ray.data=vector;
  for (int i = 1; i <= m_size; i++) {//TODO: zero index
    int k=m_tableau.zeroOrder(i);
    char sign=func::hardSign(this->AValue(vector, k));//Zero is ensured by check in AValue
    if (sign==0)  ray.ZeroSet.add(k);
    else if ((sign<0) && ray.feasible) {
      ray.feasible = false;
      ray.FirstInfeasIndex=i;  // the first violating inequality index
    }
  }
  if (ms_trace_vertices>=eTraceRays) {
    if (Set::ms_trace_set) {
      ray.ZeroSet.logSet("SRay ",false);
      ms_logger.logData(": ",false);
    }
    else ms_logger.logData("SRay: ",false);
    ms_logger.logData(ray.data);
    if (ms_normalised_rays) ms_logger.logNormalisedData(ray.data,0);
    MatrixS dist(1,m_size);
    for (int i=0;i<m_size;i++) dist.coeffRef(0,i)=this->AValue(vector, i);
    ms_logger.logData("dist: ",false);
    ms_logger.logData(dist);
  }
  return ray.feasible;
}

template <class scalar>
void VertexEnumerator<scalar>::AddRay(const MatrixS &vector)
{
  m_rayList.push_back(Ray<scalar>(m_size,m_dimension));
  TotalRayCount++;
  if (StoreRay(vector, m_rayList.back())) {
    FeasibleRayCount++;
    char sign=func::hardSign(vector.coeff(0,0));
    if (sign!=0) VertexCount++;
  }
}

template <class scalar>
bool VertexEnumerator<scalar>::FindInitialRays(Set &initHyperplanes, MatrixS& InitRays, std::vector<int> &pivotRows)
{
  long rank=this->FindEnumerationBasis(initHyperplanes, pivotRows);
  if (Set::ms_trace_set) initHyperplanes.logSet("Enumeration Basis");
  if (rank < this->getDimension()) {
    if (this->ms_trace_vertices>=eTraceVertices) ms_logger.logData(rank,"Low Column Rank");
    this->Error = this->LowColumnRank;
    return false;
  }
  InitRays=this->m_basisInverse;
  return true;
}

template <class scalar> void VertexEnumerator<scalar>::CreateNewRay(const MatrixS &Ray1, const MatrixS &Ray2, long row)
{
  // Create a new ray by taking a linear combination of two rays
  scalar v1 = abs(this->AValue(Ray1, row));
  scalar v2 = abs(this->AValue(Ray2, row));
  refScalar den=func::toLower(v1+v2);
  v1/=den;
  v2/=den;
  MatrixS NewRay = Ray1 * v2 + Ray2 * v1;
  this->Normalize(NewRay);
  AddRay(NewRay);
}

template <class scalar>
bool VertexEnumerator<scalar>::EvaluateRay(const long row)
/* Evaluate the ith component of the vector  A x Rays and
   rearrange the linked list so that the infeasible rays
   with respect to  row  will be placed consecutively from
   First. Also for all feasible rays, "positive" rays and
   "zero" rays will be placed consecutively.
 */
{
  m_zeroRays.clear();
  m_posRays.clear();
  m_negRays.clear();

  for (typename RayList::iterator it=m_rayList.begin();it!=m_rayList.end();) {
    if (this->ms_trace_vertices>=eTraceRays) {
      ms_logger.logData(it->data,"Evaluate ray");
    }
    it->ARay = this->AValue(it->data,row);
    char sign=func::hardSign(it->ARay);//Zero is ensured by check in AValue
    typename RayList::iterator it2=it;
    it++;
    if (sign<0)         m_negRays.splice(m_negRays.begin(),m_rayList,it2);
    else if (sign>0)    m_posRays.splice(m_posRays.begin(),m_rayList,it2);
    else                m_zeroRays.splice(m_zeroRays.begin(),m_rayList,it2);
  }
  int negCount=m_negRays.size();
  int posCount=m_posRays.size();
  int zeroCount=m_zeroRays.size();
  m_rayList.splice(m_rayList.end(),m_negRays);
  m_rayList.splice(m_rayList.end(),m_posRays);
  m_rayList.splice(m_rayList.end(),m_zeroRays);
  m_posHead=m_rayList.begin();
  std::advance(m_posHead,negCount);
  m_zeroHead=m_posHead;
  std::advance(m_zeroHead,posCount);
  return ((zeroCount+posCount)>0);
}

template <class scalar>
void VertexEnumerator<scalar>::DeleteNegativeRays()
/* Eliminate the infeasible rays with respect to  row,
   and sort the zero list wrt to FirstInfeasIndex */
{
  m_zeroRays.clear();
  // Delete the infeasible rays
  for (typename RayList::iterator it=m_rayList.begin();it!=m_rayList.end();) {
    char sign=func::hardSign(it->ARay);//Zero is ensured by check in AValue
    if (sign<0) it=m_rayList.erase(it);
    else if (sign==0) {
      // Sort the zero rays wrt to FirstInfeasIndex
      //Zero case leaves possible negative rays in the list which creates more work but remains sound.
      typename RayList::iterator it2=m_zeroRays.begin();
      for (;it2!=m_zeroRays.end();it2++) {
        if (it2->FirstInfeasIndex>=it->FirstInfeasIndex) break;
      }
      typename RayList::iterator it3=it;
      it++;
      m_zeroRays.splice(it2,m_rayList,it3);
    }
    else it++;
  }
  int posCount=m_rayList.size();
  int zeroCount=m_zeroRays.size();
  m_rayList.splice(m_rayList.end(),m_zeroRays);
  m_posHead=m_rayList.begin();
  m_zeroHead=m_rayList.begin();
  std::advance(m_zeroHead,posCount);
}

template <class scalar>
void VertexEnumerator<scalar>::validateRays(const bool over)
{
  for (typename RayList::iterator it=m_rayList.begin();it!=m_rayList.end();it++) {
    MatrixS &p=it->data;
    char sign=func::hardSign(p.coeff(0,0)); //No normalization on zero, which is fine.
    if (sign!=0) p/=p.coeff(0,0);
  }
}

template <class scalar>
void VertexEnumerator<scalar>::logRays()
{
  if (ms_trace_vertices>=eTraceRays) {
    ms_logger.logData("Rays");
    for (typename RayList::iterator it=m_rayList.begin();it!=m_rayList.end();it++) {
      if (Set::ms_trace_set) {
        it->ZeroSet.logSet("Ray ",false);
        ms_logger.logData(": ",false);
        ms_logger.logData(it->data);
      }
      else {
        ms_logger.logData(it->data,"Ray: ");
      }
      if (ms_normalised_rays) ms_logger.logNormalisedData(it->data,0);
    }
  }
}

template <class scalar>
void VertexEnumerator<scalar>::ConditionalAddEdge(const Ray<scalar> &Ray1, const Ray<scalar> &Ray2, const typename RayList::iterator &first, long iter)
{
  long fii1=Ray1.FirstInfeasIndex;
  long fii2=Ray2.FirstInfeasIndex;
  const Ray<scalar> &Rmin=(fii1<fii2) ? Ray1 : Ray2;
  const Ray<scalar> &Rmax=(fii1<fii2) ? Ray2 : Ray1;
  long fmin=Rmin.FirstInfeasIndex;
  long fmax=Rmax.FirstInfeasIndex;
  const Set &Zmin = Rmin.ZeroSet;
  const Set &Zmax = Rmax.ZeroSet;

  if ((fmin==fmax) || (Zmax.member(m_tableau.zeroOrder(fmin)))) {//TODO: zero index
    if (ms_trace_vertices>=eTraceEdges) {
      ms_logger.logData("Unfeasible Edge ");
      Zmin.logSet("Min ",false);
      ms_logger.logData(m_tableau.zeroOrder(fmin),"[");
      ms_logger.logData("]: ",false);
      ms_logger.logData(Rmin.data);
      Zmax.logSet("Max ",false);
      ms_logger.logData(m_tableau.zeroOrder(fmax),"[");
      ms_logger.logData("]: ",false);
      ms_logger.logData(Rmax.data);
    }
    return;
  }
  // the pair will be separated at the iteration fmin
  if (this->CheckAdjacency(Rmin,Rmax,first,iter)){
    AdjacentRays<scalar> *NewEdge=new AdjacentRays<scalar>(Rmax,Rmin);
    // save the one remains in iteration fmin in the first & the one deleted in iteration fmin in the second
    EdgeCount++;
    TotalEdgeCount++;
    if (Edges[fmin]!=NULL) NewEdge->Next=Edges[fmin];
    Edges[fmin]=NewEdge;
    if (ms_trace_vertices>=eTraceEdges) ms_logger.logData("Adjacent ",false);
  }
  if (ms_trace_vertices>=eTraceEdges) {
    ms_logger.logData("Edge:");
    ms_logger.logData(Rmax.data);
    ms_logger.logData(Rmin.data);
  }
}

template <class scalar>
void VertexEnumerator<scalar>::CreateInitialEdges()
{
  for (typename RayList::iterator it=m_rayList.begin();it!=m_rayList.end();it++) {
    for (typename RayList::iterator it2=it;it2!=m_rayList.end();it2++) {
      if (it->FirstInfeasIndex!=it2->FirstInfeasIndex) {
        ConditionalAddEdge(*it, *it2, m_rayList.begin(), m_dimension);
      }
    }
  }
  this->logRays();
}

template <class scalar>
void VertexEnumerator<scalar>::UpdateEdges(const typename RayList::iterator &first, long iter)
/* This procedure must be called after the ray list is sorted
   by EvaluateARay2 so that FirstInfeasIndex's are monotonically
   increasing.
*/
{
  for (typename RayList::iterator it=first;it!=m_rayList.end();it++) {
    typename RayList::iterator it2=it;
    it2++;
    for (;it2!=m_rayList.end();it2++){
      if  (it2->FirstInfeasIndex > it->FirstInfeasIndex){
        for (;it2!=m_rayList.end();it2++){
          ConditionalAddEdge(*it,*it2,first, iter);
        }
        break;
      }
    }
  }
}

template <class scalar>
bool VertexEnumerator<scalar>::AddNewHyperplane(long row, long iter)
{
  AdjacentRays<scalar> *EdgePtr, *EdgePtr0;
  long fii1, fii2;

  // Check feasibility of rays w.r.t. row and sort them ( -, +, 0).
  if (!this->EvaluateRay(row)) {
    VertexCount=0;
    return true;   // All rays are infeasible, and the computation must stop
  }
  
  EdgePtr=Edges[iter];
  while (EdgePtr!=NULL){
    const Ray<scalar> &Ray1=EdgePtr->Ray1;
    const Ray<scalar> &Ray2=EdgePtr->Ray2;
    fii1=Ray1.FirstInfeasIndex;
    if (Ray1.data.cols()==Ray2.data.cols()) {
      this->CreateNewRay(Ray1.data, Ray2.data, row);
      fii2=m_rayList.rbegin()->FirstInfeasIndex;
      if (fii1 != fii2) this->ConditionalAddEdge(Ray1,*m_rayList.rbegin(),m_posHead,iter);
    }
    else {
      EdgePtr0=EdgePtr;
    }
    EdgePtr0=EdgePtr;
    EdgePtr=EdgePtr->Next;
    delete EdgePtr0;
    EdgeCount--;
  }
  Edges[iter]=NULL;
  this->DeleteNegativeRays();
    
  AddedHyperplanes.add(row);
  if ((iter<m_size) && (m_zeroHead!=m_rayList.end())) {
      UpdateEdges(m_zeroHead, iter);
  }

  if (m_rayList.size()==FeasibleRayCount) return true;
  return false;
}

template <class scalar>
bool VertexEnumerator<scalar>::CheckAdjacency(const Ray<scalar> &RMin, const Ray<scalar> &RMax,const typename RayList::iterator &first,long iter)
{
  Face1.intersect(RMin.ZeroSet, RMax.ZeroSet);
  for (int i=iter+1; i < RMin.FirstInfeasIndex; i++){
    if (Face1.member(m_tableau.zeroOrder(i))){
      return false;
    }
  }
  Face.intersect(Face1, AddedHyperplanes);
  if (Face.cardinality()< m_dimension - 2) return false;

  for (typename RayList::iterator it=first;it!=m_rayList.end();it++) {
    if ((&*it != &RMin) && (&*it != &RMax)) {
      Face1.intersect(it->ZeroSet, AddedHyperplanes);
      if (Face.isSubset(Face1)) return false;
    }
  }
  return true;
}

template <class scalar>
int VertexEnumerator<scalar>::createHyperplaneList()
{
  for (int iteration = m_dimension+1;iteration <= m_size;iteration++) {//TODO: zero index
    long row=this->SelectNextHyperplane(WeaklyAddedHyperplanes);
    bool result=AddNewHyperplane(row, iteration);
    AddedHyperplanes.add(row);
    WeaklyAddedHyperplanes.add(row);
    if (result) return iteration;
  }
  return m_size+1;
}

template <class scalar>
void VertexEnumerator<scalar>::enumerate()
{
  MatrixS InitRays(m_dimension,m_dimension);
  std::vector<int> InitRayIndex; // 0 if the corr. ray is for generator of an extreme line
  this->Error=this->None;
  func::setZero(this->m_zero);

  InitialHyperplanes.initializeEmpty(m_size);
  AddedHyperplanes.initializeEmpty(m_size);
  WeaklyAddedHyperplanes.initializeEmpty(m_size);
  Face.initializeEmpty(m_size);   // used in CheckAdjacency
  Face1.initializeEmpty(m_size);  // used in CheckAdjacency

  this->ComputeRowOrderVector(LexMin);

  TotalRayCount = 0;
  FeasibleRayCount = 0;
  VertexCount = 0;
  EdgeCount=0; // active edge count
  TotalEdgeCount=0; // active edge count
  Edges.assign(m_size+1,(AdjacentRays<scalar>*)NULL);

  if (ms_trace_vertices>=eTraceRays) ms_logger.logData(this->m_tableau,"Enumerate");
  if (this->FindInitialRays(InitialHyperplanes, InitRays, InitRayIndex)) {
    initialize(InitRays, InitRayIndex);
    createHyperplaneList();
  }
  else if (ms_trace_vertices>=eTraceRays) ms_logger.logData("Initial rays not found");
  if (ms_trace_vertices>=eTraceRays) this->logRays();
}

template <class scalar>
void VertexEnumerator<scalar>::initialize(const MatrixS &InitialRays, std::vector<int> &pivotRows)
{
  MatrixS Vector1;
  Set ZSet(m_size);

  AddedHyperplanes.copy(InitialHyperplanes);
  WeaklyAddedHyperplanes.copy(InitialHyperplanes);
  this->m_tableau.UpdateRowOrderVector(InitialHyperplanes);
  for (int r = 1; r <= m_dimension; r++) {
    Vector1=InitialRays.col(r-1).transpose();
    this->Normalize(Vector1);
    this->ZeroIndexSet(Vector1, ZSet);
    AddRay(Vector1);
    if (pivotRows[r]==0) {
      Vector1=-Vector1;
      AddRay(Vector1);
    }
  }
  CreateInitialEdges();
}

template <class scalar>
typename VertexEnumerator<scalar>::RayList& VertexEnumerator<scalar>::findVertices(MatrixS &faces,MatrixS &supports,char reverse)
{
  this->Conversion = reverse ? ExtToIne : IneToExt;

  /* Initialization of global variables */
  try {
    this->Error=this->None;
    load(faces,supports);
    enumerate();
    this->validateRays();
    this->logRays();
  }
  catch(std::string error) {
    ms_logger.logData(error);
    m_rayList.clear();
  }
  return m_rayList;
}

#ifdef USE_LDOUBLE
  #ifdef USE_SINGLES
    template class VertexEnumerator<long double>;
  #endif
  #ifdef USE_INTERVALS
    template class VertexEnumerator<ldinterval>;
  #endif
#endif
#ifdef USE_MPREAL
  #ifdef USE_SINGLES
    template class VertexEnumerator<mpfr::mpreal>;
  #endif
  #ifdef USE_INTERVALS
    template class VertexEnumerator<mpinterval>;
  #endif
#endif

}
