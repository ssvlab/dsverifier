//Authors: Dario Cattaruzza, Alessandro Abate, Peter Schrammel, Daniel Kroening
//University of Oxford 2016
//This code is based on Komei Fukuda's ccd implementation and as such supplied under the GPL license agreement (see license.txt)

#include "Tableau.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

namespace abstract{

template <class scalar>  MatToStr<scalar> Tableau<scalar>::ms_logger(true);
template <class scalar>  MatToStr<scalar> Tableau<scalar>::ms_decoder(false);
template <class scalar>  traceTableau_t Tableau<scalar>::ms_trace_tableau=eTraceNoTableau;
template <class scalar>  bool Tableau<scalar>::ms_trace_errors=false;
template <class scalar>  bool Tableau<scalar>::ms_trace_time=false;

template <class scalar>
Tableau<scalar>::Tableau(const int size,const int dimension):
  m_dimension(dimension+1),
  m_size(size),
  m_zero(func::ms_weakZero),
  m_objectiveRow(0),
  m_evidenceRow(-1),
  m_evidenceCol(-1),
  Conversion(IneToExt),
  m_faces(0,dimension),
  m_supports(0,1),
  m_tableau(m_size,m_dimension),
  m_basisInverse(m_dimension,m_dimension),
  Error(None)
{
  m_basicVars.reserve(m_dimension);
  m_nonBasicRow.reserve(size+1);
}
	
template <class scalar>
Tableau<scalar>::~Tableau()
{
}

template <class scalar>
inline scalar Tableau<scalar>::AValue(const MatrixS &vector, const long row)
{ /*return the ith component of the vector  A x vector */
  scalar temp(m_tableau.coeff(row,0)*vector.coeff(0,0));
  refScalar width,maxWidth=func::toWidth(temp);
  for (int j = 1; j < m_dimension; j++) {
    width=func::madd(temp,m_tableau.coeff(row,j),vector.coeff(0,j));
    if (width>maxWidth) maxWidth=width;
  }
  char sign=func::hardSign(temp);
  if (sign==0) {
    maxWidth-=func::ms_epsilon;
    if ((func::toUpper(temp)<maxWidth) || (-func::toLower(temp)<maxWidth)) {
      maxWidth=maxTightWidth(vector,row);
      if ((func::toUpper(temp)<maxWidth) || (-func::toLower(temp)<maxWidth)) {
        /*if (maxWidth<func::ms_weakEpsilon) {
          /*scalar mult=func::ms_1+func::setpm(maxWidth);
          vector*=mult;
          return ATightValue(vector,row);
        }*/
        func::imprecise(temp,maxWidth);
      }
    }
  }
  return temp;
}

template <class scalar>
inline scalar Tableau<scalar>::ATightValue(const MatrixS &vector, const long row)
{ /*return the ith component of the vector  A x vector */
  scalar temp(m_tableau.coeff(row,0)*vector.coeff(0,0));
  refScalar width,maxWidth=func::tightWidth(m_tableau.coeff(row,0),vector.coeff(0,0));
  for (int j = 1; j < m_dimension; j++) {
    width=func::tightMadd(temp,m_tableau.coeff(row,j),vector.coeff(0,j));
    if (width>maxWidth) maxWidth=width;
  }
  char sign=func::hardSign(temp);
  if (sign==0) {
    maxWidth/=2;
    if ((func::toUpper(temp)<maxWidth) || (-func::toLower(temp)<maxWidth)) {
      func::imprecise(temp,maxWidth);
    }
  }
  return temp;
}

template <class scalar>
inline typename Tableau<scalar>::refScalar Tableau<scalar>::maxTightWidth(const MatrixS &vector, const long row)
{ /*return the minimum required width for full inclusion */
  refScalar width,maxWidth=func::tightWidth(m_tableau.coeff(row,0),vector.coeff(0,0));
  for (int j = 1; j < m_dimension; j++) {
    width=func::tightWidth(m_tableau.coeff(row,j),vector.coeff(0,j));
    if (width>maxWidth) maxWidth=width;
  }
  return maxWidth/2;
}

///Relaxes the current basis by the given interval to cope with imprecisions
template <class scalar>
void Tableau<scalar>::relaxBasis(scalar width,int col)
{
  scalar mult=func::setpm(width);
  if (col>=0) {
    refScalar magOrder=10;
    for (int row=0;row<m_basisInverse.rows();row++) {
      m_basisInverse.coeffRef(row,col)+=mult*m_basisInverse.coeff(row,col);
      m_basisInverse.coeffRef(row,col)+=magOrder*(m_basisInverse.coeff(row,col)-func::toCentre(m_basisInverse.coeff(row,col)));
    }
  }
  else m_basisInverse+=mult*m_basisInverse;
}

template <class scalar>
void Tableau<scalar>::Normalize(MatrixS &vector)
{
  refScalar temp;
  refScalar max = 0;
  for (int j = 0; j < m_dimension; j++) {
    temp = func::toUpper(abs(vector.coeff(0,j)));
    if (temp > max) max = temp;
  }
  vector /= max;
}


template <class scalar>
void Tableau<scalar>::ZeroIndexSet(const MatrixS &vector, Set &ZeroSet)
{
  scalar temp;
  ZeroSet.clear();
  for (int i = 0; i < m_size; i++) {
    temp = AValue(vector, i);
    if (func::isNearZero(temp,m_zero)) ZeroSet.add(i);
  }
}

template <class scalar>
inline scalar Tableau<scalar>::entry(const int row, const int col)
/* Computes Tableau(row)*B^-1(col) */
{
  scalar temp(m_tableau.coeff(row,0) * m_basisInverse.coeff(0,col));
  refScalar width,maxWidth=func::toWidth(temp);
  for (int j=1; j< m_dimension; j++) {
    width=func::madd(temp,m_tableau.coeff(row,j),m_basisInverse.coeff(j,col));
    if (width>maxWidth) maxWidth=width;
  }
  char sign=func::hardSign(temp);
  if ((sign==0) && (maxWidth>0)) {
    maxWidth/=2;
    if ((func::toUpper(temp)<maxWidth) || (-func::toLower(temp)<maxWidth)) {
      maxWidth=maxTightWidth(row,col);
      if ((func::toUpper(temp)<maxWidth) || (-func::toLower(temp)<maxWidth)) {
        if (maxWidth<func::ms_zero) {
          relaxBasis(maxWidth,col);
          return tightEntry(row,col);
        }
        func::imprecise(temp,maxWidth);
      }
    }
  }
  return temp;
}

template <class scalar>
inline scalar Tableau<scalar>::tightEntry(const int row, const int col,const int iteration)
/* Computes Tableau(row)*B^-1(col) */
{
  scalar temp(m_tableau.coeff(row,0) * m_basisInverse.coeff(0,col));
  refScalar width,maxWidth=func::tightWidth(m_tableau.coeff(row,0),m_basisInverse.coeff(0,col));
  for (int j=1; j< m_dimension; j++) {
    width=func::tightMadd(temp,m_tableau.coeff(row,j),m_basisInverse.coeff(j,col));
    if (width>maxWidth) maxWidth=width;
  }
  char sign=func::hardSign(temp);
  if (sign==0) {
    maxWidth/=2;
    maxWidth=maxTightWidth(row,col);
    if ((func::toUpper(temp)<maxWidth) || (-func::toLower(temp)<maxWidth)) {
      if (maxWidth<func::ms_weakEpsilon) {
        relaxBasis(maxWidth,col);
        return tightEntry(row,col,iteration+1);
      }
      func::imprecise(temp,maxWidth);
    }
  }
  return temp;
}

template <class scalar>
inline typename Tableau<scalar>::refScalar Tableau<scalar>::maxTightWidth(const int row, const int col)
/* Computes Tableau(row)*B^-1(col) */
{
  int nonzero=func::isZero(m_tableau.coeff(row,0)*m_basisInverse.coeff(0,col)) ? 0 : 1;
  refScalar width,maxWidth=func::tightWidth(m_tableau.coeff(row,0),m_basisInverse.coeff(0,col));
  for (int j=1; j< m_dimension; j++) {
    width=func::tightWidth(m_tableau.coeff(row,j),m_basisInverse.coeff(j,col));
    nonzero+=func::isZero(m_tableau.coeff(row,j)*m_basisInverse.coeff(j,col)) ? 0 : 1;
    if (width>maxWidth) maxWidth=width;
  }
  if (nonzero<=1) return 0;
  return maxWidth/2;
}

template <class scalar>
bool Tableau<scalar>::SelectPivot(const long rowmax, const Set &noPivotRow, const Set &noPivotCol, pivot_t &pivot)
/* Select a position (pivot) in the matrix X.T such that (X.T)[pivot.row][pivot.col] is nonzero
   The choice is feasible, i.e., not on NopivotRow and NopivotCol, and
   best with respect to the specified roworder 
 */
{
  long rtemp;
  Set rowexcluded(noPivotRow);
  scalar Xtemp;

  for (rtemp=rowmax;rtemp<m_size;rtemp++) {
    rowexcluded.add(rtemp);   /* cannot pivot on any row > rmax */
  }
  while(true) {
    rtemp=-1;
    for (int i=1;i<=m_size;i++) {
      if (!rowexcluded.member(m_tableau.zeroOrder(i))){
        rtemp=m_tableau.zeroOrder(i);
        break;
      }
    }
    if (rtemp>=0) {
      pivot.row=rtemp;
      for (pivot.col=0;pivot.col < m_dimension;pivot.col++) {
        if (!noPivotCol.member(pivot.col)) {
          Xtemp=entry(pivot.row,pivot.col);
          char sign=func::softSign(Xtemp);//hardSign(Xtemp);//Zero is ensured by check in entry
          if (sign!=0) return true;
        }
      }
      rowexcluded.add(rtemp);
    }
    else {
      pivot.row = -1;
      pivot.col = -1;
      return false;
    }
  }
  return true;
}

template <class scalar>
void Tableau<scalar>::ColumnPivot(const pivot_t &pivot)
/* Update the Transformation matrix T with the pivot operation on (row,col)
   This procedure performs a implicit pivot operation on the tableau by
   updating the dual basis inverse.
 */
{
  scalar Xtemp;
  refScalar Xtemp0 = func::toCentre(entry(pivot.row,pivot.col));
  for (int j = 0; j < m_dimension; j++) {
    if (j != pivot.col) {
      Xtemp = entry(pivot.row,j) / Xtemp0;
      for (int j1 = 0; j1 < m_dimension; j1++) {
        func::msub(m_basisInverse.coeffRef(j1,j),m_basisInverse.coeff(j1,pivot.col),Xtemp);
      }
    }
  }
  m_basisInverse.col(pivot.col) /= Xtemp0;
  if (ms_trace_tableau>eTraceTableau) logBasis(pivot.row,pivot.col);
}

template <class scalar>
void Tableau<scalar>::logBasis(int row,int col)
{
  if (ms_trace_tableau>=eTracePivots) {
    std::stringstream buffer;
    buffer << "r=" << row << "[" << m_basicVars[row] << "]";
    buffer << ",c=" << col << "[" << m_nonBasicRow[col] << "]";
    if (ms_trace_tableau>=eTraceBasis) {
      ms_logger.logData(m_basisInverse,buffer.str());
    }
    else if (ms_trace_tableau>=eTraceSimplex){
      MatrixS matrix=m_basisInverse.col(0).transpose();
      ms_logger.logData(matrix,buffer.str());
    }
    if (ms_trace_tableau>=eTraceEntries) {
      MatrixS matrix=m_tableau*m_basisInverse;
      ms_logger.logData(matrix,"Entries");
    }
  }
}

/// Indicates if the tableau is empty
template <class scalar>
bool Tableau<scalar>::isEmpty() const
{
  if (m_faces.rows()<=0) return true;
  if (m_faces.cols()<=0) return true;
  return (m_size == ((Conversion!=ExtToIne) ? 1 : 0));
}

template <class scalar>
void Tableau<scalar>::logTableau(const std::string parameters,bool force)
{
  if ((ms_trace_tableau>=eTraceTableau) || force) {
    if (parameters.length()>0) {
      std::stringstream stream;
      stream << getName();
      stream << ": " << parameters;
      ms_logger.logData(m_faces,m_supports,stream.str());
    }
    else ms_logger.logData(m_faces,m_supports,getName());
  }
}

template <class scalar>
void Tableau<scalar>::logBasic()
{
  ms_logger.logData(m_basicVars,"Basic: ");
}

template <class scalar>
void Tableau<scalar>::logNonBasic()
{
    ms_logger.logData(m_nonBasicRow,"NonBasic: ");
}

template <class scalar>
void Tableau<scalar>::ColumnPivotAndUpdate(const pivot_t &pivot)
{
  ColumnPivot(pivot);
  long entering=m_nonBasicRow[pivot.col];
  m_basicVars[pivot.row]=pivot.col;          // the nonbasic variable r corresponds to column s
  m_nonBasicRow[pivot.col]=pivot.row;        // the nonbasic variable on s column is r
  if (entering>=0) m_basicVars[entering]=-1; // original variables have negative index and should not affect the row index
}

template <class scalar>
int Tableau<scalar>::FindEnumerationBasis(Set& rowSelected,std::vector<int> &pivotRows)
{
  Set colSelected(m_dimension);
  pivot_t pivot;

  pivotRows.assign(m_dimension+1,0);
  rowSelected.clear();
  m_basisInverse=MatrixS::Identity(m_dimension,m_dimension);
  for (int i=0;i<m_dimension;i++) {   // Find a set of rows for a basis
      if (!SelectPivot(m_size, rowSelected, colSelected, pivot)) return i;
      rowSelected.add(pivot.row);
      colSelected.add(pivot.col);
      pivotRows[pivot.col+1]=pivot.row+1;    // pivotRows[s] stores the corr. row index //TODO: zero index
      ColumnPivot(pivot);
  }
  return m_dimension;
}

template <class scalar>
int Tableau<scalar>::FindLPBasis()
{ /* Find a LP basis using Gaussian pivots.
     If the problem has an LP basis,
     the procedure returns m_evidenceCol=-1 if LPSundecided and an LP basis.
     If the constraint matrix A (excluding the rhs and objective) is not
     column indepent, there are two cases.  If the dependency gives a dual
     inconsistency, this returns the evidence column m_evidenceCol.  Otherwise, this returns an LP basis of size less than n_size.  Columns j
     that do not belong to the basis (i.e. cannot be chosen as pivot because
     they are all zero) will be indicated in nbindex vector: nbindex[j] will
     be negative and set to -j.
  */
  if (ms_trace_tableau>=eTracePivots) ms_logger.logData("Feasibility Basis");
  ResetTableau();
  Set RowSelected(m_size);
  Set ColSelected(m_dimension);
  RowSelected.add(m_objectiveRow);
  ColSelected.add(RHSCol);
  pivot_t pivot;
  int rank=m_dimension;
  m_evidenceCol=-1;
  for (int i=0;i<m_dimension;i++) {   /* Find a set of rows for a basis */
    if (!SelectPivot(m_size, RowSelected, ColSelected, pivot))
    {
      rank=i;
      for (int j=1;j<m_dimension; j++) {//Skip RHSCol
        if (m_nonBasicRow[j]<0){
          if (!func::isZero(entry(m_objectiveRow,j),m_zero)) {  /* dual inconsistent */
            m_evidenceCol=j;
            break;
          }
        }
      }
      /* dependent columns but not dual inconsistent. */
      break;
    }
    if (ms_trace_tableau>=eTraceEntries) {
      RowSelected.logSet("Available Rows:",true);
      ColSelected.logSet("Available Cols:",true);
    }
    RowSelected.add(pivot.row);
    ColSelected.add(pivot.col);
    ColumnPivotAndUpdate(pivot);
  }
  return rank;
}

/// Restarts the Simplex process at the current vertex
template <class scalar>
int Tableau<scalar>::Rebase()
{
  std::vector<int> pivots=m_nonBasicRow;
  if (ms_trace_tableau>=eTracePivots) ms_logger.logData("Rebase");
  ResetTableau();
  Set noPivotCol(m_dimension);
  int rank=0;
  pivot_t pivot;
  scalar Xtemp;
  for (int i=0;i<pivots.size();i++) {
    pivot.row=pivots[i];
    if (pivot.row>0) {
      for (pivot.col=0;pivot.col < m_dimension;pivot.col++) {
        if (!noPivotCol.member(pivot.col)) {
          Xtemp=entry(pivot.row,pivot.col);
          char sign=func::softSign(Xtemp);
          if (sign!=0) break;
        }
      }
      if (pivot.col==m_dimension) {//should never happen
        func::imprecise(Xtemp,m_zero);
        return rank;
      }
      noPivotCol.add(pivot.col);
      ColumnPivotAndUpdate(pivot);
      rank++;
    }
  }
  return rank;
}

template <class scalar>
void Tableau<scalar>::ResetTableau()
{
  for (int col=0; col<m_dimension; col++) m_nonBasicRow[col]=-col;
  m_nonBasicRow[RHSCol]=0;/* RHS is already in nonbasis and is considered to be associated with the zero-th row of input. */
  m_basisInverse=MatrixS::Identity(m_dimension,m_dimension);
  if (ms_trace_tableau>=eTracePivots) ms_logger.logData("Reset Tableau");

  /* Set the bflag according to nbindex */
  for (int row=0; row<=m_size; row++) m_basicVars[row]=-1; /* all basic variables have index -1 */
  m_basicVars[ m_objectiveRow]=0;                          /* bflag of the objective variable is 0, different from other basic variables which have -1 */
  for (int col=0; col<m_dimension; col++) {
    if (m_nonBasicRow[col]>0) m_basicVars[m_nonBasicRow[col]]=col; /* bflag of a nonbasic variable is its column number */
  }
}

template <class scalar>
void Tableau<scalar>::ComputeRowOrderVector(const OrderType &type)
{
  m_tableau.ComputeRowOrderVector(type);
}

template <class scalar>
bool Tableau<scalar>::load(const MatrixS &faces,const MatrixS &supports,const bool transpose)
{
  int numFaces=transpose ? faces.cols() : faces.rows();
  if (supports.rows()!=numFaces) return false;
  Error=None;
  m_isNormalised=false;
  m_size =numFaces+ ((Conversion!=ExtToIne) ? 1 : 0);
  m_objectiveRow=m_size-1;
  m_evidenceRow=-1;
  m_evidenceCol=-1;
  if (&faces!=&m_faces) {
    if (transpose) m_faces=faces.transpose();
    else           m_faces=faces;
  }
  if (&supports!=&m_supports) m_supports=supports;
  m_dimension=m_faces.cols()+1;
  if (m_faces.rows()>0) {
    refScalar max=func::toUpper(m_faces.coeff(0,0));
    refScalar min=func::toLower(m_faces.coeff(0,0));
    for (int row=0;row<m_faces.rows();row++) {
      for (int col=0;col<m_faces.cols();col++) {
        if (func::toUpper(m_faces.coeff(row,col))>max) max=func::toUpper(m_faces.coeff(row,col));
        if (func::toLower(m_faces.coeff(row,col))<min) min=func::toLower(m_faces.coeff(row,col));
      }
    }
    if (-min>max) max=-min;
    m_zero=max*func::ms_weakEpsilon;
  }
  else m_zero=0;
  m_tableau.resize(m_size,m_dimension);
  for (int row=0;row<m_faces.rows();row++) {
    m_tableau.coeffRef(row,0)=supports.coeff(row,0);
    for (int col=0;col<m_faces.cols();col++) m_tableau.coeffRef(row,col+1)=-m_faces.coeff(row,col);
  }
  m_basicVars.resize(m_size+1);
  m_nonBasicRow.resize(m_dimension);
  if (Conversion==IneToExt) {
    m_tableau.block(faces.rows(),1,1,faces.cols())=MatrixS::Zero(1,faces.cols()); /*artificial row for x_1 >= 0*/
    m_tableau.coeffRef(faces.rows(),0) = 1.0;
  }
  return true;
}

/// Relaxes the polyhedra to its outer boundaries (when using interval representations)
template <class scalar>
void Tableau<scalar>::toOuter(bool force)
{
  for (int i=0;i<m_supports.rows();i++) {
    m_tableau.coeffRef(i,RHSCol)=func::toUpper(m_supports.coeff(i,0));
    if (force) {
      for (int j=0;j<m_supports.cols();j++) {
        m_supports.coeffRef(i,j)=func::toUpper(m_supports.coeff(i,j));
      }
    }
  }
}

/// Constrains the polyhedra to its inner boundaries (when using interval representations)
template <class scalar>
void Tableau<scalar>::toInner(bool force)
{
  for (int i=0;i<m_supports.rows();i++) {
    m_tableau.coeffRef(i,RHSCol)=func::toLower(m_supports.coeff(i,0));
    if (force) m_supports.coeffRef(i,0)=func::toLower(m_supports.coeff(i,0));
  }
}


#ifdef USE_LDOUBLE
  #ifdef USE_SINGLES
    template class Tableau<long double>;
  #endif
  #ifdef USE_INTERVALS
    template class Tableau<ldinterval>;
  #endif
#endif
#ifdef USE_MPREAL
  #ifdef USE_SINGLES
    template class Tableau<mpfr::mpreal>;
  #endif
  #ifdef USE_INTERVALS
    template class Tableau<mpinterval>;
  #endif
#endif

}
