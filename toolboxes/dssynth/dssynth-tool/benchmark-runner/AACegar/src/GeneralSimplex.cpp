/* dplex.c:  dual simplex method c-code
   written by Komei Fukuda, fukuda@ifor.math.ethz.ch
   Version 0.61, December 1, 1997
*/

/* dplex.c : C-Implementation of the dual simplex method for
   solving an LP: max/min  c^T x subject to  x in P, where
   P= {x :  b - A x >= 0}.  
   Please read COPYING (GNU General Public Licence) and
   the manual cddman.tex for detail.
*/

#include "GeneralSimplex.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

namespace abstract {
template <class scalar>  GeneralSimplex<scalar>::GeneralSimplex(int size,int dimension)  :
    DualSimplex<scalar>(size,dimension)
{}

template <class scalar>
bool GeneralSimplex<scalar>::FastSelectPivot(const long rowmax,const Set &noPivotRow, const Set &noPivotCol, pivot_t &pivot)
/* Select a position (pivot) in the matrix X.T such that (X.T)[pivot.row][pivot.col] is nonzero
   The choice is feasible, i.e., not on NopivotRow and NopivotCol, and
   best with respect to the specified roworder
 */
{
  scalar Xtemp;
  for (int i=0;i<rowmax;i++) {
    if (!noPivotRow.member(i)) {
      pivot.row=i;
      Xtemp=this->entry(pivot.row,0);
      if (func::isNegative(Xtemp)) {
        for (pivot.col=1;pivot.col < this->m_dimension;pivot.col++) {
          if (!noPivotCol.member(pivot.col)) {
            Xtemp=this->entry(pivot.row,pivot.col);
            if (!func::isNearZero(Xtemp,this->m_zero)) return true;
          }
        }
      }
    }
  }
  for (int i=0;i<rowmax;i++) {
    if (!noPivotRow.member(i)) {
      pivot.row=i;
      for (pivot.col=0;pivot.col < this->m_dimension;pivot.col++) {
        if (!noPivotCol.member(pivot.col)) {
          Xtemp=this->entry(pivot.row,pivot.col);
          char sign=func::hardSign(Xtemp);
          if (sign!=0) return true;
        }
      }
    }
  }
  pivot.row = -1;
  pivot.col = -1;
  return false;
}

template <class scalar>
bool GeneralSimplex<scalar>::SelectPrimalSimplexPivot(pivot_t &pivot)
{ /* selects a dual simplex pivot (*r, *s) if the current
     basis is dual feasible and not optimal. If not dual feasible,
     the procedure returns false and m_status=LPSundecided.
     If Phase1=true, the RHS column will be considered as the negative
     of the column of the largest variable (==m_size).  For this case, it is assumed
     that the caller used the auxiliary row (with variable m_size) to make the current
     dictionary dual feasible before calling this routine so that the nonbasic
     column for m_size corresponds to the auxiliary variable.
  */
  scalar val;
  pivot.col=-1;
  pivot.row=-1;
  this->m_status=LPSundecided;
  MatrixS cost=this->m_tableau.row(this->m_objectiveRow)*this->m_basisInverse;
  if (this->ms_trace_basis>eTracePivots) {
    this->ms_logger.logData(cost,"costs:");
  }
  bool optimal=true;
  int selCol=-1;
  refScalar maxCoeff=-func::ms_infinity;
  for (int col=1; col<this->m_dimension; col++) {// ignore RHSCol
    refScalar coeff=func::toUpper(cost.coeff(0,col));
    if ((coeff<0) && (coeff>maxCoeff)) {
      maxCoeff=func::toLower(cost.coeff(0,col));
      selCol=col;
    }
  }

  if (selCol>=0) {
    optimal=false;
    refScalar minrat=func::ms_infinity,rat;
    for (int row=0;row<this->m_objectiveRow;row++) {
      if (this->m_basicVars[row]<0) {
        val=this->entry(row,selCol);
        if (func::isPositive(val)) {
          //The zero case would result in a pivot move by a value inside the interval of the hyperplane
          rat=func::toUpper(this->entry(row,RHSCol)/val);
          if (rat<minrat) {
            minrat=rat;
            pivot.row=row;
          }
        }
      }
    }
    if (pivot.row>=0) {
      pivot.col=selCol;
      return true;
    }
  }
  this->m_status=optimal ? Optimal : Inconsistent;
  return false;
}

template <class scalar>
int GeneralSimplex<scalar>::FindPrimalFeasibleBasis()
{ /* Find a dual feasible basis using Phase I of Dual Simplex method.
     If the problem is dual feasible,
     the procedure returns m_status=LPSundecided and a dual feasible
     basis.   If the problem is dual infeasible, this returns
     m_status=DualInconsistent and the evidence column.
  */

  long rank=0;
  pivot_t pivot;
  this->m_evidenceCol=-1;
  this->m_status=LPSundecided;
  while (this->SelectPrimalSimplexPivot(pivot))  {
    this->ColumnPivotAndUpdate(pivot);
    rank++;
  }
  return rank;
}

template <class scalar>
int GeneralSimplex<scalar>::FindFeasLPBasis()
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
  if (this->ms_trace_basis) this->ms_logger.logData("Feasibility Basis");
  this->ResetTableau();
  Set RowSelected(this->m_size);
  Set ColSelected(this->m_dimension);
  RowSelected.add(this->m_objectiveRow);
  ColSelected.add(RHSCol);
  pivot_t pivot;
  int rank=this->m_dimension;
  this->m_evidenceCol=-1;
  for (int i=0;i<this->m_dimension;i++) {   /* Find a set of rows for a basis */
    if (!FastSelectPivot(this->m_size, RowSelected, ColSelected, pivot))
    {
      rank=i;
      for (int j=1;j<this->m_dimension; j++) {//Skip RHSCol
        if (this->m_nonBasicRow[j]<0){
          if (!func::isZero(this->entry(this->m_objectiveRow,j),this->m_zero)) {  /* dual inconsistent */
            this->m_evidenceCol=j;
            break;
          }
        }
      }
      /* dependent columns but not dual inconsistent. */
      break;
    }
    if (this->ms_trace_basis>=eTraceAll) {
      RowSelected.logSet("Available Rows:",true);
      ColSelected.logSet("Available Cols:",true);
    }
    RowSelected.add(pivot.row);
    ColSelected.add(pivot.col);
    this->ColumnPivotAndUpdate(pivot);
  }
  return rank;
}

template <class scalar>
void GeneralSimplex<scalar>::SetSolutions(scalar* sol, scalar* dsol)
/* 
Assign the solution vectors to sol, dsol, *optvalue after solving
the LP.
*/
{
  scalar sw;

  switch (this->m_status){
  case Optimal:
    for (int j=0;j<this->m_dimension; j++) {
      sol[j]=this->m_basisInverse.coeff(j,RHSCol);
      dsol[j]=-this->entry( this->m_objectiveRow,j);
    }
    break;
  case Inconsistent:
    for (int j=0;j<this->m_dimension; j++) {
      sol[j]=this->m_basisInverse.coeff(j,RHSCol);
      dsol[j]=-this->entry(this->m_evidenceRow,j);
    }
    break;
  case DualInconsistent:
    for (int j=0;j<this->m_dimension; j++) {
      sol[j]=this->m_basisInverse.coeff(j,this->m_evidenceCol);
      dsol[j]=-this->entry(this->m_objectiveRow,j);
    }
    break;

  case StrucDualInconsistent:
    if (func::isPositive(this->entry(this->m_objectiveRow, this->m_evidenceCol))) sw=1;
    else sw=-1;
    for (int j=0;j<this->m_dimension; j++) {
      sol[j]=sw*this->m_basisInverse.coeff(j,this->m_evidenceCol);
      dsol[j]=-this->entry(this->m_objectiveRow,j);
    }
    break;
  default:break;
  }
}

template <class scalar>
scalar GeneralSimplex<scalar>::solve(ConversionType_t lpconv, SolverType_t solver, int UseExisting, scalar *sol, scalar *dsol)
/* 
When LP is inconsistent then m_evidenceRow returns the evidence row.
When LP is dual-inconsistent then m_evidenceCol returns the evidence column.
*/
{
  scalar result;
  if (lpconv==LPMin) this->m_tableau.row(this->m_objectiveRow)=-this->m_tableau.row(this->m_objectiveRow);
  if (solver==eCrissCross)    ;//result=CrissCrossMaximize(UseExisting,optvalue, sol, dsol,m_evidenceRow, m_evidenceCol, iter);
  else                       result=processMaximize(UseExisting);
  SetSolutions(sol, dsol);
  if (lpconv==LPMin) {
    result=-result;
    this->m_tableau.row(this->m_objectiveRow)=-this->m_tableau.row(this->m_objectiveRow);
    for (int j=0; j<this->m_dimension; j++) dsol[j]=-dsol[j];
  }
  return result;
}

template <class scalar>
scalar GeneralSimplex<scalar>::processMaximize(int UseExisting)
/*
When LP is inconsistent then *re returns the evidence row.
When LP is dual-inconsistent then *se returns the evidence column.
*/
{
  boost::timer timer;
  long rank=0;
  long maxpivots, maxpivfactor=70;
  this->Error=this->None;
  func::setZero(this->m_zero);
  maxpivots=maxpivfactor*this->m_dimension;  /* maximum pivots to be performed before cc pivot is applied. */
  /* Initializing control variables. */

  this->m_evidenceRow=-1;
  this->m_evidenceCol=-1;

  this->m_iterations=0;

  if (this->ms_trace_basis>=eTraceResults) {
    this->ms_logger.logData("Maximise");
    this->ms_logger.logData(this->m_tableau);
  }

  this->m_iterations+=this->FindFeasBasis(UseExisting);
  if (this->ms_trace_time && !UseExisting) {
    this->logPivotCount(timer.elapsed()*1000,"Find Feasible:");
  }
  this->m_status=LPSundecided;
  bool dual=true;
  if (dual) {
    GeneralSimplex<scalar>::processMaximize(UseExisting);
  }
  else if (this->m_evidenceCol<0) this->m_iterations+=FindPrimalFeasibleBasis();
  this->m_iterations+=rank;
  if (this->ms_trace_time) {
    this->logPivotCount(timer.elapsed()*1000,"Find Support:");
  }
  return this->entry( this->m_objectiveRow,RHSCol);
}


#ifdef USE_LDOUBLE
  #ifdef USE_SINGLES
    template class GeneralSimplex<long double>;
  #endif
  #ifdef USE_INTERVALS
    template class GeneralSimplex<ldinterval>;
  #endif
#endif
#ifdef USE_MPREAL
  #ifdef USE_SINGLES
    template class GeneralSimplex<mpfr::mpreal>;
  #endif
  #ifdef USE_INTERVALS
    template class GeneralSimplex<mpinterval>;
  #endif
#endif

}
