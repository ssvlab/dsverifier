//Authors: Dario Cattaruzza, Alessandro Abate, Peter Schrammel, Daniel Kroening
//University of Oxford 2016
//This code is supplied under the BSD license agreement (see license.txt)

#include "RowSort.h"
#include <map>
#include <math.h>

namespace abstract {
 /*  quicksorts matrix rows based on the column order (first column is most significant)
 */
template <class scalar>
scalar SortedMatrix<scalar>::ms_one(1);

template <class scalar>
SortedMatrix<scalar>::SortedMatrix(int rows,int cols) : Eigen::Matrix<scalar,Eigen::Dynamic,Eigen::Dynamic>(rows,cols) {}

template <class scalar>
bool SortedMatrix<scalar>::SmallerAngleRow(const int row1,const int row2)
{
  for (int j=0;j<cols()-1;j++) {
    scalar angle1=coeff(row1,j)/coeff(row1,j+1);
    angle1=atan(angle1);
    if (func::isNegative(coeff(row1,j))) angle1=func::const_pi(ms_one)-angle1;
    if (func::isNegative(coeff(row1,j+1))) angle1+=func::const_pi(ms_one);
    scalar angle2=coeff(row2,j)/coeff(row2,j+1);
    angle2=atan(angle2);
    if (func::isNegative(coeff(row2,j))) angle2=func::const_pi(ms_one)-angle2;
    if (func::isNegative(coeff(row2,j+1))) angle2+=func::const_pi(ms_one);
    angle1-=angle2;
    char sign=func::hardSign(angle1);
    if (sign!=0) return (sign<0);
  }
  return false;
}

template <class scalar>
bool SortedMatrix<scalar>::LargerAngleRow(const int row1,const int row2)
{
  for (int j=0;j<cols()-1;j++) {
    scalar angle1=coeff(row1,j)/coeff(row1,j+1);
    angle1=atan(angle1);
    if (coeff(row1,j)<func::ms_hardZero) angle1+=func::const_pi(ms_one);
    scalar angle2=coeff(row2,j)/coeff(row2,j+1);
    angle2=atan(angle2);
    if (func::isNegative(coeff(row2,j))) angle2+=func::const_pi(ms_one);
    angle1-=angle2;
    char sign=func::hardSign(angle1);
    if (sign!=0) return (sign>0);
  }
  return false;
}

template <class scalar>
bool SortedMatrix<scalar>::SmallerRow(const int row1,const int row2)
{
  for (int j=0;j<cols();j++) {
    scalar result=coeff(row1,j)-coeff(row2,j);
    char sign=func::hardSign(result);
    if (sign!=0) return (sign<0);
  }
  return false;
}

template <class scalar>
bool SortedMatrix<scalar>::LargerRow(const int row1,const int row2)
{
  for (int j=0;j<cols();j++) {
    scalar result=coeff(row1,j)-coeff(row2,j);
    char sign=func::hardSign(result);
    if (sign!=0) return (sign > 0);
  }
  return false;
}

template <class scalar>
long SortedMatrix<scalar>::subPartition(std::vector<int> &order, const long p, const long r)
{
  long i,j,tmp;
  i=p-1;
  j=r+1;
  while (true){
    while (LargerRow(order[--j]-1,order[p]-1));
    while (SmallerRow(order[++i]-1,order[p]-1));
    if (i<j){
      tmp=order[i];
      order[i]=order[j];
      order[j]=tmp;
    }
    else return j;
  }
}

template <class scalar>
long SortedMatrix<scalar>::anglePartition(std::vector<int> &order, const long p, const long r)
{
  long i,j,temp;
  i=p-1;
  j=r+1;
  while (true){
    while (LargerAngleRow(order[--j]-1,order[p]-1));
    while (SmallerAngleRow(order[++i]-1,order[p]-1));
    if (i<j){
      temp=order[i];
      order[i]=order[j];
      order[j]=temp;
    }
    else return j;
  }
}

template <class scalar>
void SortedMatrix<scalar>::QuickMatrixSort(std::vector<int> &order, const long p, const long r)
{
  if (p < r){
    long q = subPartition(order, p, r);
    QuickMatrixSort(order, p, q);
    QuickMatrixSort(order, q+1, r);
  }
}

template <class scalar>
void SortedMatrix<scalar>::QuickAngleSort()
{
  std::multimap<refScalar,int> preorder;
  MatrixS centre=this->colwise().sum()/rows();
  for (int row=0;row<rows();row++) {
    scalar angle=(coeff(row,1)-centre.coeff(0,1))/(coeff(row,0)-centre.coeff(0,0));
    angle=atan(angle);
    scalar dif=coeff(row,0)-centre.coeff(0,0);
    if (func::isNegative(dif)) angle+=func::const_pi(ms_one);
    preorder.insert(std::pair<refScalar,int>(func::toCentre(angle),row));
  }
  m_order.resize(rows());
  typename std::map<refScalar,int>::iterator it=preorder.begin();
  for (unsigned int row=0;row<rows() && row<preorder.size();row++,it++) {
    m_order[row]=it->second;
  }
}

template <class scalar>
void SortedMatrix<scalar>::ComputeRowOrderVector(const OrderType &type)
{
  int numRows=rows()+1;
  if (m_order.size()<numRows+1) m_order.resize(numRows+1);
  switch(type) {
  case MaxIndex:
    for(int i=1; i<=numRows; i++) m_order[i]=numRows-i;
    break;
  case MinIndex:
    for(int i=1; i<=numRows; i++) m_order[i]=i;
    break;
  case LexMin:
    for(int i=0; i<=numRows; i++) m_order[i]=i;
    QuickMatrixSort(m_order, 1, rows());
    break;
  case LexMax:
    for(int i=0; i<=numRows; i++) m_order[i]=i;
    QuickMatrixSort(m_order, 1, rows());
    for(int i=1; i<=rows()/2;i++){   /* just reverse the order */
      int temp=m_order[i];
      m_order[i]=m_order[numRows-i];
      m_order[numRows-i]=temp;
    }
    break;
  case LexAngle:
    QuickAngleSort();
    break;
  }
}

template <class scalar>
void SortedMatrix<scalar>::UpdateRowOrderVector(const Set &priorityRows)
// Update the RowOrder vector to shift selected rows in highest order.
{
  bool found=true;
  long rr=priorityRows.cardinality();
  for (int i=1; i<=rr; i++){
    found=false;
    for (int j=i; j<m_order.size() && !found; j++){
      int oj=m_order[j];
      if (priorityRows.member(oj-1)){//TODO: zero index
        found=true;
        if (j>i) {
          /* shift everything lower: pOrder[i]->pOrder[i+1]..pOrder[j1-1]->pOrder[j1] */
          for (int k=j; k>=i; k--) m_order[k]=m_order[k-1];
          m_order[i]=oj;
        }
      }
    }
    if (!found) return;
  }
}

#ifdef USE_LDOUBLE
  #ifdef USE_SINGLES
    template class SortedMatrix<long double>;
  #endif
  #ifdef USE_INTERVALS
    template class SortedMatrix<ldinterval>;
  #endif
#endif
#ifdef USE_MPREAL
  #ifdef USE_SINGLES
    template class SortedMatrix<mpfr::mpreal>;
  #endif
  #ifdef USE_INTERVALS
    template class SortedMatrix<mpinterval>;
  #endif
#endif

}
