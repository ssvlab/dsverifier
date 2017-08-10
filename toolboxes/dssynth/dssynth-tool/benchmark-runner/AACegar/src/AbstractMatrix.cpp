//Authors: Dario Cattaruzza, Alessandro Abate, Peter Schrammel, Daniel Kroening
//University of Oxford 2016
//This code is supplied under the BSD license agreement (see license.txt)

#include <math.h>

#include <Eigen/Eigenvalues>

#include <boost/timer.hpp>

#include "AbstractMatrix.h"

namespace abstract{

/// Constructs an empty buffer
template <class scalar>
AbstractMatrix<scalar>::AbstractMatrix(int dimension) : AccelMatrix<scalar>(dimension),
  //m_roundDimension(dimension),
  m_abstractDynamics(dimension),
  m_abstractInputDynamics(dimension),
  m_abstractRoundDynamics(dimension)
  {
    m_abstractDynamics.setName("Abstract Dynamics");
    m_abstractRoundDynamics.setName("Round Dynamics");
  }

/// Changes the default dimension of the system
template <class scalar>
void AbstractMatrix<scalar>::changeDimensions(const int dimension)
{
  AccelMatrix<scalar>::changeDimensions(dimension);
  m_abstractDynamics.changeDimension(dimension);
  m_abstractInputDynamics.changeDimension(dimension);
  m_abstractRoundDynamics.changeDimension(dimension);
}

template <class scalar>
void AbstractMatrix<scalar>::swapValues(scalar &low,scalar &high)
{
  scalar temp=low;
  low=high;
  high=temp;
}

template <class scalar>
inline void AbstractMatrix<scalar>::checkRange(const int row,const powerS iteration,const scalar mag,scalar &min,scalar &max)
{
  if (func::isPositive(min-max)) {
    scalar temp=min;
    min=max;
    max=temp;
  }
  if ((m_zeniths[row]>0) && (m_zeniths[row]<=iteration)) {
    scalar mid=condPow(mag,m_zeniths[row],row);
    if (func::isPositive(mid-max)) max=mid;
    if (m_zeniths[row]<iteration) {
      mid=condPow(mag,m_zeniths[row]+1,row);
      if (func::toLower(mid)>func::toUpper(max)) max=mid;
    }
  }
  if (m_freq[row]>0) {
    if (m_freq[row]<=iteration) {
    }
    else if (row>m_dimension) {
      if (func::toUpper(min)<-func::toLower(max)) min=-max;
      else max=-min;
    }
  }
}

template <class scalar>
typename JordanMatrix<scalar>::complexS AbstractMatrix<scalar>::condPow(const complexS &coef,const powerS n,int row)
{
  if (row>=m_dimension) {
    if (m_inputType==eParametricInputs) row-=m_dimension;
    else {
      scalar mag=ms_one-func::pow(func::norm2(coef),n);
      if (func::isPositive(mag)) {
        return complexS(mag*this->m_foldedBinomialMultipliers.coeff(row-m_dimension),0);
      }
    }
    if (m_jordanIndex[row]==0) {
      if (func::isZero(func::norm2(coef-ms_complexOne))) return complexS(n,0);
      return this->ms_complexOne-func::c_pow(coef,n);
    }
    if (n<m_jordanIndex[row]) return complexS(0,0);
    if (func::isZero(func::norm2(coef-ms_complexOne))) {
      if (n<=m_jordanIndex[row]) return complexS(0,0);
      complexS mult(binomial(n,m_jordanIndex[row]+1),0);
      return mult;
    }
    complexS mult(binomial(n,m_jordanIndex[row]),0);
    return -mult*func::c_pow(coef,n-m_jordanIndex[row]);
  }
  if (m_jordanIndex[row]==0) {
    if (n==1) return coef;
    return func::c_pow(coef,n);
  }
  if (n<m_jordanIndex[row]) return complexS(0,0);
  complexS mult(binomial(n,m_jordanIndex[row]),0);
  return mult*func::c_pow(coef,n-m_jordanIndex[row]);
}

template <class scalar>
scalar AbstractMatrix<scalar>::condPow(const scalar &coef,const powerS n,int row)
{
  if (row>=m_dimension) {
    if (m_inputType==eParametricInputs) row-=m_dimension;
    else {
      scalar mag=ms_one-func::pow(coef,n);
      if (func::isPositive(mag)) {
        return mag*this->m_foldedBinomialMultipliers.coeff(row-m_dimension);
      }
    }
    if (m_jordanIndex[row]==0) {
      if (abs(coef-ms_one)<m_zero) return scalar(n);
      return ms_one-func::pow(coef,n);
    }
    if (n<m_jordanIndex[row])  return 0;
    if (abs(coef-ms_one)<m_zero) {
      if (n<=m_jordanIndex[row])  return 0;
      return binomial(n,m_jordanIndex[row]+1);
    }
    return -binomial(n,m_jordanIndex[row])*func::pow(coef,n-m_jordanIndex[row]);
  }

  if (m_jordanIndex[row]==0) return func::pow(coef,n);
  if (n<m_jordanIndex[row])  return 0;
  return binomial(n,m_jordanIndex[row])*func::pow(coef,n-m_jordanIndex[row]);
}

template <class scalar>
scalar AbstractMatrix<scalar>::diffPow(const scalar &coef,const powerS n,int row)
{
  if (row>=m_dimension) {
    if (m_inputType==eParametricInputs) row-=m_dimension;
    else {
      scalar mag=ms_one-func::pow(coef,n);
      if (func::isPositive(mag)) {
        mag=func::pow(coef,n)-func::pow(coef,n+1);
        return mag*this->m_foldedBinomialMultipliers.coeff(row-m_dimension);
      }
    }
    if (m_jordanIndex[row]==0) {
      if (abs(coef-ms_one)<m_zero) return ms_one;
      return func::pow(coef,n)-func::pow(coef,n+1);
    }
    if (n<m_jordanIndex[row])  return 0;
    if (abs(coef-ms_one)<m_zero) {
      if (n<m_jordanIndex[row])  return 0;
      return binomial(n,m_jordanIndex[row]);
    }
    return (binomial(n,m_jordanIndex[row])-binomial(n+1,m_jordanIndex[row])*coef)*func::pow(coef,n-m_jordanIndex[row]);
  }
  if (m_jordanIndex[row]==0) return (coef-ms_one)*func::pow(coef,n);
  if (n<m_jordanIndex[row])  return 0;
  return (binomial(n+1,m_jordanIndex[row])*coef-binomial(n,m_jordanIndex[row]))*func::pow(coef,n-m_jordanIndex[row]);
}

/// Calculates the expected iteration of coefficient to obtain value
template <class scalar>
typename AbstractMatrix<scalar>::powerS AbstractMatrix<scalar>::condLog(const refScalar &coef,refScalar value,int row,powerS &range)
{
  range=0;
  if (row>=m_dimension) {
    if (m_inputType==eParametricInputs) row-=m_dimension;
    else {
      if (func::isPositive(value)) value/=func::toCentre(this->m_foldedBinomialMultipliers.coeff(row-m_dimension));
      value=func::ms_1-value;
      if (func::isNegative(value)) return 0;
      return func::toInt(func::log(coef,value));
    }
    if (m_jordanIndex[row]==0) {
      if (abs(coef-ms_one)<m_zero) return func::toInt(value);
      value=func::ms_1-value;
      if (func::isNegative(value)) return 0;
      return func::toInt(func::log(coef,value));
    }
    range=m_jordanIndex[row]-1;
    if (func::isZero(value)) return 1;
    if (abs(coef-ms_one)<m_zero) {
      for (int i=2;i<=m_jordanIndex[row]+1;i++) value*=i;
      refScalar expected=pow(value,1/m_jordanIndex[row]+1);
      return func::toInt(expected);
    }
    value=-value;
    return binomialCondLog(coef,value,row,range);
  }
  if (m_jordanIndex[row]==0) return func::toInt(func::log(coef,value));
  range=m_jordanIndex[row]-1;
  if (func::isZero(value)) return 1;
  return binomialCondLog(coef,value,row,range);
}

/// Calculates the expected iteration of coefficient to obtain value for a jordan diagonal
template <class scalar>
typename AbstractMatrix<scalar>::powerS AbstractMatrix<scalar>::binomialCondLog(const refScalar &coef,refScalar value,int row,powerS &range)
{
  for (int i=2;i<=m_jordanIndex[row];i++) value*=i;
  value/=func::pow(coef,m_jordanIndex[row]);
  powerS low,high;
  if (coef>func::ms_1) {
    high=func::toInt(func::log(coef,value));
    refScalar bin=func::toCentre(binomial(high,m_jordanIndex[row]));
    low=func::toInt(func::log(coef,value/bin));
  }
  else {
    low=func::toInt(pow(value,1/m_jordanIndex[row]));
    refScalar bin=func::toCentre(binomial(low,m_jordanIndex[row]));
    high=func::toInt(func::log(coef,value/bin));
  }
  range=high-low;
  return low;
}


template <class scalar>
void AbstractMatrix<scalar>::fillDirection(const int row,const scalar &min,const scalar &max)
{
  m_directions.coeffRef(row,m_pos)=1;
  m_supports.coeffRef(m_pos++,0)=max;
  m_directions.coeffRef(row,m_pos)=-1;
  m_supports.coeffRef(m_pos++,0)=-min;
}

template <class scalar>
void AbstractMatrix<scalar>::fillDirections(const int row1,const int row2,const scalar &dir1,const scalar &dir2,const scalar &min,const scalar &max)
{
  m_supports.coeffRef(m_pos,0)=-min;
  m_directions.coeffRef(row1,m_pos)=-dir1;
  m_directions.coeffRef(row2,m_pos++)=-dir2;
  m_supports.coeffRef(m_pos,0)=max;
  m_directions.coeffRef(row1,m_pos)=dir1;
  m_directions.coeffRef(row2,m_pos++)=dir2;
}

template <class scalar>
void AbstractMatrix<scalar>::fillConjugateSupport(const int row1,const int row2,const scalar &angle,const scalar &max)
{
  m_directions.coeffRef(row1,m_pos)=func::cosine(angle);
  m_directions.coeffRef(row2,m_pos)=func::sine(angle);
  m_supports.coeffRef(m_pos++)=max;
}

template <class scalar>
void AbstractMatrix<scalar>::fillSupportFromPoints(const int row1,const int row2,const scalar &x1,const scalar &y1,const scalar &x2,const scalar &y2,const scalar &xRef,const scalar &yRef)
{
  scalar angle=func::invtan(y2-y1,x2-x1)+func::const_pi(this->ms_half);
  scalar max=y1*func::sine(angle)+x1*func::cosine(angle);
  scalar out=yRef*func::sine(angle)+xRef*func::cosine(angle);
  out-=max;
  if (func::isPositive(out)) {
    angle+=func::const_pi(ms_one);
    max=-max;
  }
  fillConjugateSupport(row1,row2,angle,max);
}

template <class scalar>
void AbstractMatrix<scalar>::fillTangentSupport(const int row1,const int row2,const scalar &mag1,const scalar &mag2,powerS iter)
{
  scalar angle=func::const_pi(this->ms_half);
  scalar quotient=mag2/mag1;
  scalar slope=iter;
  slope*=func::pow(quotient,iter-1);
  char sign=func::hardSign(mag1);
  if (sign==0) return;
  if (sign<0)  angle+=atan(slope)+func::ms_pi;
  else         angle+=atan(slope);
  scalar max=func::pow(mag2,iter)*func::sine(angle)+func::pow(mag1,iter)*func::cosine(angle);
  fillConjugateSupport(row1,row2,angle,max);
}

template <class scalar>
void AbstractMatrix<scalar>::fillSupportFromIter(const int row1,const int row2,const scalar &mag1,const scalar &mag2,powerS iter)
{
  scalar xdif=diffPow(mag1,iter,row1);
  scalar ydif=diffPow(mag2,iter,row2);
  scalar angle=func::invtan(ydif,xdif)+func::const_pi(this->ms_half);
  scalar x1=condPow(mag1,iter,row1);
  scalar y1=condPow(mag2,iter,row2);
  scalar max=y1*func::sine(angle)+x1*func::cosine(angle);
  scalar out=mag2*func::sine(angle)+mag1*func::cosine(angle);
  out-=max;
  if (func::isPositive(out)) {
    angle+=func::const_pi(ms_one);
    max=-max;
  }
  fillConjugateSupport(row1,row2,angle,max);
}

template <class scalar>
void AbstractMatrix<scalar>::fillSemiConjugateSupportFromIter(const int row1,const int row2,const scalar &mag1,const scalar &mag2,powerS iter)
{
  powerS iter2=iter+1;
  scalar x1=condPow(mag1,iter,row1);
  scalar y1=condPow(mag2,iter,row2);
  scalar x2=condPow(mag1,iter2,row1);
  scalar y2=condPow(mag2,iter2,row2);
  fillSupportFromPoints(row1,row2,x1,y1,x2,y2,mag1,mag2);
  fillSupportFromPoints(row1,row2,-x1,y1,-x2,y2,-mag1,mag2);
}

template <class scalar>
bool AbstractMatrix<scalar>::fillLastConjugateSupportFromPoints(int row1,int row2,scalar x1,scalar y1,scalar x2,scalar y2,scalar xRef,scalar yRef,bool hasLastQuarter)
{
  scalar angle=func::invtan(y2-y1,x2-x1)+func::const_pi(this->ms_half);
  scalar max=y1*func::sine(angle)+x1*func::cosine(angle);
  scalar out=yRef*func::sine(angle)+xRef*func::cosine(angle);
  char sign=func::hardSign(max);
  if (sign<0) {//the vector is in the wrong direction rotate 180deg
    angle+=func::const_pi(ms_one);
    max=-max;
    out=-out;
  }
  else if (sign==0) {// Undecided, overapproximate by error.
    max+=m_zero;
    out+=m_zero;
  }
  out-=max;
  sign=func::hardSign(out);
  if (hasLastQuarter) {
    //If the next to last point would be excluded by the restriction, ignore
    if (sign>0) return false;
    //If the next to last point is undecided around the restriction, make sure it remains inside
    if (sign==0) max+=m_zero;
  }
  else if (sign>0) {
    //Revert the bound to close the half-circle
    angle+=func::const_pi(ms_one);
    max=y1*func::sine(angle)+x1*func::cosine(angle);
  }
  else if (sign==0) {
    //should never happen
    func::imprecise(out,func::ms_hardZero);
    return false;
  }
  else {
    //should never happen but no error
    return false;
  }
  fillConjugateSupport(row1,row2,angle,max);
  return true;
}

template <class scalar>
bool AbstractMatrix<scalar>::testConjugateSupportFromPoints(const scalar &x1,const scalar &y1,const scalar &x2,const scalar &y2,const scalar &xExt,const scalar &yExt)
{
  scalar angle=func::invtan(y2-y1,x2-x1)+func::const_pi(this->ms_half);
  scalar max=y1*func::sine(angle)+x1*func::cosine(angle);
  scalar ext=yExt*func::sine(angle)+xExt*func::cosine(angle);
  char sign=func::hardSign(max);
  if (sign<0) {
    angle+=func::const_pi(ms_one);
    max=-max;
    ext=-ext;
  }
  else if (sign==0) {
    //should never happen
    func::imprecise(max,func::ms_hardZero);
  }
  ext-=max;
  sign=func::hardSign(ext);
  if (sign>0) return false;
  //if ext is zero, the case is dealt when calculating the support function
  return true;
}

template <class scalar>
bool AbstractMatrix<scalar>::fillConjugateSupportFromPoints(const int row1,const int row2,const scalar &x1,const scalar &y1,const scalar &x2,const scalar &y2,const scalar &xExt,const scalar &yExt)
{
  scalar angle=func::invtan(y2-y1,x2-x1)+func::const_pi(this->ms_half);
  scalar max=y1*func::sine(angle)+x1*func::cosine(angle);
  scalar ext=yExt*func::sine(angle)+xExt*func::cosine(angle);
  char sign=func::hardSign(max);
  if (sign<0) {
    angle+=func::const_pi(ms_one);
    max=-max;
    ext=-ext;
  }
  else if (sign==0) {
    //should never happen
    func::imprecise(max,func::ms_hardZero);
  }
  ext-=max;
  sign=func::hardSign(ext);
  if (sign>0) return false;
  if (sign==0) max+=m_zero;
  fillConjugateSupport(row1,row2,angle,max);
  return true;
}

template <class scalar>//template <>
void AbstractMatrix<scalar>::fillLinearSupportFromPoints(const int row1,const int row2,const scalar &min1,const scalar &max1,const scalar &min2,const scalar &max2)
{
  scalar angle=func::invtan(max2-min2,max1-min1)+func::const_pi(this->ms_half);
  scalar min=min2*func::sine(angle)+min1*func::cosine(angle);
  scalar max=max2*func::sine(angle)+max1*func::cosine(angle);
  max=func::getHull(min,max);
  min=-func::toLower(max)+this->m_largeZero;
  max=func::toUpper(max)+this->m_largeZero;
  fillConjugateSupport(row1,row2,angle,max);
  angle+=func::const_pi(ms_one);
  fillConjugateSupport(row1,row2,angle,min);
}

template <class scalar>
void AbstractMatrix<scalar>::fillQuadraticSupport(int row1,int row2,scalar mag1,scalar mag2,powerS iteration,int precision)
{
  precision=(1<<precision)-2;//Number of faces grows geometrically
  if (precision<1) return;
  refScalar iterationStep=iteration/precision;
  if (iterationStep<1) {
    iterationStep=1;
    precision=iteration;
  }
  int firstIter=1;
  if (m_jordanIndex[row1%m_dimension]>0) firstIter=m_jordanIndex[row1%m_dimension];
  if (m_jordanIndex[row2%m_dimension]>0) firstIter=m_jordanIndex[row2%m_dimension];
  for (int i=firstIter;i<precision;i++) {
    powerS iter1=func::toInt(floor(i*iterationStep));
    if (iter1==1) {
      powerS iter2=iter1+1;
      scalar x1=condPow(mag1,iter1,row1);
      scalar y1=condPow(mag2,iter1,row2);
      scalar x2=condPow(mag1,iter2,row1);
      scalar y2=condPow(mag2,iter2,row2);
      scalar x3=condPow(mag1,3,row1);
      scalar y3=condPow(mag2,3,row2);
      fillSupportFromPoints(row1,row2,x1,y1,x2,y2,x3,y3);
    }
    else fillSupportFromIter(row1,row2,mag1,mag2,iter1);
  //    else fillSupportFromPoints(row1,row2,x1,y1,x2,y2,mag1,mag2);
  }
}

template <class scalar>
void AbstractMatrix<scalar>::fillQuadraticConjugateSupport(int row1,int row2,scalar mag1,scalar mag2,powerS iteration,int precision)
{
  scalar x1=condPow(mag1,1,row1);
  scalar y1=condPow(mag2,1,row2);
  scalar x2=condPow(mag1,iteration,row1);
  scalar y2=condPow(mag2,iteration,row2);
  scalar x3=condPow(mag1,2,row1);
  scalar y3=condPow(mag2,2,row2);
  scalar angle=func::invtan(y2-y1,x2-x1)-func::const_pi(this->ms_half);//invtan returns positive values (0-pi)
  scalar max=y1*func::sine(angle)+x1*func::cosine(angle);
  scalar out=y3*func::sine(angle)+x3*func::cosine(angle);
  max-=out;
  if (func::isNegative(max)) {//TODO: need to split for jordan blocks
    precision=(1<<precision)-2;//Number of faces grows geometrically
    if (precision<1) return;
    scalar mag=condPow(mag1,iteration,row1);
    while(func::isZero(mag)) {
      iteration>>=2;
      mag=condPow(mag1,iteration,row1);
    }
    refScalar iterationStep=iteration/precision;
    if (iterationStep<1) {
      iterationStep=1;
      precision=iteration;
    }
    for (int i=1;i<precision;i++) {
      powerS iter1=func::toInt(floor(i*iterationStep));
      powerS iter2=iter1+1;
      scalar x1=condPow(mag1,iter1,row1);
      scalar y1=condPow(mag2,iter1,row2);
      scalar x2=condPow(mag1,iter2,row1);
      scalar y2=condPow(mag2,iter2,row2);
      if (iter1==1) {
        scalar x3=condPow(mag1,3,row1);
        scalar y3=condPow(mag2,3,row2);
        fillSupportFromPoints(row1,row2,x1,y1,x2,y2,x3,y3);
        fillSupportFromPoints(row1,row2,-x1,y1,-x2,y2,-x3,y3);
      }
      else {
        fillSupportFromPoints(row1,row2,x1,y1,x2,y2,mag1,mag2);
        fillSupportFromPoints(row1,row2,-x1,y1,-x2,y2,-mag1,mag2);
      }
    }
    return;
  }
  fillSupportFromPoints(row1,row2,x1,y1,x2,y2,0,y1);
  fillSupportFromPoints(row1,row2,-x1,y1,-x2,y2,0,y1);
}

template <class scalar>
void AbstractMatrix<scalar>::findConjugateSupports(const int row,const complexS &coef,const powerS iteration,int precision)
{
  if (m_conjugatePair[row]<row) return;
  precision=(1<<precision);//Number of faces grows geometrically
  int row2=row+1;
  if (iteration<=2) {
    complexS p1=condPow(coef,1,row);
    scalar minR=p1.real();
    scalar minI=p1.imag();
    if (iteration==2) {
      complexS p2=condPow(coef,2,row);
      scalar maxR=p2.real();
      scalar maxI=p2.imag();
      fillLinearSupportFromPoints(row,row2,minR,maxR,minI,maxI);
      if (func::isPositive(minR-maxR)) swapValues(minR,maxR);
      if (func::isPositive(minI-maxI)) swapValues(minI,maxI);
      fillDirection(row,minR,maxR);
      fillDirection(row2,minI,maxI);
    }
    else {
      fillDirection(row,minR,minR);
      fillDirection(row2,minI,minI);
    }
    return;
  }
  scalar mag=func::norm2(coef);
  scalar angle=func::invtan(coef.imag(),coef.real());
  scalar quartFreq=abs(func::const_pi(this->ms_half)/angle);
  powerS offset=func::toInt(trunc(4*func::toLower(quartFreq)));
  offset+=m_jordanIndex[row]+1;
  powerS start=(func::isPositive(mag-ms_one)) ? iteration : 1;
  powerS end=func::isPositive(mag-ms_one) ? ((iteration>offset) ? iteration-offset : 0) : ((iteration>offset) ? offset : iteration);
  int dir=((end-start)>0) ? 1 : -1;
  powerS step=(end-start)/precision;
  if (step==0) step=dir;
  complexS ext=condPow(coef,start+2*dir,row);
  complexS p1=condPow(coef,start,row);
  complexS p2=condPow(coef,start+dir,row);
  fillConjugateSupportFromPoints(row,row2,p1.real(),p1.imag(),p2.real(),p2.imag(),ext.real(),ext.imag());
  start+=step;
  ext=condPow(coef,func::isPositive(mag-ms_one) ? iteration : 1,row2);
  while (abs(end-start)>=abs(step)) {
    p1=condPow(coef,start,row);
    p2=condPow(coef,start+dir,row);
    if (!fillConjugateSupportFromPoints(row,row2,p1.real(),p1.imag(),p2.real(),p2.imag(),ext.real(),ext.imag())) break;
    start+=step;
  }
  if (abs(end-start)<abs(step)) {
    p1=condPow(coef,end,row);
    p2=condPow(coef,end-dir,row);
    if ((abs(end-start)<=abs(dir)) || fillConjugateSupportFromPoints(row,row2,p1.real(),p1.imag(),p2.real(),p2.imag(),ext.real(),ext.imag())) {
      start=end;
      p1=condPow(coef,start,row);
      p2=ext;
      ext=condPow(coef,start-dir,row);
      if (fillLastConjugateSupportFromPoints(row,row2,p1.real(),p1.imag(),p2.real(),p2.imag(),ext.real(),ext.imag(),iteration>func::toInt(trunc(3*func::toLower(quartFreq))))) return;
      ext=p2;
    }
  }
  powerS base=start-step;
  end=start;
  while (abs(end-base)>1) {
    start=(base+end)/2;
    p1=condPow(coef,start,row);
    p2=condPow(coef,start+dir,row);
    if (testConjugateSupportFromPoints(p1.real(),p1.imag(),p2.real(),p2.imag(),ext.real(),ext.imag())) {
      base=start;
    }
    else {
      end=start;
    }
  }
  p1=condPow(coef,base,row);
  p2=condPow(coef,base-dir,row);
  if (testConjugateSupportFromPoints(p1.real(),p1.imag(),p2.real(),p2.imag(),ext.real(),ext.imag())) base+=dir;
  p1=condPow(coef,base,row);
  fillSupportFromPoints(row,row2,p1.real(),p1.imag(),ext.real(),ext.imag(),0,0);
}

template <class scalar>
void AbstractMatrix<scalar>::wrapUp()
{
  m_directions.conservativeResize(m_directions.rows(),m_pos);
  m_supports.conservativeResize(m_pos,1);
  m_pos=0;
}

/// Finds the point at which each jordan column is maximum
template <class scalar>
void AbstractMatrix<scalar>::findZeniths()
{
  //y_n=binomial(n,k)x^n-k = y_{n-1}*nx/(n-k) -> zenith=k/(1-x)
  m_zeniths.resize(2*m_dimension);
  for (int row=0;row<m_dimension;row++) {
    scalar mag=func::norm2(m_eigenValues.coeff(row,row));
    if ((m_jordanIndex[row]>0) && (func::isNegative(mag-ms_one))) {
      scalar zenith=m_jordanIndex[row];
      zenith/=(ms_one-mag);
      m_zeniths[row]=func::toInt(func::toLower(zenith));
    }
    else m_zeniths[row]=0;
  }
  for (int row=m_dimension;row<2*m_dimension;row++) m_zeniths[row]=0;
}

/// lists the first source row for a folded set of dimensions
template <class scalar>
void AbstractMatrix<scalar>::findUnfolded()
{
  m_unfolded.resize(m_dimension);
  int pos=0;
  for (int i=0;i<m_dimension;i++) {
    if (m_jordanIndex[i]>0) continue;
    if (m_conjugatePair[i]>=0) {
      m_unfolded[pos++]=i;
      i++;
    }
    else if (m_jordanIndex[i+1]>0) m_unfolded[pos++]=i;
  }
  m_unfolded.resize(pos);
}

/// Marks the round indices in a rounded vector array
template <class scalar>
void AbstractMatrix<scalar>::findRoundIndices(std::vector<bool> &isRoundIndex)
{
  int pos=0,mult;
  isRoundIndex.resize(m_dimension);
  for (int row=0;row<m_dimension;row+=mult,pos++) {
    mult=(m_conjugatePair[row]<0) ? 1 : 2;
    if (m_jordanIndex[row+mult]==0) isRoundIndex[pos]=(m_conjugatePair[row]>=0);
    else isRoundIndex[pos]=(m_jordanIndex[row+mult]!=0);
    while (m_jordanIndex[row+mult]!=0) row+=mult;
  }
  isRoundIndex.resize(pos);
}

/// Finds the frequency of rotation of each conjugate pair
template <class scalar>
void AbstractMatrix<scalar>::findFrequencies()
{
  m_freq.assign(2*m_dimension,0);
  for (int row=0;row<m_dimension;row++) {
    if (m_conjugatePair[row]>row) {
      complexS coef=m_eigenValues.coeff(row,row);
      scalar angle=func::invtan(coef.imag(),coef.real());
      m_freq[row++]=func::toInt(2*func::toUpper(abs(func::const_pi(ms_one)/angle)));
      m_freq[row]=m_freq[row-1];
    }
  }
}

/// Indicates if the matrix dynamics are divergent
/// @param strict if true returns only true if no eigenvalues are convergent
template <class scalar>
bool AbstractMatrix<scalar>::isDivergent(const bool strict)
{
  for (int i=0;i<m_dimension;i++) {
    scalar eigenNorm=func::norm2(m_eigenValues.coeff(i,i));
    char sign=func::hardSign(eigenNorm-ms_one);
    if (strict && (sign<0)) return false;
    else if (!strict && (sign>=0)) return true;
  }
  return strict;
}

/// Finds the coefficient, magnitude, maximum, and minimum for the abstract vector at row
template <class scalar>
void AbstractMatrix<scalar>::findCoeffBounds(int row,powerS iteration,complexS &coef,scalar &mag,scalar &min,scalar &max)
{
  if (row<m_dimension) coef=m_eigenValues.coeff(row,row);
  else {
    int transRow=(m_inputType==eVariableInputs) ? m_unfolded[row-m_dimension] : row-m_dimension;
    coef=m_eigenValues.coeff(transRow,transRow);
    if ((m_inputType==eVariableInputs) && (func::isPositive(func::norm2(coef)-ms_one))) {
      coef=m_foldedEigenValues(transRow,transRow);
    }
  }
  if ((row>=m_dimension) && (m_inputType==eVariableInputs)) {
    scalar rad=func::norm2(coef);
    mag=condPow(rad,1,row);
    max=condPow(rad,iteration,row);
  }
  else if (m_conjugatePair[row]<0) {
    mag=condPow(coef.real(),1,row);
    max=condPow(coef.real(),iteration,row);
  }
  else if (iteration>2) {
    mag=func::norm2(condPow(coef,1,row));
    max=func::norm2(condPow(coef,iteration,row));
  }
  else if (m_conjugatePair[row]>row) {
    mag=condPow(coef,1,row).real();
    max=condPow(coef,iteration,row).real();
  }
  else {
    mag=-condPow(coef,1,row).imag();
    max=-condPow(coef,iteration,row).imag();
  }
  min=mag;
  if ((m_conjugatePair[row]<0) && ((row<m_dimension) || (m_inputType!=eVariableInputs)) && (func::isNegative(coef.real()))) {
    scalar sum=min+max;
    char sign=func::hardSign(sum);
    if (sign>0) min=-max;
    else if (sign<0) max=-min;
  }
}

/// Retrieves the last calculated dynamics
template <class scalar>
AbstractPolyhedra<scalar>& AbstractMatrix<scalar>::getAbstractDynamics(const inputType_t inputType)
{
  if (inputType==eVariableInputs) return m_abstractRoundDynamics;
  if (this->m_hasOnes && inputType==eParametricInputs) return m_abstractInputDynamics;
  return m_abstractDynamics;
}

/// retrieves the abstract dynamics matrix for a given iteration (n)
template <class scalar>
AbstractPolyhedra<scalar>& AbstractMatrix<scalar>::getAbstractDynamics(const powerS iteration,int precision,const inputType_t inputType,const bool normalised)
{
  boost::timer timer;
  if (precision<1) precision=1;
  int dimension=m_dimension;
  m_inputType=inputType;
  if (inputType==eVariableInputs) dimension+=m_foldedEigenValues.rows();
  else if (this->m_hasOnes && (inputType==eParametricInputs)) dimension+=m_dimension;
  m_maxIterations=iteration;
  if (m_hasMultiplicities) this->calculateBoundedEigenError(calculateMaxIterations(iteration));
  findUnfolded();
  findZeniths();
  findFrequencies();
  AbstractPolyhedra<scalar>& result=getAbstractDynamics(inputType);
  result.changeDimension(dimension);
  m_supports.resize((1<<precision)*dimension*dimension,1);
  m_directions=MatrixS::Zero(dimension,(1<<precision)*dimension*dimension);
  m_pos=0;
  complexS coef,coef2;
  for (int row=0;row<dimension;row++)
  {
    scalar mag,min1,max1;
    findCoeffBounds(row,iteration,coef,mag,min1,max1);
    scalar min=min1;
    scalar max=max1;
    checkRange(row,iteration,func::norm2(coef),min,max);
    if (m_conjugatePair[row]>=0) findConjugateSupports(row,coef,iteration,precision);
    else {
      if (func::isNegative(coef.real())) {
        scalar sum=min+max;
        char sign=func::hardSign(sum);
        if (sign>0) min=-max;
        else if (sign<0) max=-min;
      }
      fillDirection(row,min,max);
    }
    if ((iteration<2) || (precision<2)) continue;
    for (int row2=row+((m_conjugatePair[row]>row) ? 2 : 1);row2<dimension;row2++) {
      if (row2<m_dimension) coef2=m_eigenValues.coeff(row2,row2);
      else {
        int transRow2=(inputType==eVariableInputs) ? m_unfolded[row2-m_dimension]: row2-m_dimension;
        coef2=m_eigenValues.coeff(transRow2,transRow2);//TODO: the rotations and dilations are wrong at the row value
        if ((inputType==eVariableInputs) && (func::isPositive(func::norm2(coef2)-ms_one))) {
          coef=m_foldedEigenValues(transRow2,transRow2);
        }
      }
      scalar mag2,min2,max2;
      findCoeffBounds(row2,iteration,coef2,mag2,min2,max2);
      if (m_conjugatePair[row]>=0) {
        if (m_conjugatePair[row2]<0) {
          fillQuadraticConjugateSupport(row,row2,func::norm2(coef),func::norm2(coef2),iteration,precision);
        }
      }
      else if (m_conjugatePair[row2]>=0) {
        fillQuadraticConjugateSupport(row2,row,func::norm2(coef2),func::norm2(coef),iteration,precision);
      }
      else if ((iteration==2) || func::isZero(max1-max2)) {
        fillLinearSupportFromPoints(row,row2,min1,max1,min2,max2);
      }
      else {
        scalar out1=condPow(min1,2,row);
        scalar out2=condPow(min2,2,row2);
        fillSupportFromPoints(row,row2,min1,min2,max1,max2,out1,out2);
        fillQuadraticSupport(row,row2,func::norm2(coef),func::norm2(coef2),iteration,precision);
      }
    }
  }
  wrapUp();
  if (m_hasMultiplicities) {
    scalar high=func::pow(ms_one+m_error,iteration);
    scalar low=func::pow(ms_one-m_error,iteration);
    scalar relaxation=func::getHull(high,low);
    m_supports*=relaxation;
  }
  result.load(m_directions,m_supports,true);
  if (normalised) result.normalise();
  result.setCalculationTime(timer.elapsed()*1000);
  if (ms_trace_dynamics>=eTraceAbstraction) {
    std::stringstream stream;
    stream << "s=" << iteration << ",l=" << precision;
    result.logTableau(stream.str());
  }
  return result;
}

/// Adds a set of supports at the given iteration in order to refine the abstraction
template <class scalar>
bool AbstractMatrix<scalar>::addSupportsAtIteration(AbstractPolyhedra<scalar>& dynamics,powerS iteration,powerS max)
{
  boost::timer timer;
  complexS coef,coef2;
  int dimension=m_dimension;
  if (m_inputType==eVariableInputs) dimension+=m_foldedEigenValues.rows();
  else if (this->m_hasOnes && (m_inputType==eParametricInputs)) dimension+=m_dimension;

  m_supports.resize(dimension*dimension,1);
  m_directions=MatrixS::Zero(dimension,dimension*dimension);
  m_pos=0;
  for (int row=0;row<dimension;row++)
  {
    if (row<m_dimension) coef=m_eigenValues.coeff(row,row);
    else {
      int transRow=(m_inputType==eVariableInputs) ? m_unfolded[row-m_dimension] : row-m_dimension;
      coef=m_eigenValues.coeff(transRow,transRow);
      if ((m_inputType==eVariableInputs) && (func::isPositive(func::norm2(coef)-ms_one))) {
        coef=m_foldedEigenValues(transRow,transRow);
      }
    }
    if (m_conjugatePair[row]>row) {
      scalar mag=func::norm2(coef);
      int dir=(func::isPositive(mag-ms_one)) ? -1 : 1;
      complexS ext=condPow(coef,func::isPositive(mag-ms_one) ? max : 1,row+1);
      complexS p1=condPow(coef,iteration,row);
      complexS p2=condPow(coef,iteration+dir,row);
      fillConjugateSupportFromPoints(row,row+1,p1.real(),p1.imag(),p2.real(),p2.imag(),ext.real(),ext.imag());
    }
    for (int row2=row+((m_conjugatePair[row]>row) ? 2 : 1);row2<dimension;row2++) {
        if (row2<m_dimension) coef2=m_eigenValues.coeff(row2,row2);
        else {
          int transRow2=(m_inputType==eVariableInputs) ? m_unfolded[row2-m_dimension]: row2-m_dimension;
          coef2=m_eigenValues.coeff(transRow2,transRow2);//TODO: the rotations and dilations are wrong at the row value
          if ((m_inputType==eVariableInputs) && (func::isPositive(func::norm2(coef2)-ms_one))) {
            coef2=m_foldedEigenValues(transRow2,transRow2);
          }
        }
        if (m_conjugatePair[row]>=0) {
          if (m_conjugatePair[row2]<0) {
            fillSemiConjugateSupportFromIter(row,row2,func::norm2(coef),func::norm2(coef2),iteration);
          }
        }
        else if (m_conjugatePair[row2]>=0) {
          fillSemiConjugateSupportFromIter(row2,row,func::norm2(coef2),func::norm2(coef),iteration);
        }
        else {
          fillSupportFromIter(row,row2,func::norm2(coef),func::norm2(coef2),iteration);
        }
      }
  }
  wrapUp();
  if (m_hasMultiplicities) {
    scalar high=func::pow(ms_one+m_error,iteration);
    scalar low=func::pow(ms_one-m_error,iteration);
    scalar relaxation=func::getHull(high,low);
    m_supports*=relaxation;
  }
  dynamics.addDirection(m_directions,m_supports);
  dynamics.addCalculationTime(timer.elapsed()*1000);
  if (ms_trace_dynamics>=eTraceAbstraction) {
    std::stringstream stream;
    stream << "abs supports n=" << iteration;
    dynamics.logTableau(stream.str());
  }
  return true;
}

/// Calculates the number of iterations necessary to reacha fixpoint
template <class scalar>
typename AbstractMatrix<scalar>::powerS AbstractMatrix<scalar>::calculateMaxIterations(powerS max)
{
  powerS result=0;
  findZeniths();
  findFrequencies();
  for (int i=0;i<m_dimension;i++) {
    complexS coef=m_eigenValues.coeff(i,i);
    if (func::isPositive(func::norm2(coef)-ms_one)) return (max<func::ms_infPower) ? max: func::ms_infPower;
    if (m_zeniths[i]+m_freq[i]>result) result=m_zeniths[i]+m_freq[i];
  }
  return (max<result) ? max : result;
}

/// Finds the corresponding iteration that generates dynamics close to the given point
template <class scalar>
void AbstractMatrix<scalar>::findIterations(MatrixS &point,powerList &iterations)
{
  complexS coef;
  powerS range;
  for (int col=0;col<point.cols();col++) {
    if (col<m_dimension) coef=m_eigenValues.coeff(col,col);
    else {
      int transCol=(m_inputType==eVariableInputs) ? m_unfolded[col-m_dimension] : col-m_dimension;
      coef=m_eigenValues.coeff(transCol,transCol);
      if ((m_inputType==eVariableInputs) && (func::isPositive(func::norm2(coef)-ms_one))) {
        coef=m_foldedEigenValues(transCol,transCol);
      }
    }
    refScalar rad=func::toCentre(func::norm2(coef));
    powerS iteration;
    if (m_conjugatePair[col]>=0) {
      scalar coeffAngle=func::invtan(coef.imag(),coef.real());
      scalar angle=func::invtan(point.coeff(0,col+1),point.coeff(0,col));
      char dir=func::hardSign(coeffAngle);
      char sign=func::hardSign(angle);
      if (sign!=dir) {
        if (sign<0) angle+=func::ms_2_pi;
        else        angle-=func::ms_2_pi;
      }
      refScalar finalSteps=func::toCentre(angle/coeffAngle);
      iteration=func::toInt(finalSteps);
      if (rad>func::ms_1) {
        refScalar tau=func::toCentre(coeffAngle/func::ms_2_pi);
        tau*=m_maxIterations;
        powerS mult=func::toInt(tau);
        mult*=m_freq[col];
        iteration+=mult;
      }
      col++;
    }
    else if ((rad>func::ms_1) || (!func::isZero(point.coeff(0,col)))) {
      iteration=condLog(rad,func::toCentre(point.coeff(0,col)),col,range);
    }
    else {
      iteration=0;
      range=0;
    }
    while ((iteration<2) && (range>0)) {
      range--;
      iteration++;
    }
    if (iteration>1) iterations[iteration]=range;//TODO: should merge when there is equality
  }
}

#ifdef USE_LDOUBLE
  #ifdef USE_SINGLES
    template class AbstractMatrix<long double>;
  #endif
  #ifdef USE_INTERVALS
    template class AbstractMatrix<ldinterval>;
  #endif
#endif
#ifdef USE_MPREAL
  #ifdef USE_SINGLES
    template class AbstractMatrix<mpfr::mpreal>;
  #endif
  #ifdef USE_INTERVALS
    template class AbstractMatrix<mpinterval>;
  #endif
#endif

}
