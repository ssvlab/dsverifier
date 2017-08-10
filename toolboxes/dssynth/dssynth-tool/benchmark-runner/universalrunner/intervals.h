//Warning! this file assumes rounding to -\infty
#ifndef INTERVALS_H_
#define INTERVALS_H_

#include "control_types.h"

typedef  __plant_precisiont control_typet;

control_typet _zero = 0.0; //added in by EP
control_typet _one = 1.0;

struct intervalt
{
  control_typet low;
  control_typet high;
};

control_typet _dbl_max;
control_typet _dbl_min;
signed long int _fxp_max;
signed long int _fxp_min;
signed long int _fxp_one;
control_typet _dbl_lsb;

void get_bounds()
{
  if(FRAC_BITS >= 31)
    _fxp_one = 2147483647l;
  else
   _fxp_one = (1 << FRAC_BITS);
  _dbl_lsb=_one/(1 << FRAC_BITS);
  _fxp_min = -(1 << (FRAC_BITS + INT_BITS -1));
  _fxp_max = (1 << (FRAC_BITS + INT_BITS-1))-1;
  _dbl_max = (1 << (INT_BITS-1))-1;//Integer part
  _dbl_max += (_one-_dbl_lsb);//Fractional part
  _dbl_min = -_dbl_max;
}

signed long int fxp_control_typet_to_fxp(control_typet value)
{
  signed long int tmp;
  control_typet ftemp=value * _fxp_one;
  tmp = ftemp;
  control_typet residue=ftemp - tmp;
  if(value < _zero && (residue != _zero))
  {
    ftemp = ftemp - _one;
    tmp = ftemp;
  }
  return tmp;
}

void fxp_check(control_typet *value)
{
#ifdef __CPROVER
  control_typet tmp_value=*value;
  if (tmp_value < _zero) tmp_value=-tmp_value;
  verify_assume((~_dbl_max&tmp_value)==0);
#else
  *value=fxp_control_typet_to_fxp(*value);
  *value/=_fxp_one;
#endif
}

struct intervalt fxp_interval_check(struct intervalt value)
{
  fxp_check(&(value.low));
  fxp_check(&(value.high));
  value.high+=_dbl_lsb;
  return value;
}

#define interval_cast(x) (struct intervalt) {.low=x, .high=x}
#define interval_lessthan(x,y) (x.high < y.low)
#define interval_lessthanzero(x) (x.high<0)
#define interval_lessthan_equal_to_zero(x) (x.high <= 0)
#define interval_greaterthan(x,y) (x.low > y.high)
#define interval_equaltozero(x) (x.high==0.0 && x.high==0.0)

#define zero_interval (struct intervalt) {.low=_zero, .high=_zero}
#define one_interval (struct intervalt) {.low=1.0, .high=1.0}
#define minusone_interval (struct intervalt) {.low=-1.0, .high=-1.0}
#define interval_add(x,y) (struct intervalt){.low=x.low+y.low, .high=x.high+y.high}
#define interval_sub(x,y) (struct intervalt) {.low=x.low-y.high, .high=x.high-y.low}

struct intervalt neg_interval(struct intervalt x)
{
  struct intervalt z;
  z.low=-x.high;
  z.high=-x.low;
  return z;
}

struct intervalt abs_interval(struct intervalt x)
{
    struct intervalt z;

    if(x.high < _zero)
    {
      z.high = -x.low;
      z.low = 0;
      return z;
    }
     else if(x.low < _zero && x.high >= _zero)
     {
       z.low = 0;
       if(-x.low > x.high)
         z.high = -z.low;
       else
         z.high = x.high;
       return z;
     }

  return x;
}

/*inline */struct intervalt interval_mult(struct intervalt x,struct intervalt y)
{
  struct intervalt z;
  z.low=x.low*y.low;
  z.high=(-x.low)*y.low;
  z.high=-z.high;
  control_typet second=x.low*y.high;
  control_typet third=x.high*y.low;
  control_typet fourth=x.high*y.high;

  if (second<z.low) z.low=second;
  else if (second>=z.high)
  {  
    z.high=(-x.low)*y.high;
    z.high=-z.high;
  }
  if (third<z.low) z.low=third;
  else if (third>=z.high) 
  {
    z.high=(-x.high)*y.low;
    z.high=-z.high;
  }
  if (fourth<z.low) z.low=fourth;
  else if (fourth>=z.high) 
  {
    z.high=(-x.high)*y.high;
    z.high=-z.high;
  }
  return z;
}

/*inline */struct intervalt interval_fxp_mult(struct intervalt x,struct intervalt y)
{
  long long int xlow=x.low*_fxp_one;
  long long int xhigh=x.high*_fxp_one;
  long long int ylow=y.low*_fxp_one;
  long long int yhigh=y.high*_fxp_one;
  
  struct intervalt z;
  long long int zlow=xlow*ylow;
  long long int zhigh=(-xlow)*ylow;
  zhigh=-zhigh;
  long long int second=xlow*yhigh;
  long long int third=xhigh*ylow;
  long long int fourth=xhigh*yhigh;

  if (second<zlow) zlow=second;
  else if (second>=zhigh)
  {  
    zhigh=(-xlow)*yhigh;
    zhigh=-zhigh;
  }
  if (third<zlow) zlow=third;
  else if (third>=zhigh) 
  {
    zhigh=(-xhigh)*ylow;
    zhigh=-zhigh;
  }
  if (fourth<zlow) zlow=fourth;
  else if (fourth>=zhigh) 
  {
    zhigh=(-xhigh)*yhigh;
    zhigh=-zhigh;
  }
  z.low=(zlow/_fxp_one);
  z.low/=_fxp_one;
  z.high=(zhigh/_fxp_one+1);
  z.high/=_fxp_one;
  return z;
}

typedef struct intervalt int_vectort[NSTATES];
typedef struct intervalt int_matrixt[NSTATES][NSTATES];

void matrix_vector_mult(int_matrixt A,int_vectort x)
{
  int i,j;
  int_vectort y;
  for (i=0;i<NSTATES;i++)
  {
    y[i]=interval_mult(A[i][0],x[0]);
    for (j=1;j<NSTATES;j++)
    {
      y[i]=interval_add(y[i],interval_mult(A[i][j],x[j]));
    }
  }
  for (i=0;i<NSTATES;i++)
  {
    x[i].low=y[i].low;
    x[i].high=y[i].high;
  }
}

void closed_fxp_mult(const int_matrixt A,const int_vectort B,const int_vectort K,int_vectort x)
{
  int i,j,row;
  int_vectort x2;
  control_typet kx[NSTATES][4];
  for (i=0;i<NSTATES;i++)
  {
    long long int xtemp=x[i].low*_fxp_one;
    long long int ktemp=K[i].low*_fxp_one;
    kx[i][0]=(xtemp*ktemp/_fxp_one);
    kx[i][0]/=_fxp_one;
    ktemp=K[i].high*_fxp_one;
    kx[i][1]=(xtemp*ktemp/_fxp_one);
    kx[i][1]/=_fxp_one;
    xtemp=x[i].high*_fxp_one;
    kx[i][3]=(xtemp*ktemp/_fxp_one);
    kx[i][3]/=_fxp_one;
    ktemp=K[i].low*_fxp_one;
    kx[i][2]=(xtemp*ktemp/_fxp_one);
    kx[i][2]/=_fxp_one;
  }
  double up=1e-12;
  
  for (row=0;row<NSTATES;row++)
  {
    control_typet akx[16];
    struct intervalt coeffs[NSTATES];
    x2[row].low=0;
    x2[row].high=0;
    for (i=0;i<NSTATES;i++)
    {
      akx[0] =A[row][i].low*x[i].low-B[row].low*kx[i][0];
      akx[1] =A[row][i].low*x[i].low-B[row].low*kx[i][1]; 
      akx[2] =A[row][i].low*x[i].low-B[row].high*kx[i][0];
      akx[3] =A[row][i].low*x[i].low-B[row].high*kx[i][1];
      akx[4] =A[row][i].high*x[i].low-B[row].low*kx[i][0];
      akx[5] =A[row][i].high*x[i].low-B[row].low*kx[i][1];
      akx[6] =A[row][i].high*x[i].low-B[row].high*kx[i][0];
      akx[7] =A[row][i].high*x[i].low-B[row].high*kx[i][1];
      akx[8] =A[row][i].low*x[i].high-B[row].low*kx[i][2];
      akx[9] =A[row][i].low*x[i].high-B[row].low*kx[i][3];
      akx[10]=A[row][i].low*x[i].high-B[row].low*kx[i][2];
      akx[11]=A[row][i].low*x[i].high-B[row].low*kx[i][3];
      akx[12]=A[row][i].high*x[i].high-B[row].low*kx[i][2];
      akx[13]=A[row][i].high*x[i].high-B[row].low*kx[i][3];
      akx[14]=A[row][i].high*x[i].high-B[row].low*kx[i][2];
      akx[15]=A[row][i].high*x[i].high-B[row].low*kx[i][3];
      coeffs[i].low=akx[0];
      coeffs[i].high=akx[0];
      for (j=1;j<16;j++)
      {
        if (akx[j]<coeffs[i].low) coeffs[i].low=akx[j];
        else if (akx[j]>coeffs[i].high) coeffs[i].high=akx[j];
      }
      coeffs[i].high+=up;
      x2[row].low+=coeffs[i].low;
      x2[row].high+=coeffs[i].high;
    }
  }
  for (i=0;i<NSTATES;i++)
  {
    x[i].low=x2[i].low;
    x[i].high=x2[i].high;
  }
}

/*inline */struct intervalt interval_posDiv(struct intervalt x, struct intervalt y)
{
  struct intervalt z;
  if (x.high<=_zero)
  {
    z.high=(-x.high)/y.high;
    z.high=-z.high;
    z.low=x.low/y.low;
  }
  else if (x.low>=_zero)
  {
    z.low=x.low/y.high;
    z.high=(-x.high)/y.low;
    z.high=-z.high;
  }
  else
  {
    z.low=x.low/y.low;
    z.high=(-x.high)/y.low;
    z.high=-z.high;
  }
  return z;
}

#if (NSTATES==2)
  #include "int_2Inverse.h"
#else
  #include "int_3Inverse.h"
#endif

void bound_error(int_matrixt closed_loop,int_vectort K,int_vectort error_coeffs)
{
  int i,j;
  int_matrixt IminA;
  for (i=0;i<NSTATES;i++)
  {
    for (j=0;j<NSTATES;j++)
    {
      IminA[i][j].low=-closed_loop[i][j].high;
      IminA[i][j].high=-closed_loop[i][j].low;
    }
    IminA[i][i].low+=1;
    IminA[i][i].high+=1;
  }
  int_matrixt invIminA;
  inverse(invIminA,IminA);
  struct intervalt noise;
  if (K[0].low>=0)
  {
   noise.low=K[0].low;
   noise.high=K[0].high;
  }
  else
  {
   noise.low=-K[0].high;
   noise.high=-K[0].low;
  }
  for (i=1;i<NSTATES;i++)
  {
   if (K[i].low>=0)
   {
    noise.low+=K[i].low;
    noise.high+=K[i].high;
   }
   else
   {
    noise.low-=K[i].high;
    noise.high-=K[i].low;    
   }   
  }
  noise.low*=_dbl_lsb;
  noise.high*=_dbl_lsb;
  for (i=0;i<NSTATES;i++)
  {
    error_coeffs[i].low=invIminA[i][0].low;
    error_coeffs[i].high=invIminA[i][0].high;
    for (j=1;j<NSTATES;j++)
    {
      error_coeffs[i]=interval_add(error_coeffs[i],invIminA[i][j]);
    }
    error_coeffs[i]=interval_mult(error_coeffs[i],noise);
  }
  
}

#endif /*INTERVALS_H_*/
