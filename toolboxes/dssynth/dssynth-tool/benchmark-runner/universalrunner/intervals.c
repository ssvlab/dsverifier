#include <stdio.h>
struct implt
{
  int int_bits;
  int frac_bits;
};
struct implt impl={ .int_bits=8, .frac_bits=8};

#define add(z,x,y) z.low=x.low+y.low; z.high=x.high+y.high;
#define sub(z,x,y) z.low=x.low-y.high; z.high=x.high-y.low;

#define INT_BITS 8
#define FRAC_BITS 8

#define NSTATES 2

typedef double __plant_precisiont;
#include "intervals.h"

int main()
{
  struct intervalt tmp;
  int_matrixt a;
  int_vectort x,y,k,b;
  struct intervalt v,z;
  x[0].low=2.1;
  x[0].high=3.7;
  y[0].low=-2;
  y[0].high=1;
  v.low=-1.5;
  v.high=-.8;
  get_bounds();
  printf("dbl_lsb=%lf\n",_dbl_lsb);
  printf("fxp_one=%d\n",(int)_fxp_one);
  printf("x=[%lf,%lf]\n",x[0].low,x[0].high);
  printf("y=[%lf,%lf]\n",y[0].low,y[0].high);
  printf("v=[%lf,%lf]\n",v.low,v.high);
  z=interval_mult(x[0],x[0]);
  printf("x*x=[%lf,%lf]=[4.41,13.69]\n",z.low,z.high);
  add(z,x[0],x[0]);
  printf("x+x=[%lf,%lf]\n",z.low,z.high);
  sub(z,x[0],y[0]);
  printf("x-y=[%lf,%lf]\n",z.low,z.high);
  z=interval_mult(x[0],y[0]);
  printf("x*y=[%lf,%lf]=[-7.4,3.7]\n",z.low,z.high);
  z=interval_mult(x[0],v);
  printf("x*v=[%lf,%lf]=[-5.55,-1.68]\n",z.low,z.high);
  z=interval_posDiv(x[0],x[0]);
  printf("x/x=[%lf,%lf]\n",z.low,z.high);
  z=interval_posDiv(y[0],x[0]);
  printf("y/x=[%lf,%lf]\n",z.low,z.high);
  z=interval_posDiv(v,x[0]);
  printf("v/x=[%lf,%lf]\n",z.low,z.high);
  z=interval_posDiv(x[0],x[0]);
  printf("x/x=[%lf,%lf]\n",z.low,z.high);
  
  x[0]=fxp_interval_check(x[0]);
  printf("fxp_x=[%lf,%lf]\n",x[0].low,x[0].high);
  z=interval_mult(x[0],x[0]);
  printf("x*x=[%lf,%lf]\n",z.low,z.high);
  z=interval_fxp_mult(x[0],x[0]);
  printf("fxp_x*fxp_x=[%lf,%lf]\n",z.low,z.high);
  int i,j;
  /*for (i=-100;i<100;i++)
  {
    for (j=-100;j<100;j++)
    {
      x[0].low=i*.1;
      y[0].low=j*.1;  
      x[0].high=x[0].low+j*j*.001;
      y[0].high=y[0].low+.1;
      z=interval_mult(x,y);
      printf("[%lf,%lf]*[%lf,%lf]=[%lf,%lf]\n",x[0].low,x[0].high,y[0].low,y[0].high,z.low,z.high);
    }
  }*/
  
  x[0].low=1;
  x[0].high=1;
  x[1].low=1;
  x[1].high=1;
  k[0].low=2.95703125;
  k[0].high=2.95703125;
  k[1].low=-0.03125;
  k[1].high=-0.03125;
  a[0][0].low=2.2553;
  a[0][0].high=2.2553;
  a[0][1].low=-1;
  a[0][1].high=-1;
  a[1][0].low=1;
  a[1][0].high=1;
  a[1][1].low=0;
  a[1][1].high=0;
  b[0].low=.5;
  b[0].high=.5;
  b[1].low=0;
  b[1].high=0;
  printf("a=[(%lf,%lf) (%lf,%lf)\n  (%lf,%lf) (%lf,%lf)]\n",a[0][0].low,a[0][0].high,a[0][1].low,a[0][1].high,a[1][0].low,a[1][0].high,a[1][1].low,a[1][1].high);
  printf("b=[(%lf,%lf) (%lf,%lf)]\n",b[0].low,b[0].high,b[1].low,b[1].high);
  printf("k=[(%lf,%lf) (%lf,%lf)]\n",k[0].low,k[0].high,k[1].low,k[1].high);
  
/*  for (i=0;i<20;i++)
  {
    add(y[0],interval_mult(a[0][0],x[0]),interval_mult(a[0][1],x[1]));
    add(y[1],interval_mult(a[1][0],x[0]),interval_mult(a[1][1],x[1]));
    printf("y=[%lf,%lf],[%lf,%lf]\n",y[0].low,y[0].high,y[1].low,y[1].high);
    add(y[0],interval_fxp_mult(k[0],x[0]),interval_fxp_mult(k[1],x[1]));
    sub(x[0],y[0],interval_mult(b[0],y[0]));
    sub(x[1],y[1],interval_mult(b[1],y[0]));
    printf("x=[%lf,%lf],[%lf,%lf]\n",x[0].low,x[0].high,x[1].low,x[1].high);
  }*/

/*  printf("discrete\n");
  printf("a=[(%lf,%lf) (%lf,%lf)\n  (%lf,%lf) (%lf,%lf)]\n",a[0][0].low,a[0][0].high,a[0][1].low,a[0][1].high,a[1][0].low,a[1][0].high,a[1][1].low,a[1][1].high);
  printf("b=[(%lf,%lf) (%lf,%lf)]\n",b[0].low,b[0].high,b[1].low,b[1].high);
  printf("k=[(%lf,%lf) (%lf,%lf)]\n",k[0].low,k[0].high,k[1].low,k[1].high);
  x[0].low=1;
  x[0].high=1;
  x[1].low=1;
  x[1].high=1;
  for (i=0;i<100;i++)
  {
    closed_fxp_mult(a,b,k,x);
    printf("x=[%lf,%lf],[%lf,%lf]\n",x[0].low,x[0].high,x[1].low,x[1].high);
  }*/
  
  printf("continuous\n");
  x[0].low=1;
  x[0].high=1;
  x[1].low=1;
  x[1].high=1;
  sub(a[0][0],a[0][0],interval_mult(b[0],k[0]));
  sub(a[0][1],a[0][1],interval_mult(b[0],k[1]));
  sub(a[1][0],a[1][0],interval_mult(b[1],k[0]));
  sub(a[1][1],a[1][1],interval_mult(b[1],k[1]));
  printf("a=[(%lf,%lf) (%lf,%lf);(%lf,%lf) (%lf,%lf)]\n",a[0][0].low,a[0][0].high,a[0][1].low,a[0][1].high,a[1][0].low,a[1][0].high,a[1][1].low,a[1][1].high);
  for (i=0;i<500;i++)
  {
    matrix_vector_mult(a,x);
/*    add(y[0],interval_mult(a[0][0],x[0]),interval_mult(a[0][1],x[1]));
    add(y[1],interval_mult(a[1][0],x[0]),interval_mult(a[1][1],x[1]));
    x[0].low=y[0].low;
    x[0].high=y[0].high;
    x[1].low=y[1].low;
    x[1].high=y[1].high;*/
    printf("x=[%.9f,%.9f],[%.9f,%.9f]\n",x[0].low,x[0].high,x[1].low,x[1].high);
  }
  int_vectort error_coeffs;
  bound_error(a,k,error_coeffs);
  printf("error=[%f,%f]",error_coeffs[0].low,error_coeffs[0].high);
  for (i=0;i<NSTATES;i++)
  {
    printf(",[%f,%f]",error_coeffs[i].low,error_coeffs[i].high);
  }
  printf("\n");
  return 0;
}