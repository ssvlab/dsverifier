#include <assert.h>
#include <stdbool.h>

/*#define __CONTROLLER_DEN_SIZE 3
#define __CONTROLLER_NUM_SIZE 3
#define __PLANT_DEN_SIZE 2
#define __PLANT_NUM_SIZE 1
#define SOLUTION_DEN_SIZE 3
#define SOLUTION_NUM_SIZE 3*/
#include "sizes.h"
#define __OPENLOOP_DEN_SIZE (__CONTROLLER_DEN_SIZE+__PLANT_DEN_SIZE-1)
#define __OPENLOOP_NUM_SIZE (__CONTROLLER_NUM_SIZE+__PLANT_NUM_SIZE-1)

#define __NORMALIZED
#ifdef __CPROVER
#ifndef _FIXEDBV
  #ifndef _EXPONENT_WIDTH
    #define _EXPONENT_WIDTH 16
  #endif
  #ifndef _FRACTION_WIDTH
  #define _FRACTION_WIDTH 11
  #endif
  typedef __CPROVER_floatbv[_EXPONENT_WIDTH][_FRACTION_WIDTH] control_floatt;
  control_floatt _imp_max=(((1 <<(_EXPONENT_WIDTH-1))-1)<<1)+1;
#else
  #ifndef _CONTROL_FLOAT_WIDTH
    #define _CONTROL_FLOAT_WIDTH 16
  #endif
  #ifndef _CONTORL_RADIX_WIDTH
    #define _CONTORL_RADIX_WIDTH _CONTROL_FLOAT_WIDTH / 2
  #endif
  typedef __CPROVER_fixedbv[_CONTROL_FLOAT_WIDTH][_CONTORL_RADIX_WIDTH] control_floatt;
  control_floatt _imp_max=(((1 <<(_CONTROL_FLOAT_WIDTH-1))-1)<<1)+1;
#endif
  typedef unsigned char cnttype;
#else
  typedef double control_floatt;
  typedef unsigned int cnttype;
  #include <stdlib.h>
//  #include <iostream>
  #include <stdio.h> 
#endif

control_floatt one_lsb_error(void) {
  const unsigned long int one = 1;
  return *(const control_floatt *)&one;
}

struct anonymous0
{
  cnttype int_bits;
  cnttype frac_bits;
};

struct anonymous2
{
  control_floatt denBot[SOLUTION_DEN_SIZE];
  control_floatt denTop[SOLUTION_DEN_SIZE];
  cnttype den_size;
  control_floatt numBot[SOLUTION_NUM_SIZE];
  control_floatt numTop[SOLUTION_NUM_SIZE];
  cnttype num_size;
};

struct anonymous3
{
  control_floatt den[SOLUTION_DEN_SIZE];
  control_floatt den_uncertainty[SOLUTION_DEN_SIZE];
  cnttype den_size;
  control_floatt num[SOLUTION_NUM_SIZE];
  control_floatt num_uncertainty[SOLUTION_NUM_SIZE];
  cnttype num_size;
};

control_floatt _dbl_max;
control_floatt _dbl_min;
signed long int _fxp_max;
signed long int _fxp_min;
signed long int _fxp_one;
control_floatt _dbl_lsb;
control_floatt _plant_norm;

struct anonymous0 impl={ .int_bits=_CONTROLER_INT_BITS, .frac_bits=_CONTROLER_FRAC_BITS};

#include "plant.h"
/*struct anonymous3 plant={ .den={ 1.0, -9.998000e-1, 0.0}, .den_uncertainty={0.0, 0.0, 0.0}, .den_size=2,
.num={ 2.640000e-2, 0.0, 0.0}, .num_uncertainty={0.0, 0.0, 0.0}, .num_size=1};*/
/*.den={ 1.0, -3.32481248817168, 1.64872127070013 }, .den_size=3,
    .num={ 0.548693198268086, -0.886738807003861, 0.0 }, .num_size=2};*/

struct anonymous2 plant_cbmc;
struct anonymous3 controller_cbmc;

#ifdef __CPROVER
extern struct anonymous3 controller;
#else
#include "controller.h"
/*struct anonymous3 controller = { .den={ -32768, -28676, -32768 }, .den_uncertainty={0.0, 0.0, 0.0}, .den_size=3,
 .num={ 256, -209.5625, 0 }, .num_uncertainty={0.0, 0.0, 0.0}, .num_size=3};*/
//struct anonymous3 controller = { .den={ 32620.5, -153.4375, 0 },.den_size=2,
// .num={ -32758.125, -3143.0625, 0.0 }, .num_size=2};
#endif

void __DSVERIFIER_assume(_Bool expression)
{
#ifdef __CPROVER
  __CPROVER_assume(expression != (_Bool)0);
#endif
}

void __DSVERIFIER_assert(_Bool expression)
{
  /* assertion expression */
  assert(expression != (_Bool)0);
}

void initialization()
{
  __DSVERIFIER_assume(impl.int_bits+impl.frac_bits < 32);
#ifdef __NORMALIZED
  _fxp_one = 1 << (impl.frac_bits + impl.int_bits);
  _dbl_lsb=1.0/(1 << impl.frac_bits + impl.int_bits);
  _fxp_min = -(1 << (impl.frac_bits + impl.int_bits -1));
  _fxp_max = (1 << (impl.frac_bits + impl.int_bits-1))-1;
  _dbl_max = (1.0-_dbl_lsb);//Fractional part
#else
  _fxp_one = (1 << impl.frac_bits);
  _dbl_lsb=1.0/(1 << impl.frac_bits);
  _fxp_min = -(1 << (impl.frac_bits + impl.int_bits -1));
  _fxp_max = (1 << (impl.frac_bits + impl.int_bits-1))-1;
  _dbl_max = (1 << (impl.int_bits-1))-1;//Integer part
  _dbl_max += (1.0-_dbl_lsb);//Fractional part
#endif
  _dbl_min = -_dbl_max;
}

int validation()
{
  cnttype i;
  control_floatt max=0;
  for (i=0;i<plant.num_size;i++) {
    if (plant.num[i]>max) max=plant.num[i];
    else if (-plant.num[i]>max) max=-plant.num[i];
  }
  for (i=0;i<plant.den_size;i++) {
    if (plant.den[i]>max) max=plant.den[i];
    else if (-plant.den[i]>max) max=-plant.den[i];
  }
  unsigned int max_int=max;
#ifdef __NORMALIZED
  cnttype mult_bits=1;
#else
  cnttype mult_bits=12;
#endif
  while (max_int>0) 
  {
    mult_bits++;
    max_int>>=1;
  }
  _plant_norm=1<<mult_bits;
#ifdef __CPROVER 
  #ifndef _FIXEDBV
    #ifdef __NORMALIZED
      __DSVERIFIER_assume((impl.int_bits+impl.frac_bits<=_FRACTION_WIDTH) && (mult_bits<_EXPONENT_WIDTH));
    #else
      __DSVERIFIER_assume((impl.frac_bits<=_FRACTION_WIDTH) && (impl.int_bits+mult_bits<_EXPONENT_WIDTH));
    #endif
  #else
    #ifdef __NORMALIZED
      __DSVERIFIER_assume((impl.int_bits+impl.frac_bits<=_CONTORL_RADIX_WIDTH) && (mult_bits<_CONTROL_FLOAT_WIDTH));
    #else
      __DSVERIFIER_assume((impl.frac_bits<=_CONTORL_RADIX_WIDTH) && (impl.int_bits+mult_bits<_CONTROL_FLOAT_WIDTH));
    #endif
  #endif
  __DSVERIFIER_assume((__CONTROLLER_DEN_SIZE == controller.den_size) && (__CONTROLLER_NUM_SIZE == controller.num_size) && (plant.num_size != 0) && (impl.int_bits != 0));
#else
  if (!((__CONTROLLER_DEN_SIZE == controller.den_size) && (__CONTROLLER_NUM_SIZE == controller.num_size) && (plant.num_size != 0) && (impl.int_bits != 0))) return 10;
#endif
  unsigned int zero_count = 0;
  for( i=0; i < __CONTROLLER_DEN_SIZE; i++)
  {
    const control_floatt value=controller.den[i];
    if(value == 0.0) ++zero_count;
#ifdef __CPROVER 
    __DSVERIFIER_assume(value <= _dbl_max);
    __DSVERIFIER_assume(value >= _dbl_min);
#else
    if(value > _dbl_max) return 10;
    if(value < _dbl_min) return 10;
#endif
  }
#ifdef __CPROVER
  __DSVERIFIER_assume(zero_count < __CONTROLLER_DEN_SIZE);
#else
  if (zero_count == __CONTROLLER_DEN_SIZE) return 10;
#endif
  zero_count = 0;
  for(i = 0 ; i < __CONTROLLER_NUM_SIZE; i++)
  {
    const control_floatt value=controller.num[i];
    if(value == 0.0) ++zero_count;
#ifdef __CPROVER 
    __DSVERIFIER_assume(value <= _dbl_max);
    __DSVERIFIER_assume(value >= _dbl_min);
#else
    if (value > _dbl_max) return 10;
    if (value < _dbl_min) return 10;
#endif
  }
#ifdef __CPROVER
  __DSVERIFIER_assume(zero_count < __CONTROLLER_NUM_SIZE);
#else
  if (zero_count == __CONTROLLER_NUM_SIZE) return 10;
#endif
  return 0;
}

#ifndef __CPROVER 
void print_poly(control_floatt *bot,control_floatt *top,cnttype n)
{
  cnttype i;
  for (i=0;i<n;i++)
  {
    printf("[%f,%f]z%d ", bot[i], -top[i], n-i-1);
    //std::cout << "[" << bot[i] << "," << -top[i] << "]z" << n-i-1 << " ";
  }
  puts("");
  //std::cout << std::endl;
}
#endif

void call_closedloop_verification_task()
{
  cnttype i;
  plant_cbmc.num_size=plant.num_size;
  for(i = 0;i < plant.num_size; i++)
  {
#ifdef __NORMALIZED
    control_floatt value=plant.num[i]/_plant_norm;
    control_floatt factor=(value * plant.num_uncertainty[i]) / 100.0;
    factor = (factor < 0.0) ? -factor : factor;
    factor += one_lsb_error();
    plant_cbmc.numBot[i]=value-factor;
    plant_cbmc.numTop[i]=-value-factor;
#else
    control_floatt factor=(plant.num[i] * plant.num_uncertainty[i]) / 100.0;
    factor = (factor < 0.0) ? -factor : factor;
    factor += one_lsb_error();
    plant_cbmc.numBot[i]=plant.num[i]-factor;
    plant_cbmc.numTop[i]=-plant.num[i]-factor;
#endif
  }
  plant_cbmc.den_size=plant.den_size;
  for(i = 0;i < plant.den_size; i++)
  {
#ifdef __NORMALIZED
    control_floatt value=plant.den[i]/_plant_norm;
    control_floatt factor=(value * plant.den_uncertainty[i]) / 100.0;
    factor = (factor < 0.0) ? -factor : factor;
    factor += one_lsb_error();
    plant_cbmc.denBot[i]=value-factor;
    plant_cbmc.denTop[i]=-value-factor;  
#else
    control_floatt factor=(plant.den[i] * plant.den_uncertainty[i]) / 100.0;
    factor = (factor < 0.0) ? -factor : factor;
    factor += one_lsb_error();
    plant_cbmc.denBot[i]=plant.den[i]-factor;
    plant_cbmc.denTop[i]=-plant.den[i]-factor;  
#endif
  }
}

signed int check_stability_closedloop(control_floatt *denBot,control_floatt *denTop, cnttype n)
{
  cnttype columns=n;
  control_floatt top[n][n],bot[n][n];
  cnttype i;
  cnttype j;
  control_floatt sumBot=0.0,sumTop=0.0;
#ifdef __CPROVER
  __DSVERIFIER_assume(denBot[0] > 0.0);
  __DSVERIFIER_assume(-denTop[n - 1] < denBot[0]);
  __DSVERIFIER_assume(-denBot[n - 1] < denBot[0]);
#else
  printf("m[0]=[%f,%f]>0\n", denBot[0], -denTop[0]);
  //std::cout << "m[0]=[" << denBot[0] << "," << -denTop[0] << "]>0" << std::endl;
  printf("fabs(m[%d]=[%f,%f])<m[0]=[%f,%f]\n", n-1, denBot[n-1], -denTop[n-1], denBot[0], -denTop[0]);
  //std::cout << "fabs(m[" << n-1 << "]=[" << denBot[n-1] << "," << -denTop[n-1] << "])<" << "m[0]=[" << denBot[0] << "," << -denTop[0] << "]" << std::endl;
  if (!(denBot[0] > 0.0)) return 0;
  if (!(-denTop[n - 1] < denBot[0])) return 0;
  if (!(-denBot[n - 1] < denBot[0])) return 0;
#endif
  for(i = 0 ; i < n; i+=2) sumBot+=denBot[i];
  for(i = 1 ; i < n; i+=2) sumTop+=denTop[i];
#ifdef __CPROVER
  __DSVERIFIER_assume(sumBot > -sumTop);
#else
  printf("sumEven-sumOdd=[%f,?]>0\n", sumBot+sumTop);
  //std::cout << "sumEven-sumOdd=[" << sumBot+sumTop << ",?]>0" << std::endl;
  if (!(sumBot > -sumTop)) return 0;
#endif
  for(i = 1 ; i < n; i+=2) sumBot+=denBot[i];
#ifdef __CPROVER
  __DSVERIFIER_assume(sumBot > 0);
#else
  printf("sum=[%f,?]>0\n", sumBot);
  //std::cout << "sum=[" << sumBot << ",?]>0" << std::endl;
  if (!(sumBot > 0)) return 0;
#endif
  
  for(j=0;j<columns;j++) 
  {
    top[0][j]=denTop[j];
    bot[0][j]=denBot[j];
  }
  columns--;
  control_floatt botF,topF;
  for(i=1;i<n; i++)
  {
    control_floatt *oldBot=bot[i-1];
    control_floatt *oldTop=top[i-1];
    //denominator is always >0
    if (oldBot[columns]>=0)
    {
      botF=oldBot[columns]/-oldTop[0];
      topF=oldTop[columns]/oldBot[0];
    }
    else {
      if (oldTop[columns]>=0) botF=oldBot[columns]/oldBot[0]; 
      else                    botF=oldBot[columns]/(-oldTop[0]);
      topF=oldTop[columns]/(-oldTop[0]);
    }
    control_floatt botRes,topRes;
    control_floatt errorF=(topF+botF);
    for( j = 0; j<columns; j++)
    {  
      botRes=oldBot[columns-j]*botF;
      control_floatt error=(oldTop[columns-j]+oldBot[columns-j]);
      topRes=oldBot[columns-j]*errorF+(botF-errorF)*error;
      if (topRes<0) topRes-=botRes;
      else
      {
        botRes-=topRes;
        topRes=(-oldBot[columns-j])*botF; 
      }
      top[i][j] = oldTop[j] + botRes;
      bot[i][j] = oldBot[j] + topRes;
    }
#ifdef __CPROVER
    __DSVERIFIER_assume(bot[i][0] >= 0.0);
#else
  printf("m[%d]=[%f,%f]>0\n", i, bot[i][0], -top[i][0]);
	//std::cout << "m[" << i << "]=[" << bot[i][0] << "," << -top[i][0] << "]>0" << std::endl;
    if (!(bot[i][0] >= 0.0)) return 0;
#endif		
    columns--;
  }
  return 1;
}

signed long int fxp_control_floatt_to_fxp(control_floatt value)
{
  signed long int tmp;
  control_floatt ftemp=value * _fxp_one;
  tmp = ftemp;
  control_floatt residue=ftemp - tmp;
  if(value < 0.0 && (residue != 0.0))
  {
    ftemp = ftemp - 1.0;
    tmp = ftemp;
  }
  return tmp;
}

void fxp_check(control_floatt *value)
{
#ifdef __CPROVER
  __DSVERIFIER_assume((~_dbl_max&*value)==0);
#else
  *value=fxp_control_floatt_to_fxp(*value);
  *value/=_fxp_one;
#endif
}

void fxp_check_array(control_floatt *f, signed int N)
{
  for(cnttype i=0; i < N; i++) fxp_check(&f[i]);
}

void poly_mult(control_floatt *a, cnttype Na, control_floatt *bBot, control_floatt *bTop,cnttype Nb, control_floatt *ans_bot, control_floatt *ans_top,cnttype Nans)
{
  cnttype i;
  cnttype j;
  cnttype k;
  Nans = Na + Nb - 1;
  for(i = 0 ; i<Nans; i++) ans_bot[i] = ans_top[i] = 0.0;
  for(i = 0; i < Na; i++)
  {
    for( j = 0; j < Nb; j++)
    {
      k = Na + Nb -i -j -2;
      if (a[Na - i - 1]>=0)
      {
        ans_bot[k]+=a[Na - i - 1]*bBot[Nb - j - 1];
        ans_top[k]+=a[Na - i - 1]*bTop[Nb - j - 1];
      }  
      else
      {
        ans_bot[k]+=a[Na - i - 1]*(-bTop[Nb - j - 1]);
        ans_top[k]+=a[Na - i - 1]*(-bBot[Nb - j - 1]);
      }
    }
  }
}

void poly_sum(control_floatt *a, cnttype Na, control_floatt *b, cnttype Nb, control_floatt *ans, cnttype Nans)
{
  cnttype i;
  Nans--;
  Na--;
  Nb--;
  for (i=0;i<=Na;i++) ans[Nans-i]=a[Na-i];
  for (i=Na+1;i<=Nans;i++) ans[Nans-i]=0;
  for (i=0;i<=Nb;i++) ans[Nans-i]+=b[Nb-i];
}

void ft_closedloop_series(control_floatt *ans_num_bot, control_floatt *ans_num_top, cnttype Nans_num, control_floatt *ans_den_bot, control_floatt *ans_den_top, cnttype Nans_den)
{
  Nans_num = __CONTROLLER_NUM_SIZE + plant.num_size -1;
  Nans_den = __CONTROLLER_DEN_SIZE + plant.den_size -1;
  control_floatt den_mult_bot[Nans_den];
  control_floatt den_mult_top[Nans_den];
  poly_mult(controller.num, __CONTROLLER_NUM_SIZE, plant_cbmc.numBot, plant_cbmc.numTop, plant.num_size, ans_num_bot, ans_num_top, Nans_num);
  poly_mult(controller.den, __CONTROLLER_DEN_SIZE, plant_cbmc.denBot, plant_cbmc.denTop, plant.den_size, den_mult_bot,den_mult_top, Nans_den);
  poly_sum(ans_num_bot,Nans_num,den_mult_bot,Nans_den,ans_den_bot,Nans_den);
  poly_sum(ans_num_top,Nans_num,den_mult_top,Nans_den,ans_den_top,Nans_den);
}

int verify_stability_closedloop_using_dslib(void)
{
  fxp_check_array(controller.num,__CONTROLLER_NUM_SIZE);
  fxp_check_array(controller.den,__CONTROLLER_DEN_SIZE);

  cnttype i;
  cnttype ans_num_size=__CONTROLLER_NUM_SIZE + plant.num_size-1;
  control_floatt ans_num_bot[ans_num_size];
  control_floatt ans_num_top[ans_num_size];
  cnttype ans_den_size=__CONTROLLER_DEN_SIZE + plant.den_size-1;
  control_floatt ans_den_bot[ans_den_size];
  control_floatt ans_den_top[ans_den_size];
#ifdef __CPROVER
  __CPROVER_array_set(ans_num_bot,0.0);
  __CPROVER_array_set(ans_num_top,0.0);
  __CPROVER_array_set(ans_den_bot,0.0);
  __CPROVER_array_set(ans_den_top,0.0);
#else
  for (int i=0;i<ans_num_size;i++) ans_num_bot[i]=ans_num_top[i]=3;
  for (int i=0;i<ans_den_size;i++) ans_den_bot[i]=ans_den_top[i]=3;
#endif
/*  for (i=0;i<__CONTROLLER_NUM_SIZE;i++) ans_den_top[i]=-controller.num[i];
  signed int return_value1=check_stability_closedloop(controller.num,ans_den_top, __CONTROLLER_NUM_SIZE);
#ifdef __CPROVER  
  __DSVERIFIER_assume(!(return_value1 == 0));
#else
  if (return_value1 == 0) return 10;
#endif*/
  ft_closedloop_series(ans_num_bot,ans_num_top, ans_num_size, ans_den_bot,ans_den_top, ans_den_size);
  signed int return_value2=check_stability_closedloop(ans_den_bot,ans_den_top, ans_den_size);
#ifdef __CPROVER    
  __DSVERIFIER_assume(!(return_value2 == 0));
#else
  fputs("plant_num=", stdout);
  //std::cout << "plant_num=";
  print_poly(plant_cbmc.numBot,plant_cbmc.numTop, plant_cbmc.num_size);
  fputs("plant_den=", stdout);
  //std::cout << "plant_den=";
  print_poly(plant_cbmc.denBot,plant_cbmc.denTop, plant_cbmc.den_size);
  fputs("ans=", stdout);
  //std::cout << "ans=";
  print_poly(ans_den_bot,ans_den_top, ans_den_size);
  if (return_value2 == 0) return 10;
  return 0;
#endif
}

// main
int main()
{
  struct anonymous3 __trace_controller = controller;
  struct anonymous3 __trace_plant = plant;
  initialization();
  int result=validation();
#ifndef __CPROVER
  if (result!=0) return result;
#endif
  call_closedloop_verification_task();
  result=verify_stability_closedloop_using_dslib();
#ifdef __CPROVER
  __DSVERIFIER_assert(0);
#endif
  return result;
}
