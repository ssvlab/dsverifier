/**
 * DSVerifier - Digital Systems Verifier
 *
 * Federal University of Amazonas - UFAM
 *
 * Authors:       Hussama Ismail <hussamaismail@gmail.com>
 *                Iury Bessa     <iury.bessa@gmail.com>
 *                Renato Abreu   <renatobabreu@yahoo.com.br>
 *
 * ------------------------------------------------------
 *
 * initialize internal variables
 *
 * ------------------------------------------------------
 */
#ifndef DSVERIFIER_CORE_INITIALIZATION_H
#define DSVERIFIER_CORE_INITIALIZATION_H

#include <float.h>

extern digital_system ds;
extern digital_system plant;
extern digital_system control;
extern implementation impl;
extern filter_parameters filter;
extern hardware hw;

/** function to set the necessary parameters to DSVerifier FXP library */
void initialization()
{
#if(ARITHMETIC == FIXEDBV)
  if(impl.frac_bits >= FXP_WIDTH)
  {
    printf("impl.frac_bits must be less than the word-width!");
    __DSVERIFIER_assert(0);
  }
  if(impl.int_bits >= FXP_WIDTH - impl.frac_bits)
  {
    printf("impl.int_bits must be less than the word-width "
        "subtracted by the precision!\n");
    __DSVERIFIER_assert(0);
  }

  if(impl.frac_bits >= 31)
  {
    _fxp_one = 0x7fffffff;
  }
  else
  {
    _fxp_one = (0x00000001 << impl.frac_bits);
  }

  _fxp_half = (0x00000001 << (impl.frac_bits - 1));
  _fxp_minus_one = -(0x00000001 << impl.frac_bits);
  _fxp_min = -(0x00000001 << (impl.frac_bits + impl.int_bits - 1));
  _fxp_max = (0x00000001 << (impl.frac_bits + impl.int_bits - 1)) - 1;
  _fxp_fmask = ((((int32_t) 1) << impl.frac_bits) - 1);
  _fxp_imask = ((0x80000000) >> (FXP_WIDTH - impl.frac_bits - 1));

  _dbl_min = _fxp_min;
  _dbl_min /= (1 << impl.frac_bits);
  _dbl_max = _fxp_max;
  _dbl_max /= (1 << impl.frac_bits);
#elif(ARITHMETIC == FLOATBV)
  _fp_min = FLT_MIN;
  _fp_max = FLT_MAX;
#endif

  /* check if the scale exists */
  if((impl.scale == 0) || (impl.scale == 1))
  {
    impl.scale = 1;
    return;
  }

  /** applying scale in dynamical range */
  if(impl.min != 0)
  {
    impl.min = impl.min / impl.scale;
  }

  if(impl.max != 0)
  {
    impl.max = impl.max / impl.scale;
  }
}
#endif // DSVERIFIER_CORE_INITIALIZATION_H
