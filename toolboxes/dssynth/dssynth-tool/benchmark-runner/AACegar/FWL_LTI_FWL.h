
control_floatt _dbl_max;
control_floatt _dbl_min;
signed long int _fxp_max;
signed long int _fxp_min;
signed long int _fxp_one;
control_floatt _dbl_lsb;
control_floatt _transform_error;
control_floatt _sum_error;

void get_bounds()
{
  verify_assume((impl.int_bits+impl.frac_bits) < 32);
  if(impl.frac_bits >= 31)
    _fxp_one = 2147483647l;
  else
    _fxp_one = (1 << impl.frac_bits);
  _dbl_lsb=_one/(1 << impl.frac_bits);
  _fxp_min = -(1 << (impl.frac_bits + impl.int_bits -1));
  _fxp_max = (1 << (impl.frac_bits + impl.int_bits-1))-1;
  _dbl_max = (1 << (impl.int_bits-1))-1;//Integer part
  _dbl_max += (_one-_dbl_lsb);//Fractional part
  _dbl_min = -_dbl_max;
#ifndef __CPROVER
  printf("%f",_dbl_max);
#endif
#ifdef __CHECK_FP
  _transform_error=2*_dbl_lsb*_DIMENSION*_DIMENSION;
  _sum_error=2*_transform_error*_DIMENSION;
#else
  _transform_error=0;
  _sum_error=0;
#endif
}

int check_vector_bounds(vectort coeffs)
{
  cnttype i;
  for( i=0; i < _DIMENSION; i++)
  {
    const control_floatt value=coeffs[i];
#ifdef __CPROVER 
    verify_assume(value <= _dbl_max);
    verify_assume(value >= _dbl_min);
#else
    if(value > _dbl_max) return 10;
    if(value < _dbl_min) return 10;
#endif  
  }
  return 0;
}

int check_matrix_bounds(matrixt coeffs)
{
  cnttype i,j;
  for(i=0; i < _DIMENSION; i++)
  {
    for(j=0; j < _DIMENSION; j++)
    {
      const control_floatt value=coeffs[i][j];
#ifdef __CPROVER 
      verify_assume(value <= _dbl_max);
      verify_assume(value >= _dbl_min);
#else
      if(value > _dbl_max) return 10;
      if(value < _dbl_min) return 10;
#endif  
    }
  }
  return 0;
}

int check_bits()
{
#ifdef __CPROVER 
  #ifndef _FIXEDBV
    verify_assert((impl.frac_bits<=_FRACTION_WIDTH) && (impl.int_bits+impl.mult_bits<_EXPONENT_WIDTH));
  #else
    verify_assert((impl.frac_bits<=_CONTORL_RADIX_WIDTH) && (impl.int_bits+impl.mult_bits<_CONTROL_FLOAT_WIDTH));
  #endif
  verify_assert(impl.int_bits != 0);
#else
  if (impl.int_bits == 0) return 10;
#endif
  return 0;
}

signed long int fxp_control_floatt_to_fxp(control_floatt value)
{
  signed long int tmp;
  control_floatt ftemp=value * _fxp_one;
  tmp = ftemp;
  control_floatt residue=ftemp - tmp;
  if(value < _zero && (residue != _zero))
  {
    ftemp = ftemp - _one;
    tmp = ftemp;
  }
  return tmp;
}

void fxp_check(control_floatt *value)
{
#ifdef __CPROVER
  control_floatt tmp_value=*value;
  if (tmp_value < _zero) tmp_value=-tmp_value;
  verify_assume((~_dbl_max&tmp_value)==0);
#else
  *value=fxp_control_floatt_to_fxp(*value);
  *value/=_fxp_one;
#endif
}

void fxp_check_coeffs(vectort f)
{
  cnttype i;
  for(i=0; i < _DIMENSION; i++) fxp_check(&f[i]);
}

int validation()
{
  cnttype i;
  control_floatt max=0;
  for (i=0;i<_DIMENSION;i++) {
    if (plant.coeffs[i]>max) max=plant.coeffs[i];
    else if (-plant.coeffs[i]>max) max=-plant.coeffs[i];
  }
  unsigned int max_int=max;
  cnttype mult_bits=12;
  while (max_int>0) 
  {
    mult_bits++;
    max_int>>=1;
  }
  fxp_check_coeffs(controller);
  boundController();
  return check_bits()+check_vector_bounds(controller);  
}
