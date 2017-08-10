struct coeff_nondett
{
  vectort coeffs;
};

struct transform_nondett
{
  control_floatt coeffs[_DIMENSION][_DIMENSION];
};

#ifdef __CPROVER 
control_floatt nondet_double();
#else
control_floatt nondet_double()
{
  return _zero;
}
#endif

//Accounts for errors in the CCF coefficients row
void make_nondet_coeffs(vectort src,vectort uncertainty,vectort dst)
{
  cnttype i=0;
  for(i = 0; i < _DIMENSION; i++)
#ifdef __CPROVER 
    if(uncertainty[i] > _zero)
    {
      control_floatt min=src[i] -uncertainty[i];
      control_floatt max=src[i] +uncertainty[i];
      dst[i] = nondet_double();
      verify_assume(dst[i] >= min);
      verify_assume(dst[i] <= max);
    }
    else
#endif
      dst[i] = src[i];
}

//Accounts for errors in the transform matrix
void make_nondet_transform(matrixt src,matrixt uncertainty,matrixt dst)
{
  cnttype i=0,j=0;
  for(i = 0; i < _DIMENSION; i++)
    for(j = 0; j < _DIMENSION; j++)
#ifdef __CPROVER 
      if(uncertainty[i][j] > _zero)
      {
        control_floatt min=src[i][j]-uncertainty[i][j];
        control_floatt max=src[i][j]+uncertainty[i][j];
        dst[i][j] = nondet_double();
        verify_assume(dst[i][j] >= min);
        verify_assume(dst[i][j] <= max);
      }
      else
#endif
        dst[i][j] = src[i][j];
}
