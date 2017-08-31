#define _DIMENSION 3
typedef control_floatt vectort[_DIMENSION];
typedef control_floatt matrixt[_DIMENSION][_DIMENSION];
struct coefft
{
  vectort coeffs;
  vectort uncertainty;
};
struct transformt
{
  matrixt coeffs;
  matrixt uncertainty;
};
