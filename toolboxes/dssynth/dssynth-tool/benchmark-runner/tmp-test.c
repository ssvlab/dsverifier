#ifdef _FIXEDBV
typedef __CPROVER_fixedbv[64][32] __plant_typet;
#else
typedef __CPROVER_floatbv[64][32] __plant_typet;
#endif

#ifdef _NONDET
extern __plant_typet _AminusBK[3][3];
extern __plant_typet _AminusBK_0_2;
extern __plant_typet _AminusBK_1_0;
extern __plant_typet _AminusBK_2_1;
#else
__plant_typet _AminusBK[3][3]={ { (__plant_typet)-0.009500026702880859375, (__plant_typet)0.075686991214752197265625, (__plant_typet)-195.853967010974884033203125 }, { (__plant_typet)0.125, (__plant_typet)0.0, (__plant_typet)0.0 }, { (__plant_typet)0.0, (__plant_typet)0.015625, (__plant_typet)0.0 } };
__plant_typet _AminusBK_0_2=(__plant_typet)-195.853967010974884033203125;
__plant_typet _AminusBK_1_0=(__plant_typet)0.125;
__plant_typet _AminusBK_2_1=(__plant_typet)0.015625;
#endif

int main(void) {
#ifdef _NONDET
  __CPROVER_assume(_AminusBK[0][0] == (__plant_typet)-0.009500026702880859375);
  __CPROVER_assume(_AminusBK[0][1] == (__plant_typet)0.075686991214752197265625);
  __CPROVER_assume(_AminusBK[0][2] == (__plant_typet)-195.853967010974884033203125);
  __CPROVER_assume(_AminusBK[1][0] == (__plant_typet)0.125);
  __CPROVER_assume(_AminusBK[1][1] == (__plant_typet)0.0);
  __CPROVER_assume(_AminusBK[1][2] == (__plant_typet)0.0);
  __CPROVER_assume(_AminusBK[2][0] == (__plant_typet)0.0);
  __CPROVER_assume(_AminusBK[2][1] == (__plant_typet)0.015625);
  __CPROVER_assume(_AminusBK[2][2] == (__plant_typet)0.0);
  __CPROVER_assume(_AminusBK_0_2 == (__plant_typet)-195.853967010974884033203125);
  __CPROVER_assume(_AminusBK_1_0 == (__plant_typet)0.125);
  __CPROVER_assume(_AminusBK_2_1 == (__plant_typet)0.015625);
#endif
  __plant_typet tmp4a=_AminusBK[0][2] * (_AminusBK[1][0] * _AminusBK[2][1]);
  __plant_typet tmp4b=_AminusBK_0_2 * (_AminusBK_1_0 * _AminusBK_2_1);
  __CPROVER_assert(0 == 1, "");
  return 0;
}
