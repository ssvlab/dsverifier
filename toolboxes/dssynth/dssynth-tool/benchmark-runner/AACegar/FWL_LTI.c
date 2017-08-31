#include <assert.h>
#include <stdbool.h>

#include "FWL_LTI.h"
#include "types.h"
#include "system.h"

void print_vector(char *name,vectort vector)
{
#ifndef __CPROVER
  cnttype i;
  printf("%s ",name);
  for (i=0;i<_DIMENSION;i++) printf("%f ", vector[i]);
  puts("");
#endif
}

void print_matrix(char *name,matrixt matrix)
{
#ifndef __CPROVER
  cnttype i,j;
  printf("%s ",name);
  for (i=0;i<_DIMENSION;i++) 
  {
    for (j=0;j<_DIMENSION;j++) printf("%f ", matrix[i][j]);
    printf(";");
  }
  puts("");
#endif
}

vectort plant_cbmc,controller_cbmc;
matrixt transform_cbmc;
matrixt loop_cbmc;

#include "FWL_LTI_FWL.h"
#include "FWL_LTI_nondet.h"
#include "FWL_LTI_stability.h"

#ifdef _NUM_ITERATIONS
  #include "FWL_LTI_iters.h"
#endif

//struct implt impl={ .int_bits=_CONTROLER_INT_BITS, .frac_bits=_CONTROLER_FRAC_BITS};

int initialization()
{
  get_bounds();
  int result=validation();
  return result;
}

#ifndef __CPROVER 
void print_poly(control_floatt *pol,cnttype n)
{
  cnttype i;
  for (i=0;i<n;i++)
  {
    printf("%fz%d ", pol[i], n-i-1);
  }
  puts("");
}
#endif

// main
int main()
{
  cnttype i;
  int result=initialization();
#ifndef __CPROVER
  if (result!=0) return 10;
#endif
  make_closed_loop();
#ifndef __CPROVER
  print_matrix("dynamics",dynamics);
  print_matrix("loop",loop_cbmc);
#endif  
  result=check_stability_closedloop(plant_cbmc);
#ifndef __CPROVER
  printf("stability=%d\n",result);
#endif 
#ifdef _NUM_ITERATIONS
  if (result>0) checkIterations(loop_cbmc);
#endif
#ifdef __CPROVER
  __CPROVER_array_copy(controller_cbmc, controller);
  verify_assert(0);
#else
  printf("end\n");
  if (result != 0) return 10;
#endif
  return 0;
}
