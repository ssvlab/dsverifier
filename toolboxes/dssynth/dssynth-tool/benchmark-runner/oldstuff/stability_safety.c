#include <stdio.h>

typedef __CPROVER_fixedbv[24][12] __CPROVER_EIGEN_fixedbvt;

#define NSTATES 3u
#define NINPUTS 1u
#define NOUTPUTS 1u
#define INITIALSTATE_UPPERBOUND 1.0
#define INITIALSTATE_LOWERBOUND 0.0
#define NUMBERLOOPS 10 //number of timesteps to check safety spec over
#define INT_BITS 2
#define FRAC_BITS 3
#define SAFE_STATE_UPPERBOUND 100
#define SAFE_STATE_LOWERBOUND -100

typedef __CPROVER_fixedbv[INT_BITS+FRAC_BITS][FRAC_BITS] __CPROVER_fxp_t;

typedef struct {
    __CPROVER_EIGEN_fixedbvt const A[NSTATES][NSTATES];
    __CPROVER_EIGEN_fixedbvt const B[NSTATES][NINPUTS];
    __CPROVER_EIGEN_fixedbvt const C[NOUTPUTS][NSTATES];
    __CPROVER_EIGEN_fixedbvt const D[NOUTPUTS][NINPUTS];

    __CPROVER_EIGEN_fixedbvt states[NSTATES];
    __CPROVER_EIGEN_fixedbvt outputs[NOUTPUTS];
    __CPROVER_EIGEN_fixedbvt inputs[NINPUTS];
    __CPROVER_EIGEN_fixedbvt const K[NINPUTS][NSTATES];
    __CPROVER_EIGEN_fixedbvt const ref[NINPUTS];
} digital_system_state_space;

__CPROVER_EIGEN_fixedbvt nondet_double(void);

extern __CPROVER_fxp_t K_fxp[NINPUTS][NSTATES];


const digital_system_state_space _controller=
{
    .A = { { 4.6764e-166,0.0,0.0}, { 5.1253e-144,0.0,0.0}, { 0,2.5627e-144,0.0} },
    .B = { { 0.125},{0.0},{0.0} },
    .C = { { 0.16,-5.9787e-32,0.0 } },
    .D = { { 0.0 } },
    .K = { { nondet_double(), nondet_double(), nondet_double() } },
    //.K = { { 1072.1259765625, 1665.046875, -2047.998779296875 } },
    .inputs = { { 1.0 } },
};

__CPROVER_EIGEN_fixedbvt _AminusBK[NSTATES][NSTATES];

#define __CPROVER_EIGEN_MATRIX_SIZE 3u
#define __CPROVER_EIGEN_POLY_SIZE (__CPROVER_EIGEN_MATRIX_SIZE + 1u)
//const __CPROVER_EIGEN_fixedbvt __CPROVER_EIGEN_TEST_A[__CPROVER_EIGEN_MATRIX_SIZE][__CPROVER_EIGEN_MATRIX_SIZE] = { { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 }, { 1.0, 0.0, 0.0 } };
//extern const __CPROVER_EIGEN_fixedbvt __CPROVER_EIGEN_TEST_A[__CPROVER_EIGEN_MATRIX_SIZE][__CPROVER_EIGEN_MATRIX_SIZE];
__CPROVER_EIGEN_fixedbvt __CPROVER_EIGEN_poly[__CPROVER_EIGEN_POLY_SIZE];


__CPROVER_EIGEN_fixedbvt internal_pow(__CPROVER_EIGEN_fixedbvt a, unsigned int b){
   unsigned int i;
   __CPROVER_EIGEN_fixedbvt acc = 1.0;
   for (i=0; i < b; i++){
    acc = acc*a;
   }
   return acc;
}

__CPROVER_EIGEN_fixedbvt internal_abs(__CPROVER_EIGEN_fixedbvt a){
   return a < 0 ? -a : a;
}



int check_stability(void){
#define __a __CPROVER_EIGEN_poly
#define __n __CPROVER_EIGEN_POLY_SIZE
   int lines = 2 * __n - 1;
   int columns = __n;
   __CPROVER_EIGEN_fixedbvt m[lines][__n];
   int i,j;

   /* to put current values in stability counter-example
    * look for current_stability (use: --no-slice) */
   __CPROVER_EIGEN_fixedbvt current_stability[__n];
   for (i=0; i < __n; i++){
     current_stability[i] = __a[i];
   }

   /* check the first constraint condition F(1) > 0 */
   __CPROVER_EIGEN_fixedbvt sum = 0;
   for (i=0; i < __n; i++){
     sum += __a[i];
   }
   if (sum <= 0){
  printf("[DEBUG] the first constraint of Jury criteria failed: (F(1) > 0)");
     return 0;
   }

   /* check the second constraint condition F(-1)*(-1)^n > 0 */
   sum = 0;
   for (i=0; i < __n; i++){
    sum += __a[i] * internal_pow(-1, __n-1-i);
   }
   sum = sum * internal_pow(-1, __n-1);
   if (sum <= 0){
    printf("[DEBUG] the second constraint of Jury criteria failed: (F(-1)*(-1)^n > 0)");
    return 0;
   }

   /* check the third constraint condition abs(a0 < an*(z^n)  */
   if (internal_abs(__a[__n-1]) > __a[0]){
     printf("[DEBUG] the third constraint of Jury criteria failed: (abs(a0) < a_{n}*z^{n})");
     return 0;
   }

   /* check the fourth constraint of condition (Jury Table) */
   for (i=0; i < lines; i++){
    for (j=0; j < columns; j++){
     m[i][j] = 0;
    }
   }
   for (i=0; i < lines; i++){
    for (j=0; j < columns; j++){
     if (i == 0){
      m[i][j] = __a[j];
      continue;
     }
     if (i % 2 != 0 ){
       int x;
       for(x=0; x<columns;x++){
        m[i][x] = m[i-1][columns-x-1];
       }
       columns = columns - 1;
      j = columns;
     }else{
      m[i][j] = m[i-2][j] - (m[i-2][columns] / m[i-2][0]) * m[i-1][j];
     }
    }
   }
   int first_is_positive =  m[0][0] >= 0 ? 1 : 0;
   for (i=0; i < lines; i++){
    if (i % 2 == 0){
     int line_is_positive = m[i][0] >= 0 ? 1 : 0;
     if (first_is_positive != line_is_positive){
      return 0;
     }
     continue;
    }
   }
   return 1;
}


// P(s)=(s-m11)*(s-m22)*(s-m33) - m13*m31*(s-m22) - m12*m21*(s-m33) - m23*m32*(s-m11) - m12*m23*m31 - m13*m21*m32
// P(s)=s^3 + (-m_11 - m_22 - m_33) * s^2 +  (m_11*m_22 + m_11*m_33 - m_12*m_21 - m_13*m_31 + m_22*m_33 - m_23*m_32) * s - m_11*m_22*m_33 + m_11*m_23*m_32 + m_12*m_21*m_33 - m_12*m_23*m_31 - m_13*m_21*m_32 + m_13*m_22*m_31
void __CPROVER_EIGEN_charpoly(void) {
#define __m _AminusBK
  //                        m_11*m_22*m_33                    + m_11*m_23*m_32                    + m_12*m_21*m_33                    - m_12*m_23*m_31                    - m_13*m_21*m_32                    + m_13*m_22*m_31
  __CPROVER_EIGEN_poly[3] = __m[0][0] * __m[1][1] * __m[2][2] + __m[0][0] * __m[1][2] * __m[2][1] + __m[0][1] * __m[1][0] * __m[2][2] - __m[0][1] * __m[1][2] * __m[2][0] - __m[0][2] * __m[1][0] * __m[2][1] + __m[0][2] * __m[1][1] * __m[2][0];
  //                        (m_11*m_22            + m_11*m_33             - m_12*m_21             - m_13*m_31             + m_22*m_33             - m_23*m_32) * s
  __CPROVER_EIGEN_poly[2] = __m[0][0] * __m[1][1] + __m[0][0] * __m[2][2] - __m[0][1] * __m[1][0] - __m[0][2] * __m[2][0] + __m[1][1] * __m[2][2] - __m[1][2] * __m[2][1];
  //                        (-m_11     - m_22      - m_33) * s^2
  __CPROVER_EIGEN_poly[1] = -__m[0][0] - __m[1][1] - __m[2][2];
  // s^3
  __CPROVER_EIGEN_poly[0] = 1.0;
}

/*void init(void) {
  _controller.K[0][0] = nondet_double();
  _controller.K[0][1] = nondet_double();
  _controller.K[0][2] = nondet_double();
}*/

__CPROVER_EIGEN_fixedbvt __CPROVER_EIGEN_matrix_multiplication_result[4][4];

void double_sub_matrix(void/* unsigned int lines, unsigned int columns, __CPROVER_EIGEN_fixedbvt m1[4][4], __CPROVER_EIGEN_fixedbvt m2[4][4], __CPROVER_EIGEN_fixedbvt result[4][4]*/){
#define __sm_lines NSTATES
#define __sm_columns NSTATES
#define __sm_m1 _controller.A
#define __sm_m2 __CPROVER_EIGEN_matrix_multiplication_result
#define __sm_m3 _AminusBK
 unsigned int i, j;
    for (i = 0; i < __sm_lines; i++){
     for (j = 0; j < __sm_columns; j++){
       __sm_m3[i][j] = __sm_m1[i][j] - __sm_m2[i][j];

     }
 }
}

void double_matrix_multiplication(void/* unsigned int i1, unsigned int j1, unsigned int i2, unsigned int j2, __CPROVER_EIGEN_fixedbvt m1[4][4], __CPROVER_EIGEN_fixedbvt m2[4][4], __CPROVER_EIGEN_fixedbvt m3[4][4]*/){
#define __mm_i1 NSTATES
  //unsigned int __mm_i1;
#define __mm_j1 NINPUTS
  //unsigned int __mm_j1;
#define __mm_i2 NINPUTS
  //unsigned int __mm_i2;
#define __mm_j2 NSTATES
  //unsigned int __mm_j2;
#define __mm_m1 _controller.B
  //__CPROVER_EIGEN_fixedbvt __mm_m1[4][4];
#define __mm_m2 _controller.K
  //__CPROVER_EIGEN_fixedbvt __mm_m2[4][4];
#define __mm_m3 __CPROVER_EIGEN_matrix_multiplication_result
  //__CPROVER_EIGEN_fixedbvt __mm_m3[4][4];

 unsigned int i, j, k;
    if (__mm_j1 == __mm_i2) {

        for (i=0;i<__mm_i1; i++) {
            for (j=0; j<__mm_j2; j++) {
                for (k=0; k<__mm_j1; k++) {

                  __CPROVER_EIGEN_fixedbvt mult = (__mm_m1[i][k] * __mm_m2[k][j]);


                    __mm_m3[i][j] = __mm_m3[i][j] + (__mm_m1[i][k] * __mm_m2[k][j]);

                }
            }
        }
    } else {
        printf("\nError! Operation invalid, please enter with valid matrices.\n");
    }
}


void closed_loop(void)
{
  // B*K
  double_matrix_multiplication();
//A - BK
  double_sub_matrix();

}

void inputs_equal_ref_minus_k_times_states(void)
  {
    __CPROVER_fxp_t states_fxp[NSTATES];
    __CPROVER_fxp_t result_fxp[NINPUTS];

    int k, i;
    for(k=0; k<NSTATES;k++)
      {states_fxp[k]= (__CPROVER_fxp_t)_controller.states[k];}

    for(i=0; i<NINPUTS; i++)
    {
      for(k=0; k<NSTATES; k++)
        { result_fxp[i] = result_fxp[i] + (K_fxp[i][k] * states_fxp[k]);}
    }

    for(i=0; i<NINPUTS; i++)
     {
        _controller.inputs[i] = _controller.ref[i] - (__CPROVER_EIGEN_fixedbvt)result_fxp[i];
     }
  }

void states_equals_A_states_plus_B_inputs(void)
 {
   __CPROVER_EIGEN_fixedbvt result[NSTATES];
   int i,k;
   for(i=0; i<NSTATES; i++)
   {
    result[i]=0;
    for(k=0; k<NSTATES; k++)
         {result[i] = result[i] + _controller.A[i][k]*_controller.states[k];}
    for(k=0; k<NSTATES; k++)
         {result[i] = result[i] +_controller.B[i][k]*_controller.inputs[k];}
   }
   for(i=0; i<NSTATES; i++)
       {_controller.states[i] = result[i];}

 }

int check_safety(void)
{
  int i,j,k;

  for(j=0; j<NSTATES; j++)//set initial states
  {
    _controller.states[j] = nondet_double();
    __CPROVER_assume(_controller.states[j]<INITIALSTATE_UPPERBOUND & _controller.states[j]>INITIALSTATE_LOWERBOUND);
  }

  for(i=0; i<NINPUTS;i++)
  {
    for(j=0; j<NSTATES; j++)//convert controller to fixed point
      { K_fxp[i][j]= (__CPROVER_fxp_t)_controller.K[i][j];}
  }

  for(k=0; k<NUMBERLOOPS; k++)
  {
    inputs_equal_ref_minus_k_times_states(); //update inputs one time step
    states_equals_A_states_plus_B_inputs(); //update states one time step

    for(i=0; i<NSTATES; i++)
    {
      if(_controller.states[i]>SAFE_STATE_UPPERBOUND || _controller.states[i]<SAFE_STATE_LOWERBOUND)
        {return 0;}
    }
  }
  return 1;
}


int main(void) {
  //init();
  closed_loop();
  __CPROVER_EIGEN_charpoly();
  //__CPROVER_assert(check_stability(), "");
  __CPROVER_assume(check_stability() != 0);
  __CPROVER_assume(check_safety() != 0);
  __CPROVER_EIGEN_fixedbvt __trace_K0 = _controller.K[0][0];
  __CPROVER_EIGEN_fixedbvt __trace_K1 = _controller.K[0][1];
  __CPROVER_EIGEN_fixedbvt __trace_K2 = _controller.K[0][2];
  __CPROVER_assert(0 == 1, "");
  return 0;
}
