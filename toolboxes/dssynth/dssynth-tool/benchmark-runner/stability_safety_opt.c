#include <stdio.h>
#include "dcmotor_ss_disc1.h"



#define INITIALSTATE_UPPERBOUND (__CPROVER_EIGEN_fixedbvt)0.5
#define INITIALSTATE_LOWERBOUND (__CPROVER_EIGEN_fixedbvt)-0.5
#define SAFE_STATE_UPPERBOUND (__CPROVER_EIGEN_fixedbvt)1
#define SAFE_STATE_LOWERBOUND (__CPROVER_EIGEN_fixedbvt)-1

typedef __CPROVER_fixedbv[INT_BITS+FRAC_BITS][FRAC_BITS] __CPROVER_fxp_t;
__CPROVER_EIGEN_fixedbvt _AminusBK[NSTATES][NSTATES];


__CPROVER_EIGEN_fixedbvt _AminusBK[NSTATES][NSTATES];
extern const __CPROVER_fxp_t K_fxp[NSTATES];
__CPROVER_EIGEN_fixedbvt _controller_inputs;
extern __CPROVER_EIGEN_fixedbvt _controller_states[NSTATES];

#define __CPROVER_EIGEN_POLY_SIZE (NSTATES + 1u)
__CPROVER_EIGEN_fixedbvt __CPROVER_EIGEN_poly[__CPROVER_EIGEN_POLY_SIZE];


__CPROVER_EIGEN_fixedbvt internal_pow(__CPROVER_EIGEN_fixedbvt a, unsigned int b){

   __CPROVER_EIGEN_fixedbvt acc = 1.0;
   for (int i=0; i < b; i++){
    acc = acc*a;
   }
   return acc;
}

__CPROVER_EIGEN_fixedbvt internal_abs(__CPROVER_EIGEN_fixedbvt a){
   return a < 0 ? -a : a;
}

int check_stability(void){

#if NSTATES==1
  if(_AminusBK[0][0] > 1 || _AminusBK[0][0] < -1)
    {return 0;}
  else
    {return 1;}
#endif

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

#define __m _AminusBK
void __CPROVER_EIGEN_charpoly_2(void) { //m00*m11 - m10*m11 - m00*x - m11*x + x^2

  __CPROVER_EIGEN_poly[2] = __m[0][0]*__m[1][1] - __m[1][0]*__m[1][1];

  __CPROVER_EIGEN_poly[1] = -__m[0][0] - __m[1][1];
  // s^2
  __CPROVER_EIGEN_poly[0] = 1.0;
}


// P(s)=(s-m11)*(s-m22)*(s-m33) - m13*m31*(s-m22) - m12*m21*(s-m33) - m23*m32*(s-m11) - m12*m23*m31 - m13*m21*m32
// P(s)=s^3 + (-m_11 - m_22 - m_33) * s^2 +  (m_11*m_22 + m_11*m_33 - m_12*m_21 - m_13*m_31 + m_22*m_33 - m_23*m_32) * s - m_11*m_22*m_33 + m_11*m_23*m_32 + m_12*m_21*m_33 - m_12*m_23*m_31 - m_13*m_21*m_32 + m_13*m_22*m_31
void __CPROVER_EIGEN_charpoly_3(void) {

  //                        m_11*m_22*m_33                    + m_11*m_23*m_32                    + m_12*m_21*m_33                    - m_12*m_23*m_31                    - m_13*m_21*m_32                    + m_13*m_22*m_31
  __CPROVER_EIGEN_poly[3] = __m[0][0] * __m[1][1] * __m[2][2] + __m[0][0] * __m[1][2] * __m[2][1] + __m[0][1] * __m[1][0] * __m[2][2] - __m[0][1] * __m[1][2] * __m[2][0] - __m[0][2] * __m[1][0] * __m[2][1] + __m[0][2] * __m[1][1] * __m[2][0];
  //                        (m_11*m_22            + m_11*m_33             - m_12*m_21             - m_13*m_31             + m_22*m_33             - m_23*m_32) * s
  __CPROVER_EIGEN_poly[2] = __m[0][0] * __m[1][1] + __m[0][0] * __m[2][2] - __m[0][1] * __m[1][0] - __m[0][2] * __m[2][0] + __m[1][1] * __m[2][2] - __m[1][2] * __m[2][1];
  //                        (-m_11     - m_22      - m_33) * s^2
  __CPROVER_EIGEN_poly[1] = -__m[0][0] - __m[1][1] - __m[2][2];
  // s^3
  __CPROVER_EIGEN_poly[0] = 1.0;
}


void __CPROVER_EIGEN_charpoly_4(void) {

 __CPROVER_EIGEN_poly[4] = __m[0][0]*__m[1][1]*__m[2][2]*__m[3][3] - __m[0][0]*__m[1][1]*__m[2][3]*__m[3][2] - __m[0][0]*__m[1][2]*__m[2][1]*__m[3][3] + __m[0][0]*__m[1][2]*__m[2][3]*__m[3][1] + __m[0][0]*__m[1][3]*__m[2][1]*__m[3][2] -
    __m[0][0]*__m[1][3]*__m[2][2]*__m[3][1] - __m[0][1]*__m[1][0]*__m[2][2]*__m[3][3] + __m[0][1]*__m[1][0]*__m[2][3]*__m[3][2] + __m[0][1]*__m[1][2]*__m[2][0]*__m[3][3] - __m[0][1]*__m[1][2]*__m[2][3]*__m[3][0] -
    __m[0][1]*__m[1][3]*__m[2][0]*__m[3][2] + __m[0][1]*__m[1][3]*__m[2][2]*__m[3][0] + __m[0][2]*__m[1][0]*__m[2][1]*__m[3][3] - __m[0][2]*__m[1][0]*__m[2][3]*__m[3][1] - __m[0][2]*__m[1][1]*__m[2][0]*__m[3][3] +
    __m[0][2]*__m[1][1]*__m[2][3]*__m[3][0] + __m[0][2]*__m[1][3]*__m[2][0]*__m[3][1] - __m[0][2]*__m[1][3]*__m[2][1]*__m[3][0] - __m[0][3]*__m[1][0]*__m[2][1]*__m[3][2] + __m[0][3]*__m[1][0]*__m[2][2]*__m[3][1] +
    __m[0][3]*__m[1][1]*__m[2][0]*__m[3][2] - __m[0][3]*__m[1][1]*__m[2][2]*__m[3][0] - __m[0][3]*__m[1][2]*__m[2][0]*__m[3][1] + __m[0][3]*__m[1][2]*__m[2][1]*__m[3][0];


__CPROVER_EIGEN_poly[3] = - __m[0][0]*__m[1][1]*__m[2][2]  + __m[0][0]*__m[1][2]*__m[2][1]  + __m[0][1]*__m[1][0]*__m[2][2] -   __m[0][1]*__m[1][2]*__m[2][0]  -  __m[0][2]*__m[1][0]*__m[2][1]  + __m[0][2]*__m[1][1]*__m[2][0]
    - __m[0][0]*__m[1][1]*__m[3][3]  + __m[0][0]*__m[1][3]*__m[3][1]  + __m[0][1]*__m[1][0]*__m[3][3] - __m[0][1]*__m[1][3]*__m[3][0] -  __m[0][3]*__m[1][0]*__m[3][1]  + __m[0][3]*__m[1][1]*__m[3][0]
    - __m[0][0]*__m[2][2]*__m[3][3]  + __m[0][0]*__m[2][3]*__m[3][2]  + __m[0][2]*__m[2][0]*__m[3][3] - __m[0][2]*__m[2][3]*__m[3][0] - __m[0][3]*__m[2][0]*__m[3][2]  + __m[0][3]*__m[2][2]*__m[3][0]
    - __m[1][1]*__m[2][2]*__m[3][3]  + __m[1][1]*__m[2][3]*__m[3][2]  + __m[1][2]*__m[2][1]*__m[3][3] - __m[1][2]*__m[2][3]*__m[3][1] -  __m[1][3]*__m[2][1]*__m[3][2]  + __m[1][3]*__m[2][2]*__m[3][1];


  __CPROVER_EIGEN_poly[2] = + __m[0][0]*__m[1][1]  - __m[0][1]*__m[1][0]  +  __m[0][0]*__m[2][2]  - __m[0][2]*__m[2][0] +  __m[0][0]*__m[3][3]  - __m[0][3]*__m[3][0] + __m[1][1]*__m[2][2]  -
   __m[1][2]*__m[2][1] +  __m[1][1]*__m[3][3] - __m[1][3]*__m[3][1] +  __m[2][2]*__m[3][3]  - __m[2][3]*__m[3][2];


  __CPROVER_EIGEN_poly[1] = - __m[3][3] - __m[2][2] - __m[1][1] - __m[0][0];
  __CPROVER_EIGEN_poly[0] = 1.0;
}

void __CPROVER_EIGEN_charpoly(void){
#if NSTATES==1
  __CPROVER_assume(_AminusBK[0][0] < 1 && _AminusBK[0][0] > -1);
#elif  NSTATES==2
  __CPROVER_EIGEN_charpoly_2();
#elif  NSTATES==3
  __CPROVER_EIGEN_charpoly_3();
#elif  NSTATES==4
  __CPROVER_EIGEN_charpoly_4();
#endif
}

void A_minus_B_K()
{
  /*for(i=0; i<NSTATES; i++)
   {
    for(j=0; j<NSTATES; j++)
      {_AminusBK[i][j] = _controller_A[i][j];}
   }*/
  __CPROVER_array_copy(_AminusBK, _controller_A);

  for (int i=0;i<NSTATES; i++) { //rows of B
      for (int j=0; j<NSTATES; j++) { //columns of K
             // _AminusBK[i][j] -= _controller_B[i] * _controller_K[j];
        _AminusBK[i][j] -= _controller_B[i] * (__CPROVER_EIGEN_fixedbvt)K_fxp[j];
          }
      }
}

void closed_loop(void)
{
  A_minus_B_K();
}

void inputs_equal_ref_minus_k_times_states(void)
  {
    __CPROVER_fxp_t states_fxp[NSTATES];
    //inputs_equal_ref_minus_k_times_states_result_fxp
    //single input
    __CPROVER_fxp_t result_fxp=0;

     for(int k=0; k<NSTATES; k++)
        { result_fxp += (K_fxp[k] * (__CPROVER_fxp_t)_controller_states[k]);}

        _controller_inputs = -(__CPROVER_EIGEN_fixedbvt)result_fxp;
        __CPROVER_assume(_controller_inputs<INPUT_UPPERBOUND && _controller_inputs>INPUT_LOWERBOUND);
  }

__CPROVER_EIGEN_fixedbvt states_equals_A_states_plus_B_inputs_result[NSTATES];

void states_equals_A_states_plus_B_inputs(void)
 {
   __CPROVER_array_set(states_equals_A_states_plus_B_inputs_result, (__CPROVER_EIGEN_fixedbvt)0.0);
   //int i,k;
   for(int i=0; i<NSTATES; i++)
   {
     //states_equals_A_states_plus_B_inputs_result[i]=0;
    for(int k=0; k<NSTATES; k++) {
      states_equals_A_states_plus_B_inputs_result[i] += _controller_A[i][k]*_controller_states[k];
      states_equals_A_states_plus_B_inputs_result[i] += _controller_B[i]*_controller_inputs;
    }
   }
   __CPROVER_array_copy(_controller_states, states_equals_A_states_plus_B_inputs_result);
   /*for(i=0; i<NSTATES; i++)
       {_controller_states[i] = states_equals_A_states_plus_B_inputs_result[i];}*/
  __CPROVER_assume( _controller_states[0]<SAFE_STATE_UPPERBOUND && _controller_states[0]>SAFE_STATE_LOWERBOUND);
  __CPROVER_assume( _controller_states[1]<SAFE_STATE_UPPERBOUND && _controller_states[1]>SAFE_STATE_LOWERBOUND);
  __CPROVER_assume( _controller_states[2]<SAFE_STATE_UPPERBOUND && _controller_states[2]>SAFE_STATE_LOWERBOUND);
 }




int check_safety(void)
{

  for(int j=0; j<NSTATES; j++)//set initial states and reference
  {
    __CPROVER_EIGEN_fixedbvt __state0 = _controller_states[0];
     __CPROVER_EIGEN_fixedbvt __state1 = _controller_states[1];
     __CPROVER_EIGEN_fixedbvt __state2 = _controller_states[2];
    __CPROVER_assume(_controller_states[j]<INITIALSTATE_UPPERBOUND && _controller_states[j]>INITIALSTATE_LOWERBOUND);
    __CPROVER_assume(_controller_states[j]!=(__CPROVER_EIGEN_fixedbvt)0.0);

  }

  for(int k=0; k<NUMBERLOOPS; k++)
  {
    inputs_equal_ref_minus_k_times_states(); //update inputs one time step
    states_equals_A_states_plus_B_inputs(); //update states one time step

    for(int i=0; i<NSTATES; i++)
    {
      if(_controller_states[i]>SAFE_STATE_UPPERBOUND || _controller_states[i]<SAFE_STATE_LOWERBOUND)
        {return 0;}
    }
  }
  return 1;
}


int main(void) {
  //init();
  closed_loop(); //calculate A - BK
  __CPROVER_EIGEN_charpoly(); //get eigen values
  __CPROVER_assume(check_stability() != 0);
  __CPROVER_assume(check_safety() !=0);
 /* __CPROVER_EIGEN_fixedbvt __trace_K0 = _controller_K[0];
  __CPROVER_EIGEN_fixedbvt __trace_K1 = _controller_K[1];
  __CPROVER_EIGEN_fixedbvt __trace_K2 = _controller_K[2];*/
  __CPROVER_EIGEN_fixedbvt __trace_fxpK0 = K_fxp[0];
  __CPROVER_EIGEN_fixedbvt __trace_fxpK1 = K_fxp[1];
  __CPROVER_EIGEN_fixedbvt __trace_fxpK2 = K_fxp[2];

  __CPROVER_assert(0 == 1, "");
  return 0;
}
