#include <stdlib.h>
#include <stdio.h>



typedef __CPROVER_fixedbv[24][12] __CPROVER_EIGEN_fixedbvt;
//typedef double __CPROVER_EIGEN_fixedbvt;
typedef int64_t fxp_t;

#define NUMBERLOOPS 20u
#define LIMIT 4u
#define FLAG 0u //flag = 0 if D is null
extern __CPROVER_EIGEN_fixedbvt initial_states;
#define INITIALSTATE_UPPERBOUND 1.0
#define INITIALSTATE_LOWERBOUND 0.0

typedef struct {
   int int_bits;
   int frac_bits;
} implementation;

typedef struct {
    //__CPROVER_EIGEN_fixedbvt A[3][3];
    __CPROVER_EIGEN_fixedbvt const A[4][4];
    //__CPROVER_EIGEN_fixedbvt B[3][1];
    __CPROVER_EIGEN_fixedbvt const B[4][4];
    //__CPROVER_EIGEN_fixedbvt C[1][3];
    __CPROVER_EIGEN_fixedbvt const C[4][4];
    //__CPROVER_EIGEN_fixedbvt D[1][1];
    __CPROVER_EIGEN_fixedbvt const D[4][4];
    __CPROVER_EIGEN_fixedbvt states[4][4];
    __CPROVER_EIGEN_fixedbvt outputs[4][4];
    __CPROVER_EIGEN_fixedbvt inputs[4][4];
    __CPROVER_EIGEN_fixedbvt const K[4][4];
} digital_system_state_space;

__CPROVER_EIGEN_fixedbvt nondet_double(void);


const implementation impl = {
 .int_bits = 2,
 .frac_bits = 3
};

#define NSTATES 3u
#define NINPUTS 1u
#define NOUTPUTS 1u

static const double scale_factor[31] = { 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0,
        128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0, 32768.0,
        65536.0, 131072.0, 262144.0, 524288.0, 1048576.0, 2097152.0, 4194304.0,
        8388608.0, 16777216.0, 33554432.0, 67108864.0, 134217728.0,
        268435456.0, 536870912.0, 1073741824.0 };

static const double scale_factor_inv[31] = { 1.0, 0.5, 0.25, 0.125, 0.0625,
        0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 0.0009765625,
        0.00048828125, 0.000244140625, 0.0001220703125, 0.00006103515625,
        0.000030517578125, 0.000015258789063, 0.000007629394531,
        0.000003814697266, 0.000001907348633, 0.000000953674316,
        0.000000476837158, 0.000000238418579, 0.000000119209290,
        0.000000059604645, 0.000000029802322, 0.000000014901161,
        0.000000007450581, 0.000000003725290, 0.000000001862645,
        0.000000000931323 };


digital_system_state_space _controller=
{
    .A = { { 0.974,0.0,0.0,0.0 }, { 1.0,0.0,0.0,0.0 }, { 0.0,0.0,0.0, 0.0 }, { 0.0,0.0,0.0, 0.0 }},
    .B = { { 0.25,0.0,0.0,0.0 }, { 0.0,0.0,0.0,0.0 }, { 0.0,0.0,0.0,0.0 },{ 0.0,0.0,0.0,0.0 } },
    .C = { { 0.157,0.008,0.0,0.0 } },
    .D = { { 0.0 } },
    .K = { { nondet_double(), nondet_double(), nondet_double() } },
    .inputs = { { 1.0 } },
};



/* adds two matrices */
void double_add_matrix( unsigned int lines,  unsigned int columns, __CPROVER_EIGEN_fixedbvt m1[4][4], __CPROVER_EIGEN_fixedbvt m2[4][4], __CPROVER_EIGEN_fixedbvt result[4][4]){
    unsigned int i, j;
    for (i = 0; i < lines; i++){
        for (j = 0; j < columns; j++){
            result[i][j] = m1[i][j] + m2[i][j];
        }
    }
}

/* multiplies two matrices */
void double_matrix_multiplication( unsigned int i1, unsigned int j1, unsigned int i2, unsigned int j2,
                  __CPROVER_EIGEN_fixedbvt m1[4][4], __CPROVER_EIGEN_fixedbvt m2[4][4], __CPROVER_EIGEN_fixedbvt m3[4][4]){

    unsigned int i, j, k;
    if (j1 == i2) {
        for (i=0; i<i1; i++) {
            for (j=0; j<j2; j++) {
                m3[i][j] = 0;
            }
        }

        for (i=0;i<i1; i++) {
            for (j=0; j<j2; j++) {
                for (k=0; k<j1; k++) {
                  __CPROVER_EIGEN_fixedbvt mult = (m1[i][k] * m2[k][j]);
                    m3[i][j] = m3[i][j] + (m1[i][k] * m2[k][j]);
                }
            }
        }
    } else {
        printf("\nError! Operation invalid, please enter with valid matrices.\n");
    }
}


fxp_t fxp_double_to_fxp(__CPROVER_EIGEN_fixedbvt value) {
    fxp_t tmp;
    double ftemp = value * scale_factor[impl.frac_bits];
    tmp = (fxp_t) ftemp;
    return tmp;
}

__CPROVER_EIGEN_fixedbvt fxp_to_double(fxp_t fxp) {
  __CPROVER_EIGEN_fixedbvt f;
    int f_int = (int) fxp;
    f = f_int * scale_factor_inv[impl.frac_bits];
    return f;
}

fxp_t fxp_quantize(fxp_t aquant) {
    return (fxp_t) aquant;
}

fxp_t fxp_mult(fxp_t amult, fxp_t bmult) {
    fxp_t tmpmult, tmpmultprec;
    tmpmult = (fxp_t)((fxp_t)(amult)*(fxp_t)(bmult));
    if (tmpmult >= 0) {
        tmpmultprec = (tmpmult + ((tmpmult & 1 << (impl.frac_bits - 1)) << 1)) >> impl.frac_bits;
    } else {
        tmpmultprec = -(((-tmpmult) + (((-tmpmult) & 1 << (impl.frac_bits - 1)) << 1)) >> impl.frac_bits);
    }
    tmpmultprec = fxp_quantize(tmpmultprec);
    return tmpmultprec;
}


fxp_t fxp_add(fxp_t aadd, fxp_t badd) {
    fxp_t tmpadd;
    tmpadd = ((fxp_t)(aadd) + (fxp_t)(badd));
    tmpadd = fxp_quantize(tmpadd);
    return tmpadd;
}


/* adds two matrices, fixed point version */
void fxp_add_matrix( unsigned int lines,  unsigned int columns, fxp_t m1[4][4], fxp_t m2[4][4], fxp_t result[4][4]){
    unsigned int i, j;
    for (i = 0; i < lines; i++)
        for (j = 0; j < columns; j++) {
        result[i][j] = fxp_add(m1[i][j] , m2[i][j]);
    }
}

/* multiplies two matrices, fixed point version */
void fxp_matrix_multiplication( unsigned int i1, unsigned int j1, unsigned int i2, unsigned int j2, fxp_t m1[4][4], fxp_t m2[4][4], fxp_t m3[4][4]){
    unsigned int i, j, k;
    if (j1 == i2) { //Checking if the multiplication is possible
        // Initialising Matrix 3
        for (i=0; i<i1; i++) {
            for (j=0; j<j2; j++) {
                m3[i][j] = 0;
            }
        }
        //Calculating multiplication result
        for (i=0;i<i1; i++) {
            for (j=0; j<j2; j++) {
                for (k=0; k<j1; k++) {
                    m3[i][j] = fxp_add( m3[i][j], fxp_mult(m1[i][k] , m2[k][j]));
                }
            }
        }
    } else {
        printf("\nError! Operation invalid, please enter with valid matrices.\n");
    }
}



/* subtracts two matrices */
void double_sub_matrix( unsigned int lines,  unsigned int columns, __CPROVER_EIGEN_fixedbvt m1[4][4],
                  __CPROVER_EIGEN_fixedbvt m2[4][4], __CPROVER_EIGEN_fixedbvt result[4][4]){
    unsigned int i, j;
    for (i = 0; i < lines; i++){
        for (j = 0; j < columns; j++){
            result[i][j] = m1[i][j] - m2[i][j];
        }
    }
}

extern fxp_t K_fxp[4][4];

fxp_ss_closed_loop_quantization_error(){

  __CPROVER_EIGEN_fixedbvt reference[4][4] = { { 0.0,0.0,0.0,0.0 }, { 0.0,0.0,0.0,0.0 }, { 0.0,0.0,0.0,0.0 }, { 0.0,0.0,0.0,0.0 }};
  __CPROVER_EIGEN_fixedbvt result1[4][4] = { { 0.0,0.0,0.0,0.0 }, { 0.0,0.0,0.0,0.0 }, { 0.0,0.0,0.0,0.0 }, { 0.0,0.0,0.0,0.0 }};
  __CPROVER_EIGEN_fixedbvt result2[4][4]= { { 0.0,0.0,0.0,0.0 }, { 0.0,0.0,0.0,0.0 }, { 0.0,0.0,0.0,0.0 }, { 0.0,0.0,0.0,0.0 }};

    fxp_t states_fxp[4][4];
    fxp_t result_fxp[4][4] = { {0,0,0,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0}};
    unsigned int k;

      ////// inputs = reference - K * states
              for(k=0; k<NSTATES;k++){
                    states_fxp[k][0]= fxp_double_to_fxp(_controller.states[k][0]);
               }

            fxp_matrix_multiplication(NOUTPUTS,NSTATES,NSTATES,1,K_fxp,states_fxp,result_fxp);

            for(k=0; k<NOUTPUTS;k++){
                    result1[k][0]= fxp_to_double(result_fxp[k][0]);
            }

           double_sub_matrix(NINPUTS,1,reference,result1, _controller.inputs);

        /////output = C*states + D * inputs
            double_matrix_multiplication(NOUTPUTS,NSTATES,NSTATES,1,_controller.C,_controller.states,result1);
            if(FLAG == 1)
                {double_matrix_multiplication(NOUTPUTS,NSTATES,NINPUTS,1,_controller.D,_controller.inputs,result2);
                }
            double_add_matrix(NOUTPUTS,1,result1,result2,_controller.outputs);

        /////states = A*states + B*inputs
            double_matrix_multiplication(NSTATES,NSTATES,NSTATES,1,_controller.A,_controller.states,result1);
           double_matrix_multiplication(NSTATES,NINPUTS,NINPUTS,1,_controller.B,_controller.inputs,result2);
            double_add_matrix(NSTATES,1,result1,result2,_controller.states);
}


int main()
{
  __CPROVER_EIGEN_fixedbvt upper_bound = 1000;
  __CPROVER_EIGEN_fixedbvt lower_bound = -1000;
  int k;
  int i,j,l;

  for(l=0; l<NSTATES; l++)//set initial states
    {_controller.states[l][0] = nondet_double();
    __CPROVER_assume(_controller.states[l][0]<INITIALSTATE_UPPERBOUND & _controller.states[l][0]>INITIALSTATE_LOWERBOUND);
    }

  for(i=0; i<NINPUTS;i++){ //convert controller to fixed point
      for(j=0; j<NSTATES;j++){
          K_fxp[i][j]= fxp_double_to_fxp(_controller.K[i][j]);
      }
  }
  printf("Initial states %f %f\n",_controller.states[0][0], _controller.states[1][0]);


  for(k=0; k<NUMBERLOOPS; k++)
      {fxp_ss_closed_loop_quantization_error(); //one time step
      for(i=0; i<NSTATES; i++){
        printf("Fail states %f %f\n",_controller.states[0][0], _controller.states[1][0]);
        __CPROVER_assert(_controller.states[i][0]<upper_bound && _controller.states[i][0]>lower_bound, "");
        }
      }



 printf("Final states %f %f\n",_controller.states[0][0], _controller.states[1][0]);
__CPROVER_assert(0, " Successful");

return 0;
}
