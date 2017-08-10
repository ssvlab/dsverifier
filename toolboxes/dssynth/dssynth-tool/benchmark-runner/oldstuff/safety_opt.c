

typedef __CPROVER_fixedbv[24][12] __CPROVER_EIGEN_fixedbvt;
__CPROVER_EIGEN_fixedbvt nondet_double(void);

#define NSTATES 4u
#define NINPUTS 1u
#define NOUTPUTS 1u
#define INITIALSTATE_UPPERBOUND 1.0
#define INITIALSTATE_LOWERBOUND 0.0
#define NUMBERLOOPS 10
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

  }digital_system_state_space;



  digital_system_state_space _controller=
  {
      .A = { { 0.974,0.0},{ 1.0,0.0}},
      .B = { { 0.25},{0.0} },
     .C = { { 0.157,0.008} },
      .D = { { 0.0 } },
      .K = { 0.5, 1 },
     // .K = { nondet_double(), nondet_double() },
      .inputs = { { 1.0 } },
      .ref = {{0.0}},
  };

  extern __CPROVER_fxp_t K_fxp[NINPUTS][NSTATES];

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

  /*void outputs_equals_C_states_plus_D_inputs(void) //this is not needed becuase our safety spec is bounds on the states.
  {
    int i,j,k;


   for(i=0; i<NOUTPUTS; i++)
   {
     _controller.outputs[i]=0;
     for(j=0; j<NSTATES; j++)
         {_controller.outputs[i] = _controller.outputs[i] +
                     _controller.C[i][j]*_controller.states[j];}
     for(k=0; k<NINPUTS; k++)
     { _controller.outputs[i] = _controller.outputs[i] +
           _controller.D[i][j]*_controller.inputs[j];}

     }
  }*/

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


  fxp_ss_closed_loop_quantization_error(){

        ////// inputs = reference - K * states
              inputs_equal_ref_minus_k_times_states();
          /////output = C*states + D * inputs
             // outputs_equals_C_states_plus_D_inputs(); //we are using state feedback, we don't need the output

          /////states = A*states + B*inputs
              states_equals_A_states_plus_B_inputs();
  }

  int main()
  {
    int i,j,k;

    for(j=0; j<NSTATES; j++)//set initial states
      {_controller.states[j] = nondet_double();
      __CPROVER_assume(_controller.states[j]<INITIALSTATE_UPPERBOUND & _controller.states[j]>INITIALSTATE_LOWERBOUND);
      }

    for(i=0; i<NINPUTS;i++){
      for(j=0; j<NSTATES; j++)//convert controller to fixed point
      { K_fxp[i][j]= (__CPROVER_fxp_t)_controller.K[i][j];}
    }
  
    for(k=0; k<NUMBERLOOPS; k++)
        {fxp_ss_closed_loop_quantization_error(); //one time step
        for(i=0; i<NSTATES; i++){
         //
         __CPROVER_assert(_controller.states[i]<SAFE_STATE_UPPERBOUND && _controller.states[i]>SAFE_STATE_LOWERBOUND, "");
          }
        }

   //printf("Final states %f %f\n",_controller.states[0], _controller.states[1]);
  __CPROVER_assert(0, " Successful");

  return 0;
  }
