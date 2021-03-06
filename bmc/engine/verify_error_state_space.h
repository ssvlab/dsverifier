/**
 * # DSVerifier - Digital Systems Verifier (Error)
 * #
 * #                Universidade Federal do Amazonas - UFAM
 * #
 * # Authors:       Iury Bessa     <iury.bessa@gmail.com>
 * #                Hussama Ismail <hussamaismail@gmail.com>
 * #                Felipe Monteiro <felipemonteiro@ufam.edu.br>
 * #
 * # Contributors: Lennon Chaves <lennon.correach@gmail.com>
 * #
 * # ------------------------------------------------------
 * #
 * # For UNCERTAINTY: For use uncertainty, It's only permitted
 * # to use DFI, DFII and TDFII forms.
 * #
 * # ------------------------------------------------------
 */
#ifndef DSVERIFIER_ENGINE_VERIFY_ERROR_STATE_SPACE_H
#define DSVERIFIER_ENGINE_VERIFY_ERROR_STATE_SPACE_H

extern digital_system_state_space _controller;
extern double error_limit;
extern int closed_loop;
double new_state[LIMIT][LIMIT];
double new_stateFWL[LIMIT][LIMIT];
digital_system_state_space _controller_fxp;
digital_system_state_space _controller_double;

double ss_system_quantization_error(fxp_t inputs)
{
  digital_system_state_space __backupController;
  int i;
  int j;

  _controller.inputs[0][0] = inputs;

  for(i = 0; i < nStates; i++)
  {
    for(j = 0; j < nStates; j++)
    {
      __backupController.A[i][j] = (_controller.A[i][j]);
    }
  }

  for(i = 0; i < nStates; i++)
  {
    for(j = 0; j < nInputs; j++)
    {
      __backupController.B[i][j] = (_controller.B[i][j]);
    }
  }

  for(i = 0; i < nOutputs; i++)
  {
    for(j = 0; j < nStates; j++)
    {
      __backupController.C[i][j] = (_controller.C[i][j]);
    }
  }

  for(i = 0; i < nOutputs; i++)
  {
    for(j = 0; j < nInputs; j++)
    {
      __backupController.D[i][j] = (_controller.D[i][j]);
    }
  }

  for(i = 0; i < nStates; i++)
  {
    for(j = 0; j < 1; j++)
    {
      __backupController.states[i][j] = (_controller.states[i][j]);
    }
  }

  for(i = 0; i < nInputs; i++)
  {
    for(j = 0; j < 1; j++)
    {
      __backupController.inputs[i][j] = (_controller.inputs[i][j]);
    }
  }

  for(i = 0; i < nOutputs; i++)
  {
    for(j = 0; j < 1; j++)
    {
      __backupController.outputs[i][j] = (_controller.outputs[i][j]);
    }
  }

  double __quant_error = 0.0;

  for(i = 0; i < nStates; i++)
  {
    for(j = 0; j < 1; j++)
    {
      _controller.states[i][j] = (new_state[i][j]);
    }
  }

  double output_double = double_state_space_representation();

  for(i = 0; i < nStates; i++)
  {
    for(j = 0; j < 1; j++)
    {
      new_state[i][j] = (_controller.states[i][j]);
    }
  }

  __backupController.inputs[0][0] = inputs;

  for(i = 0; i < nStates; i++)
  {
    for(j = 0; j < nStates; j++)
    {
      _controller.A[i][j] = __backupController.A[i][j];
    }
  }

  for(i = 0; i < nStates; i++)
  {
    for(j = 0; j < nInputs; j++)
    {
      _controller.B[i][j] = __backupController.B[i][j];
    }
  }

  for(i = 0; i < nOutputs; i++)
  {
    for(j = 0; j < nStates; j++)
    {
      _controller.C[i][j] = __backupController.C[i][j];
    }
  }

  for(i = 0; i < nOutputs; i++)
  {
    for(j = 0; j < nInputs; j++)
    {
      _controller.D[i][j] = __backupController.D[i][j];
    }
  }

  for(i = 0; i < nStates; i++)
  {
    for(j = 0; j < 1; j++)
    {
      _controller.states[i][j] = __backupController.states[i][j];
    }
  }

  for(i = 0; i < nInputs; i++)
  {
    for(j = 0; j < 1; j++)
    {
      _controller.inputs[i][j] = __backupController.inputs[i][j];
    }
  }

  for(i = 0; i < nOutputs; i++)
  {
    for(j = 0; j < 1; j++)
    {
      _controller.outputs[i][j] = __backupController.outputs[i][j];
    }
  }

  for(i = 0; i < nStates; i++)
  {
    for(j = 0; j < 1; j++)
    {
      _controller.states[i][j] = (new_stateFWL[i][j]);
    }
  }

  double output_fxp = fxp_state_space_representation();

  for(i = 0; i < nStates; i++)
  {
    for(j = 0; j < 1; j++)
    {
      new_stateFWL[i][j] = (_controller.states[i][j]);
    }
  }

  __quant_error = output_double - output_fxp;

  return __quant_error;
}

double fxp_ss_closed_loop_quantization_error(double reference)
{
  double reference_aux[LIMIT][LIMIT];
  double result1[LIMIT][LIMIT];
  double temp_result1[LIMIT][LIMIT];
  double result2[LIMIT][LIMIT];
  double temp_states[LIMIT][LIMIT];
  fxp_t K_fxp[LIMIT][LIMIT];
  fxp_t states_fxp[LIMIT][LIMIT];
  fxp_t result_fxp[LIMIT][LIMIT];
  unsigned int i;
  unsigned int j;
  unsigned int k;
  // flag is 0 if matrix D is null matrix, otherwise flag is 1
  short unsigned int flag = 0;

  for(i = 0; i < nOutputs; i++)
  {    // check if matrix D is a null matrix
    for(j = 0; j < nInputs; j++)
    {
      if(_controller_fxp.D[i][j] != 0)
      {
        flag = 1;
      }
    }
  }

  for(i = 0; i < LIMIT; i++)
  {
    for(j = 0; j < LIMIT; j++)
    {
      reference_aux[i][j] = 0;
      K_fxp[i][j] = 0;
    }
  }

  for(i = 0; i < nInputs; i++)
  {
    reference_aux[i][0] = reference;
  }

  for(i = 0; i < LIMIT; i++)
  {
    states_fxp[i][0] = 0;
  }

  for(i = 0; i < nStates; i++)
  {
    K_fxp[0][i] = fxp_double_to_fxp(_controller_fxp.K[0][i]);
  }

  for(i = 0; i < LIMIT; i++)
  {
    for(j = 0; j < LIMIT; j++)
    {
      result1[i][j] = 0;
      result2[i][j] = 0;
    }
  }

  ////// inputs = reference - K * states

  // result 1 = first element of k * outputs
  for(k = 0; k < nStates; k++)
  {
    states_fxp[k][0] = fxp_double_to_fxp(_controller_fxp.states[k][0]);
  }

  fxp_matrix_multiplication(nOutputs, nStates, nStates, 1, K_fxp, states_fxp,
      result_fxp);

  fxp_t reference_fxp[LIMIT][LIMIT];
  fxp_t result_fxp2[LIMIT][LIMIT];

  for(k = 0; k < nInputs; k++)
  {
    reference_fxp[k][0] = fxp_double_to_fxp(fxp_quantize(reference_aux[k][0]));
  }

  // inputs = reference - result1
  fxp_sub_matrix(nInputs, 1, reference_fxp, result_fxp, result_fxp2);

  for(k = 0; k < nInputs; k++)
  {
    _controller_fxp.inputs[k][0] = fxp_to_double(
        fxp_quantize(result_fxp2[k][0]));
  }

  // output = C*states + D * inputs

  // result1 = C * states
  double_matrix_multiplication(nOutputs, nStates, nStates, 1, _controller_fxp.C,
      _controller_fxp.states, result1);

  // result2 = D * inputs
  if(flag == 1)
  {
    double_matrix_multiplication(nOutputs, nInputs, nInputs, 1,
        _controller_fxp.D, _controller_fxp.inputs, result2);
  }

  // outputs = result 1 + result 2 = C*states + D * inputs
  double_add_matrix(nOutputs, 1, result1, result2, _controller_fxp.outputs);

  // states = A*states + B*inputs

  // result1 = A * states
  double_matrix_multiplication(nStates, nStates, nStates, 1, _controller_fxp.A,
      _controller_fxp.states, result1);

  // result2 = B*inputs
  double_matrix_multiplication(nStates, nInputs, nInputs, 1, _controller_fxp.B,
      _controller_fxp.inputs, result2);

  // states = result 1 + result 2 =A*states + B*inputs
  double_add_matrix(nStates, 1, result1, result2, _controller_fxp.states);

  return _controller_fxp.outputs[0][0];
}

double ss_closed_loop_quantization_error(double reference)
{
  double reference_aux[LIMIT][LIMIT];
  double result1[LIMIT][LIMIT];
  double result2[LIMIT][LIMIT];
  unsigned int i;
  unsigned int j;
  // flag is 0 if matrix D is null matrix, otherwise flag is 1
  short unsigned int flag = 0;

  for(i = 0; i < nOutputs; i++)
  {    // check if matrix D is a null matrix
    for(j = 0; j < nInputs; j++)
    {
      if(_controller_double.D[i][j] != 0)
      {
        flag = 1;
      }
    }
  }

  for(i = 0; i < nInputs; i++)
  {
    for(j = 0; j < 1; j++)
    {
      reference_aux[i][j] = reference;
    }
  }

  for(i = 0; i < LIMIT; i++)
  {
    for(j = 0; j < LIMIT; j++)
    {
      result1[i][j] = 0;
      result2[i][j] = 0;
    }
  }

  // inputs = reference - K * states

  // result 1 = first element of k * outputs
  double_matrix_multiplication(nOutputs, nStates, nStates, 1,
      _controller_double.K, _controller_double.states, result1);

  // inputs = reference - result1
  double_sub_matrix(nInputs, 1, reference_aux, result1,
      _controller_double.inputs);

  // output = C*states + D * inputs

  // result1 = C * states0
  double_matrix_multiplication(nOutputs, nStates, nStates, 1,
      _controller_double.C, _controller_double.states, result1);

  // result2 = D * inputs
  if(flag == 1)
  {
    double_matrix_multiplication(nOutputs, nInputs, nInputs, 1,
        _controller_double.D, _controller_double.inputs, result2);
  }

  // outputs = result 1 + result 2 = C*states + D * inputs
  double_add_matrix(nOutputs, 1, result1, result2, _controller_double.outputs);

  // states = A*states + B*inputs

  // result1 = A * states
  double_matrix_multiplication(nStates, nStates, nStates, 1,
      _controller_double.A, _controller_double.states, result1);

  // result2 = B*inputs
  double_matrix_multiplication(nStates, nInputs, nInputs, 1,
      _controller_double.B, _controller_double.inputs, result2);

  // states = result 1 + result 2 =A*states + B*inputs
  double_add_matrix(nStates, 1, result1, result2, _controller_double.states);

  return _controller_double.outputs[0][0];
}

int verify_error_state_space(void)
{
  int i, j;

  for(i = 0; i < nStates; i++)
  {
    for(j = 0; j < 1; j++)
    {
      new_state[i][j] = (_controller.states[i][j]);
    }
  }

  for(i = 0; i < nStates; i++)
  {
    for(j = 0; j < 1; j++)
    {
      new_stateFWL[i][j] = (_controller.states[i][j]);
    }
  }

  _controller_fxp = _controller;
  _controller_double = _controller;
  set_overflow_mode = 0;

  fxp_t x[K_SIZE];
  fxp_t min_fxp = fxp_double_to_fxp(impl.min);
  fxp_t max_fxp = fxp_double_to_fxp(impl.max);
  double nondet_constant_input = nondet_double();

  __DSVERIFIER_assume(
      (nondet_constant_input >= min_fxp) && (nondet_constant_input <= max_fxp));

  for(i = 0; i < K_SIZE; ++i)
  {
    x[i] = nondet_constant_input;
  }

  double __quant_error;

  if(closed_loop)
  {
    for(i = 0; i < K_SIZE; ++i)
    {
      __quant_error = ss_closed_loop_quantization_error(x[i])
          - fxp_ss_closed_loop_quantization_error(x[i]);

      assert(
          (__quant_error < error_limit)
              && (__quant_error > ((-1) * error_limit)));
    }
  }
  else
  {
    for(i = 0; i < K_SIZE; i++)
    {
      __quant_error = ss_system_quantization_error(x[i]);

      assert(
          (__quant_error < error_limit)
              && (__quant_error > ((-1) * error_limit)));
    }
  }

  return 0;
}
#endif // DSVERIFIER_ENGINE_VERIFY_ERROR_STATE_SPACE_H
