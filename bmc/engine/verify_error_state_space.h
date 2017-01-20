/**
# DSVerifier - Digital Systems Verifier (Error)
#
#                Universidade Federal do Amazonas - UFAM
#
# Authors:       Iury Bessa     <iury.bessa@gmail.com>
#                Hussama Ismail <hussamaismail@gmail.com>
#                Felipe Monteiro <felipemonteiro@ufam.edu.br>
# ------------------------------------------------------
#
# For UNCERTAINTY: For use uncertainty, It's only permitted
# to use DFI, DFII and TDFII forms.
#
# ------------------------------------------------------
 */

extern digital_system_state_space _controller;
extern double error_limit;
extern int closed_loop;

double new_state[LIMIT][LIMIT];
double new_stateFWL[LIMIT][LIMIT];

double ss_system_quantization_error(fxp_t inputs){

	digital_system_state_space __backupController;
	
	int i;
	int j;

			_controller.inputs[0][0] = inputs;

			for(i=0; i<nStates;i++){
				for(j=0; j<nStates;j++){
					__backupController.A[i][j]= (_controller.A[i][j]);
				}
			}

			for(i=0; i<nStates;i++){
				for(j=0; j<nInputs;j++){
					__backupController.B[i][j]= (_controller.B[i][j]);
				}
			}

			for(i=0; i<nOutputs;i++){
				for(j=0; j<nStates;j++){
					__backupController.C[i][j]= (_controller.C[i][j]);
				}
			}

			for(i=0; i<nOutputs;i++){
				for(j=0; j<nInputs;j++){
					__backupController.D[i][j]= (_controller.D[i][j]);
				}
			}

			for(i=0; i<nStates;i++){
				for(j=0; j<1;j++){
					__backupController.states[i][j]= (_controller.states[i][j]);
				}
			}

			for(i=0; i<nInputs;i++){
				for(j=0; j<1;j++){
					__backupController.inputs[i][j]= (_controller.inputs[i][j]);
				}
			}

			for(i=0; i<nOutputs;i++){
				for(j=0; j<1;j++){
					__backupController.outputs[i][j]= (_controller.outputs[i][j]);
				}
			}

			double __quant_error = 0.0;

			for(i=0; i<nStates;i++){
				for(j=0; j<1;j++){
					_controller.states[i][j]= (new_state[i][j]);
				}
			}

			double output_double = double_state_space_representation();
			
			for(i=0; i<nStates;i++){
				for(j=0; j<1;j++){
					new_state[i][j]= (_controller.states[i][j]);
				}
			}

			__backupController.inputs[0][0] = inputs;

			for(i=0; i<nStates;i++){
				for(j=0; j<nStates;j++){
					_controller.A[i][j] = __backupController.A[i][j];
				}
			}

			for(i=0; i<nStates;i++){
				for(j=0; j<nInputs;j++){
					_controller.B[i][j] = __backupController.B[i][j];
				}
			}

			for(i=0; i<nOutputs;i++){
				for(j=0; j<nStates;j++){
					_controller.C[i][j] = __backupController.C[i][j];
				}
			}

			for(i=0; i<nOutputs;i++){
				for(j=0; j<nInputs;j++){
					_controller.D[i][j] = __backupController.D[i][j];
				}
			}

			for(i=0; i<nStates;i++){
				for(j=0; j<1;j++){
					_controller.states[i][j] = __backupController.states[i][j];
				}
			}

			for(i=0; i<nInputs;i++){
				for(j=0; j<1;j++){
					_controller.inputs[i][j] = __backupController.inputs[i][j];
				}
			}

			for(i=0; i<nOutputs;i++){
				for(j=0; j<1;j++){
					_controller.outputs[i][j] = __backupController.outputs[i][j];
				}
			}

			for(i=0; i<nStates;i++){
				for(j=0; j<1;j++){
					_controller.states[i][j]= (new_stateFWL[i][j]);
				}
			} 
			double output_fxp = fxp_state_space_representation();
			
			for(i=0; i<nStates;i++){
				for(j=0; j<1;j++){
					new_stateFWL[i][j]= (_controller.states[i][j]);
				}
			}

			__quant_error = output_double - output_fxp;

			return __quant_error;
}

double ss_closed_loop_quantization_error(){

	double reference[LIMIT][LIMIT];
	double result1[LIMIT][LIMIT];
	double result2[LIMIT][LIMIT];
	unsigned int i;
	unsigned int j;
	short unsigned int flag = 0; // flag is 0 if matrix D is null matrix, otherwise flag is 1

	for(i=0; i<nOutputs;i++){ // check if matrix D is a null matrix
		for(j=0; j<nInputs;j++){
			if(_controller.D[i][j] != 0){
				flag = 1;
			}
		}
	}

	for(i=0; i<nInputs;i++){
		for(j=0; j<1;j++){
			reference[i][j]= (_controller.inputs[i][j]);
		}
	}

	for(i=0; i<LIMIT;i++){
		for(j=0; j<LIMIT;j++){
			result1[i][j]=0;
			result2[i][j]=0;
		}
	}

	  for (i = 1; i < K_SIZE; i++) {

	    ////// inputs = reference - K * states
	        //result 1 = first element of k * outputs
	        double_matrix_multiplication(nOutputs,nStates,nStates,1,_controller.K,_controller.states,result1);
	        //inputs = reference - result1
	        double_sub_matrix(nInputs,1,reference,result1, _controller.inputs);

	    /////output = C*states + D * inputs
	        //result1 = C * states
	        double_matrix_multiplication(nOutputs,nStates,nStates,1,_controller.C,_controller.states,result1);
	        //result2 = D * inputs
	        if(flag == 1)
	            double_matrix_multiplication(nOutputs,nInputs,nInputs,1,_controller.D,_controller.inputs,result2);
	        //outputs = result 1 + result 2 = C*states + D * inputs
	        double_add_matrix(nOutputs,1,result1,result2,_controller.outputs);

	    /////states = A*states + B*inputs
	        //result1 = A * states
	        double_matrix_multiplication(nStates,nStates,nStates,1,_controller.A,_controller.states,result1);
	        //result2 = B*inputs
	        double_matrix_multiplication(nStates,nInputs,nInputs,1,_controller.B,_controller.inputs,result2);
	        //states = result 1 + result 2 =A*states + B*inputs
	        double_add_matrix(nStates,1,result1,result2,_controller.states);

	        }

	return _controller.outputs[0][0];
}

double fxp_ss_closed_loop_quantization_error(){

	double reference[LIMIT][LIMIT];
	double result1[LIMIT][LIMIT];
	double result2[LIMIT][LIMIT];
	fxp_t K_fpx[LIMIT][LIMIT];
	fxp_t states_fxp[LIMIT][LIMIT];
	fxp_t result_fxp[LIMIT][LIMIT];
	unsigned int i;
	unsigned int j;
	unsigned int k;
	short unsigned int flag = 0; // flag is 0 if matrix D is null matrix, otherwise flag is 1

	for(i=0; i<nOutputs;i++){ // check if matrix D is a null matrix
		for(j=0; j<nInputs;j++){
			if(_controller.D[i][j] != 0){
				flag = 1;
			}
		}
	}

	for(i=0; i<nInputs;i++){
		for(j=0; j<1;j++){
			reference[i][j]= (_controller.inputs[i][j]);
		}
	}

	for(i=0; i<nInputs;i++){
		for(j=0; j<nOutputs;j++){
			K_fpx[i][j]=0;
		}
	}

	for(i=0; i<nOutputs;i++){
		for(j=0; j<1;j++){
			outputs_fxp[i][j]=0;
		}
	}

	for(i=0; i<LIMIT;i++){
		for(j=0; j<LIMIT;j++){
			result_fxp[i][j]=0;
		}
	}

	for(i=0; i<nInputs;i++){
		for(j=0; j<nOutputs;j++){
			K_fpx[i][j]= fxp_double_to_fxp(_controller.K[i][j]);
		}
	}

	for(i=0; i<LIMIT;i++){
		for(j=0; j<LIMIT;j++){
			result1[i][j]=0;
			result2[i][j]=0;
		}
	}

	for (i = 1; i < K_SIZE; i++) {

		double_matrix_multiplication(nOutputs,nStates,nStates,1,_controller.C,_controller.states,result1);

		if(flag == 1){
			double_matrix_multiplication(nOutputs,nInputs,nInputs,1,_controller.D,_controller.inputs,result2);
		}

		double_add_matrix(nOutputs,
				1,
				result1,
				result2,
				_controller.outputs);

		for(k=0; k<nOutputs;k++){
			for(j=0; j<1;j++){
				outputs_fxp[k][j]= fxp_double_to_fxp(_controller.outputs[k][j]);
			}
		}

		fxp_matrix_multiplication(nInputs,nOutputs,nOutputs,1,K_fpx,outputs_fxp,result_fxp);

		for(k=0; k<nInputs;k++){
			for(j=0; j<1;j++){
				result1[k][j]= fxp_to_double(result_fxp[k][j]);
			}
		}

		printf("### fxp: U (before) = %.9f", _controller.inputs[0][0]);
		printf("### fxp: reference = %.9f", reference[0][0]);
		printf("### fxp: result1 = %.9f", result1[0][0]);
		printf("### fxp: reference - result1 = %.9f", (reference[0][0] - result1[0][0]));

		double_sub_matrix(nInputs,
				1,
				reference,
				result1,
				_controller.inputs);

		printf("### fxp: Y = %.9f", _controller.outputs[0][0]);
		printf("### fxp: U (after) = %.9f \n### \n### ", _controller.inputs[0][0]);

		double_matrix_multiplication(nStates,nStates,nStates,1,_controller.A,_controller.states,result1);
		double_matrix_multiplication(nStates,nInputs,nInputs,1,_controller.B,_controller.inputs,result2);

		double_add_matrix(nStates,
				1,
				result1,
				result2,
				_controller.states);
	}

	return _controller.outputs[0][0];

}


int verify_error_state_space(void){
	int i,j;

	for(i=0; i<nStates;i++){
		for(j=0; j<1;j++){
	  	new_state[i][j]= (_controller.states[i][j]);
		}
	}

	for(i=0; i<nStates;i++){
		for(j=0; j<1;j++){
	 	 new_stateFWL[i][j]= (_controller.states[i][j]);
		}
	}
	
	overflow_mode = 0;
	
	fxp_t x[K_SIZE];

	fxp_t min_fxp = fxp_double_to_fxp(impl.min);
	fxp_t max_fxp = fxp_double_to_fxp(impl.max);

	double nondet_constant_input = nondet_double();

	__DSVERIFIER_assume(nondet_constant_input >= min_fxp && nondet_constant_input <= max_fxp);
	for (i = 0; i < K_SIZE; ++i) {
		x[i] = nondet_constant_input;
	}

	double __quant_error;

	if(closed_loop){
		__quant_error = ss_closed_loop_quantization_error() - fxp_ss_closed_loop_quantization_error();
	} else {
	for (i=0; i < K_SIZE; i++)
	{
	  __quant_error = ss_system_quantization_error(x[i]);
	}

	}

	assert(__quant_error < error_limit && __quant_error > ((-1)*error_limit));
        // assert(__quant_error == 0);
	return 0;
}
