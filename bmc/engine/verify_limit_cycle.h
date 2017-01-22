/**
 * DSVerifier - Digital Systems Verifier (Limit Cycle)
 *
 * Federal University of Amazonas - UFAM
 *
 * Authors:       Hussama Ismail <hussamaismail@gmail.com>
 *
 * ------------------------------------------------------
 *
 * ------------------------------------------------------
*/

extern digital_system ds;
extern implementation impl;
extern digital_system_state_space _controller;

extern int nStates;
extern int nInputs;
extern int nOutputs;

int verify_limit_cycle_state_space(void){

	/* setting up variables */
	double stateMatrix[LIMIT][LIMIT];
	double outputMatrix[LIMIT][LIMIT];
	double arrayLimitCycle[LIMIT];

	double result1[LIMIT][LIMIT];
	double result2[LIMIT][LIMIT];

	int i, j, k;

	/* initializing variables */
	for(i=0; i<LIMIT;i++){
		for(j=0; j<LIMIT;j++){
			result1[i][j]=0;
			result2[i][j]=0;
			stateMatrix[i][j]=0;
			outputMatrix[i][j]=0;
		}
	}

	/* first system iteration */
	double_matrix_multiplication(nOutputs,nStates,nStates,1,_controller.C,_controller.states,result1);
	double_matrix_multiplication(nOutputs,nInputs,nInputs,1,_controller.D,_controller.inputs,result2);

	double_add_matrix(nOutputs,
			1,
			result1,
			result2,
			_controller.outputs);

	k = 0;

	/* remaining system iterations */
	for (i = 1; i < K_SIZE; i++) {
		double_matrix_multiplication(nStates,nStates,nStates,1,_controller.A,_controller.states,result1);
		double_matrix_multiplication(nStates,nInputs,nInputs,1,_controller.B,_controller.inputs,result2);

		double_add_matrix(nStates,
				1,
				result1,
				result2,
				_controller.states);

		double_matrix_multiplication(nOutputs,nStates,nStates,1,_controller.C,_controller.states,result1);
		double_matrix_multiplication(nOutputs,nInputs,nInputs,1,_controller.D,_controller.inputs,result2);

		double_add_matrix(nOutputs,
				1,
				result1,
				result2,
				_controller.outputs);

		/* adding states and outputs in matrices */
		int l;
		for(l = 0; l < nStates; l++){
			stateMatrix[l][k] = _controller.states[l][0];
		}
		for(l = 0; l < nOutputs; l++){
			stateMatrix[l][k] = _controller.outputs[l][0];
		}
		k++;
	}

	printf("#matrix STATES -------------------------------");
	print_matrix(stateMatrix,nStates,K_SIZE); //DEBUG

	printf("#matrix OUTPUTS -------------------------------");
	print_matrix(outputMatrix,nOutputs,K_SIZE); //DEBUG
	assert(0);
	/* checking limit cycle for states */
	for(i=0; i<nStates;i++){
		for(j=0; j<K_SIZE;j++){
			arrayLimitCycle[j] = stateMatrix[i][j];
		}
		double_check_persistent_limit_cycle(arrayLimitCycle,K_SIZE);
	}

	/* checking limit cycle for outputs */
	for(i=0; i<nOutputs;i++){
		for(j=0; j<K_SIZE;j++){
			arrayLimitCycle[j] = outputMatrix[i][j];
		}
		double_check_persistent_limit_cycle(arrayLimitCycle,K_SIZE);
	}

	assert(0);
}

int verify_limit_cycle(void){

	overflow_mode = WRAPAROUND;

	int i;
	int Set_xsize_at_least_two_times_Na = 2 * ds.a_size;
	printf("X_SIZE must be at least 2 * ds.a_size");
	__DSVERIFIER_assert(X_SIZE_VALUE >= Set_xsize_at_least_two_times_Na);

	/* check the realization */
	#if	((REALIZATION == DFI) || (REALIZATION == DFII) || (REALIZATION == TDFII))
		fxp_t a_fxp[ds.a_size];
		fxp_t b_fxp[ds.b_size];
		/* quantize the denominator using fxp */
		fxp_double_to_fxp_array(ds.a, a_fxp, ds.a_size);
		/* quantize the numerator using fxp */
		fxp_double_to_fxp_array(ds.b, b_fxp, ds.b_size);
	#elif ((REALIZATION == DDFI)||(REALIZATION == DDFII)||(REALIZATION == TDDFII))
		double da[ds.a_size];
		double db[ds.b_size];
		get_delta_transfer_function_with_base(ds.b, db, ds.b_size,ds.a, da, ds.a_size, impl.delta);
		fxp_t a_fxp[ds.a_size];
		fxp_t b_fxp[ds.b_size];
		/* quantize delta denominators using fxp */
		fxp_double_to_fxp_array(da, a_fxp, ds.a_size);
		/* quantize delta numerator using fxp */
		fxp_double_to_fxp_array(db, b_fxp, ds.b_size);
	#elif ((REALIZATION == CDFI) || (REALIZATION == CDFII) || (REALIZATION == CTDFII))
		double a_cascade[100];
		int a_cascade_size;
		double b_cascade[100];
		int b_cascade_size;
		/* generate cascade realization for digital system */
		__DSVERIFIER_generate_cascade_controllers(&ds, a_cascade, a_cascade_size, b_cascade, b_cascade_size);
		fxp_t ac_fxp[100];
		fxp_t bc_fxp[100];
		/* quantize cascade denominators */
		fxp_double_to_fxp_array(a_cascade, ac_fxp, a_cascade_size);
		/* quantize cascade numerators */
		fxp_double_to_fxp_array(b_cascade, bc_fxp, b_cascade_size);
	#elif ((REALIZATION == CDDFI) || (REALIZATION == CDDFII) || (REALIZATION == CTDDFII))
		double da_cascade[100];
		int a_cascade_size;
		double db_cascade[100];
		int b_cascade_size;
		/* generate cascade realization with delta for the digital system */
		__DSVERIFIER_generate_cascade_delta_controllers(&ds, da_cascade, a_cascade_size, db_cascade, b_cascade_size, impl.delta);
		fxp_t ac_fxp[100];
		fxp_t bc_fxp[100];
		/* quantize cascade denominators */
		fxp_double_to_fxp_array(da_cascade, ac_fxp, a_cascade_size);
		/* quantize cascade numerators */
		fxp_double_to_fxp_array(db_cascade, bc_fxp, b_cascade_size);
	#endif

	fxp_t y[X_SIZE_VALUE];
	fxp_t x[X_SIZE_VALUE];

	fxp_t min_fxp = fxp_double_to_fxp(impl.min);
	fxp_t max_fxp = fxp_double_to_fxp(impl.max);

	/* prepare inputs (all possibles values in dynamical range) */
	fxp_t xaux[ds.b_size];
	int nondet_constant_input = nondet_int();
	__DSVERIFIER_assume(nondet_constant_input >= min_fxp && nondet_constant_input <= max_fxp);
	for (i = 0; i < X_SIZE_VALUE; ++i) {
		x[i] = nondet_constant_input;
		y[i] = 0;
	}
	for (i = 0; i < ds.b_size; ++i) {
		xaux[i] = nondet_constant_input;
	}

	int Nw = 0;
	#if ((REALIZATION == CDFI) || (REALIZATION == CDFII) || (REALIZATION == CTDFII) || (REALIZATION == CDDFII) || (REALIZATION == CDDFII) || (REALIZATION == CTDDFII))
		Nw = a_cascade_size > b_cascade_size ? a_cascade_size : b_cascade_size;
	#else
		Nw = ds.a_size > ds.b_size ? ds.a_size : ds.b_size;
	#endif

	fxp_t yaux[ds.a_size];
	fxp_t y0[ds.a_size];

	fxp_t waux[Nw];
	fxp_t w0[Nw];

	#if (REALIZATION == DFI || REALIZATION == CDFI || REALIZATION == DDFI || REALIZATION == CDDFI)
		for (i = 0; i < ds.a_size; ++i) {
			yaux[i] = nondet_int();
			__DSVERIFIER_assume(yaux[i] >= min_fxp && yaux[i] <= max_fxp);
			y0[i] = yaux[i];
		}
	#else
		for (i = 0; i < Nw; ++i) {
			waux[i] = nondet_int();
			__DSVERIFIER_assume(waux[i] >= min_fxp && waux[i] <= max_fxp);
			w0[i] = waux[i];
		}
	#endif

	fxp_t xk, temp;
	fxp_t *aptr, *bptr, *xptr, *yptr, *wptr;

	int j;
	for(i=0; i<X_SIZE_VALUE; ++i){

		/* direct form I realization */
		#if ((REALIZATION == DFI) || (REALIZATION == DDFI))
			shiftL(x[i], xaux, ds.b_size);
			y[i] = fxp_direct_form_1(yaux, xaux, a_fxp, b_fxp, ds.a_size, ds.b_size);
			shiftL(y[i], yaux, ds.a_size);
		#endif

		/* direct form II realization */
		#if ((REALIZATION == DFII) || (REALIZATION == DDFII))
			shiftR(0, waux, Nw);
			y[i] = fxp_direct_form_2(waux, x[i], a_fxp, b_fxp, ds.a_size, ds.b_size);
		#endif

		/* transposed direct form II realization */
		#if ((REALIZATION == TDFII) || (REALIZATION == TDDFII))
			y[i] = fxp_transposed_direct_form_2(waux, x[i], a_fxp, b_fxp, ds.a_size, ds.b_size);
		#endif

		/* cascade direct form I realization (or delta cascade) */
		#if ((REALIZATION == CDFI) || (REALIZATION == CDDFI))
			assert((Nw % 3) == 0 && a_cascade_size == b_cascade_size);
			xk = x[i];
			for (j = 0; j < a_cascade_size; j += 3) {
				aptr = &ac_fxp[j];
				bptr = &bc_fxp[j];
				xptr = &xaux[j];
				yptr = &yaux[j];
				shiftL(xk, xptr, 3);
				y[i] = fxp_direct_form_1(yptr, xptr, aptr, bptr, 3, 3);
				shiftL(y[i], yptr, 3);
				xk = y[i];
			}
		#endif

		/* cascade direct form II realization (or delta cascade) */
		#if ((REALIZATION == CDFII) || (REALIZATION == CDDFII))
			assert((Nw % 3) == 0 && a_cascade_size == b_cascade_size);
			for (j = 0; j < a_cascade_size; j += 3) {
				aptr = &ac_fxp[j];
				bptr = &bc_fxp[j];
				wptr = &waux[j];
				shiftR(0, wptr, 3);
				y[i] = fxp_direct_form_2(wptr, xk, aptr, bptr, 3, 3);
				xk = y[i];
			}
		#endif

		/* cascade transposed direct form II realization (or delta cascade) */
		#if ((REALIZATION == CTDFII) || (REALIZATION == CTDDFII))
			assert((Nw % 3) == 0 && a_cascade_size == b_cascade_size);
			xk = x[i];
			for (j = 0; j < a_cascade_size; j += 3) {
				aptr = &ac_fxp[j];
				bptr = &bc_fxp[j];
				wptr = &waux[j];
				y[i] = fxp_transposed_direct_form_2(wptr, xk, aptr, bptr, 3, 3);
				xk = y[i];
			}
		#endif
	}

	/* check oscillations in produced output */
	fxp_check_persistent_limit_cycle(y, X_SIZE_VALUE);

	return 0;
}
