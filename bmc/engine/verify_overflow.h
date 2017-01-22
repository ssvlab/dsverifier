/**
 * DSVerifier - Digital Systems Verifier (Overflow)
 *
 * Federal University of Amazonas - UFAM
 *
 * Authors:       Renato Abreu   <renatobabreu@yahoo.com.br>
 *		 		  Iury Bessa     <iury.bessa@gmail.com>
 *                Hussama Ismail <hussamaismail@gmail.com>
 *
 * ------------------------------------------------------
 *
 * For UNCERTAINTY: Not supported yet
 *
 * ------------------------------------------------------
*/

int nondet_int();
float nondet_float();

extern digital_system ds;
extern implementation impl;

int verify_overflow(void) {

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
		double da_cascade[100];F
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

	fxp_t min_fxp = fxp_double_to_fxp(impl.min);
	fxp_t max_fxp = fxp_double_to_fxp(impl.max);

	fxp_t y[X_SIZE_VALUE];
	fxp_t x[X_SIZE_VALUE];

	/* prepare inputs with the possibles values (min ~ max) */
	int i;
	for (i = 0; i < X_SIZE_VALUE; ++i) {
		y[i] = 0;
		x[i] = nondet_int();
		__DSVERIFIER_assume(x[i] >= min_fxp && x[i] <= max_fxp);
	}

	int Nw = 0;
	#if ((REALIZATION == CDFI) || (REALIZATION == CDFII) || (REALIZATION == CTDFII) || (REALIZATION == CDDFII) || (REALIZATION == CDDFII) || (REALIZATION == CTDDFII))
		Nw = a_cascade_size > b_cascade_size ? a_cascade_size : b_cascade_size;
	#else
		Nw = ds.a_size > ds.b_size ? ds.a_size : ds.b_size;
	#endif

	fxp_t yaux[ds.a_size];
	fxp_t xaux[ds.b_size];
	fxp_t waux[Nw];

	for (i = 0; i < ds.a_size; ++i) {
		yaux[i] = 0;
	}
	for (i = 0; i < ds.b_size; ++i) {
		xaux[i] = 0;
	}
	for (i = 0; i < Nw; ++i) {
		waux[i] = 0;
	}

	fxp_t xk, temp;
	fxp_t *aptr, *bptr, *xptr, *yptr, *wptr;

	int j;
	for (i = 0; i < X_SIZE_VALUE; ++i) {

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
		#if ((REALIZATION == TDFII) || (REALIZATION ==TDDFII))
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
	/* enable detect overflow (in outputs) */
	overflow_mode = DETECT_OVERFLOW;
	fxp_verify_overflow_array(y, X_SIZE_VALUE);

	return 0;
}
