/**
 * DSVerifier - Digital Systems Verifier
 *
 * Federal University of Amazonas - UFAM
 *
 * Authors:       Hussama Ismail <hussamaismail@gmail.com>
 *                Iury Bessa     <iury.bessa@gmail.com>
 *                Renato Abreu   <renatobabreu@yahoo.com.br>
 *
 * ------------------------------------------------------
 *
 * delta operator transformation
 *
 * ------------------------------------------------------
*/

#include <stdlib.h>
#include <assert.h>

/** direct form I realization in fixed point */
fxp32_t fxp_direct_form_1(fxp32_t y[], fxp32_t x[], fxp32_t a[], fxp32_t b[], int Na,	int Nb) {
	fxp32_t *a_ptr, *y_ptr, *b_ptr, *x_ptr;
	fxp32_t sum = 0;
	a_ptr = &a[1];
	y_ptr = &y[Na - 1];
	b_ptr = &b[0];
	x_ptr = &x[Nb - 1];
	int i, j;

	for (i = 0; i < Nb; i++) {
		sum = fxp_add(sum, fxp_mult(*b_ptr++, *x_ptr--));
	}

	for (j = 1; j < Na; j++) {
		sum = fxp_sub(sum, fxp_mult(*a_ptr++, *y_ptr--));
	}
	return sum;
}

/** direct form II realization in fixed point */
fxp32_t fxp_direct_form_2(fxp32_t w[], fxp32_t x, fxp32_t a[], fxp32_t b[], int Na,	int Nb) {
	fxp32_t *a_ptr, *b_ptr, *w_ptr;
	fxp32_t sum = 0;
	a_ptr = &a[1];
	b_ptr = &b[0];
	w_ptr = &w[1];
	int k, j;

	for (j = 1; j < Na; j++) {
		w[0] = fxp_sub(w[0], fxp_mult(*a_ptr++, *w_ptr++));
	}
	w[0] = fxp_add(w[0], x); //w[0] += x;
	w_ptr = &w[0];
	for (k = 0; k < Nb; k++) {
		sum = fxp_add(sum, fxp_mult(*b_ptr++, *w_ptr++));
	}
	return sum;
}

/** transposed direct form II realization in fixed point */
fxp32_t fxp_transposed_direct_form_2(fxp32_t w[], fxp32_t x, fxp32_t a[], fxp32_t b[], int Na, int Nb) {
	fxp32_t *a_ptr, *b_ptr;
	fxp32_t yout = 0;
	a_ptr = &a[1];
	b_ptr = &b[0];
	int Nw = Na > Nb ? Na : Nb;
	yout = fxp_add(fxp_mult(*b_ptr++, x), w[0]);
	int j;

	for (j = 0; j < Nw - 1; j++) {
		w[j] = w[j + 1];
		if (j < Na - 1) {
			w[j] = fxp_sub(w[j], fxp_mult(*a_ptr++, yout));
		}
		if (j < Nb - 1) {
			w[j] = fxp_add(w[j], fxp_mult(*b_ptr++, x));
		}
	}

	return yout;
}

/** direct form I realization using double precision */
double double_direct_form_1(double y[], double x[], double a[], double b[], int Na, int Nb) {
	double *a_ptr, *y_ptr, *b_ptr, *x_ptr;
	double sum = 0;
	a_ptr = &a[1];
	y_ptr = &y[Na - 1];
	b_ptr = &b[0];
	x_ptr = &x[Nb - 1];
	int i, j;
	for (i = 0; i < Nb; i++) {
		sum += *b_ptr++ * *x_ptr--;
	}
	for (j = 1; j < Na; j++) {
		sum -= *a_ptr++ * *y_ptr--;
	}
	return sum;
}

/** direct form II realization using double precision */
double double_direct_form_2(double w[], double x, double a[], double b[], int Na, int Nb) {
	double *a_ptr, *b_ptr, *w_ptr;
	double sum = 0;
	a_ptr = &a[1];
	b_ptr = &b[0];
	w_ptr = &w[1];
	int k, j;
	for (j = 1; j < Na; j++) {
		w[0] -= *a_ptr++ * *w_ptr++;
	}
	w[0] += x;
	w_ptr = &w[0];
	for (k = 0; k < Nb; k++) {
		sum += *b_ptr++ * *w_ptr++;
	}
	return sum;
}

/** transposed direct form II realization using double precision */
double double_transposed_direct_form_2(double w[], double x, double a[], double b[], int Na, int Nb) {
	double *a_ptr, *b_ptr;
	double yout = 0;
	a_ptr = &a[1];
	b_ptr = &b[0];
	int Nw = Na > Nb ? Na : Nb;
	yout = (*b_ptr++ * x) + w[0];
	int j;
	for (j = 0; j < Nw - 1; j++) {
		w[j] = w[j + 1];
		if (j < Na - 1) {
			w[j] -= *a_ptr++ * yout;
		}
		if (j < Nb - 1) {
			w[j] += *b_ptr++ * x;
		}
	}
	return yout;
}

/** direct form I realization using float precision */
float float_direct_form_1(float y[], float x[], float a[], float b[], int Na, int Nb) {
	float *a_ptr, *y_ptr, *b_ptr, *x_ptr;
	float sum = 0;
	a_ptr = &a[1];
	y_ptr = &y[Na - 1];
	b_ptr = &b[0];
	x_ptr = &x[Nb - 1];
	int i, j;

	for (i = 0; i < Nb; i++) {
		sum += *b_ptr++ * *x_ptr--;
	}

	for (j = 1; j < Na; j++) {
		sum -= *a_ptr++ * *y_ptr--;
	}
	return sum;
}

/** direct form II realization using float precision */
float float_direct_form_2(float w[], float x, float a[], float b[], int Na, int Nb) {
	float *a_ptr, *b_ptr, *w_ptr;
	float sum = 0;
	a_ptr = &a[1];
	b_ptr = &b[0];
	w_ptr = &w[1];
	int k, j;

	for (j = 1; j < Na; j++) {
		w[0] -= *a_ptr++ * *w_ptr++;
	}
	w[0] += x;
	w_ptr = &w[0];
	for (k = 0; k < Nb; k++) {
		sum += *b_ptr++ * *w_ptr++;
	}
	return sum;
}

/** transposed direct form II realization using float precision */
float float_transposed_direct_form_2(float w[], float x, float a[], float b[], int Na, int Nb) {
	float *a_ptr, *b_ptr;
	float yout = 0;
	a_ptr = &a[1];
	b_ptr = &b[0];
	int Nw = Na > Nb ? Na : Nb;
	yout = (*b_ptr++ * x) + w[0];
	int j;
	for (j = 0; j < Nw - 1; j++) {
		w[j] = w[j + 1];
		if (j < Na - 1) {
			w[j] -= *a_ptr++ * yout;
		}
		if (j < Nb - 1) {
			w[j] += *b_ptr++ * x;
		}
	}
	return yout;
}
