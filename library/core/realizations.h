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

extern digital_system ds;
extern hardware hw;
extern int generic_timer;

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

/* direct form I realization using double precision and WCET analysis of assembly code generated by MSP430 CCS compiler */
double double_direct_form_1_MSP430(double y[], double x[], double a[], double b[], int Na, int Nb){
	/* timer1 += 40; */
	int timer1 = OVERHEAD;
	double *a_ptr, *y_ptr, *b_ptr, *x_ptr;									/* timer1 += 8;  */
	double sum = 0;															/* timer1 += 4;  */
	a_ptr = &a[1];															/* timer1 += 2;  */
	y_ptr = &y[Na-1];														/* timer1 += 13; */
	b_ptr = &b[0];															/* timer1 += 1;  */
	x_ptr = &x[Nb-1];														/* timer1 += 13; */
	int i, j;																/* timer1 += 1;  */
	timer1 += 91;		/* (40+42+9) */
	for (i = 0; i < Nb; i++){												/* timer1 += 9;  */
		sum += *b_ptr++ * *x_ptr--;											/* timer1 += 35  */
		timer1 += 47;	/* (12+35);  */
	}																		/* timer1 += 12; */
	for (j = 1; j < Na; j++){												/* timer1 += 3;  */
		sum -= *a_ptr++ * *y_ptr--;											/* timer1 += 37; */
		timer1 += 57;	/* (37+20);  */
	}																		/* timer1 += 20; */
	timer1 += 3;		/* (3+7);    */
	assert((double) timer1 * hw.cycle <= ds.sample_time);
	return sum;																/* timer1 += 7;  */
}

/* direct form 2 realization using double precision and WCET analysis of assembly code generated by MSP430 CCS compiler */
double double_direct_form_2_MSP430(double w[], double x, double a[], double b[], int Na, int Nb) {
	/* timer1 += 40; */
	int timer1 = OVERHEAD;
	double *a_ptr, *b_ptr, *w_ptr;											/* timer1 += 7;  */
	double sum = 0;															/* timer1 += 4;  */
	a_ptr = &a[1];															/* timer1 += 7;  */
	b_ptr = &b[0];
	w_ptr = &w[1];															/* timer1 += 2;  */
	int k, j;																/* timer1 += 2;  */
	timer1 += 71;	/* (40+22+9) */
	for (j = 1; j < Na; j++) {												/* timer1 += 9;  */
		w[0] -= *a_ptr++ * *w_ptr++;										/* timer1 += 42; */
		timer1 += 54;	/* (42+12) */
	}																		/* timer1 += 12; */
	w[0] += x;																/* timer1 += 21; */
	w_ptr = &w[0];															/* timer1 += 1;  */
	for (k = 0; k < Nb; k++) {												/* timer1 += 9;  */
		sum += *b_ptr++ * *w_ptr++;											/* timer1 += 34; */
		timer1 += 46;	/* (34+12) */
	}																		/* timer1 += 12; */
	timer1 += 38;	/* (21+1+9+7) */
	assert((double) timer1 * hw.cycle <= ds.sample_time);
	return sum;																/* timer1 += 7;  */
}

/* transposed direct form 2 realization using double precision and WCET analysis of assembly code generated by MSP430 CCS compiler */
double double_transposed_direct_form_2_MSP430(double w[], double x, double a[], double b[], int Na, int Nb) {
	/* timer1 += 40; */
	int timer1 = OVERHEAD;
	double *a_ptr, *b_ptr;													/* timer1 += 6;  */
	double yout = 0;														/* timer1 += 3;  */
	a_ptr = &a[1];															/* timer1 += 7;  */
	b_ptr = &b[0];
	int Nw = Na > Nb ? Na : Nb;												/* timer1 += 10; */
	yout = (*b_ptr++ * x) + w[0];											/* timer1 += 36; */
	int j;
	timer1 += 105;	/* (40+62+3) */
	for (j = 0; j < Nw - 1; j++) {											/* timer1 += 3;  */
		w[j] = w[j + 1];													/* timer1 += 12; */
		if (j < Na - 1) {													/* timer1 += 9;  */
			w[j] -= *a_ptr++ * yout;										/* timer1 += 34; */
			timer1 += 41;	/* (34+7) */
		}																	/* timer1 += 7;  */
		if (j < Nb - 1) {													/* timer1 += 13; */
			w[j] += *b_ptr++ * x;											/* timer1 += 38; */
			timer1 += 38;	/* (38) */
		}
		timer1 += 54;	/* (12+9+13+20) */
	}																		/* timer1 += 20; */
	timer1 += 7;	/* (7) */
	assert((double) timer1 * hw.cycle <= ds.sample_time);
	return yout;															/* timer1 += 7;  */
}

/* direct form I realization using double precision and WCET analysis of assembly code generated by an generic compiler */
double generic_timing_double_direct_form_1(double y[], double x[], double a[], double b[], int Na, int Nb){
	generic_timer += ((1 * hw.assembly.push) + (7 * hw.assembly.mov));
	double *a_ptr, *y_ptr, *b_ptr, *x_ptr;
	double sum = 0;
	a_ptr = &a[1];
	y_ptr = &y[Na-1];
	b_ptr = &b[0];
	x_ptr = &x[Nb-1];
	generic_timer += ((1 * hw.assembly._xor) + (11 * hw.assembly.mov) + (3 * hw.assembly.add) + (2 * hw.assembly.lpm) + (2 * hw.assembly.clt) + (2 * hw.assembly.asr));
	int i, j;
	generic_timer += ((1 * hw.assembly.mov) + (1 * hw.assembly.jmp));
	for (i = 0; i < Nb; i++){
		sum += *b_ptr++ * *x_ptr--;
		generic_timer += ((9 * hw.assembly.mov) + (2 * hw.assembly.lpm) + (1 * hw.assembly.mul) + (2 * hw.assembly.add) + (1 * hw.assembly.cp) + (1 * hw.assembly.jmp));
	}
	generic_timer += ((1 * hw.assembly.mov) + (1 * hw.assembly.jmp));
	for (j = 1; j < Na; j++){
		sum -= *a_ptr++ * *y_ptr--;
		generic_timer += ((10 * hw.assembly.mov) + (2 * hw.assembly.lpm) + (1 * hw.assembly.sub) + (1 * hw.assembly.mul) + (1 * hw.assembly.add) + (1 * hw.assembly.cp) + (1 * hw.assembly.jmp));
	}
	generic_timer += ((1 * hw.assembly.mov) + (1 * hw.assembly.pop));
	return sum;
}

/* direct form 2 realization using double precision and WCET analysis of assembly code generated by an generic compiler */
double generic_timing_double_direct_form_2(double w[], double x, double a[], double b[], int Na, int Nb) {
	generic_timer += ((7 * hw.assembly.mov) + (1 * hw.assembly.push));
	double *a_ptr, *b_ptr, *w_ptr;
	double sum = 0;
	a_ptr = &a[1];
	b_ptr = &b[0];
	w_ptr = &w[1];
	int k, j;
	generic_timer += ((1 * hw.assembly._xor) + (1 * hw.assembly.mov) + (2 * hw.assembly.add));
	generic_timer += ((1 * hw.assembly.mov) + (1 * hw.assembly.jmp));
	for (j = 1; j < Na; j++) {
		w[0] -= *a_ptr++ * *w_ptr++;
		generic_timer += ((12 * hw.assembly.mov) + (1 * hw.assembly.mul) + (1 * hw.assembly.sub) + (2 * hw.assembly.lpm) + (1 * hw.assembly.add) + (1 * hw.assembly.cp) + (1 * hw.assembly.jmp));
	}
	w[0] += x;
	w_ptr = &w[0];
	generic_timer += ((6 * hw.assembly.mov) + (1 * hw.assembly.add));
	generic_timer += ((1 * hw.assembly.mov) + (1 * hw.assembly.jmp));
	for (k = 0; k < Nb; k++) {
		sum += *b_ptr++ * *w_ptr++;
		generic_timer += ((9 * hw.assembly.mov) + (2 * hw.assembly.add) + (1 * hw.assembly.mul) + (2 * hw.assembly.lpm) + (1 * hw.assembly.cp) + (1 * hw.assembly.jmp));
	}
	generic_timer += ((1 * hw.assembly.mov) + (1 * hw.assembly.pop));
	return sum;
}

/* transposed direct form 2 realization using double precision and WCET analysis of assembly code generated by an generic compiler */
double generic_timing_double_transposed_direct_form_2(double w[], double x, double a[], double b[], int Na, int Nb) {
	generic_timer += ((7 * hw.assembly.mov) + (1 * hw.assembly.push));
	double *a_ptr, *b_ptr;
	double yout = 0;
	a_ptr = &a[1];
	b_ptr = &b[0];
	generic_timer += ((6 * hw.assembly.mov) + (1 * hw.assembly.add));
	int Nw = Na > Nb ? Na : Nb;
	yout = (*b_ptr++ * x) + w[0];
	generic_timer += ((10 * hw.assembly.mov) + (1 * hw.assembly.add) + (1 * hw.assembly.mul) + (1 * hw.assembly.cp) + (1 * hw.assembly.lpm));
	int j;
	generic_timer += ((1 * hw.assembly.mov) + (1 * hw.assembly.jmp));
	for (j = 0; j < Nw - 1; j++) {
		w[j] = w[j + 1];
		generic_timer += ((6 * hw.assembly.mov) + (3 * hw.assembly.add) + (2 * hw.assembly.clt) + (2 * hw.assembly.lpm));
		if (j < Na - 1) {
			w[j] -= *a_ptr++ * yout;
		}
		generic_timer += ((11 * hw.assembly.mov) + (2 * hw.assembly.sub) + (2 * hw.assembly.add) + (2 * hw.assembly.clt) + (3 * hw.assembly.lpm) + (1 * hw.assembly.cp) + (1 * hw.assembly.jmp) + (1 * hw.assembly.mul));
		if (j < Nb - 1) {
			w[j] += *b_ptr++ * x;
		}
		generic_timer += ((15 * hw.assembly.mov) + (2 * hw.assembly.sub) + (4 * hw.assembly.add) + (1 * hw.assembly.clt) + (3 * hw.assembly.lpm) + (2 * hw.assembly.cp) + (2 * hw.assembly.jmp) + (1 * hw.assembly.mul));
	}
	generic_timer += ((1 * hw.assembly.mov) + (1 * hw.assembly.pop));
	return yout;
}
