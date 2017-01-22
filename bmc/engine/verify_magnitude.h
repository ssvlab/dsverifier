/**
 * DSVerifier - Digital Systems Verifier
 *
 * 				  Federal University of Amazonas - UFAM
 *
 * Authors:       Daniel Mello <dani-dmello@hotmail.com>
 *                
 * ------------------------------------------------------
 *
 * Engine needed for magnitude verification of highpass and lowpass filters
 *
 * ------------------------------------------------------
 */


extern filter_parameters filter;
extern implementation impl;
extern digital_system ds;

#define M_PI     3.14159265358979323846
#define SINE_precision 6
#define LOWPASS 1
#define HIGHPASS 2

/*
 *  Generates magnitude response from transfer function
 */
void resp_mag(double* num, int lnum, double* den, int lden, double* res, int N) {

	double w;
	int m, i;
	double out_numRe[N + 1];
	double out_numIm[N + 1];
	double out_denRe[N + 1];
	double out_denIm[N + 1];
	double old_out_Re;
	double zero_test;
	for (w = 0, i = 0; w <= M_PI; w += M_PI / N, ++i) {
		out_numRe[i] = num[0];
		out_numIm[i] = 0;
		for (m = 1; m < lnum; ++m) {
			old_out_Re = out_numRe[i];
			out_numRe[i] = cosTyl(w, SINE_precision) * out_numRe[i] - sinTyl(w, SINE_precision) * out_numIm[i] + num[m];
			out_numIm[i] = sinTyl(w, SINE_precision) * old_out_Re + cosTyl(w, SINE_precision) * out_numIm[i];
		}
		out_denRe[i] = den[0];
		out_denIm[i] = 0;

		for (m = 1; m < lden; ++m) {
			old_out_Re = out_denRe[i];
			out_denRe[i] = cosTyl(w, SINE_precision) * out_denRe[i] - sinTyl(w, SINE_precision) * out_denIm[i] + den[m];
			out_denIm[i] = sinTyl(w, SINE_precision) * old_out_Re + cosTyl(w, SINE_precision) * out_denIm[i];
		}

		res[i] = sqrt3(out_numRe[i] * out_numRe[i] + out_numIm[i] * out_numIm[i]); 
	    zero_test = sqrt3(out_denRe[i] * out_denRe[i] + out_denIm[i] * out_denIm[i]);
	    __DSVERIFIER_assume(zero_test != 0);
		res[i] = res[i] / zero_test;
	}
}

/*
 * Magnitude verifier 
 */
 int verify_magnitude(void) {

	int freq_response_samples = 100;
	double w;
	double w_incr = 1.0 / freq_response_samples;
  	double res[freq_response_samples+1];
  	int i,j;

  	/* quantize "a" array using fxp */

	fxp_t a_fxp[ds.a_size];
	fxp_double_to_fxp_array(ds.a, a_fxp, ds.a_size);
	double _a[ds.a_size];
	fxp_to_double_array(_a, a_fxp, ds.a_size);

  	/* quantize "b" array using fxp */

	fxp_t b_fxp[ds.b_size];
	fxp_double_to_fxp_array(ds.b, b_fxp, ds.b_size);
	double _b[ds.b_size];
	fxp_to_double_array(_b, b_fxp, ds.b_size);

 /* generates magnitude response of the quantized TF, placing the result in the "res" array*/

  resp_mag(ds.b, ds.b_size, ds.a, ds.a_size, res, freq_response_samples);

	/* generates magnitude response, placing the result in the "res" array*/

	if (filter.type == LOWPASS) { //lowpass
		for (i = 0, w = 0; (w <= 1.0); ++i, w += w_incr) {
			if (w <= filter.wp) {
				__DSVERIFIER_assert_msg(res[i] >= filter.Ap, "|----------------Passband Failure-------------|");
			} else if (w == filter.wc) {
				__DSVERIFIER_assert_msg(res[i] <= filter.Ac, "|-------------Cutoff Frequency Failure--------|");
			} else if ((w >= filter.wr) && (w <= 1)) {
				__DSVERIFIER_assert_msg(res[i] <= filter.Ar, "|----------------Stopband Failure-------------|");
			}
		}
	} else if (filter.type == HIGHPASS) { //highpass
		for (i = 0, w = 0; (w <= 1.0); ++i, w += w_incr) {
			if (w <= filter.wr) {
				__DSVERIFIER_assert_msg(res[i] <= filter.Ar, "|----------------Stopband Failure-------------|");
			} else if (w == filter.wc) {
				__DSVERIFIER_assert_msg(res[i] <= filter.Ac, "|-------------Cutoff Frequency Failure--------|");
			} else if ((w > filter.wp) && (w <= 1)) {
				__DSVERIFIER_assert_msg(res[i] >= filter.Ap, "|----------------Passband Failure-------------|");
			}
		}
	} else {
		__DSVERIFIER_assert(0);	
	}
	return 0;
}

