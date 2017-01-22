/**
 * DSVerifier - Digital Systems Verifier
 *
 * Federal University of Amazonas - UFAM
 *
 * Authors:       Hussama Ismail <hussamaismail@gmail.com>
 *                Iury Bessa     <iury.bessa@gmail.com>
 *                Renato Abreu   <renatobabreu@yahoo.com.br>
 *                Felipe Monteiro <felipemonteiro@ufam.edu.br>
 *
 * ------------------------------------------------------
 *
 * This file contains the definitions for DSVerifier and
 * its properties verification. Here, we have the available
 * realizations, properties constants, overflow handling,
 * rounding mode, input and output formats, and etc.
 *
 * ------------------------------------------------------
*/

/** model checkers */
#define NONE	0
#define ESBMC	1
#define CBMC	2
#ifndef BMC
	#define BMC	1
#endif

/** required parameters */
#define LIMIT 4

/* realizations (use: -DREALIZATION=DIRECTFORMI) */
#define DIRECTFORMI	1
#define DIRECTFORMII	2
#define TRANSPOSEDDIRECTFORMII	3
#define DELTADIRECTFORMI	4
#define DELTADIRECTFORMII	5
#define DELTATRANSPOSEDDIRECTFORMII	6
#define DIRECTFORMICASCADE	7
#define DIRECTFORMIICASCADE	8
#define TRANSPOSEDDIRECTFORMIICASCADE	9
#define DELTADIRECTFORMICASCADE	10
#define DELTADIRECTFORMIICASCADE	11
#define DELTATRANSPOSEDDIRECTFORMIICASCADE	12

/* nickname for realizations (use: -DREALIZATION=DFI) */
#define DFI	1
#define DFII	2
#define TDFII	3
#define DDFI	4
#define DDFII	5
#define TDDFII	6
#define CDFI	7
#define CDFII	8
#define CTDFII	9
#define CDDFI	10
#define CDDFII	11
#define CTDDFII	12

/* property verification (use: -DPROPERTY=OVERFLOW) */
#define NOTHING	0
#define ZERO_INPUT_LIMIT_CYCLE	1
#define LIMIT_CYCLE	2
#define OVERFLOW	3
#define TIMING_MSP430	4
#define TIMING	5
#define ERROR	6
#define STABILITY	7
#define MINIMUM_PHASE	8
#define STABILITY_CLOSED_LOOP	9
#define LIMIT_CYCLE_CLOSED_LOOP	10
#define QUANTIZATION_ERROR_CLOSED_LOOP	11
#define QUANTIZATION_ERROR	12
#define CONTROLLABILITY	13
#define OBSERVABILITY	14
#define LIMIT_CYCLE_STATE_SPACE	15
#define SAFETY_STATE_SPACE	16
#define FILTER_MAGNITUDE_DET 17
#define FILTER_MAGNITUDE_NON_DET 18

/** Check Required Parameters */
#ifndef PROPERTY
	#define PROPERTY              	          0
#endif
#ifndef REALIZATION
	#define REALIZATION                     	0
#endif
#ifndef X_SIZE
	#define X_SIZE                         		0
#endif
#ifndef K_SIZE
	#define K_SIZE                        	  0
#endif
#ifndef EXPECTED_ERROR
	#define EXPECTED_ERROR                	  -1
#endif

/** processor parameters (DEPRECATED) */
#define CLOCK                               16000000
#define CYCLE                               1 / CLOCK
#ifndef SAMPLERATE
	#define SAMPLERATE                    	  100
#endif
#define DEADLINE                            1 / SAMPLERATE
#define OVERHEAD                            0

/** overflow mode */
#define DETECT_OVERFLOW                     1
#define SATURATE                            2
#define WRAPAROUND                          3
#ifndef OVERFLOW_MODE
	#define OVERFLOW_MODE					1
#endif

/** rounding mode */
#define ROUNDING                            0
#define FLOOR                               1
#define CEIL                                2
#ifndef ROUNDING_MODE
	#define ROUNDING_MODE               	  1
#endif

/** k induction */
#define K_INDUCTION         1
#ifndef K_INDUCTION_MODE
	#define K_INDUCTION_MODE               	  0
#endif

/** overflow, x_size, and round parameters */
int X_SIZE_VALUE = 0;
int overflow_mode = DETECT_OVERFLOW;
int rounding_mode = ROUNDING;

/** connection mode for closed-loop verification */
#define SERIES                              1
#define FEEDBACK                            2
#ifndef CONNECTION_MODE
	#define CONNECTION_MODE               	  0
#endif

/** error mode for digital and closed-loop verification */
#define RELATIVE							1
#define ABSOLUTE							2
#ifndef ERROR_MODE
	#define ERROR_MODE                     	0
#endif


/** device model */
#define MSP430                              1

/* digital system in transfer function structure */
typedef struct {
  double a[100];
  int a_size;
  double b[100];
  int b_size;
  double sample_time;
  double a_uncertainty[100];
  double b_uncertainty[100];
} digital_system;

/** digital system in space state structure */

typedef struct {
    double A[LIMIT][LIMIT];
    double B[LIMIT][LIMIT];
    double C[LIMIT][LIMIT];
    double D[LIMIT][LIMIT];
    double states[LIMIT][LIMIT];
    double outputs[LIMIT][LIMIT];
    double inputs[LIMIT][LIMIT];
    double K[LIMIT][LIMIT];
    unsigned int nStates;
    unsigned int nInputs;
    unsigned int nOutputs;
} digital_system_state_space;

/** implementation structure */
typedef struct {
   int int_bits;
   int frac_bits;
   double max;
   double min;
   int default_realization;
   double delta;
   int scale;
   double max_error;
} implementation;

/** assembly structure */
typedef struct {
	int push;
	int in;
	int sbiw;
	int cli;
	int out;
	int std;
	int ldd;
	int subi;
	int sbci;
	int lsl;
	int rol;
	int add;
	int adc;
	int adiw;
	int rjmp;
	int mov;
	int sbc;
	int ld;
	int rcall;
	int cp;
	int cpc;
	int ldi;
	int brge;
	int pop;
	int ret;
	int st;
	int brlt;
	int cpi;
} instructions;

/** hardware structure */
typedef struct {
   long clock;
   int device;
   double cycle;
   instructions assembly;
} hardware;

typedef struct{
  float Ap, Ar, Ac;
  float wp, wc, wr; 
  int type; 
}filter_parameters;
