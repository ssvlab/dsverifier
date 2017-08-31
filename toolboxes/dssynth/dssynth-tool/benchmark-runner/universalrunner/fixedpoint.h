/*
 * fixedpoint.h

 *
 *  Created on: 18 Jan 2017
 *      Author: elipol
 */

#ifndef FIXEDPOINT_H_
#define FIXEDPOINT_H_
#include "control_types.h"
#include "benchmark.h"

typedef int fxp_t;
typedef double double_t;
typedef struct {
   int int_bits;
   int frac_bits;
} implementation;

const implementation impl = {
 .int_bits =8,//FRAC_BITS,
 .frac_bits =8,// INT_BITS,
};

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


fxp_t fxp_double_to_fxp(double value) {
    fxp_t tmp;
    double ftemp = value * scale_factor[impl.frac_bits];
    tmp = (fxp_t) ftemp;
    return tmp;
}

double_t fxp_to_double(fxp_t fxp) {
  double_t f;
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



#endif /* FIXEDPOINT_H_ */
