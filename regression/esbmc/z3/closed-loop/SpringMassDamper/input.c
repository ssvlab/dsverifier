#include <dsverifier.h>

digital_system plant = {
   .b = { 5.000E-5, 5.000E-5 },     /* sampled plant numerator coefficients */
   .b_uncertainty = { 0, 0 },         /* uncertainty numerator coefficients */
   .b_size = 2,                       /* number of coefficients in numerator */
   .a = { 1.000, -2.000, 1.0001 }, /* sampled plant denominator coefficients */
   .a_uncertainty = { 0, 0, 0 },      /* uncertainty numerator coefficients */
   .a_size = 3                        /* number of coefficients in denominator */
};

digital_system controller = {            
   .b = {1.280000E3, -1.354240E3, 0.074879E3},    /* digital control numerator coefficients */
   .b_size = 3,                       /* number of coefficients in numerator */   
   .a = {1.0000, -1.4990, 0.4995},            /* digital control denominator coefficients */
   .a_size = 3                        /* number of coefficients in denominator */   
};

implementation impl = {
    .int_bits = 3,                      /* integer part of the fxp representation (for controller) */
    .frac_bits = 8,                   /* precision part of the fxp representation (for controller) */
 };



