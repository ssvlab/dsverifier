//lp12EST(6)

#include <dsverifier.h>

digital_system ds = {
            .b =  { 2.026075069571502e-04     ,     0 ,   -1.098484710840275e-05 ,        0  ,   2.026075069571502e-04},
            .b_size = 5,
            .a =  {1.000000000000000e+00 ,   -7.147060721024445e-16  ,   1.970942734098978e+00 ,   -6.435824095873954e-16 ,    9.713589339600007e-01},
            .a_size = 5
         };

implementation impl = {
         .int_bits = 5,
         .frac_bits = 10,
         .min = -1.6,
         .max = 1.6,
};


filter_parameters filter = {
        .Ap = 0.000100000001, 
        .Ar = 0.000100000001,

         .w1r = 0.3, 
         .w2r = 0.7,

         .w1p = 0.31,
         .w2p = 0.69,

        // .1st_wr = 0.29,
        // .2nd_wr = 0.71,

         .type = 3
};