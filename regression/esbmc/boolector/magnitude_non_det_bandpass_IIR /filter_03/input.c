//lp12EST(6)

#include <dsverifier.h>

digital_system ds = {
            .b =  { 2.277266369349034e-13     ,   0 ,   -1.366359821609420e-12    ,    0  ,   3.415899554023551e-12    ,    0   , -4.554532738698068e-12   ,   0   ,  3.415899554023551e-12  ,   0   , -1.366359821609420e-12    ,    0  ,   2.277266369349034e-13},
            .b_size = 13,
            .a =  {1.000000000000000e+00 ,   -6.941885892095623e+00   ,  2.601848694489667e+01  ,  -6.533397865544730e+01   ,  1.210830910551153e+02   , -1.724526841801021e+02  ,   1.930725951150498e+02 ,   -1.707170714838558e+02   ,  1.186581254288614e+02   , -6.338114380460735e+01   ,  2.498676199956173e+01  ,  -6.599520511872571e+00  ,  9.411133395899519e-01},
            .a_size = 13
         };

implementation impl = {
         .int_bits = 5,
         .frac_bits = 10,
         .min = -1.6,
         .max = 1.6,
};


filter_parameters filter = {
         .Ac =  0.707945784384138, 
         .Ap =  0.707945784384138, 
         .Ar =  0.707945784384138, 

         .w1c = 0.3, 
         .w2c = 0.305,

         .w1p = 0.3,
         .w2p = 0.305,

         .w1r = 0.29,
         .w2r = 0.31,

        // .1st_wr = 0.29,
        // .2nd_wr = 0.71,

         .type = 3
};