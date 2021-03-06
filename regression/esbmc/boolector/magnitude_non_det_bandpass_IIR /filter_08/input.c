//lp12EST(6)

#include <dsverifier.h>

digital_system ds = {
            .b =  { 9.959664241583719e-04  ,   0 ,  -1.044873334270529e-03 ,  4.336808689942018e-19  ,   2.185605925563302e-03 ,   0 ,   -2.043335730056855e-03  ,  8.673617379884035e-19  ,   2.185605925563303e-03  ,   4.336808689942018e-19 ,   -1.044873334270528e-03 ,     0  ,   9.959664241583719e-04},
            .b_size = 13,
            .a =  {1.000000000000000e+00 ,   -3.219646771412954e-15 ,    3.927097427770937e+00 ,   -8.354428260304303e-15   ,  6.681136843916320e+00  ,  -8.965050923848139e-15  ,   6.237796647169016e+00  ,  -4.857225732735060e-15   ,  3.353006307889214e+00 ,   -1.249000902703301e-15   ,  9.801693157381465e-01   , -1.179611963664229e-16  ,   1.214164659706075e-01},
            .a_size = 13
         };

implementation impl = {
         .int_bits = 10,
         .frac_bits = 16,
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