//hp12E(17)

#include <dsverifier.h>
digital_system ds = {
    .b =  {0.019714212127711,  -0.130150853632434 ,  0.465026879705444 , -1.129140452132243,   2.044181162139295 , -2.878921071462923  ,3.220567842849763 , -2.878921071462923  , 2.044181162139295  ,-1.129140452132243  , 0.465026879705444 , -0.130150853632434   , 0.019714212127711},
    .b_size = 13,
    .a = { 1.000000000000000,  -0.687258403731591,   3.986148977732428,  -1.206465726475966,   5.973061286205805,   0.131177764397140 ,  4.732677656400860,  1.341121599484313 ,  2.449709824949071 ,  0.889980681234946 ,  0.903327424002456 ,  0.182009411230337 ,  0.180472896901802 }, 
    .a_size = 13
};

filter_parameters filter = {
    .Ap = 0.891250938133746, 
    .Ar = 0.891250938133746,
    .wp = 0.4, 
    .wr = 0.39,
    .type = 2
};
implementation impl = {
    .int_bits = 5,
    .frac_bits = 10,
    .min = -1.6,
    .max = 1.6,
};
