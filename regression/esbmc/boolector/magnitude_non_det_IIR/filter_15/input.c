//lp4ESTE(15)

#include <dsverifier.h>

digital_system ds = {
    .b =  {9.997654267080493e-05,    -3.974983405582690e-04,5.950509698093377e-04,    -3.974983405582690e-04,9.997654267080493e-05 },
    .b_size = 5,
    .a = {  1.000000000000000,    -3.987375811540735,5.962378493164128,    -3.962627864739390, 9.876251913897993e-01},
    .a_size = 5
};

filter_parameters filter = {
    .Ap = 0.891250938133746, 
    .Ar = 0.891250938133746,
    .wp = 0.0041, 
    .wr = 0.01,
    .type = 1
};
implementation impl = {
    .int_bits = 5,
    .frac_bits = 10,
    .min = -1.6,
    .max = 1.6,
};
