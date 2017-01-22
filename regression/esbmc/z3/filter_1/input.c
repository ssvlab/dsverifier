#include <dsverifier.h>

digital_system ds = {
    .b = {0.206572083826148,  0.413144167652296, 0.206572083826148},
    .b_size = 3,
    .a = {1.000000000000000, -0.369527377351241, 0.195815712655833},
    .a_size = 3
};


implementation impl = {
	.int_bits = 2,
	.frac_bits = 5,
	.min = -1.6,
	.max = 1.6,
	.max_error = 0
};


filter_parameters filter = {
    .Ap =  0.8, 
    .Ac =  0.707945784384138, 
    .Ar = 0.5,
    .wp = 0.3, 
    .wc = 0.4, 
    .wr = 0.5,
    .type = LOW_PASS
};