//flp10ESTHann

#include <dsverifier.h>

digital_system ds = {
    .b = {     0,   0.019091354650918,   0.069086981455607,   0.130898947251178,   0.180913395299018,   0.200018642686557, 0.180913395299018,   0.130898947251178,   0.069086981455607,   0.019091354650918,                   0},
    .b_size = 11,
    .a = {1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    .a_size = 11
};

filter_parameters filter = {
//    .Ap =  0.501187233627272,
    .Ac = 0.501187233627272,
 //   .Ar = 0.501187233627272,
 //   .wp = 0,
    .wc = 0.0041,
 //   .wr = 0.1041,
    .type = 1
};

implementation impl = {
    .int_bits = 5,
    .frac_bits = 10,
    .min = -1.6,
    .max = 1.6,
};


