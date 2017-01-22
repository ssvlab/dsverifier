#include <dsverifier.h>

          digital_system ds = {
            .b = {4.244336814021699e-05, 8.488673628043397e-05, 4.244336814021699e-05},
            .b_size = 3,
            .a = { 1.000000000000000, -1.981488509144574, 9.816582826171342e-01},
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