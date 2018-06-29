#include <dsverifier.h>

digital_system ds = {
	.b = { 0.0096, -0.009 },
	.b_size = 2,
	.a = { 0.02, 0.0 },
	.a_size = 2,
	.sample_time = 0.02
};
	
implementation impl = {
	.int_bits = 5,
	.frac_bits = 11,
	.max = 1.0,
	.min = -1.0
};	
