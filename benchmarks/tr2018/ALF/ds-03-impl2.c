#include <dsverifier.h>

digital_system ds = {
	.b = { 110.0, -100.0 },
	.b_size = 2,
	.a = { 1.0, 0.0 },
	.a_size = 2,
	.sample_time = 0.02
};
	
implementation impl = {
	.int_bits = 9,
	.frac_bits = 7,
	.max = 1.0,
	.min = -1.0
};	
