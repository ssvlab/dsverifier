#include <dsverifier.h>

digital_system ds = {
	.b = { 2002.0, -4000.0, 1998.0 },
	.b_size = 3,
	.a = { 1.0, 0.0, -1.0 },
	.a_size = 3,
	.sample_time = 0.001
};
	
implementation impl = {
	.int_bits = 10,
	.frac_bits = 6,
	.max = 1.0,
	.min = -1.0
};	
