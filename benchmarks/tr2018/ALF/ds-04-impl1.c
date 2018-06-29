#include <dsverifier.h>

digital_system ds = {
	.b = { 135.0, -260.0, 125.0 },
	.b_size = 3,
	.a = { 1.0, -1.0, 0.0 },
	.a_size = 3,
	.sample_time = 0.02
};
	
implementation impl = {
	.int_bits = 8,
	.frac_bits = 8,
	.max = 1.0,
	.min = -1.0
};	
