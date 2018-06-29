#include <dsverifier.h>

digital_system ds = {
	.b = { 0.93, -0.87 },
	.b_size = 2,
	.a = { 1.0, 1.0 },
	.a_size = 2,
	.sample_time = 0.02
};
	
implementation impl = {
	.int_bits = 4,
	.frac_bits = 12,
	.max = 1.0,
	.min = -1.0
};	
