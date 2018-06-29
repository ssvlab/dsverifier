#include <dsverifier.h>

digital_system ds = {
	.b = { 1.5, -0.5 },
	.b_size = 2,
	.a = { 1.0, 0.0 },
	.a_size = 2,
	.sample_time = 0.02
};

implementation impl = {
	.int_bits = 2,
	.frac_bits = 14,
	.max = 1.0,
	.min = -1.0
};
