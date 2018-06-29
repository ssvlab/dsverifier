#include <dsverifier.h>

digital_system controller = { 
	.b = {  0 , 0.0625 },
	.b_uncertainty = {  0.05 , 0.05 },
	.b_size =  2,
	.a = {  0.51758 , -0.49902 },
	.a_uncertainty = {  0.05 , 0.05 },
	.a_size =  2,
	.sample_time = 2
};

implementation impl = { 
	.int_bits =  2,
	.frac_bits =   6,
	.max =  1.000000,
	.min =  -1.000000
	};

digital_system plant = { 
	.b = {  0.02 , -3.8303e-176 },
	.b_uncertainty = {  0.05 , 0.05 },
	.b_size =  2,
	.a = {  1 , -4.6764e-166 },
	.a_size =  2, 
	.a_uncertainty = {  0.05 , 0.05 }
	};

