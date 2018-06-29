#include <dsverifier.h>

digital_system controller = { 
	.b = {  0 , 0 , 0 , 0 , 0 , 0.5 , -0.96875 , -0.875 , -0.5 },
	.b_uncertainty = {  0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 },
	.b_size =  9,
	.a = {  0.0011902 , 0.00085449 , 0.00015259 , 0.00033569 , 0 , 0 , 0 , 0 , 0 },
	.a_uncertainty = {  0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 },
	.a_size =  9,
	.sample_time = 1.000000e-02
};

implementation impl = { 
	.max_error =  0.050000,
	.int_bits =  13,
	.frac_bits =   3,
	.max =  1.000000,
	.min =  -1.000000
	};

digital_system plant = { 
	.b = {  0 , 0 , 0 , 0 , 0 , 0 , 0 , 0.0001929 , 6.814e-09 },
	.b_uncertainty = {  0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 },
	.b_size =  9,
	.a = {  1 , -0.6921 , 0 , 0 , 0 , 0 , 0 , 0 , 0 },
	.a_size =  9, 
	.a_uncertainty = {  0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 }
	};

