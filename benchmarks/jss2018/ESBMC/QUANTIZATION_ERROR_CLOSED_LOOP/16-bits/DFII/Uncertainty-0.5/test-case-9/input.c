#include <dsverifier.h>

digital_system controller = { 
	.b = {  0 , -0.022461 , 0 , 0 , 0 },
	.b_uncertainty = {  0.005 , 0.005 , 0.005 , 0.005 , 0.005 },
	.b_size =  5,
	.a = {  0.13867 , 0 , 0 , 0 , 0 },
	.a_uncertainty = {  0.005 , 0.005 , 0.005 , 0.005 , 0.005 },
	.a_size =  5,
	.sample_time = 2
};

implementation impl = { 
	.max_error =  0.050000,
	.int_bits =  8,
	.frac_bits =   8,
	.max =  1.000000,
	.min =  -1.000000
	};

digital_system plant = { 
	.b = {  0 , 0.25 , 0.5 , 0.25 , -4.3341e-07 },
	.b_uncertainty = {  0.005 , 0.005 , 0.005 , 0.005 , 0.005 },
	.b_size =  5,
	.a = {  1 , 5.915e-06 , 1.048e-11 , -4.9349e-63 , 7.0168e-225 },
	.a_size =  5, 
	.a_uncertainty = {  0.005 , 0.005 , 0.005 , 0.005 , 0.005 }
	};

