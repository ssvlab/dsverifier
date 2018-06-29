#include <dsverifier.h>

digital_system controller = { 
	.b = {  0.0039062 , 0.00097656 },
	.b_uncertainty = {  0.015 , 0.015 },
	.b_size =  2,
	.a = {  0.31348 , -0.00097656 },
	.a_uncertainty = {  0.015 , 0.015 },
	.a_size =  2,
	.sample_time = 2.000000e-01
};

implementation impl = { 
	.max_error =  0.050000,
	.int_bits =  15,
	.frac_bits =   16,
	.max =  1.000000,
	.min =  -1.000000
	};

digital_system plant = { 
	.b = {  0 , 0.00097541 },
	.b_uncertainty = {  0.015 , 0.015 },
	.b_size =  2,
	.a = {  1 , -0.9512 },
	.a_size =  2, 
	.a_uncertainty = {  0.015 , 0.015 }
	};

