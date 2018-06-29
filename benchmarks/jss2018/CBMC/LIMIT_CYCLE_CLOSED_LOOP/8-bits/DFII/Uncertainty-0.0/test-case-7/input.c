#include <dsverifier.h>

digital_system controller = { 
	.b = {  0 , 0.83594 , 0.26562 , -0.96875 },
	.b_uncertainty = {  0 , 0 , 0 , 0 },
	.b_size =  4,
	.a = {  0.94531 , 0.90625 , -0.15625 , -0.12305 },
	.a_uncertainty = {  0 , 0 , 0 , 0 },
	.a_size =  4,
	.sample_time = 1.000000e-03
};

implementation impl = { 
	.int_bits =  7,
	.frac_bits =   1,
	.max =  1.000000,
	.min =  -1.000000
	};

digital_system plant = { 
	.b = {  0 , 0.125 , 0.125 },
	.b_uncertainty = {  0 , 0 , 0 },
	.b_size =  3,
	.a = {  1 , -2 , 1 },
	.a_size =  3, 
	.a_uncertainty = {  0 , 0 , 0 }
	};

