#include <dsverifier.h>

digital_system controller = { 
	.b = {  0 , -0.96484 , 0.9834 },
	.b_uncertainty = {  0 , 0 , 0 },
	.b_size =  3,
	.a = {  0.88965 , -0.875 , 0 },
	.a_uncertainty = {  0 , 0 , 0 },
	.a_size =  3,
	.sample_time = 2
};

implementation impl = { 
	.int_bits =  2,
	.frac_bits =   6,
	.max =  1.000000,
	.min =  -1.000000
	};

digital_system plant = { 
	.b = {  0 , 0.2039 , 0.2039 },
	.b_uncertainty = {  0 , 0 , 0 },
	.b_size =  3,
	.a = {  1 , 1.2 , 1 },
	.a_size =  3, 
	.a_uncertainty = {  0 , 0 , 0 }
	};

