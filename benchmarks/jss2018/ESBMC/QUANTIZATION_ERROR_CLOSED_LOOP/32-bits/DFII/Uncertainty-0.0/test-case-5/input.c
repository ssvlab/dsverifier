#include <dsverifier.h>

digital_system controller = { 
	.b = {  0 , -0.00097656 , 0 , 0 },
	.b_uncertainty = {  0 , 0 , 0 , 0 },
	.b_size =  4,
	.a = {  0.76172 , 0 , 0 , 0 },
	.a_uncertainty = {  0 , 0 , 0 , 0 },
	.a_size =  4,
	.sample_time = 2
};

implementation impl = { 
	.max_error =  0.050000,
	.int_bits =  15,
	.frac_bits =   16,
	.max =  1.000000,
	.min =  -1.000000
	};

digital_system plant = { 
	.b = {  0 , 15.1315 , 17.86 , 17.4549 },
	.b_uncertainty = {  0 , 0 , 0 , 0 },
	.b_size =  4,
	.a = {  1 , -2.6207 , 2.3586 , -0.657 },
	.a_size =  4, 
	.a_uncertainty = {  0 , 0 , 0 , 0 }
	};

