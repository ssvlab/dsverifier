#include <dsverifier.h>

digital_system controller = { 
	.b = {  -4.4366 , 9.177 , -3.6362 , -5.1444 , 5.9167 , -2.2791 , 0.31329 },
	.b_uncertainty = {  0.005 , 0.005 , 0.005 , 0.005 , 0.005 , 0.005 , 0.005 },
	.b_size =  7,
	.a = {  1 , -0.23339 , -1.5195 , 0.73999 , 0.51029 , -0.41403 , 0.073294 },
	.a_uncertainty = {  0.005 , 0.005 , 0.005 , 0.005 , 0.005 , 0.005 , 0.005 },
	.a_size =  7,
	.sample_time = 5.000000e-01
};

implementation impl = { 
	.max_error =  0.050000,
	.int_bits =  8,
	.frac_bits =   8,
	.max =  1.000000,
	.min =  -1.000000
	};

digital_system plant = { 
	.b = {  0 , 0.54869 , -0.88674 },
	.b_uncertainty = {  0.005 , 0.005 , 0.005 },
	.b_size =  3,
	.a = {  1 , -3.3248 , 1.6487 },
	.a_size =  3, 
	.a_uncertainty = {  0.005 , 0.005 , 0.005 }
	};

