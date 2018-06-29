#include <dsverifier.h>

digital_system controller = { 
	.b = {  -2.7056 , 4.9189 , -2.9898 , 0.60746 },
	.b_uncertainty = {  0.015 , 0.015 , 0.015 , 0.015 },
	.b_size =  4,
	.a = {  1 , -0.24695 , -0.80001 , 0.35681 },
	.a_uncertainty = {  0.015 , 0.015 , 0.015 , 0.015 },
	.a_size =  4,
	.sample_time = 5.000000e-01
};

implementation impl = { 
	.int_bits =  8,
	.frac_bits =   8,
	.max =  1.000000,
	.min =  -1.000000
	};

digital_system plant = { 
	.b = {  0 , 0.33528 , -0.55879 },
	.b_uncertainty = {  0.015 , 0.015 , 0.015 },
	.b_size =  3,
	.a = {  1 , -1.8906 , 0.7788 },
	.a_size =  3, 
	.a_uncertainty = {  0.015 , 0.015 , 0.015 }
	};

