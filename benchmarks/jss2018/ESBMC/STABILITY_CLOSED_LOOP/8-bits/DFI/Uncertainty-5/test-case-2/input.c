#include <dsverifier.h>

digital_system controller = { 
	.b = {  0 , -0.34668 , 0.015625 },
	.b_uncertainty = {  0.05 , 0.05 , 0.05 },
	.b_size =  3,
	.a = {  0.5 , 0.19922 , 0 },
	.a_uncertainty = {  0.05 , 0.05 , 0.05 },
	.a_size =  3,
	.sample_time = 2
};

implementation impl = { 
	.int_bits =  1,
	.frac_bits =   7,
	.max =  1.000000,
	.min =  -1.000000
	};

digital_system plant = { 
	.b = {  0 , 0.1898 , 0.00018027 },
	.b_uncertainty = {  0.05 , 0.05 , 0.05 },
	.b_size =  3,
	.a = {  1 , -0.9012 , -1.0006e-16 },
	.a_size =  3, 
	.a_uncertainty = {  0.05 , 0.05 , 0.05 }
	};

