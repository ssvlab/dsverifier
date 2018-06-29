#include <dsverifier.h>

digital_system controller = { 
	.b = {  11.9255 , -11.8089 },
	.b_uncertainty = {  0.05 , 0.05 },
	.b_size =  2,
	.a = {  1 , -1.0729 },
	.a_uncertainty = {  0.05 , 0.05 },
	.a_size =  2,
	.sample_time = 1.000000e-02
};

implementation impl = { 
	.int_bits =  8,
	.frac_bits =   8,
	.max =  1.000000,
	.min =  -1.000000
	};

digital_system plant = { 
	.b = {  0 , 0.01 , -0.010101 },
	.b_uncertainty = {  0.05 , 0.05 , 0.05 },
	.b_size =  3,
	.a = {  1 , -2.0103 , 1.0101 },
	.a_size =  3, 
	.a_uncertainty = {  0.05 , 0.05 , 0.05 }
	};

