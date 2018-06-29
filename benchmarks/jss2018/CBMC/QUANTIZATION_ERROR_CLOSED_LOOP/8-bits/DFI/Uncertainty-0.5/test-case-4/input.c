#include <dsverifier.h>

digital_system controller = { 
	.b = {  -0.58054 , 0.91977 , 0.11871 , -0.95193 },
	.b_uncertainty = {  0.005 , 0.005 , 0.005 , 0.005 },
	.b_size =  4,
	.a = {  0.71887 , -0.38751 , -0.41592 , 0.43729 },
	.a_uncertainty = {  0.005 , 0.005 , 0.005 , 0.005 },
	.a_size =  4,
	.sample_time = 1.000000e-02
};

implementation impl = { 
	.max_error =  0.050000,
	.int_bits =  3,
	.frac_bits =   5,
	.max =  1.000000,
	.min =  -1.000000
	};

digital_system plant = { 
	.b = {  0 , -0.01285 , 0.02582 , -0.01293 },
	.b_uncertainty = {  0.005 , 0.005 , 0.005 , 0.005 },
	.b_size =  4,
	.a = {  1 , -2.99 , 2.983 , -0.9929 },
	.a_size =  4, 
	.a_uncertainty = {  0.005 , 0.005 , 0.005 , 0.005 }
	};

