#include <dsverifier.h>

digital_system controller = {
        .b = { 0, 0.0625 },
        .b_uncertainty = { 0.0, 0.0 },
        .b_size = 2,
        .a = { 0.517578125, -0.4990234375 },
        .a_uncertainty = { 0.0, 0.0 },
        .a_size = 2,
        .sample_time = 2,
};

implementation impl = {
        .int_bits = 3,
        .frac_bits = 7,
        .max = 1.0,
        .min = -1.0,
        .scale = 1,
};

digital_system plant = {
        .b = { 0.0200f, -3.8303E-176 },
        .b_uncertainty = { 0.0, 0.0 },
        .b_size = 2,
        .a = { 1.0f, -4.6764E-166 },
        .a_uncertainty = { 0.0, 0.0 },
        .a_size = 2,
};

