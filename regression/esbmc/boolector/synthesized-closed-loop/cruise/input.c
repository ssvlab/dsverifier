#include <dsverifier.h>

digital_system controller = {
        .b = {  0.00390625, 0.0009765625 },
        .b_uncertainty = { 0.0, 0.0 },
        .b_size = 2,
        .a = { 0.3134765625, -0.0009765625 },
        .a_uncertainty = { 0.0, 0.0 },
        .a_size = 2,
        .sample_time = 0.2
};

implementation impl = {
        .int_bits = 3,
        .frac_bits = 7,
        .max = 1.0,
        .min = -1.0,
        .scale = 1,
};

digital_system plant = {
        .b = { 9.7541E-04 },
        .b_uncertainty = { 0.0 },
        .b_size = 1,
        .a = { 1.0f, -0.9512f },
        .a_uncertainty = { 0.0, 0.0 },
        .a_size = 2,
};

