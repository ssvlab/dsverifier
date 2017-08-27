#include <dsverifier.h>

digital_system controller = {
        .b = {  2.72f, -4.153, 1.896 },
        .b_uncertainty = { 0.0, 0.0, 0.0 },
        .b_size = 3,
        .a = { 1.0f, -1.844, 0.8496 },
        .a_uncertainty = { 0.0, 0.0, 0.0 },
        .a_size = 3,
        .sample_time = 0.2
};

implementation impl = {
        .int_bits = 4,
        .frac_bits = 16,
        .max = 1.0,
        .min = -1.0,
        .scale = 1,
};

digital_system plant = {
        .b = { 0.0264f },
        .b_uncertainty = { 0.0 },
        .b_size = 1,
        .a = { 1.0f, -0.9998f },
        .a_uncertainty = { 0.0, 0.0 },
        .a_size = 2,
};

