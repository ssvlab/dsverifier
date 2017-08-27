#include <dsverifier.h>

digital_system controller = {
        .b = { 2.72, -4.153, 1.896 },
        .b_size = 3,
        .a = { 1.0, -1.844, 0.8496},
        .a_size = 3,
        .sample_time = 0.2
};

implementation impl = {
        .int_bits = 3,
        .frac_bits = 16,
        .delta = 0.25,
        .max = 1.0,
        .min = -1.0,
        .scale = 1
};

digital_system plant = {
        .b = { 0.0264f },
        .b_size = 1,
        .a = { 1.0f, -0.9998f },
        .a_size = 2
};

