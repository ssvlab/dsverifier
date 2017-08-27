#include <dsverifier.h>

digital_system controller = {
        .b = { 0.5, -0.96875, -0.875, -0.5},
        .b_uncertainty = { 0.0, 0.0, 0.0, 0.0 },
        .b_size = 4,
        .a = { 0.001190185546875, 0.0008544921875, 0.000152587890625, 0.000335693359375, 0, 0, 0, 0, 0 },
        .a_uncertainty = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 , 0.0 },
        .a_size = 9,
        .sample_time = 0.01,
};

implementation impl = {
        .int_bits = 4,
        .frac_bits = 11,
        .max = 1.0,
        .min = -1.0,
        .scale = 1,
};

digital_system plant = {
        .b = { 0.0001929f, 6.814e-09f },
        .b_uncertainty = { 0.0, 0.0 },
        .b_size = 2,
        .a = { 1.0f, -0.6921f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
        .a_uncertainty = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
        .a_size = 9,
 };
