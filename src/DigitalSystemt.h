/**
 * \file DigitalSystemt.h
 *
 * \brief //TODO
 *
 * Authors: Felipe R. Monteiro <rms.felipe@gmail.com>
 *
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.md', which is part of this source code package.
 */

#ifndef SRC_DIGITAL_SYSTEMT_H_
#define SRC_DIGITAL_SYSTEMT_H_

class DigitalSystemt
{
    unsigned int desired_x_size;
    Stringst dsv_strings;

public:
    DigitalSystemt();
    virtual ~DigitalSystemt();
};

#endif /* SRC_DIGITAL_SYSTEMT_H_ */

enum Properties { OVERFLOW, LIMIT_CYCLE, ZERO_INPUT_LIMIT_CYCLE, ERROR,
    TIMING, TIMING_MSP430, STABILITY, STABILITY_CLOSED_LOOP,
    LIMIT_CYCLE_CLOSED_LOOP, QUANTIZATION_ERROR_CLOSED_LOOP,
    MINIMUM_PHASE, QUANTIZATION_ERROR, CONTROLLABILITY, OBSERVABILITY,
    LIMIT_CYCLE_STATE_SPACE, SAFETY_STATE_SPACE, FILTER_PHASE_NON_DET,
    FILTER_MAGNITUDE_NON_DET, FILTER_MAGNITUDE_DET, FILTER_PHASE_DET }; // We need to identify properties for state-space system and transfer functions

enum Rounding { ROUNDING, FLOOR, CEIL }; // all

enum Overflow { DETECT_OVERFLOW, SATURATE, WRAPAROUND }; // all

enum Realizations { DFI, DFII, TDFII, DDFI, DDFII, TDDFII }; // transfer functions

enum BoundedModelCheckers { ESBMC, CBMC }; // all

enum ConnectionMode { SERIES, FEEDBACK }; // transfer function

enum ArithmeticMode { FIXEDBV, FLOATBV }; // all

enum WordlengthMode { BITS16, BITS32, BITS64 }; // all

enum ErrorMode { ABSOLUTE, RELATIVE }; // all

