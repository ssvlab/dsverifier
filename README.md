[![Build Status][build_img]][travis]

# DSVerifier (Digital-Systems Verifier) 

DSVerifier is a verification tool for digital systems.
In particular, DSVerifier employs bounded model checking (BMC) techniques
based on satisfiability modulo theories (SMT) and boolean satisfiability (SAT),
which allows engineers to verify the occurrence of design errors,
due to the finite word-length (FWL) effects employed in fixed-point digital filters
and controllers.

Supported properties:

###### For Digital-Systems:
  * Overflow
  * Limit Cycle
  * Timing
  * Error
  * Stability
  * Minimum Phase
  * Settling time
  * Overshoot

###### For Digital-Systems in Closed-loop:
  * Stability
  * Limit Cycle
  * Quantization Error
  * Settling time
  * Overshoot

## Configuration

(1) Before using DSVerifier, we need to configure an environment variable 
called DSVERIFIER_HOME. You should add it to your .bashrc file as follows:

     export DSVERIFIER_HOME='path to dsverifier folder'

After that, you should save it and use the following command:

     $ source .bashrc

(2) You should download the (desired) version of CBMC or ESBMC
executables for DSVerifier. This package contains CBMC v5.6 and 
ESBMC v4.0. Please, add them to $DSVERIFIER_HOME/model-checker

CBMC and ESBMC recommended versions for DSVerifier v2.0.3 are:

* CBMC v5.6: http://www.cprover.org/cbmc/download/cbmc-5-6-linux-64.tgz
* ESBMC v4.1.0: http://esbmc.org/binaries/esbmc-v4.1.0-linux-static-64.tgz

(3) You need to install the Eigen library (e.g., eigen3, eigen3-static, 
and eigen3-devel depending on your distribution).

(4) Make sure that you have installed GCC version 5.4 (or higher).

## Command line mode

###### Verification of Digital-Systems

  It is necessary to generate a verification file using the following format:

  ```c
  #include <dsverifier.h>

  digital_system ds = {
    .a = { 1.0, 1.068, 0.1239 },
    .a_size = 3,
    .b = { 2.813, -0.0163, -1.872 },
    .b_size = 3
  };

  implementation impl = {
    .int_bits = 4,
    .frac_bits = 10,
    .min = -5.0,
    .max = 5.0
  };
  ```

  Execution:

      ./dsverifier ds.c --realization <r> --property <j> --x-size <k>

  where *ds.c* is the digital-system specification file, *r* is the chosen
  realization, *j* is the property to be verified, and *k* is the verification
  bound.

* Properties: OVERFLOW, LIMIT_CYCLE, ZERO_INPUT_LIMIT_CYCLE, TIMING, ERROR,
              STABILITY, and MINIMUM_PHASE.
* Realizations: DFI, DFII, TDFII, DDFI, DDFII, and TDDFII.

###### Verification of Digital-Systems in Closed-loop

  It is necessary to generate a verification file using the following format:

  ```c
  #include <dsverifier.h>

  digital_system plant = {
    .b = { 0.00018604, 0.000172886972 },
    .b_uncertainty = { 5, 5 },
    .b_size = 2,
    .a = { 1.0,-1.7989, 0.80248974 },
    .a_uncertainty = { 5, 5, 5 },
    .a_size = 3
  };

  digital_system controller = {
    .b = { 1.422, -1.1360358, -1.41689538972, 1.14114041028 },
    .b_size = 4,
    .a = { 1.0, -1.0307, -0.861428, 0.892128 },
    .a_size = 4
  };

  implementation impl = {
    .int_bits = 16,
    .frac_bits = 12,
    .min = -1.0,
    .max = 1.0,
    .max_error = 5
  };
  ```

  Execution:

      ./dsverifier <file> --connection-mode <cm> --property <j>
                          --realization <r> --x-size <k> --bmc CBMC

  where *file* is the digital-system specification file, *cm* is the chosen
  connection-mode, *j* is the property to be verified, *r* is the desired
  realization, and *k* is the verification bound.

* Connection mode: SERIES and FEEDBACK

* Properties: STABILITY_CLOSED_LOOP, LIMIT_CYCLE_CLOSED_LOOP,
              and QUANTIZATION_ERROR_CLOSED_LOOP

* Realizations: DFI, DFII, and TDFII.

## Graphical User Interface

To use DSVerifier GUI, it is necessary to have at least Java Runtime Environment
Version 8.0 Update 40 (jre1.8.0_40)

Execution:

     java -jar dsverifier-gui.jar

[build_img]: https://travis-ci.org/ssvlab/dsverifier.svg?branch=master
[travis]: https://travis-ci.org/ssvlab/dsverifier
