/**
 * \file Messagest.cpp
 *
 * \brief Print standard messages about DSVerifier.
 *
 * Authors: Lennon C. Chaves <lennon.correach@gmail.com>
 *          Lucas C. Cordeiro <lucasccordeiro@gmail.com>
 *
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.md', which is part of this source code package.
 */

#ifndef SRC_MESSAGEST_CPP
#define SRC_MESSAGEST_CPP

#include "src/Messagest.h"

void Messagest::Messagest()
{
    //TODO
}

void Messagest::help()
{
  std::cout << std::endl;
  std::cout << "* * *           DSVerifier " << DSVERIFIER_VERSION << "          * * *" << std::endl;
  std::cout << "" << std::endl;
  std::cout << "Usage:                       Purpose:" << std::endl;
  std::cout << "" << std::endl;
  std::cout << "dsverifier [-h] [--help]           show help" << std::endl;
  std::cout << "dsverifier file.c ...              source file name" << std::endl;
  std::cout << "" << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "" << std::endl;
  std::cout << "--realization <r>            set the realization for the Digital-System" << std::endl;
  std::cout << "                             (for Digital-Systems: DFI, DFII, TDFII, DDFI, DDFII, and TDDFII)" << std::endl;
  std::cout << "                             (for Digital-Systems in Closed-loop: DFI, DFII, and TDFII)" << std::endl;
  std::cout << "--property <p>               set the property to check in order to find violations" << std::endl;
  std::cout << "                             (for Digital-Systems: OVERFLOW, LIMIT_CYCLE, ZERO_INPUT_LIMIT_CYCLE, ERROR, TIMING, STABILITY, and MINIMUM_PHASE)" << std::endl;
  std::cout << "                             (for Digital-Systems in Closed-loop: STABILITY_CLOSED_LOOP, LIMIT_CYCLE_CLOSED_LOOP, and QUANTIZATION_ERROR_CLOSED_LOOP)" << std::endl;
  std::cout << "--x-size <k>                 set the bound of verification" << std::endl;
  std::cout << "--unlimited-x-size           configure the verifier to employ k-induction verification (for ESBMC only)" << std::endl;
  std::cout << "--rounding-mode <rm>         set the rounding mode (ROUNDING, FLOOR, or CEIL)" << std::endl;
  std::cout << "--error-mode <em>            set the error mode (ABSOLUTE or RELATIVE), default is RELATIVE" << std::endl;
  std::cout << "--overflow-mode <om>         set the overflow mode (DETECT_OVERFLOW, SATURATE, or WRAPAROUND)" << std::endl;
  std::cout << "--connection-mode <cm>       set the connection mode for the closed-loop system (SERIES or FEEDBACK)" << std::endl;
  std::cout << "--arithmetic-mode            set the arithmetic mode (FIXEDBV or FLOATBV)" << std::endl;
  std::cout << "--wordlength <bits>          set the word-length for FLOATBV, integer followed by {16, 32, 64}" << std::endl;
  std::cout << "--bmc <b>                    set the BMC back-end for DSVerifier (ESBMC or CBMC, default is CBMC)" << std::endl;
  std::cout << "--solver <s>                 use the specified solver in BMC back-end (e.g., boolector, z3, yices, cvc4, and minisat)" << std::endl;
  std::cout << "--timeout <t>                configure time limit, integer followed by {s,m,h} (for ESBMC only)" << std::endl;
  std::cout << "--tf2ss                      converts a transfer function representation of a given system to an equivalent state-space representation" << std::endl;
  std::cout << "--show-ce-data               shows initial states, inputs, and outputs extracted from counterexample" << std::endl;
  std::cout << "--preprocess                 preprocess the digital-system file to be verified" << std::endl;
  std::cout << "" << std::endl;
  exit(0);
}

void Messagest::cplus_print_fxp_array_elements(const char * name,
                                               fxp_t * v,
                                               int n)
{
  printf("%s = {", name);

  for(int i=0; i < n; i++)
    printf(" %lld ", v[i]);

  printf("}\n");
}

void Messagest::cplus_print_array_elements(const char * name,
                                           double * v,
                                           int n)
{
  printf("%s = {", name);

  for(int i=0; i < n; i++)
  {
    printf(" %.16f ", v[i]);
  }

  printf("}\n");
}

void Messagest::cplus_print_array_elements_ignoring_empty(const char * name,
                                                          double * v,
                                                          int n)
{
  if(n > 0)
    cplus_print_array_elements(name, v, n);
}

void Messagest::show_required_argument_message(std::string parameter)
{
  std::cerr << parameter << " option requires one argument." << std::endl;
  exit(1);
}

void Messagest::show_underflow_message()
{
  std::cout << "UNDERFLOW detected: An underflow has occurred "
      "after delta transformation"
      << std::endl;
}

void Messagest::show_delta_not_representable()
{
  std::cout << "DSVerifier cannot represent this value in "
      "delta-form using the given precision"
      << std::endl;
}

void Messagest::show_verification_error()
{
  std::cout << std::endl << "VERIFICATION ERROR" << std::endl;
}

void Messagest::show_verification_successful()
{
  std::cout << std::endl << "VERIFICATION SUCCESSFUL" << std::endl;
}

void Messagest::show_verification_failed()
{
  std::cout << std::endl << "VERIFICATION FAILED" << std::endl;
}

#endif /* SRC_MESSAGEST_CPP */
