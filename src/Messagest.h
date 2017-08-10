/**
 * \file Messagest.h
 *
 * \brief Print standard messages about DSVerifier.
 *
 * Authors: Lennon C. Chaves <lennon.correach@gmail.com>
 * 			    Lucas C. Cordeiro <lucasccordeiro@gmail.com>
 *
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.md', which is part of this source code package.
 */

#ifndef DSVERIFIER_MESSAGEST_H
#define DSVERIFIER_MESSAGEST_H

#include "version.h"

class Messagest
{
 public:
  void help();

  void cplus_print_fxp_array_elements(const char * name,
                                      fxp_t * v,
                                      int n);
  void cplus_print_array_elements(const char * name,
                                  double * v,
                                  int n);
  void cplus_print_array_elements_ignoring_empty(const char * name,
                                                 double * v,
                                                 int n);
  void show_required_argument_message(std::string parameter);
  void show_underflow_message();
  void show_delta_not_representable();
  void show_verification_error();
  void show_verification_successful();
  void show_verification_failed();
};

#endif // DSVERIFIER_MESSAGEST_H
