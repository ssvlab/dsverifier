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

#ifndef SRC_MESSAGEST_H
#define SRC_MESSAGEST_H

#include "src/Version.h"

class Messagest
{
 public:
  Messagest();
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

#endif /* SRC_DSVERIFIER_MESSAGET_H_ */
