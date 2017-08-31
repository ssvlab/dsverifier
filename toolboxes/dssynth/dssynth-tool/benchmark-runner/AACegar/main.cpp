#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "DynamicalSystem.h"
#include "ProgramOptions.h"

int main(int argc, char *argv[])
{
  functions<mpfr::mpreal>::setDefaultPrec(256);
  abstract::ProgramOptions progOptions(argc, argv);
  progOptions.process();
  return 0;
}
