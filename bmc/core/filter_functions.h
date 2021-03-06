/**
 * DSVerifier - Digital Systems Verifier
 *
 * Federal University of Amazonas - UFAM
 *
 * Authors:       Daniel Mello <dani-dmello@hotmail.com>
 *
 * ------------------------------------------------------------------------
 *
 * Common functions needed for digital filters and frequency domain operations
 *
 * ------------------------------------------------------------------------
 */
#ifndef DSVERIFIER_CORE_FILTER_FUNCTIONS_H
#define DSVERIFIER_CORE_FILTER_FUNCTIONS_H

/**
 * Sine aproximation by taylor series (max 6 terms).
 * Algorithm created aiming to avoid state explosion due to loops and recursion.
 */
double sinTyl(double x, int precision)
{
  double sine;
  double xsquared;
  double aux;

  xsquared = x * x;

  if(precision < 0)
  {
    printf("Warning: Function sinTyl from bmc/core/filter_functions.h: "
        "Precision must be a positive integer. Assuming 0 precision\n");
    precision = 0;
  }

  if(precision >= 0)
  {
    aux = 0;
    sine = aux;
    if(precision >= 1)
    {
      aux = x;
      sine += aux;
      if(precision >= 2)
      {
        aux = aux * xsquared;
        sine -= aux / 6;
        if(precision >= 3)
        {
          aux = aux * xsquared;
          sine += aux / 120;
          if(precision >= 4)
          {
            aux = aux * xsquared;
            sine -= aux / 5040;
            if(precision >= 5)
            {
              aux = aux * xsquared;
              sine += aux / 362880;
              if(precision >= 6)
              {
                aux = aux * xsquared;
                sine -= aux / 39916800;
                if(precision >= 7)
                {
                  aux = aux * xsquared;
                  sine += aux / 6227020800;
                  if(precision >= 8)
                  {
                    printf(
                        "Warning: Function sinTyl "
                            "from bmc/core/filter_functions.h: Precision "
                            "representation exceeded. Assuming maximum "
                            "precision of 7\n");
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  return sine;
}

/**
 * Cosine aproximation by taylor series (max 6 terms).
 * Algorithm created aiming to avoid state explosion due to loops and recursion.
 */
double cosTyl(double x, int precision)
{
  double cosine;
  double xsquared = x * x;
  double aux;

  if(precision < 0)
  {
    printf("Warning: Function cosTyl from bmc/core/filter_functions.h: "
        "Precision must be a positive integer. Assuming 0 precision\n");

    precision = 0;
  }

  if(precision >= 0)
  {
    aux = 0;
    cosine = aux;
    if(precision >= 1)
    {
      aux = 1;
      cosine = 1;
      if(precision >= 2)
      {
        aux = xsquared;
        cosine -= aux / 2;
        if(precision >= 3)
        {
          aux = aux * xsquared;
          cosine += aux / 24;
          if(precision >= 4)
          {
            aux = aux * xsquared;
            cosine -= aux / 720;
            if(precision >= 5)
            {
              aux = aux * xsquared;
              cosine += aux / 40320;
              if(precision >= 6)
              {
                aux = aux * xsquared;
                cosine -= aux / 3628800;
                if(precision >= 7)
                {
                  aux = aux * xsquared;
                  cosine += aux / 479001600;
                  if(precision >= 8)
                  {
                    printf(
                        "Warning: Function sinTyl "
                            "from bmc/core/filter_functions.h: Precision "
                            "representation exceeded. Assuming maximum "
                            "precision of 7\n");
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  return cosine;
}

/**
 * Arctangent aproximation by taylor series (max 4 terms).
 * Algorithm created aiming to avoid state explosion due to loops and recursion.
 */
double atanTyl(double x, int precision)
{
  double atangent;
  double xsquared = x * x;
  double aux;

  if(precision < 0)
  {
    printf("Warning: Function sinTyl from bmc/core/filter_functions.h: "
        "Precision must be a positive integer. Assuming 0 precision\n");
    precision = 0;
  }
  if(precision >= 0)
  {
    aux = 0;
    atangent = aux;
    if(precision >= 1)
    {
      aux = x;
      atangent = aux;
      if(precision >= 2)
      {
        aux = xsquared;
        atangent -= aux / 3;
        if(precision >= 3)
        {
          aux = aux * xsquared;
          atangent += aux / 5;
          if(precision >= 4)
          {
            aux = aux * xsquared;
            atangent -= aux / 7;
            if(precision >= 7)
            {
              printf(
                  "Warning: Function sinTyl from bmc/core/filter_functions.h: "
                      "Precision representation exceeded. "
                      "Assuming maximum precision of 4\n");
            }
          }
        }
      }
    }
  }

  return atangent;
}

/**
 * square root aproximation (80% faster than standard sqrt from math.h, 99.9% precision)
 */
float sqrt1(const float x)
{
  const float xhalf = 0.5f * x;

  union    // get bits for floating value
  {
    float x;
    int i;
  } u;

  u.x = x;
  u.i = 0x5f3759df - (u.i >> 1);    // gives initial guess y0

  // Newton step, repeating increases accuracy
  return x * u.x * (1.5f - xhalf * u.x * u.x);
}

/**
 * square root aproximation (100% faster than standard sqrt from math.h, 97% precison)
 */
float sqrt2(const float x)
{
  union
  {
    int i;
    float x;
  } u;

  u.x = x;
  u.i = (1 << 29) + (u.i >> 1) - (1 << 22);

  return u.x;
}

float fabsolut(float x)
{
  if(x < 0)
  {
    x = -x;
  }

  return x;
}

static float sqrt3(float val)
{
  float x = val / 10;
  float dx;
  double diff;
  double min_tol = 0.00001;
  int i, flag;

  flag = 0;

  if(val == 0)
  {
    x = 0;
  }
  else
  {
    for(i = 1; i < 20; i++)
    {
      if(!flag)
      {
        dx = (val - (x * x)) / (2.0 * x);
        x = x + dx;
        diff = val - (x * x);

        if(fabsolut(diff) <= min_tol)
        {
          flag = 1;
        }
      }
    }
  }

  return (x);
}
#endif // DSVERIFIER_CORE_FILTER_FUNCTIONS_H
