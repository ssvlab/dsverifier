/*
    Multi-precision floating point number class for C++. 
    Based on MPFI library:

    Derived from mpireal:
    Project homepage:    http://www.holoborodko.com/pavel/mpfr
    Contact e-mail:      pavel@holoborodko.com

    Copyright (c) 2008-2012 Pavel Holoborodko

    Contributors:
    Dmitriy Gubanov, Konstantin Holoborodko, Brian Gladman, 
    Helmut Jarausch, Fokko Beekhof, Ulrich Mutze, Heinz van Saanen, 
    Pere Constans, Peter van Hoof, Gael Guennebaud, Tsai Chia Cheng, 
    Alexei Zubanov, Jauhien Piatlicki, Victor Berger, John Westwood.

    ****************************************************************************
    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

    ****************************************************************************
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:
    
    1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    
    2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    
    3. The name of the author may not be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
    OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
    OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
    SUCH DAMAGE.
*/

#ifndef __MPIREAL_H__
#define __MPIREAL_H__

#include <string>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cfloat>
#include <cmath>
#include <limits>
#include "mpreal.h"

// Options
#define MPIREAL_HAVE_INT64_SUPPORT               // Enable int64_t support if possible. Available only for MSVC 2010 & GCC. 
#define MPIREAL_HAVE_MSVC_DEBUGVIEW              // Enable Debugger Visualizer for "Debug" builds in MSVC.

// Detect compiler using signatures from http://predef.sourceforge.net/
#if defined(__GNUC__) && defined(__INTEL_COMPILER)
    #define IsInf(x) isinf(x)                   // Intel ICC compiler on Linux 

#elif defined(_MSC_VER)                         // Microsoft Visual C++ 
    #define IsInf(x) (!_finite(x))                           

#else
    #define IsInf(x) std::isinf(x)              // GNU C/C++ (and/or other compilers), just hope for C99 conformance
#endif

#if defined(MPIREAL_HAVE_INT64_SUPPORT)
    
    #define MPFR_USE_INTMAX_T                   // Should be defined before mpfr.h

    #if defined(_MSC_VER)                       // MSVC + Windows
        #if (_MSC_VER >= 1600)                    
            #include <stdint.h>                 // <stdint.h> is available only in msvc2010!

        #else                                   // MPFR relies on intmax_t which is available only in msvc2010
            #undef MPIREAL_HAVE_INT64_SUPPORT    // Besides, MPFR & MPIR have to be compiled with msvc2010
            #undef MPFR_USE_INTMAX_T            // Since we cannot detect this, disable x64 by default
                                                // Someone should change this manually if needed.
        #endif

    #elif defined (__GNUC__) && defined(__linux__)
        #if defined(__amd64__) || defined(__amd64) || defined(__x86_64__) || defined(__x86_64) || defined(__ia64) || defined(__itanium__) || defined(_M_IA64)
            #undef MPIREAL_HAVE_INT64_SUPPORT    // Remove all shaman dances for x64 builds since
            #undef MPFR_USE_INTMAX_T            // GCC already supports x64 as of "long int" is 64-bit integer, nothing left to do
        #else
            #include <stdint.h>                 // use int64_t, uint64_t otherwise
        #endif

    #else
        #include <stdint.h>                     // rely on int64_t, uint64_t in all other cases, Mac OSX, etc.
    #endif

#endif 

#if defined(MPIREAL_HAVE_MSVC_DEBUGVIEW) && defined(_MSC_VER) && defined(_DEBUG)
#define MPIREAL_MSVC_DEBUGVIEW_CODE         DebugView = toString();
    #define MPIREAL_MSVC_DEBUGVIEW_DATA     std::string DebugView;
#else
    #define MPIREAL_MSVC_DEBUGVIEW_CODE 
    #define MPIREAL_MSVC_DEBUGVIEW_DATA 
#endif

#include <mpfi.h>

#if (MPFI_VERSION < MPFI_VERSION_NUM(3,0,0))
    #include <cstdlib>                          // Needed for random()
#endif

// Less important options
#define MPIREAL_DOUBLE_BITS_OVERFLOW -1          // Triggers overflow exception during conversion to double if mpireal 
                                                // cannot fit in MPIREAL_DOUBLE_BITS_OVERFLOW bits
                                                // = -1 disables overflow checks (default)
#if defined(__GNUC__)
  #define MPIREAL_PERMISSIVE_EXPR __extension__
#else
  #define MPIREAL_PERMISSIVE_EXPR
#endif

namespace mpfi {

using namespace mpfr;
class mpireal {
private:
    mpfi_t mp;
    
public:
    
    // Get default rounding mode & precision
    inline static mp_rnd_t   get_default_rnd()    {    return (mp_rnd_t)(mpfr_get_default_rounding_mode());       }
    inline static mp_prec_t  get_default_prec()   {    return mpfr_get_default_prec();                            }

    // Constructors && type conversions
    mpireal();
    mpireal(const mpireal& u);
    mpireal(const mpreal& u);
    mpireal(const mpfi_t u);
    mpireal(const mpfr_t u);
    mpireal(const mpf_t u);    
    mpireal(const mpz_t u,             mp_prec_t prec = mpireal::get_default_prec());
    mpireal(const mpq_t u,             mp_prec_t prec = mpireal::get_default_prec());
    mpireal(const double u,            mp_prec_t prec = mpireal::get_default_prec());
    mpireal(const long double u,       mp_prec_t prec = mpireal::get_default_prec());
    mpireal(const unsigned long int u, mp_prec_t prec = mpireal::get_default_prec());
    mpireal(const unsigned int u,      mp_prec_t prec = mpireal::get_default_prec());
    mpireal(const long int u,          mp_prec_t prec = mpireal::get_default_prec());
    mpireal(const int u,               mp_prec_t prec = mpireal::get_default_prec());

#if defined (MPIREAL_HAVE_INT64_SUPPORT)
    mpireal(const uint64_t u,          mp_prec_t prec = mpireal::get_default_prec());
    mpireal(const int64_t u,           mp_prec_t prec = mpireal::get_default_prec());
#endif

    mpireal(const char* s,             mp_prec_t prec = mpireal::get_default_prec(), int base = 10);
    mpireal(const std::string& s,      mp_prec_t prec = mpireal::get_default_prec(), int base = 10);

    ~mpireal();                           

    // Operations
    // =
    // +, -, *, /, ++, --, <<, >> 
    // *=, +=, -=, /=,
    // <, >, ==, <=, >=

    // =
    mpireal& operator=(const mpireal& v);
    mpireal& operator=(const mpreal& v);
    mpireal& operator=(const mpf_t v);
    mpireal& operator=(const mpz_t v);
    mpireal& operator=(const mpq_t v);
    mpireal& operator=(const long double v);
    mpireal& operator=(const double v);        
    mpireal& operator=(const unsigned long int v);
    mpireal& operator=(const unsigned int v);
    mpireal& operator=(const long int v);
    mpireal& operator=(const int v);
    mpireal& operator=(const char* s);
    mpireal& operator=(const std::string& s);

    // +
    mpireal& operator+=(const mpireal& v);
    mpireal& operator+=(const mpreal& v);
    mpireal& operator+=(const mpf_t v);
    mpireal& operator+=(const mpz_t v);
    mpireal& operator+=(const mpq_t v);
    mpireal& operator+=(const long double u);
    mpireal& operator+=(const double u);
    mpireal& operator+=(const unsigned long int u);
    mpireal& operator+=(const unsigned int u);
    mpireal& operator+=(const long int u);
    mpireal& operator+=(const int u);

#if defined (MPIREAL_HAVE_INT64_SUPPORT)
    mpireal& operator+=(const int64_t  u);
    mpireal& operator+=(const uint64_t u);
    mpireal& operator-=(const int64_t  u);
    mpireal& operator-=(const uint64_t u);
    mpireal& operator*=(const int64_t  u);
    mpireal& operator*=(const uint64_t u);
    mpireal& operator/=(const int64_t  u);
    mpireal& operator/=(const uint64_t u);
#endif 

    const mpireal operator+() const;
    mpireal& operator++ ();
    const mpireal  operator++ (int); 

    // -
    mpireal& operator-=(const mpireal& v);
    mpireal& operator-=(const mpreal& v);
    mpireal& operator-=(const mpz_t v);
    mpireal& operator-=(const mpq_t v);
    mpireal& operator-=(const long double u);
    mpireal& operator-=(const double u);
    mpireal& operator-=(const unsigned long int u);
    mpireal& operator-=(const unsigned int u);
    mpireal& operator-=(const long int u);
    mpireal& operator-=(const int u);
    const mpireal operator-() const;
    friend const mpireal operator-(const unsigned long int b, const mpireal& a);
    friend const mpireal operator-(const unsigned int b,      const mpireal& a);
    friend const mpireal operator-(const long int b,          const mpireal& a);
    friend const mpireal operator-(const int b,               const mpireal& a);
    friend const mpireal operator-(const double b,            const mpireal& a);
    mpireal& operator-- ();    
    const mpireal  operator-- (int);

    // *
    mpireal& operator*=(const mpireal& v);
    mpireal& operator*=(const mpreal& v);
    mpireal& operator*=(const mpz_t v);
    mpireal& operator*=(const mpq_t v);
    mpireal& operator*=(const long double v);
    mpireal& operator*=(const double v);
    mpireal& operator*=(const unsigned long int v);
    mpireal& operator*=(const unsigned int v);
    mpireal& operator*=(const long int v);
    mpireal& operator*=(const int v);
    
    // /
    mpireal& operator/=(const mpireal& v);
    mpireal& operator/=(const mpreal& v);
    mpireal& operator/=(const mpz_t v);
    mpireal& operator/=(const mpq_t v);
    mpireal& operator/=(const long double v);
    mpireal& operator/=(const double v);
    mpireal& operator/=(const unsigned long int v);
    mpireal& operator/=(const unsigned int v);
    mpireal& operator/=(const long int v);
    mpireal& operator/=(const int v);
    friend const mpireal operator/(const unsigned long int b, const mpireal& a);
    friend const mpireal operator/(const unsigned int b,      const mpireal& a);
    friend const mpireal operator/(const long int b,          const mpireal& a);
    friend const mpireal operator/(const int b,               const mpireal& a);
    friend const mpireal operator/(const double b,            const mpireal& a);

    //<<= Fast Multiplication by 2^u
    mpireal& operator<<=(const unsigned long int u);
    mpireal& operator<<=(const unsigned int u);
    mpireal& operator<<=(const long int u);
    mpireal& operator<<=(const int u);

    //>>= Fast Division by 2^u
    mpireal& operator>>=(const unsigned long int u);
    mpireal& operator>>=(const unsigned int u);
    mpireal& operator>>=(const long int u);
    mpireal& operator>>=(const int u);

    // Boolean Operators
    friend bool operator >  (const mpireal& a, const mpireal& b);
    friend bool operator >= (const mpireal& a, const mpireal& b);
    friend bool operator <  (const mpireal& a, const mpireal& b);
    friend bool operator <= (const mpireal& a, const mpireal& b);
    friend bool operator == (const mpireal& a, const mpireal& b);
    friend bool operator != (const mpireal& a, const mpireal& b);

    friend bool operator >  (const mpireal& a, const mpreal& b);
    friend bool operator >= (const mpireal& a, const mpreal& b);
    friend bool operator <  (const mpireal& a, const mpreal& b);
    friend bool operator <= (const mpireal& a, const mpreal& b);
    friend bool operator == (const mpireal& a, const mpreal& b);
    friend bool operator != (const mpireal& a, const mpreal& b);
    friend bool operator >  (const mpreal& a, const mpireal& b);
    friend bool operator >= (const mpreal& a, const mpireal& b);
    friend bool operator <  (const mpreal& a, const mpireal& b);
    friend bool operator <= (const mpreal& a, const mpireal& b);
    friend bool operator == (const mpreal& a, const mpireal& b);
    friend bool operator != (const mpreal& a, const mpireal& b);

    // Optimized specializations for boolean operators
    friend bool operator == (const mpireal& a, const unsigned long int b);
    friend bool operator == (const mpireal& a, const unsigned int b);
    friend bool operator == (const mpireal& a, const long int b);
    friend bool operator == (const mpireal& a, const int b);
    friend bool operator == (const mpireal& a, const long double b);
    friend bool operator == (const mpireal& a, const double b);

    // Type Conversion operators
    long            toLong      ()    const;
    unsigned long   toULong     ()    const;
    double          toDouble    ()    const;
    long double     toLDouble   ()    const;

#if defined (MPIREAL_HAVE_INT64_SUPPORT)
    int64_t         toInt64     ()    const;
    uint64_t        toUInt64    ()    const;
#endif

    // Get raw pointers so that mpireal can be directly used in raw mpfi_* functions
    ::mpfi_ptr    mpfi_ptr();
    ::mpfi_srcptr mpfi_ptr()    const;
    ::mpfi_srcptr mpfi_srcptr() const;

    // Convert mpireal to string with n significant digits in base b
    // n = 0 -> convert with the maximum available digits 
    std::string        toString(int n = 0, int b = 10, mp_rnd_t mode = mpireal::get_default_rnd()) const;

#if (MPFI_VERSION >= MPFI_VERSION_NUM(1,5,1))
    std::string        toString(const std::string& format) const;
#endif

    // Math Functions
    friend const mpireal sqr (const mpireal& v);
    friend const mpireal sqrt(const mpireal& v);
    friend const mpireal sqrt(const unsigned long int v);
    friend const mpireal cbrt(const mpireal& v);
#if (MPFI_VERSION >= MPFI_VERSION_NUM(3,0,0))
    friend const mpireal root(const mpireal& v, unsigned long int k);
#endif
    friend const mpireal pow (const mpireal& a, const mpireal& b);
    friend const mpireal pow (const mpireal& a, const mpz_t b);
    friend const mpireal pow (const mpireal& a, const unsigned long int b);
    friend const mpireal pow (const mpireal& a, const long int b);
    friend const mpireal pow (const unsigned long int a, const mpireal& b);
    friend const mpireal pow (const unsigned long int a, const unsigned long int b);
    friend const mpireal fabs(const mpireal& v);

    friend const mpireal abs(const mpireal& v);
    friend const mpireal dim(const mpireal& a, const mpireal& b);
    friend inline const mpireal mul_2ui(const mpireal& v, unsigned long int k);
    friend inline const mpireal mul_2si(const mpireal& v, long int k);
    friend inline const mpireal div_2ui(const mpireal& v, unsigned long int k);
    friend inline const mpireal div_2si(const mpireal& v, long int k);
    friend int cmpabs(const mpireal& a,const mpireal& b);
    
    friend const mpireal log  (const mpireal& v);
    friend const mpireal log2 (const mpireal& v);
    friend const mpireal log10(const mpireal& v);
    friend const mpireal exp  (const mpireal& v);
    friend const mpireal exp2 (const mpireal& v);
    friend const mpireal exp10(const mpireal& v);
    friend const mpireal log1p(const mpireal& v);
    friend const mpireal expm1(const mpireal& v);

    friend const mpireal cos(const mpireal& v);
    friend const mpireal sin(const mpireal& v);
    friend const mpireal tan(const mpireal& v);
    friend const mpireal sec(const mpireal& v);
    friend const mpireal csc(const mpireal& v);
    friend const mpireal cot(const mpireal& v);
#if (MPFI_VERSION >= MPFI_VERSION_NUM(3,0,0))
    friend int sin_cos(mpireal& s, mpireal& c, const mpireal& v);
#endif

    friend const mpireal acos  (const mpireal& v);
    friend const mpireal asin  (const mpireal& v);
    friend const mpireal atan  (const mpireal& v);
    friend const mpireal atan2 (const mpireal& y, const mpireal& x);
    friend const mpireal acot  (const mpireal& v);
    friend const mpireal asec  (const mpireal& v);
    friend const mpireal acsc  (const mpireal& v);

    friend const mpireal cosh  (const mpireal& v);
    friend const mpireal sinh  (const mpireal& v);
    friend const mpireal tanh  (const mpireal& v);
    friend const mpireal sech  (const mpireal& v);
    friend const mpireal csch  (const mpireal& v);
    friend const mpireal coth  (const mpireal& v);
    friend const mpireal acosh (const mpireal& v);
    friend const mpireal asinh (const mpireal& v);
    friend const mpireal atanh (const mpireal& v);
    friend const mpireal acoth (const mpireal& v);
    friend const mpireal asech (const mpireal& v);
    friend const mpireal acsch (const mpireal& v);

    friend const mpireal hypot (const mpireal& x, const mpireal& y);

    friend const mpireal fac_ui (unsigned long int v,  mp_prec_t prec = mpireal::get_default_prec());
    friend const mpireal eint   (const mpireal& v);

    friend const mpireal gamma    (const mpireal& v);
    friend const mpireal lngamma  (const mpireal& v);
    friend const mpireal lgamma   (const mpireal& v, int *signp = 0);
    friend const mpireal zeta     (const mpireal& v);
    friend const mpireal erf      (const mpireal& v);
    friend const mpireal erfc     (const mpireal& v);
    friend const mpireal besselj0 (const mpireal& v);
    friend const mpireal besselj1 (const mpireal& v);
    friend const mpireal besseljn (long n, const mpireal& v);
    friend const mpireal bessely0 (const mpireal& v);
    friend const mpireal bessely1 (const mpireal& v);
    friend const mpireal besselyn (long n, const mpireal& v);
    friend const mpireal fma      (const mpireal& v1, const mpireal& v2, const mpireal& v3);
    friend const mpireal fms      (const mpireal& v1, const mpireal& v2, const mpireal& v3);
    friend const mpireal agm      (const mpireal& v1, const mpireal& v2);
    friend const mpireal sum      (const mpireal tab[], unsigned long int n);
    friend int sgn(const mpireal& v); // returns -1 or +1

// MPFR 2.4.0 Specifics
#if (MPFI_VERSION >= MPFI_VERSION_NUM(2,4,0))
    friend int          sinh_cosh   (mpireal& s, mpireal& c, const mpireal& v);
    friend const mpireal li2         (const mpireal& v);
    friend const mpireal fmod        (const mpireal& x, const mpireal& y);
    friend const mpireal rec_sqrt    (const mpireal& v);

    // MATLAB's semantic equivalents
    friend const mpireal rem (const mpireal& x, const mpireal& y); // Remainder after division
    friend const mpireal mod (const mpireal& x, const mpireal& y); // Modulus after division
#endif

// MPFR 3.0.0 Specifics
#if (MPFI_VERSION >= MPFI_VERSION_NUM(3,0,0))
    friend const mpireal digamma (const mpireal& v);
    friend const mpireal ai      (const mpireal& v);
    friend const mpireal urandom (gmp_randstate_t& state);     // use gmp_randinit_default() to init state, gmp_randclear() to clear
#endif
    
    // Uniformly distributed random number generation in [0,1] using
    // Mersenne-Twister algorithm by default.
    // Use parameter to setup seed, e.g.: random((unsigned)time(NULL))
    // Check urandom() for more precise control.
    friend const mpireal random(unsigned int seed = 0);
    
    // Exponent and mantissa manipulation
#if (MPFI_VERSION >= MPFI_VERSION_NUM(3,0,0))
    friend const mpireal frexp(const mpireal& v, mp_exp_t* exp);
#endif
    friend const mpireal ldexp(const mpireal& v, mp_exp_t exp);

    // Splits mpireal value into fractional and integer parts.
    // Returns fractional part and stores integer part in n.
#if (MPFI_VERSION >= MPFI_VERSION_NUM(3,0,0))
    friend const mpireal modf(const mpireal& v, mpireal& n);    
#endif

    // Constants
    // don't forget to call mpfr_free_cache() for every thread where you are using const-functions
    friend const mpireal const_log2      (mp_prec_t prec = mpireal::get_default_prec());
    friend const mpireal const_pi        (mp_prec_t prec = mpireal::get_default_prec());
    friend const mpireal const_euler     (mp_prec_t prec = mpireal::get_default_prec());
    friend const mpireal const_catalan   (mp_prec_t prec = mpireal::get_default_prec());

    // returns +inf iff sign>=0 otherwise -inf
    friend const mpireal const_infinity(int sign = 1, mp_prec_t prec = mpireal::get_default_prec());

    // Output/ Input
    friend std::ostream& operator<<(std::ostream& os, const mpireal& v);
    friend std::istream& operator>>(std::istream& is, mpireal& v);

    // Integer Related Functions
    friend const mpireal rint (const mpireal& v);
    friend const mpireal ceil (const mpireal& v);
    friend const mpireal floor(const mpireal& v);
    friend const mpireal round(const mpireal& v);
    friend const mpireal trunc(const mpireal& v);
    friend const mpireal rint_ceil   (const mpireal& v);
    friend const mpireal rint_floor  (const mpireal& v);
    friend const mpireal rint_round  (const mpireal& v);
    friend const mpireal rint_trunc  (const mpireal& v);
#if (MPFI_VERSION >= MPFI_VERSION_NUM(3,0,0))
    friend const mpireal frac        (const mpireal& v);
    friend const mpireal remainder   (         const mpireal& x, const mpireal& y);
    friend const mpireal remquo      (long* q, const mpireal& x, const mpireal& y);
#endif

    // Miscellaneous Functions
    friend const mpireal nexttoward (const mpireal& x, const mpireal& y);
    friend const mpireal nextabove  (const mpireal& x);
    friend const mpireal nextbelow  (const mpireal& x);

    // use gmp_randinit_default() to init state, gmp_randclear() to clear
    friend const mpreal urandomb (gmp_randstate_t& state);

    // Instance Checkers
    friend bool isnan    (const mpireal& v);
    friend bool isinf    (const mpireal& v);
    friend bool isfinite (const mpireal& v);

    friend bool isnum    (const mpireal& v);
    friend bool iszero   (const mpireal& v);
    friend bool isint    (const mpireal& v);

#if (MPFI_VERSION >= MPFI_VERSION_NUM(3,0,0))
    friend bool isregular(const mpireal& v);
#endif

    // Set/Get instance properties
    inline mp_prec_t    get_prec() const;
    inline void         set_prec(mp_prec_t prec = get_default_rnd());    // Change precision with rounding mode

    // Aliases for get_prec(), set_prec() - needed for compatibility with std::complex<mpireal> interface
    inline mpireal&     setPrecision(int Precision);
    inline int          getPrecision() const;
    
    // Set mpireal to +/- inf, NaN, +/-0
    mpireal&        setInf  (int Sign = +1);    
    mpireal&        setNan  ();
    mpireal&        setZero (int Sign = +1);
    mpireal&        setSign (int Sign, mp_rnd_t RoundingMode = get_default_rnd());

    //Exponent
#if (MPFI_VERSION >= MPFI_VERSION_NUM(3,0,0))
    mp_exp_t get_exp();
    int set_exp(mp_exp_t e);
    int check_range  (int t = get_default_rnd());
    int subnormalize (int t);
#endif

    // Inexact conversion from float
    inline bool fits_in_bits(double x, int n);

    // Set/Get global properties
    static void            set_default_prec(mp_prec_t prec);
    static void            set_default_rnd(mp_rnd_t rnd_mode);

    static mp_exp_t  get_emin (void);
    static mp_exp_t  get_emax (void);
    static mp_exp_t  get_emin_min (void);
    static mp_exp_t  get_emin_max (void);
    static mp_exp_t  get_emax_min (void);
    static mp_exp_t  get_emax_max (void);
    static int       set_emin (mp_exp_t exp);
    static int       set_emax (mp_exp_t exp);

    // Efficient swapping of two mpireal values - needed for std algorithms
    friend void swap(mpireal& x, mpireal& y);
    
    friend const mpireal fmax(const mpireal& x, const mpireal& y);
    friend const mpireal fmin(const mpireal& x, const mpireal& y);

private:
    // Human friendly Debug Preview in Visual Studio.
    // Put one of these lines:
    //
    // mpfi::mpireal=<DebugView>                                ; Show value only
    // mpfi::mpireal=<DebugView>, <mp[0]._mpfr_prec,u>bits    ; Show value & precision
    // 
    // at the beginning of
    // [Visual Studio Installation Folder]\Common7\Packages\Debugger\autoexp.dat
    MPIREAL_MSVC_DEBUGVIEW_DATA
};

//////////////////////////////////////////////////////////////////////////
// Exceptions
class conversion_overflow : public std::exception {
public:
    std::string why() { return "inexact conversion from floating point"; }
};

//////////////////////////////////////////////////////////////////////////
// Constructors & converters
// Default constructor: creates mp number and initializes it to 0.
inline mpireal::mpireal() 
{ 
    mpfi_init2(mp,mpireal::get_default_prec()); 
    mpfi_set_ui(mp,0);

    MPIREAL_MSVC_DEBUGVIEW_CODE;
}

inline mpireal::mpireal(const mpireal& u) 
{
    mpfi_init2(mp,mpfi_get_prec(u.mp));
    mpfi_set(mp,u.mp);

    MPIREAL_MSVC_DEBUGVIEW_CODE;
}

inline mpireal::mpireal(const mpfi_t u)
{
    mpfi_init2(mp,mpfi_get_prec(u));
    mpfi_set(mp,u);

    MPIREAL_MSVC_DEBUGVIEW_CODE;
}

inline mpireal::mpireal(const mpfr_t u)
{
    mpfi_init2(mp,mpfr_get_prec(u));
    mpfi_set_fr(mp,u);

    MPIREAL_MSVC_DEBUGVIEW_CODE;
}

inline mpireal::mpireal(const mpf_t u)
{
    mpfi_init2(mp,(mp_prec_t) mpf_get_prec(u)); // (gmp: mp_bitcnt_t) unsigned long -> long (mpfr: mp_prec_t)
    mpfr_set_f(&mp->left,u,MPFR_RNDD);
    mpfr_set_f(&mp->right,u,MPFR_RNDU);

    MPIREAL_MSVC_DEBUGVIEW_CODE;
}

inline mpireal::mpireal(const mpz_t u, mp_prec_t prec)
{
    mpfi_init2(mp,prec);
    mpfr_set_z(&mp->left,u,MPFR_RNDD);
    mpfr_set_z(&mp->right,u,MPFR_RNDU);

    MPIREAL_MSVC_DEBUGVIEW_CODE;
}

inline mpireal::mpireal(const mpq_t u, mp_prec_t prec)
{
    mpfi_init2(mp,prec);
    mpfi_set_q(mp,u);

    MPIREAL_MSVC_DEBUGVIEW_CODE;
}

inline mpireal::mpireal(const double u, mp_prec_t prec)
{
     mpfi_init2(mp, prec);

#if (MPIREAL_DOUBLE_BITS_OVERFLOW > -1)
	if(fits_in_bits(u, MPIREAL_DOUBLE_BITS_OVERFLOW))
	{
                mpfi_set_d(mp, u);
	}else
		throw conversion_overflow();
#else
        mpfi_set_d(mp, u);
#endif

    MPIREAL_MSVC_DEBUGVIEW_CODE;
}

inline mpireal::mpireal(const long double u, mp_prec_t prec)
{ 
    mpfi_init2(mp,prec);
    mpfi_set_ld(mp,u);

    MPIREAL_MSVC_DEBUGVIEW_CODE;
}

inline mpireal::mpireal(const unsigned long int u, mp_prec_t prec)
{ 
    mpfi_init2(mp,prec);
    mpfi_set_ui(mp,u);

    MPIREAL_MSVC_DEBUGVIEW_CODE;
}

inline mpireal::mpireal(const unsigned int u, mp_prec_t prec)
{ 
    mpfi_init2(mp,prec);
    mpfi_set_ui(mp,u);

    MPIREAL_MSVC_DEBUGVIEW_CODE;
}

inline mpireal::mpireal(const long int u, mp_prec_t prec)
{ 
    mpfi_init2(mp,prec);
    mpfi_set_si(mp,u);

    MPIREAL_MSVC_DEBUGVIEW_CODE;
}

inline mpireal::mpireal(const int u, mp_prec_t prec)
{ 
    mpfi_init2(mp,prec);
    mpfi_set_si(mp,u);

    MPIREAL_MSVC_DEBUGVIEW_CODE;
}

#if defined (MPIREAL_HAVE_INT64_SUPPORT)
inline mpireal::mpireal(const uint64_t u, mp_prec_t prec)
{
    mpfi_init2(mp,prec);
    mpfi_set_uj(mp, u);

    MPIREAL_MSVC_DEBUGVIEW_CODE;
}

inline mpireal::mpireal(const int64_t u, mp_prec_t prec)
{
    mpfi_init2(mp,prec);
    mpfi_set_sj(mp, u);

    MPIREAL_MSVC_DEBUGVIEW_CODE;
}
#endif

inline mpireal::mpireal(const char* s, mp_prec_t prec, int base)
{
    mpfi_init2(mp, prec);
    mpfi_set_str(mp, s, base);

    MPIREAL_MSVC_DEBUGVIEW_CODE;
}

inline mpireal::mpireal(const std::string& s, mp_prec_t prec, int base)
{
    mpfi_init2(mp, prec);
    mpfi_set_str(mp, s.c_str(), base);

    MPIREAL_MSVC_DEBUGVIEW_CODE;
}

inline mpireal::~mpireal() 
{ 
    mpfi_clear(mp);
}                           

// internal namespace needed for template magic
namespace internal{

    // Use SFINAE to restrict arithmetic operations instantiation only for numeric types
    // This is needed for smooth integration with libraries based on expression templates, like Eigen.
    // TODO: Do the same for boolean operators.
    template <typename ArgumentType> struct result_type {};    
    
    template <> struct result_type<mpireal>             {typedef mpireal type;};    
    template <> struct result_type<mpreal>              {typedef mpireal type;};    
    template <> struct result_type<mpz_t>               {typedef mpireal type;};    
    template <> struct result_type<mpq_t>               {typedef mpireal type;};    
    template <> struct result_type<long double>         {typedef mpireal type;};    
    template <> struct result_type<double>              {typedef mpireal type;};    
    template <> struct result_type<unsigned long int>   {typedef mpireal type;};    
    template <> struct result_type<unsigned int>        {typedef mpireal type;};    
    template <> struct result_type<long int>            {typedef mpireal type;};    
    template <> struct result_type<int>                 {typedef mpireal type;};    

#if defined (MPIREAL_HAVE_INT64_SUPPORT)
    template <> struct result_type<int64_t  >           {typedef mpireal type;};    
    template <> struct result_type<uint64_t >           {typedef mpireal type;};    
#endif
}

// + Addition
template <typename Rhs> 
inline const typename internal::result_type<Rhs>::type 
    operator+(const mpireal& lhs, const Rhs& rhs){ return mpireal(lhs) += rhs;    }

template <typename Lhs> 
inline const typename internal::result_type<Lhs>::type 
    operator+(const Lhs& lhs, const mpireal& rhs){ return mpireal(rhs) += lhs;    } 

// - Subtraction
template <typename Rhs> 
inline const typename internal::result_type<Rhs>::type 
    operator-(const mpireal& lhs, const Rhs& rhs){ return mpireal(lhs) -= rhs;    }

template <typename Lhs> 
inline const typename internal::result_type<Lhs>::type 
    operator-(const Lhs& lhs, const mpireal& rhs){ return mpireal(lhs) -= rhs;    }

// * Multiplication
template <typename Rhs> 
inline const typename internal::result_type<Rhs>::type 
    operator*(const mpireal& lhs, const Rhs& rhs){ return mpireal(lhs) *= rhs;    }

template <typename Lhs> 
inline const typename internal::result_type<Lhs>::type 
    operator*(const Lhs& lhs, const mpireal& rhs){ return mpireal(rhs) *= lhs;    } 

// / Division
template <typename Rhs> 
inline const typename internal::result_type<Rhs>::type 
    operator/(const mpireal& lhs, const Rhs& rhs){ return mpireal(lhs) /= rhs;    }

template <typename Lhs> 
inline const typename internal::result_type<Lhs>::type 
    operator/(const Lhs& lhs, const mpireal& rhs){ return mpireal(lhs) /= rhs;    }

//////////////////////////////////////////////////////////////////////////
// sqrt
const mpireal sqrt(const unsigned int v);
const mpireal sqrt(const long int v);
const mpireal sqrt(const int v);
const mpireal sqrt(const long double v);
const mpireal sqrt(const double v);

//////////////////////////////////////////////////////////////////////////
// pow
const mpireal pow(const mpireal& a, const unsigned int b);
const mpireal pow(const mpireal& a, const int b);
const mpireal pow(const mpireal& a, const long double b);
const mpireal pow(const mpireal& a, const double b);

const mpireal pow(const unsigned int a, const mpireal& b);
const mpireal pow(const long int a, const mpireal& b);
const mpireal pow(const int a, const mpireal& b);
const mpireal pow(const long double a, const mpireal& b);
const mpireal pow(const double a, const mpireal& b);

const mpireal pow(const unsigned long int a, const unsigned int b);
const mpireal pow(const unsigned long int a, const long int b);
const mpireal pow(const unsigned long int a, const int b);
const mpireal pow(const unsigned long int a, const long double b);
const mpireal pow(const unsigned long int a, const double b);

const mpireal pow(const unsigned int a, const unsigned long int b);
const mpireal pow(const unsigned int a, const unsigned int b);
const mpireal pow(const unsigned int a, const long int b);
const mpireal pow(const unsigned int a, const int b);
const mpireal pow(const unsigned int a, const long double b);
const mpireal pow(const unsigned int a, const double b);

const mpireal pow(const long int a, const unsigned long int b);
const mpireal pow(const long int a, const unsigned int b);
const mpireal pow(const long int a, const long int b);
const mpireal pow(const long int a, const int b);
const mpireal pow(const long int a, const long double b);
const mpireal pow(const long int a, const double b);

const mpireal pow(const int a, const unsigned long int b);
const mpireal pow(const int a, const unsigned int b);
const mpireal pow(const int a, const long int b);
const mpireal pow(const int a, const int b);
const mpireal pow(const int a, const long double b);
const mpireal pow(const int a, const double b);

const mpireal pow(const long double a, const long double b);
const mpireal pow(const long double a, const unsigned long int b);
const mpireal pow(const long double a, const unsigned int b);
const mpireal pow(const long double a, const long int b);
const mpireal pow(const long double a, const int b);

const mpireal pow(const double a, const double b);
const mpireal pow(const double a, const unsigned long int b);
const mpireal pow(const double a, const unsigned int b);
const mpireal pow(const double a, const long int b);
const mpireal pow(const double a, const int b);

//////////////////////////////////////////////////////////////////////////
// Estimate machine epsilon for the given precision
// Returns smallest eps such that 1.0 + eps != 1.0
inline mpireal machine_epsilon(mp_prec_t prec = mpireal::get_default_prec());

// Returns smallest eps such that x + eps != x (relative machine epsilon)
inline mpireal machine_epsilon(const mpireal& x);        

// Gives max & min values for the required precision, 
// minval is 'safe' meaning 1 / minval does not overflow
// maxval is 'safe' meaning 1 / maxval does not underflow
inline mpireal minval(mp_prec_t prec = mpireal::get_default_prec());
inline mpireal maxval(mp_prec_t prec = mpireal::get_default_prec());

// 'Dirty' equality check 1: |a-b| < min{|a|,|b|} * eps
inline bool isEqualFuzzy(const mpireal& a, const mpireal& b, const mpireal& eps);

// 'Dirty' equality check 2: |a-b| < min{|a|,|b|} * eps( min{|a|,|b|} )
inline bool isEqualFuzzy(const mpireal& a, const mpireal& b);

// 'Bitwise' equality check
//  maxUlps - a and b can be apart by maxUlps binary numbers. 
inline bool isEqualUlps(const mpireal& a, const mpireal& b, int maxUlps);

//////////////////////////////////////////////////////////////////////////
//     Convert precision in 'bits' to decimal digits and vice versa.
//        bits   = ceil(digits*log[2](10))
//        digits = floor(bits*log[10](2))

inline mp_prec_t digits2bits(int d);
inline int       bits2digits(mp_prec_t b);

//////////////////////////////////////////////////////////////////////////
// min, max
const mpireal (max)(const mpireal& x, const mpireal& y);
const mpireal (min)(const mpireal& x, const mpireal& y);

//////////////////////////////////////////////////////////////////////////
// Implementation
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// Operators - Assignment
inline mpireal& mpireal::operator=(const mpireal& v)
{
    if (this != &v)
    {
		mp_prec_t tp = mpfi_get_prec(mp);
		mp_prec_t vp = mpfi_get_prec(v.mp);

		if(tp != vp){
			mpfi_clear(mp);
			mpfi_init2(mp, vp);
		}

        mpfi_set(mp, v.mp);

        MPIREAL_MSVC_DEBUGVIEW_CODE;
    }
    return *this;
}

inline mpireal& mpireal::operator=(const mpreal& v)
{
    mp_prec_t tp = mpfi_get_prec(mp);
    mp_prec_t vp = v.getPrecision();

    if (tp != vp){
        mpfi_clear(mp);
        mpfi_init2(mp, vp);
    }

    mpfi_set_fr(mp, v.mpfr_srcptr());
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator=(const mpf_t v)
{
    mpfi_set_f(mp, v);
    
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator=(const mpz_t v)
{
    mpfi_set_z(mp, v);
    
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator=(const mpq_t v)
{
    mpfi_set_q(mp, v);

    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator=(const long double v)        
{    
    mpfi_set_ld(mp, v);

    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator=(const double v)                
{   
#if (MPIREAL_DOUBLE_BITS_OVERFLOW > -1)
	if(fits_in_bits(v, MPIREAL_DOUBLE_BITS_OVERFLOW))
	{
                mpfi_set_d(mp,v);
	}else
		throw conversion_overflow();
#else
        mpfi_set_d(mp,v);
#endif

	MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator=(const unsigned long int v)    
{    
    mpfi_set_ui(mp, v);

    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator=(const unsigned int v)        
{    
    mpfi_set_ui(mp, v);

    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator=(const long int v)            
{    
    mpfi_set_si(mp, v);

    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator=(const int v)
{    
    mpfi_set_si(mp, v);

    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator=(const char* s)
{
    // Use other converters for more precise control on base & precision & rounding:
    //
    //        mpireal(const char* s,        mp_prec_t prec, int base, mp_rnd_t mode)
    //        mpireal(const std::string& s,mp_prec_t prec, int base, mp_rnd_t mode)
    //
    // Here we assume base = 10 and we use precision of target variable.

    mpfi_t t;

    mpfi_init2(t, mpfi_get_prec(mp));

    if(0 == mpfi_set_str(t, s, 10))
    {
        mpfi_set(mp, t);
        MPIREAL_MSVC_DEBUGVIEW_CODE;
    }

    mpfi_clear(t);
    return *this;
}

inline mpireal& mpireal::operator=(const std::string& s)
{
    // Use other converters for more precise control on base & precision & rounding:
    //
    //        mpireal(const char* s,        mp_prec_t prec, int base, mp_rnd_t mode)
    //        mpireal(const std::string& s,mp_prec_t prec, int base, mp_rnd_t mode)
    //
    // Here we assume base = 10 and we use precision of target variable.

    mpfi_t t;

    mpfi_init2(t, mpfi_get_prec(mp));

    if(0 == mpfi_set_str(t, s.c_str(), 10))
    {
        mpfi_set(mp, t);
        MPIREAL_MSVC_DEBUGVIEW_CODE;
    }

    mpfi_clear(t);
    return *this;
}


//////////////////////////////////////////////////////////////////////////
// + Addition
inline mpireal& mpireal::operator+=(const mpireal& v)
{
    mpfi_add(mp,mp,v.mp);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator+=(const mpf_t u)
{
    *this += mpireal(u);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator+=(const mpz_t u)
{
    mpfi_add_z(mp,mp,u);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator+=(const mpq_t u)
{
    mpfi_add_q(mp,mp,u);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator+= (const long double u)
{
    *this += mpireal(u);    
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;    
}

inline mpireal& mpireal::operator+= (const double u)
{
#if (MPFI_VERSION >= MPFI_VERSION_NUM(1,5,1))
    mpfi_add_d(mp,mp,u);
#else
    *this += mpireal(u);
#endif

    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator+=(const unsigned long int u)
{
    mpfi_add_ui(mp,mp,u);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator+=(const unsigned int u)
{
    mpfi_add_ui(mp,mp,u);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator+=(const long int u)
{
    mpfi_add_si(mp,mp,u);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator+=(const int u)
{
    mpfi_add_si(mp,mp,u);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

#if defined (MPIREAL_HAVE_INT64_SUPPORT)
inline mpireal& mpireal::operator+=(const int64_t  u){    *this += mpireal(u); MPIREAL_MSVC_DEBUGVIEW_CODE; return *this;    }
inline mpireal& mpireal::operator+=(const uint64_t u){    *this += mpireal(u); MPIREAL_MSVC_DEBUGVIEW_CODE; return *this;    }
inline mpireal& mpireal::operator-=(const int64_t  u){    *this -= mpireal(u); MPIREAL_MSVC_DEBUGVIEW_CODE; return *this;    }
inline mpireal& mpireal::operator-=(const uint64_t u){    *this -= mpireal(u); MPIREAL_MSVC_DEBUGVIEW_CODE; return *this;    }
inline mpireal& mpireal::operator*=(const int64_t  u){    *this *= mpireal(u); MPIREAL_MSVC_DEBUGVIEW_CODE; return *this;    }
inline mpireal& mpireal::operator*=(const uint64_t u){    *this *= mpireal(u); MPIREAL_MSVC_DEBUGVIEW_CODE; return *this;    }
inline mpireal& mpireal::operator/=(const int64_t  u){    *this /= mpireal(u); MPIREAL_MSVC_DEBUGVIEW_CODE; return *this;    }
inline mpireal& mpireal::operator/=(const uint64_t u){    *this /= mpireal(u); MPIREAL_MSVC_DEBUGVIEW_CODE; return *this;    }
#endif 

inline const mpireal mpireal::operator+()const    {    return mpireal(*this); }

inline const mpireal operator+(const mpireal& a, const mpireal& b)
{
        mpireal c(0, (std::max)(mpfi_get_prec(a.mpfi_ptr()), mpfi_get_prec(b.mpfi_ptr())));
        mpfi_add(c.mpfi_ptr(), a.mpfi_srcptr(), b.mpfi_srcptr());
	return c;
}

inline mpireal& mpireal::operator++() 
{
    return *this += 1;
}

inline const mpireal mpireal::operator++ (int)
{
    mpireal x(*this);
    *this += 1;
    return x;
}

inline mpireal& mpireal::operator--() 
{
    return *this -= 1;
}

inline const mpireal mpireal::operator-- (int)
{
    mpireal x(*this);
    *this -= 1;
    return x;
}

//////////////////////////////////////////////////////////////////////////
// - Subtraction
inline mpireal& mpireal::operator-=(const mpireal& v)
{
    mpfi_sub(mp,mp,v.mp);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator-=(const mpz_t v)
{
    mpfi_sub_z(mp,mp,v);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator-=(const mpq_t v)
{
    mpfi_sub_q(mp,mp,v);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator-=(const long double v)
{
    *this -= mpireal(v);    
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;    
}

inline mpireal& mpireal::operator-=(const double v)
{
#if (MPFI_VERSION >= MPFI_VERSION_NUM(1,5,1))
    mpfi_sub_d(mp,mp,v);
#else
    *this -= mpireal(v);    
#endif

    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator-=(const unsigned long int v)
{
    mpfi_sub_ui(mp,mp,v);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator-=(const unsigned int v)
{
    mpfi_sub_ui(mp,mp,v);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator-=(const long int v)
{
    mpfi_sub_si(mp,mp,v);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator-=(const int v)
{
    mpfi_sub_si(mp,mp,v);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline const mpireal mpireal::operator-()const
{
    mpireal u(*this);
    mpfi_neg(u.mp,u.mp);
    return u;
}

inline const mpireal operator-(const mpireal& a, const mpireal& b)
{
        mpireal c(0, (std::max)(mpfi_get_prec(a.mpfi_ptr()), mpfi_get_prec(b.mpfi_ptr())));
        mpfi_sub(c.mpfi_ptr(), a.mpfi_srcptr(), b.mpfi_srcptr());
	return c;
}

inline const mpireal operator-(const double  b, const mpireal& a)
{
#if (MPFI_VERSION >= MPFI_VERSION_NUM(1,5,1))
    mpireal x(0, mpfi_get_prec(a.mpfi_ptr()));
    mpfi_d_sub(x.mpfi_ptr(), b, a.mpfi_srcptr());
    return x;
#else
    mpireal x(b, mpfi_get_prec(a.mpfi_ptr()));
    x -= a;
    return x;
#endif
}

inline const mpireal operator-(const unsigned long int b, const mpireal& a)
{
    mpireal x(0, mpfi_get_prec(a.mpfi_ptr()));
    mpfi_ui_sub(x.mpfi_ptr(), b, a.mpfi_srcptr());
    return x;
}

inline const mpireal operator-(const unsigned int b, const mpireal& a)
{
    mpireal x(0, mpfi_get_prec(a.mpfi_ptr()));
    mpfi_ui_sub(x.mpfi_ptr(), b, a.mpfi_srcptr());
    return x;
}

inline const mpireal operator-(const long int b, const mpireal& a)
{
    mpireal x(0, mpfi_get_prec(a.mpfi_ptr()));
    mpfi_si_sub(x.mpfi_ptr(), b, a.mpfi_srcptr());
    return x;
}

inline const mpireal operator-(const int b, const mpireal& a)
{
    mpireal x(0, mpfi_get_prec(a.mpfi_ptr()));
    mpfi_si_sub(x.mpfi_ptr(), b, a.mpfi_srcptr());
    return x;
}

//////////////////////////////////////////////////////////////////////////
// * Multiplication
inline mpireal& mpireal::operator*= (const mpireal& v)
{
    mpfi_mul(mp,mp,v.mp);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator*=(const mpz_t v)
{
    mpfi_mul_z(mp,mp,v);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator*=(const mpq_t v)
{
    mpfi_mul_q(mp,mp,v);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator*=(const long double v)
{
    *this *= mpireal(v);    
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;    
}

inline mpireal& mpireal::operator*=(const double v)
{
#if (MPFI_VERSION >= MPFI_VERSION_NUM(1,5,1))
    mpfi_mul_d(mp,mp,v);
#else
    *this *= mpireal(v);    
#endif
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator*=(const unsigned long int v)
{
    mpfi_mul_ui(mp,mp,v);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator*=(const unsigned int v)
{
    mpfi_mul_ui(mp,mp,v);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator*=(const long int v)
{
    mpfi_mul_si(mp,mp,v);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator*=(const int v)
{
    mpfi_mul_si(mp,mp,v);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline const mpireal operator*(const mpireal& a, const mpireal& b)
{
        mpireal c(0, (std::max)(mpfi_get_prec(a.mpfi_ptr()), mpfi_get_prec(b.mpfi_ptr())));
        mpfi_mul(c.mpfi_ptr(), a.mpfi_srcptr(), b.mpfi_srcptr());
	return c;
}

//////////////////////////////////////////////////////////////////////////
// / Division
inline mpireal& mpireal::operator/=(const mpireal& v)
{
    mpfi_div(mp,mp,v.mp);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator/=(const mpz_t v)
{
    mpfi_div_z(mp,mp,v);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator/=(const mpq_t v)
{
    mpfi_div_q(mp,mp,v);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator/=(const long double v)
{
    *this /= mpireal(v);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;    
}

inline mpireal& mpireal::operator/=(const double v)
{
#if (MPFI_VERSION >= MPFI_VERSION_NUM(1,5,1))
    mpfi_div_d(mp,mp,v);
#else
    *this /= mpireal(v);    
#endif
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator/=(const unsigned long int v)
{
    mpfi_div_ui(mp,mp,v);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator/=(const unsigned int v)
{
    mpfi_div_ui(mp,mp,v);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator/=(const long int v)
{
    mpfi_div_si(mp,mp,v);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator/=(const int v)
{
    mpfi_div_si(mp,mp,v);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline const mpireal operator/(const mpireal& a, const mpireal& b)
{
        mpireal c(0, (std::max)(mpfi_get_prec(a.mpfi_ptr()), mpfi_get_prec(b.mpfi_ptr())));
        mpfi_div(c.mpfi_ptr(), a.mpfi_srcptr(), b.mpfi_srcptr());
	return c;
}

inline const mpireal operator/(const unsigned long int b, const mpireal& a)
{
    mpireal x(0, mpfi_get_prec(a.mpfi_ptr()));
    mpfi_ui_div(x.mpfi_ptr(), b, a.mpfi_srcptr());
    return x;
}

inline const mpireal operator/(const unsigned int b, const mpireal& a)
{
    mpireal x(0, mpfi_get_prec(a.mpfi_ptr()));
    mpfi_ui_div(x.mpfi_ptr(), b, a.mpfi_srcptr());
    return x;
}

inline const mpireal operator/(const long int b, const mpireal& a)
{
    mpireal x(0, mpfi_get_prec(a.mpfi_ptr()));
    mpfi_si_div(x.mpfi_ptr(), b, a.mpfi_srcptr());
    return x;
}

inline const mpireal operator/(const int b, const mpireal& a)
{
    mpireal x(0, mpfi_get_prec(a.mpfi_ptr()));
    mpfi_si_div(x.mpfi_ptr(), b, a.mpfi_srcptr());
    return x;
}

inline const mpireal operator/(const double  b, const mpireal& a)
{
#if (MPFI_VERSION >= MPFI_VERSION_NUM(1,5,1))
    mpireal x(0, mpfi_get_prec(a.mpfi_ptr()));
    mpfi_d_div(x.mpfi_ptr(), b, a.mpfi_srcptr());
    return x;
#else
    mpireal x(0, mpfi_get_prec(a.mpfi_ptr()));
    x /= a;
    return x;
#endif
}

//////////////////////////////////////////////////////////////////////////
// Shifts operators - Multiplication/Division by power of 2
inline mpireal& mpireal::operator<<=(const unsigned long int u)
{
    mpfi_mul_2ui(mp,mp,u);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator<<=(const unsigned int u)
{
    mpfi_mul_2ui(mp,mp,static_cast<unsigned long int>(u));
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator<<=(const long int u)
{
    mpfi_mul_2si(mp,mp,u);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator<<=(const int u)
{
    mpfi_mul_2si(mp,mp,static_cast<long int>(u));
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator>>=(const unsigned long int u)
{
    mpfi_div_2ui(mp,mp,u);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator>>=(const unsigned int u)
{
    mpfi_div_2ui(mp,mp,static_cast<unsigned long int>(u));
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator>>=(const long int u)
{
    mpfi_div_2si(mp,mp,u);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::operator>>=(const int u)
{
    mpfi_div_2si(mp,mp,static_cast<long int>(u));
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline const mpireal operator<<(const mpireal& v, const unsigned long int k)
{
    return mul_2ui(v,k);
}

inline const mpireal operator<<(const mpireal& v, const unsigned int k)
{
    return mul_2ui(v,static_cast<unsigned long int>(k));
}

inline const mpireal operator<<(const mpireal& v, const long int k)
{
    return mul_2si(v,k);
}

inline const mpireal operator<<(const mpireal& v, const int k)
{
    return mul_2si(v,static_cast<long int>(k));
}

inline const mpireal operator>>(const mpireal& v, const unsigned long int k)
{
    return div_2ui(v,k);
}

inline const mpireal operator>>(const mpireal& v, const long int k)
{
    return div_2si(v,k);
}

inline const mpireal operator>>(const mpireal& v, const unsigned int k)
{
    return div_2ui(v,static_cast<unsigned long int>(k));
}

inline const mpireal operator>>(const mpireal& v, const int k)
{
    return div_2si(v,static_cast<long int>(k));
}

// mul_2ui
inline const mpireal mul_2ui(const mpireal& v, unsigned long int k)
{
    mpireal x(v);
    mpfi_mul_2ui(x.mp,v.mp,k);
    return x;
}

// mul_2si
inline const mpireal mul_2si(const mpireal& v, long int k)
{
    mpireal x(v);
    mpfi_mul_2si(x.mp,v.mp,k);
    return x;
}

inline const mpireal div_2ui(const mpireal& v, unsigned long int k)
{
    mpireal x(v);
    mpfi_div_2ui(x.mp,v.mp,k);
    return x;
}

inline const mpireal div_2si(const mpireal& v, long int k)
{
    mpireal x(v);
    mpfi_div_2si(x.mp,v.mp,k);
    return x;
}

//////////////////////////////////////////////////////////////////////////
//Boolean operators
inline bool operator >  (const mpireal& a, const mpireal& b){    return (mpfi_greater_p(a.mp,b.mp)      !=0);    }
inline bool operator >= (const mpireal& a, const mpireal& b){    return (mpfi_greaterequal_p(a.mp,b.mp) !=0);    }
inline bool operator <  (const mpireal& a, const mpireal& b){    return (mpfi_less_p(a.mp,b.mp)         !=0);    }
inline bool operator <= (const mpireal& a, const mpireal& b){    return (mpfi_lessequal_p(a.mp,b.mp)    !=0);    }
inline bool operator == (const mpireal& a, const mpireal& b){    return (mpfi_equal_p(a.mp,b.mp)        !=0);    }
inline bool operator != (const mpireal& a, const mpireal& b){    return (mpfi_lessgreater_p(a.mp,b.mp)  !=0);    }

inline bool operator == (const mpireal& a, const unsigned long int b ){    return (mpfi_cmp_ui(a.mp,b) == 0);    }
inline bool operator == (const mpireal& a, const unsigned int b      ){    return (mpfi_cmp_ui(a.mp,b) == 0);    }
inline bool operator == (const mpireal& a, const long int b          ){    return (mpfi_cmp_si(a.mp,b) == 0);    }
inline bool operator == (const mpireal& a, const int b               ){    return (mpfi_cmp_si(a.mp,b) == 0);    }
inline bool operator == (const mpireal& a, const long double b       ){    return (mpfi_cmp_ld(a.mp,b) == 0);    }
inline bool operator == (const mpireal& a, const double b            ){    return (mpfi_cmp_d(a.mp,b)  == 0);    }


inline bool isnan    (const mpireal& v){    return (mpfi_nan_p(v.mp)     != 0);    }
inline bool isinf    (const mpireal& v){    return (mpfi_inf_p(v.mp)     != 0);    }
inline bool isfinite (const mpireal& v){    return (mpfi_number_p(v.mp)  != 0);    }
inline bool iszero   (const mpireal& v){    return (mpfi_zero_p(v.mp)    != 0);    }
inline bool isint    (const mpireal& v){    return (mpfi_integer_p(v.mp) != 0);    }

#if (MPFI_VERSION >= MPFI_VERSION_NUM(3,0,0))
inline bool isregular(const mpireal& v){    return (mpfi_regular_p(v.mp));}
#endif 

//////////////////////////////////////////////////////////////////////////
// Type Converters
inline long             mpireal::toLong   ()  const    {    return mpfi_get_si(mp);    }
inline unsigned long    mpireal::toULong  ()  const    {    return mpfi_get_ui(mp);    }
inline double           mpireal::toDouble ()  const    {    return mpfi_get_d (mp);    }
inline long double      mpireal::toLDouble()  const    {    return mpfi_get_ld(mp);    }

#if defined (MPIREAL_HAVE_INT64_SUPPORT)
inline int64_t      mpireal::toInt64 ()    const{    return mpfi_get_sj(mp);    }
inline uint64_t     mpireal::toUInt64()    const{    return mpfi_get_uj(mp);    }
#endif

inline ::mpfi_ptr     mpireal::mpfi_ptr()             { return mp; }
inline ::mpfi_srcptr  mpireal::mpfi_ptr()    const    { return mp; }
inline ::mpfi_srcptr  mpireal::mpfi_srcptr() const    { return mp; }

template <class T>
inline std::string toString(T t, std::ios_base & (*f)(std::ios_base&))
{
    std::ostringstream oss;
    oss << f << t;
    return oss.str();
}

#if (MPFI_VERSION >= MPFI_VERSION_NUM(1,5,1))

inline std::string mpireal::toString(const std::string& format) const
{
    char *sl = NULL;
    char *sr = NULL;

    std::string out;

    if( !format.empty() )
    {
        if (!(mpfr_asprintf(&sl,format.c_str(),mp->left) < 0)
        && !(mpfr_asprintf(&sr,format.c_str(),mp->right) < 0))
        {
            out = std::string(sl)+std::string(sr);

            mpfr_free_str(sl);
            mpfr_free_str(sr);
        }
    }

    return out;
}

#endif

inline std::string mpireal::toString(int n, int b, mp_rnd_t mode) const
{
    (void)b;
    (void)mode;

#if (MPFI_VERSION >= MPFI_VERSION_NUM(1,5,1))

    // Use MPFR native function for output
    char format[128];
    int digits;

    digits = n > 0 ? n : bits2digits(mpfi_get_prec(mp));

    sprintf(format,"%%.%dRNg",digits);        // Default format

    return toString(std::string(format));

#else

    char *s, *ns = NULL; 
    size_t slen, nslen;
    mp_exp_t exp;
    std::string out;

    if(mpfi_inf_p(mp))
    { 
        if(mpfi_sgn(mp)>0) return "+Inf";
        else               return "-Inf";
    }

    if(mpfi_zero_p(mp)) return "0";
    if(mpfi_nan_p(mp))  return "NaN";

    s  = mpfi_get_str(NULL,&exp,b,0,mp,mode);
    ns = mpfi_get_str(NULL,&exp,b,n,mp,mode);

    if(s!=NULL && ns!=NULL)
    {
        slen  = strlen(s);
        nslen = strlen(ns);
        if(nslen<=slen) 
        {
            mpfi_free_str(s);
            s = ns;
            slen = nslen;
        }
        else {
            mpfi_free_str(ns);
        }

        // Make human eye-friendly formatting if possible
        if (exp>0 && static_cast<size_t>(exp)<slen)
        {
            if(s[0]=='-')
            {
                // Remove zeros starting from right end
                char* ptr = s+slen-1;
                while (*ptr=='0' && ptr>s+exp) ptr--; 

                if(ptr==s+exp) out = std::string(s,exp+1);
                else           out = std::string(s,exp+1)+'.'+std::string(s+exp+1,ptr-(s+exp+1)+1);

                //out = string(s,exp+1)+'.'+string(s+exp+1);
            }
            else
            {
                // Remove zeros starting from right end
                char* ptr = s+slen-1;
                while (*ptr=='0' && ptr>s+exp-1) ptr--; 

                if(ptr==s+exp-1) out = std::string(s,exp);
                else             out = std::string(s,exp)+'.'+std::string(s+exp,ptr-(s+exp)+1);

                //out = string(s,exp)+'.'+string(s+exp);
            }

        }else{ // exp<0 || exp>slen
            if(s[0]=='-')
            {
                // Remove zeros starting from right end
                char* ptr = s+slen-1;
                while (*ptr=='0' && ptr>s+1) ptr--; 

                if(ptr==s+1) out = std::string(s,2);
                else         out = std::string(s,2)+'.'+std::string(s+2,ptr-(s+2)+1);

                //out = string(s,2)+'.'+string(s+2);
            }
            else
            {
                // Remove zeros starting from right end
                char* ptr = s+slen-1;
                while (*ptr=='0' && ptr>s) ptr--; 

                if(ptr==s) out = std::string(s,1);
                else       out = std::string(s,1)+'.'+std::string(s+1,ptr-(s+1)+1);

                //out = string(s,1)+'.'+string(s+1);
            }

            // Make final string
            if(--exp)
            {
                if(exp>0) out += "e+"+mpfi::toString<mp_exp_t>(exp,std::dec);
                else       out += "e"+mpfi::toString<mp_exp_t>(exp,std::dec);
            }
        }

        mpfi_free_str(s);
        return out;
    }else{
        return "conversion error!";
    }
#endif
}


//////////////////////////////////////////////////////////////////////////
// I/O
inline std::ostream& operator<<(std::ostream& os, const mpireal& v)
{
    return os<<v.toString(static_cast<int>(os.precision()));
}

inline std::istream& operator>>(std::istream &is, mpireal& v)
{
    // ToDo, use cout::hexfloat and other flags to setup base
    std::string tmp;
    is >> tmp;
    mpfi_set_str(v.mp, tmp.c_str(), 10);
    return is;
}

//////////////////////////////////////////////////////////////////////////
//     Bits - decimal digits relation
//        bits   = ceil(digits*log[2](10))
//        digits = floor(bits*log[10](2))

inline mp_prec_t digits2bits(int d)
{
    const double LOG2_10 = 3.3219280948873624;

    return (mp_prec_t) std::ceil( d * LOG2_10 );
}

inline int bits2digits(mp_prec_t b)
{
    const double LOG10_2 = 0.30102999566398119;

    return (int) std::floor( b * LOG10_2 );
}

//////////////////////////////////////////////////////////////////////////
// Set/Get number properties
inline int sgn(const mpireal& v)
{
    int l = mpfr_signbit((&v.mp->left));
    int r = mpfr_signbit((&v.mp->right));
    if (l==r) return (r>0?-1:1);
    return 0;
}

inline mpireal& mpireal::setSign(int sign, mp_rnd_t RoundingMode)
{
    mpfr_setsign(&mp->left,&mp->left,(sign < 0 ? 1 : 0),RoundingMode);
    mpfr_setsign(&mp->right,&mp->right,(sign < 0 ? 1 : 0),RoundingMode);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline int mpireal::getPrecision() const
{
    return mpfi_get_prec(mp);
}

inline mpireal& mpireal::setPrecision(int Precision)
{
    mpfi_set_prec(mp, Precision);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal& mpireal::setInf(int sign) 
{ 
    mpfi_set_inf(mp,sign);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}    

inline mpireal& mpireal::setNan() 
{
    mpfi_set_nan(mp);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpireal&    mpireal::setZero(int sign)
{

#if (MPFI_VERSION >= MPFI_VERSION_NUM(3,0,0))
    mpfi_set_zero(mp, sign);
#else
    mpfi_set_si(mp, 0);
    setSign(sign);
#endif 

    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mp_prec_t mpireal::get_prec() const
{
    return mpfi_get_prec(mp);
}

inline void mpireal::set_prec(mp_prec_t prec)
{
    mpfr_prec_round(&mp->left,prec,MPFR_RNDD);
    mpfr_prec_round(&mp->right,prec,MPFR_RNDU);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
}

#if (MPFI_VERSION >= MPFI_VERSION_NUM(3,0,0))
inline mp_exp_t mpireal::get_exp ()
{
    return mpfi_get_exp(mp);
}

inline int mpireal::set_exp (mp_exp_t e)
{
    int x = mpfi_set_exp(mp, e);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return x;
}

inline const mpireal frexp(const mpireal& v, mp_exp_t* exp)
{
    mpireal x(v);
    *exp = x.get_exp();
    x.set_exp(0);
    return x;
}
#endif

inline const mpireal ldexp(const mpireal& v, mp_exp_t exp)
{
    mpireal x(v);

    // rounding is not important since we just increasing the exponent
    mpfi_mul_2si(x.mp,x.mp,exp);
    return x;
}

inline mpireal machine_epsilon(mp_prec_t prec)
{
    /* the smallest eps such that 1 + eps != 1 */
    return machine_epsilon(mpireal(1, prec));
}

inline mpireal machine_epsilon(const mpireal& x)
{    
    /* the smallest eps such that x + eps != x */
    if ( sgn(x) < 0)
    {
        return nextabove(-x)+x;
    }else{
        return nextabove(x)-x;
    }
}

// minval is 'safe' meaning 1 / minval does not overflow
inline mpireal minval(mp_prec_t prec)
{
    /* min = 1/2 * 2^emin = 2^(emin - 1) */
    return mpireal(1, prec) << mpireal::get_emin()-1;
}

// maxval is 'safe' meaning 1 / maxval does not underflow
inline mpireal maxval(mp_prec_t prec)
{
    /* max = (1 - eps) * 2^emax, eps is machine epsilon */
    return (mpireal(1, prec) - machine_epsilon(prec)) << mpireal::get_emax(); 
}

inline bool isEqualUlps(const mpireal& a, const mpireal& b, int maxUlps)
{
  return abs(a - b) <= machine_epsilon((max)(abs(a), abs(b))) * maxUlps;
}

inline bool isEqualFuzzy(const mpireal& a, const mpireal& b, const mpireal& eps)
{
    return abs(a - b) <= (min)(abs(a), abs(b)) * eps;
}

inline bool isEqualFuzzy(const mpireal& a, const mpireal& b)
{
    return isEqualFuzzy(a, b, machine_epsilon((min)(abs(a), abs(b))));
}

#if (MPFI_VERSION >= MPFI_VERSION_NUM(3,0,0))
inline const mpireal modf(const mpireal& v, mpireal& n)
{
    mpireal frac(v);

    // rounding is not important since we are using the same number
    mpfi_frac(frac.mp,frac.mp);
    mpfi_trunc(n.mp,v.mp);
    return frac;
}

inline int mpireal::check_range (int t)
{
    return mpfi_check_range(mp,t);
}

inline int mpireal::subnormalize (int t)
{
    int r = mpfi_subnormalize(mp,t);
    MPIREAL_MSVC_DEBUGVIEW_CODE;
    return r;
}
#endif

inline mp_exp_t mpireal::get_emin (void)
{
    return mpfr_get_emin();
}

inline int mpireal::set_emin (mp_exp_t exp)
{
    return mpfr_set_emin(exp);
}

inline mp_exp_t mpireal::get_emax (void)
{
    return mpfr_get_emax();
}

inline int mpireal::set_emax (mp_exp_t exp)
{
    return mpfr_set_emax(exp);
}

inline mp_exp_t mpireal::get_emin_min (void)
{
    return mpfr_get_emin_min();
}

inline mp_exp_t mpireal::get_emin_max (void)
{
    return mpfr_get_emin_max();
}

inline mp_exp_t mpireal::get_emax_min (void)
{
    return mpfr_get_emax_min();
}

inline mp_exp_t mpireal::get_emax_max (void)
{
    return mpfr_get_emax_max();
}

//////////////////////////////////////////////////////////////////////////
// Mathematical Functions
//////////////////////////////////////////////////////////////////////////
#define MPIREAL_UNARY_MATH_FUNCTION_BODY(f)                    \
        mpireal y(0, mpfi_get_prec(x.mpfi_srcptr()));          \
        mpfi_##f(y.mpfi_ptr(), x.mpfi_srcptr());           \
        return y; 

#define MPIREAL_UNARY_RND_MATH_FUNCTION_BODY(f)                    \
        mpireal y(0, mpfi_get_prec(x.mpfi_srcptr()));          \
        mpfi_##f(y.mpfi_ptr(), x.mpfi_srcptr(),r);           \
        return y;

inline const mpireal sqr  (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(sqr );    }
inline const mpireal sqrt (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(sqrt);    }

inline const mpireal sqrt(const unsigned long int x)
{
    mpireal y;
    mpfr_sqrt_ui(&y.mp->left, x,MPFR_RNDD);
    mpfr_sqrt_ui(&y.mp->right,x,MPFR_RNDU);
    return y;
}

inline const mpireal sqrt(const unsigned int v)
{
    return sqrt(static_cast<unsigned long int>(v));
}

inline const mpireal sqrt(const long int v)
{
    if (v>=0)   return sqrt(static_cast<unsigned long int>(v));
    else        return mpireal().setNan(); // NaN  
}

inline const mpireal sqrt(const int v)
{
    if (v>=0)   return sqrt(static_cast<unsigned long int>(v));
    else        return mpireal().setNan(); // NaN
}

#if (MPFI_VERSION >= MPFI_VERSION_NUM(3,0,0))
inline const mpireal root(const mpireal& x, unsigned long int k)
{
    mpireal y(0, mpfi_get_prec(x.mpfi_srcptr()));
    mpfi_root(y.mpfi_ptr(), x.mpfi_srcptr(), k);
    return y; 
}
#endif

inline const mpireal dim(const mpireal& a, const mpireal& b)
{
    mpireal y(0, mpfi_get_prec(a.mpfi_srcptr()));
    mpfi_dim(y.mpfi_ptr(), a.mpfi_srcptr(), b.mpfi_srcptr());
    return y;
}

inline int cmpabs(const mpireal& a,const mpireal& b)
{
    return mpfi_cmpabs(a.mp,b.mp);
}

#if (MPFI_VERSION >= MPFI_VERSION_NUM(3,0,0))
inline int sin_cos(mpireal& s, mpireal& c, const mpireal& v)
{
    return mpfi_sin_cos(s.mp,c.mp,v.mp);
}
#endif

inline const mpireal sqrt  (const long double v)    {   return sqrt(mpireal(v));    }
inline const mpireal sqrt  (const double v)         {   return sqrt(mpireal(v));    }

inline const mpireal cbrt  (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(cbrt );    }
inline const mpireal fabs  (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(abs  );    }
inline const mpireal abs   (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(abs  );    }
inline const mpireal log   (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(log  );    }
inline const mpireal log2  (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(log2 );    }
inline const mpireal log10 (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(log10);    }
inline const mpireal exp   (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(exp  );    }
inline const mpireal exp2  (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(exp2 );    }
inline const mpireal exp10 (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(exp10);    }
inline const mpireal cos   (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(cos  );    }
inline const mpireal sin   (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(sin  );    }
inline const mpireal tan   (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(tan  );    }
inline const mpireal sec   (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(sec  );    }
inline const mpireal csc   (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(csc  );    }
inline const mpireal cot   (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(cot  );    }
inline const mpireal acos  (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(acos);     }
inline const mpireal asin  (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(asin);     }
inline const mpireal atan  (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(atan);     }

inline const mpireal acot  (const mpireal& v) {   return atan (1/v);                      }
inline const mpireal asec  (const mpireal& v) {   return acos (1/v);                      }
inline const mpireal acsc  (const mpireal& v) {   return asin (1/v);                      }
inline const mpireal acoth (const mpireal& v) {   return atanh(1/v);                      }
inline const mpireal asech (const mpireal& v) {   return acosh(1/v);                      }
inline const mpireal acsch (const mpireal& v) {   return asinh(1/v);                      }

inline const mpireal cosh  (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(cosh );    }
inline const mpireal sinh  (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(sinh );    }
inline const mpireal tanh  (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(tanh );    }
inline const mpireal sech  (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(sech );    }
inline const mpireal csch  (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(csch );    }
inline const mpireal coth  (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(coth );    }
inline const mpireal acosh (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(acosh);    }
inline const mpireal asinh (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(asinh);    }
inline const mpireal atanh (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(atanh);    }

#if (MPFI_VERSION >= MPFI_VERSION_NUM(3,0,0))
    inline const mpireal log1p   (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(log1p  );    }
    inline const mpireal expm1   (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(expm1  );    }
    inline const mpireal eint    (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(eint   );    }
    inline const mpireal gamma   (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(gamma  );    }
    inline const mpireal lngamma (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(lngamma);    }
    inline const mpireal zeta    (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(zeta   );    }
    inline const mpireal erf     (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(erf    );    }
    inline const mpireal erfc    (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(erfc   );    }
    inline const mpireal besselj0(const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(j0     );    }
    inline const mpireal besselj1(const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(j1     );    }
    inline const mpireal bessely0(const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(y0     );    }
    inline const mpireal bessely1(const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(y1     );    }
#endif

inline const mpireal atan2 (const mpireal& y, const mpireal& x)
{
    mpireal a;
    mp_prec_t yp, xp;

    yp = y.get_prec(); 
    xp = x.get_prec(); 

    a.set_prec(yp>xp?yp:xp);

    mpfi_atan2(a.mp, y.mp, x.mp);

    return a;
}

inline const mpireal hypot (const mpireal& x, const mpireal& y)
{
    mpireal a;
    mp_prec_t yp, xp;

    yp = y.get_prec(); 
    xp = x.get_prec(); 

    a.set_prec(yp>xp?yp:xp);

    mpfi_hypot(a.mp, x.mp, y.mp);

    return a;
}

#if (MPFI_VERSION >= MPFI_VERSION_NUM(3,0,0))
inline const mpireal remainder (const mpireal& x, const mpireal& y)
{    
    mpireal a;
    mp_prec_t yp, xp;

    yp = y.get_prec(); 
    xp = x.get_prec(); 

    a.set_prec(yp>xp?yp:xp);

    mpfi_remainder(a.mp, x.mp, y.mp);

    return a;
}

inline const mpireal remquo (long* q, const mpireal& x, const mpireal& y)
{
    mpireal a;
    mp_prec_t yp, xp;

    yp = y.get_prec(); 
    xp = x.get_prec(); 

    a.set_prec(yp>xp?yp:xp);

    mpfi_remquo(a.mp,q, x.mp, y.mp);

    return a;
}

inline const mpireal fac_ui (unsigned long int v, mp_prec_t prec)
{
    mpireal x(0, prec);
    mpfi_fac_ui(x.mp,v);
    return x;
}


inline const mpireal lgamma (const mpireal& v, int *signp)
{
    mpireal x(v);
    int tsignp;

    if(signp)   mpfi_lgamma(x.mp,signp,v.mp);
    else        mpfi_lgamma(x.mp,&tsignp,v.mp);

    return x;
}


inline const mpireal besseljn (long n, const mpireal& x)
{
    mpireal  y(0, mpfi_get_prec(x.mpfi_srcptr()));
    mpfi_jn(y.mpfi_ptr(), n, x.mpfi_srcptr());
    return y;
}

inline const mpireal besselyn (long n, const mpireal& x)
{
    mpireal  y(0, mpfi_get_prec(x.mpfi_srcptr()));
    mpfi_yn(y.mpfi_ptr(), n, x.mpfi_srcptr());
    return y;
}

inline const mpireal fma (const mpireal& v1, const mpireal& v2, const mpireal& v3)
{
    mpireal a;
    mp_prec_t p1, p2, p3;

    p1 = v1.get_prec(); 
    p2 = v2.get_prec(); 
    p3 = v3.get_prec(); 

    a.set_prec(p3>p2?(p3>p1?p3:p1):(p2>p1?p2:p1));

    mpfi_fma(a.mp,v1.mp,v2.mp,v3.mp);
    return a;
}

inline const mpireal fms (const mpireal& v1, const mpireal& v2, const mpireal& v3)
{
    mpireal a;
    mp_prec_t p1, p2, p3;

    p1 = v1.get_prec(); 
    p2 = v2.get_prec(); 
    p3 = v3.get_prec(); 

    a.set_prec(p3>p2?(p3>p1?p3:p1):(p2>p1?p2:p1));

    mpfi_fms(a.mp,v1.mp,v2.mp,v3.mp);
    return a;
}

inline const mpireal agm (const mpireal& v1, const mpireal& v2)
{
    mpireal a;
    mp_prec_t p1, p2;

    p1 = v1.get_prec(); 
    p2 = v2.get_prec(); 

    a.set_prec(p1>p2?p1:p2);

    mpfi_agm(a.mp, v1.mp, v2.mp);

    return a;
}

inline const mpireal sum (const mpireal tab[], unsigned long int n)
{
    mpireal x;
    mpfi_ptr* t;
    unsigned long int i;

    t = new mpfi_ptr[n];
    for (i=0;i<n;i++) t[i] = (mpfi_ptr)tab[i].mp;
    mpfi_sum(x.mp,t,n);
    delete[] t;
    return x;
}
#endif

//////////////////////////////////////////////////////////////////////////
// MPFI 2.4.0 Specifics
#if (MPFI_VERSION >= MPFI_VERSION_NUM(2,4,0))

inline int sinh_cosh(mpireal& s, mpireal& c, const mpireal& v)
{
    return mpfi_sinh_cosh(s.mp,c.mp,v.mp);
}

inline const mpireal li2 (const mpireal& x)
{   
    MPIREAL_UNARY_MATH_FUNCTION_BODY(li2);
}

inline const mpireal rem (const mpireal& x, const mpireal& y)
{
    /*  R = rem(X,Y) if Y != 0, returns X - n * Y where n = trunc(X/Y). */
    return fmod(x, y);
}

inline const mpireal mod (const mpireal& x, const mpireal& y)
{
    /*

    m = mod(x,y) if y != 0, returns x - n*y where n = floor(x/y)

    The following are true by convention:
    - mod(x,0) is x
    - mod(x,x) is 0
    - mod(x,y) for x != y and y != 0 has the same sign as y.    
    
    */

    if(iszero(y)) return x;
    if(x == y) return 0;

    mpireal m = x - floor(x / y) * y;
    
    m.setSign(sgn(y)); // make sure result has the same sign as Y

    return m;
}

inline const mpireal fmod (const mpireal& x, const mpireal& y)
{
    mpireal a;
    mp_prec_t yp, xp;

    yp = y.get_prec(); 
    xp = x.get_prec(); 

    a.set_prec(yp>xp?yp:xp);

    mpfi_fmod(a.mp, x.mp, y.mp);

    return a;
}

inline const mpireal rec_sqrt(const mpireal& v)
{
    mpireal x(v);
    mpfi_rec_sqrt(x.mp,v.mp);
    return x;
}
#endif //  MPFR 2.4.0 Specifics

//////////////////////////////////////////////////////////////////////////
// MPFR 3.0.0 Specifics
#if (MPFI_VERSION >= MPFI_VERSION_NUM(3,0,0))
inline const mpireal digamma (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(digamma);     }
inline const mpireal ai      (const mpireal& x) {   MPIREAL_UNARY_MATH_FUNCTION_BODY(ai);          }
#endif // MPFR 3.0.0 Specifics

//////////////////////////////////////////////////////////////////////////
// Constants
inline const mpireal const_log2 (mp_prec_t p)
{
    mpireal x(0, p);
    mpfi_const_log2(x.mpfi_ptr());
    return x;
}

inline const mpireal const_pi (mp_prec_t p)
{
    mpireal x(0, p);
    mpfi_const_pi(x.mpfi_ptr());
    return x;
}

inline const mpireal const_euler (mp_prec_t p)
{
    mpireal x(0, p);
    mpfi_const_euler(x.mpfi_ptr());
    return x;
}

inline const mpireal const_catalan (mp_prec_t p)
{
    mpireal x(0, p);
    mpfi_const_catalan(x.mpfi_ptr());
    return x;
}

inline const mpireal const_infinity (int sign, mp_prec_t p)
{
    mpireal x(0, p);
    mpfi_set_inf(x.mpfi_ptr(), sign);
    return x;
}

//////////////////////////////////////////////////////////////////////////
// Integer Related Functions
inline const mpireal ceil(const mpireal& v)
{
    mpireal x(v);
    mpfi_ceil(x.mp,v.mp);
    return x;
}

inline const mpireal floor(const mpireal& v)
{
    mpireal x(v);
    mpfi_floor(x.mp,v.mp);
    return x;
}

inline const mpireal round(const mpireal& v)
{
    mpireal x(v);
    mpfi_round(x.mp,v.mp);
    return x;
}

inline const mpireal trunc(const mpireal& v)
{
    mpireal x(v);
    mpfi_trunc(x.mp,v.mp);
    return x;
}

inline const mpireal rint       (const mpireal& x)                                         {   MPIREAL_UNARY_MATH_FUNCTION_BODY(rint      );     }
inline const mpireal rint_ceil  (const mpireal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPIREAL_UNARY_RND_MATH_FUNCTION_BODY(rint_ceil );     }
inline const mpireal rint_floor (const mpireal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPIREAL_UNARY_RND_MATH_FUNCTION_BODY(rint_floor);     }
inline const mpireal rint_round (const mpireal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPIREAL_UNARY_RND_MATH_FUNCTION_BODY(rint_round);     }
inline const mpireal rint_trunc (const mpireal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPIREAL_UNARY_RND_MATH_FUNCTION_BODY(rint_trunc);     }

#if (MPFI_VERSION >= MPFI_VERSION_NUM(3,0,0))
inline const mpireal frac       (const mpireal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPIREAL_UNARY_RND_MATH_FUNCTION_BODY(frac      );     }
#endif

//////////////////////////////////////////////////////////////////////////
// Miscellaneous Functions
inline void         swap (mpireal& a, mpireal& b)             {    mpfi_swap(a.mp,b.mp);   }
inline const mpireal (max)(const mpireal& x, const mpireal& y){    return (x>y?x:y);       }
inline const mpireal (min)(const mpireal& x, const mpireal& y){    return (x<y?x:y);       }

inline const mpireal fmax(const mpireal& x, const mpireal& y)
{
    mpireal a;
    mpfi_max(a.mp,x.mp,y.mp);
    return a;
}

inline const mpireal fmin(const mpireal& x, const mpireal& y)
{
    mpireal a;
    mpfi_min(a.mp,x.mp,y.mp);
    return a;
}

inline const mpireal nexttoward (const mpireal& x, const mpireal& y)
{
    mpireal a(x);
    mpfi_nexttoward(a.mp,y.mp);
    return a;
}

inline const mpireal nextabove  (const mpireal& x)
{
    mpireal a(x);
    mpfi_nextabove(a.mp);
    return a;
}

inline const mpireal nextbelow  (const mpireal& x)
{
    mpireal a(x);
    mpfi_nextbelow(a.mp);
    return a;
}

inline const mpreal urandomb (gmp_randstate_t& state)
{
    mpreal x;
    mpireal y;
    mpfi_urandom(x.mpfr_ptr(),y.mp,state);
    return x;
}

#if (MPFI_VERSION >= MPFI_VERSION_NUM(3,0,0))
// use gmp_randinit_default() to init state, gmp_randclear() to clear
inline const mpireal urandom (gmp_randstate_t& state)
{
    mpireal x;
    mpfi_urandom(x.mp,state);
    return x;
}
#endif 

// Uniformly distributed random number generation
// a = random(seed); <- initialization & first random number generation
// a = random();     <- next random numbers generation
// seed != 0
inline const mpireal random(unsigned int seed)
{

#if (MPFI_VERSION >= MPFI_VERSION_NUM(3,0,0))
    static gmp_randstate_t state;
    static bool isFirstTime = true;

    if(isFirstTime)
    {
        gmp_randinit_default(state);
        gmp_randseed_ui(state,0);
        isFirstTime = false;
    }

    if(seed != 0)    gmp_randseed_ui(state,seed);

    return mpfi::urandom(state);
#else
    if(seed != 0)    std::srand(seed);
    return mpfi::mpireal(std::rand()/(double)RAND_MAX);
#endif

}

//////////////////////////////////////////////////////////////////////////
// Set/Get global properties
inline void mpireal::set_default_prec(mp_prec_t prec)
{ 
    mpfr_set_default_prec(prec);
}

inline void mpireal::set_default_rnd(mp_rnd_t rnd_mode)
{ 
    mpfr_set_default_rounding_mode(rnd_mode);
}

inline bool mpireal::fits_in_bits(double x, int n)
{   
    int i;
    double t;
    return IsInf(x) || (std::modf ( std::ldexp ( std::frexp ( x, &i ), n ), &t ) == 0.0);
}

inline const mpireal pow(const mpireal& a, const mpireal& b)
{
    mpireal x(a);
    mpfi_pow(x.mp,x.mp,b.mp);
    return x;
}

inline const mpireal pow(const mpireal& a, const mpz_t b)
{
    mpireal x(b);
    mpfi_pow(x.mp,a.mp,x.mp);
    return x;
}

inline const mpireal pow(const mpireal& a, const unsigned long int b)
{
    mpireal x(a);
    mpfi_pow_ui(x.mp,x.mp,b);
    return x;
}

inline const mpireal pow(const mpireal& a, const unsigned int b)
{
    return pow(a,static_cast<unsigned long int>(b));
}

inline const mpireal pow(const mpireal& a, const long int b)
{
    mpireal x(a);
    mpfi_pow_si(x.mp,x.mp,b);
    return x;
}

inline const mpireal pow(const mpireal& a, const int b)
{
    return pow(a,static_cast<long int>(b));
}

inline const mpireal pow(const mpireal& a, const long double b)
{
    return pow(a,mpireal(b));
}

inline const mpireal pow(const mpireal& a, const double b)
{
    return pow(a,mpireal(b));
}

inline const mpireal pow(const unsigned long int a, const mpireal& b)
{
    mpireal x(a);
    mpfi_ui_pow(x.mp,a,b.mp);
    return x;
}

inline const mpireal pow(const unsigned int a, const mpireal& b)
{
    return pow(static_cast<unsigned long int>(a),b);
}

inline const mpireal pow(const long int a, const mpireal& b)
{
    if (a>=0)   return pow(static_cast<unsigned long int>(a),b);
    else        return pow(mpireal(a),b);
}

inline const mpireal pow(const int a, const mpireal& b)
{
    if (a>=0)   return pow(static_cast<unsigned long int>(a),b);
    else        return pow(mpireal(a),b);
}

inline const mpireal pow(const long double a, const mpireal& b)
{
    return pow(mpireal(a),b);
}

inline const mpireal pow(const double a, const mpireal& b)
{
    return pow(mpireal(a),b);
}

// pow unsigned long int
inline const mpireal pow(const unsigned long int a, const unsigned long int b)
{
    mpireal(x);
    mpfr_ui_pow_ui(&x.mp->left,a,b,MPFR_RNDD);
    mpfr_ui_pow_ui(&x.mp->right,a,b,MPFR_RNDU);
    return x;
}

inline const mpireal pow(const unsigned long int a, const unsigned int b)
{
    return pow(a,static_cast<unsigned long int>(b)); //mpfi_ui_pow_ui
}

inline const mpireal pow(const unsigned long int a, const long int b)
{
    if(b>0)    return pow(a,static_cast<unsigned long int>(b)); //mpfi_ui_pow_ui
    else    return pow(a,mpireal(b)); //mpfi_ui_pow
}

inline const mpireal pow(const unsigned long int a, const int b)
{
    if(b>0)    return pow(a,static_cast<unsigned long int>(b)); //mpfi_ui_pow_ui
    else    return pow(a,mpireal(b)); //mpfi_ui_pow
}

inline const mpireal pow(const unsigned long int a, const long double b)
{
    return pow(a,mpireal(b)); //mpfi_ui_pow
}

inline const mpireal pow(const unsigned long int a, const double b)
{
    return pow(a,mpireal(b)); //mpfi_ui_pow
}

// pow unsigned int
inline const mpireal pow(const unsigned int a, const unsigned long int b)
{
    return pow(static_cast<unsigned long int>(a),b); //mpfi_ui_pow_ui
}

inline const mpireal pow(const unsigned int a, const unsigned int b)
{
    return pow(static_cast<unsigned long int>(a),static_cast<unsigned long int>(b)); //mpfi_ui_pow_ui
}

inline const mpireal pow(const unsigned int a, const long int b)
{
    if(b>0) return pow(static_cast<unsigned long int>(a),static_cast<unsigned long int>(b)); //mpfi_ui_pow_ui
    else    return pow(static_cast<unsigned long int>(a),mpireal(b)); //mpfi_ui_pow
}

inline const mpireal pow(const unsigned int a, const int b)
{
    if(b>0) return pow(static_cast<unsigned long int>(a),static_cast<unsigned long int>(b)); //mpfi_ui_pow_ui
    else    return pow(static_cast<unsigned long int>(a),mpireal(b)); //mpfi_ui_pow
}

inline const mpireal pow(const unsigned int a, const long double b)
{
    return pow(static_cast<unsigned long int>(a),mpireal(b)); //mpfi_ui_pow
}

inline const mpireal pow(const unsigned int a, const double b)
{
    return pow(static_cast<unsigned long int>(a),mpireal(b)); //mpfi_ui_pow
}

// pow long int
inline const mpireal pow(const long int a, const unsigned long int b)
{
    if (a>0) return pow(static_cast<unsigned long int>(a),b); //mpfi_ui_pow_ui
    else     return pow(mpireal(a),b); //mpfi_pow_ui
}

inline const mpireal pow(const long int a, const unsigned int b)
{
    if (a>0) return pow(static_cast<unsigned long int>(a),static_cast<unsigned long int>(b));  //mpfi_ui_pow_ui
    else     return pow(mpireal(a),static_cast<unsigned long int>(b)); //mpfi_pow_ui
}

inline const mpireal pow(const long int a, const long int b)
{
    if (a>0)
    {
        if(b>0) return pow(static_cast<unsigned long int>(a),static_cast<unsigned long int>(b)); //mpfi_ui_pow_ui
        else    return pow(static_cast<unsigned long int>(a),mpireal(b)); //mpfi_ui_pow
    }else{
        return pow(mpireal(a),b); // mpfi_pow_si
    }
}

inline const mpireal pow(const long int a, const int b)
{
    if (a>0)
    {
        if(b>0) return pow(static_cast<unsigned long int>(a),static_cast<unsigned long int>(b)); //mpfi_ui_pow_ui
        else    return pow(static_cast<unsigned long int>(a),mpireal(b)); //mpfi_ui_pow
    }else{
        return pow(mpireal(a),static_cast<long int>(b)); // mpfi_pow_si
    }
}

inline const mpireal pow(const long int a, const long double b)
{
    if (a>=0)   return pow(static_cast<unsigned long int>(a),mpireal(b)); //mpfi_ui_pow
    else        return pow(mpireal(a),mpireal(b)); //mpfi_pow
}

inline const mpireal pow(const long int a, const double b)
{
    if (a>=0)   return pow(static_cast<unsigned long int>(a),mpireal(b)); //mpfi_ui_pow
    else        return pow(mpireal(a),mpireal(b)); //mpfi_pow
}

// pow int
inline const mpireal pow(const int a, const unsigned long int b)
{
    if (a>0) return pow(static_cast<unsigned long int>(a),b); //mpfi_ui_pow_ui
    else     return pow(mpireal(a),b); //mpfi_pow_ui
}

inline const mpireal pow(const int a, const unsigned int b)
{
    if (a>0) return pow(static_cast<unsigned long int>(a),static_cast<unsigned long int>(b));  //mpfi_ui_pow_ui
    else     return pow(mpireal(a),static_cast<unsigned long int>(b)); //mpfi_pow_ui
}

inline const mpireal pow(const int a, const long int b)
{
    if (a>0)
    {
        if(b>0) return pow(static_cast<unsigned long int>(a),static_cast<unsigned long int>(b)); //mpfi_ui_pow_ui
        else    return pow(static_cast<unsigned long int>(a),mpireal(b)); //mpfi_ui_pow
    }else{
        return pow(mpireal(a),b); // mpfi_pow_si
    }
}

inline const mpireal pow(const int a, const int b)
{
    if (a>0)
    {
        if(b>0) return pow(static_cast<unsigned long int>(a),static_cast<unsigned long int>(b)); //mpfi_ui_pow_ui
        else    return pow(static_cast<unsigned long int>(a),mpireal(b)); //mpfi_ui_pow
    }else{
        return pow(mpireal(a),static_cast<long int>(b)); // mpfi_pow_si
    }
}

inline const mpireal pow(const int a, const long double b)
{
    if (a>=0)   return pow(static_cast<unsigned long int>(a),mpireal(b)); //mpfi_ui_pow
    else        return pow(mpireal(a),mpireal(b)); //mpfi_pow
}

inline const mpireal pow(const int a, const double b)
{
    if (a>=0)   return pow(static_cast<unsigned long int>(a),mpireal(b)); //mpfi_ui_pow
    else        return pow(mpireal(a),mpireal(b)); //mpfi_pow
}

// pow long double 
inline const mpireal pow(const long double a, const long double b)
{
    return pow(mpireal(a),mpireal(b));
}

inline const mpireal pow(const long double a, const unsigned long int b)
{
    return pow(mpireal(a),b); //mpfi_pow_ui
}

inline const mpireal pow(const long double a, const unsigned int b)
{
    return pow(mpireal(a),static_cast<unsigned long int>(b)); //mpfi_pow_ui
}

inline const mpireal pow(const long double a, const long int b)
{
    return pow(mpireal(a),b); // mpfi_pow_si
}

inline const mpireal pow(const long double a, const int b)
{
    return pow(mpireal(a),static_cast<long int>(b)); // mpfi_pow_si
}

inline const mpireal pow(const double a, const double b)
{
    return pow(mpireal(a),mpireal(b));
}

inline const mpireal pow(const double a, const unsigned long int b)
{
    return pow(mpireal(a),b); // mpfi_pow_ui
}

inline const mpireal pow(const double a, const unsigned int b)
{
    return pow(mpireal(a),static_cast<unsigned long int>(b)); // mpfi_pow_ui
}

inline const mpireal pow(const double a, const long int b)
{
    return pow(mpireal(a),b); // mpfi_pow_si
}

inline const mpireal pow(const double a, const int b)
{
    return pow(mpireal(a),static_cast<long int>(b)); // mpfi_pow_si
}
} // End of mpfi namespace

// Explicit specialization of std::swap for mpireal numbers
// Thus standard algorithms will use efficient version of swap (due to Koenig lookup)
// Non-throwing swap C++ idiom: http://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Non-throwing_swap
namespace std
{
	// only allowed to extend namespace std with specializations
    template <>
    inline void swap(mpfi::mpireal& x, mpfi::mpireal& y)
    { 
        return mpfi::swap(x, y);
    }

    template<>
    class numeric_limits<mpfi::mpireal>
    {
    public:
        static const bool is_specialized    = true;
        static const bool is_signed         = true;
        static const bool is_integer        = false;
        static const bool is_exact          = false;
        static const int  radix             = 2;    

        static const bool has_infinity      = true;
        static const bool has_quiet_NaN     = true;
        static const bool has_signaling_NaN = true;

        static const bool is_iec559         = true;        // = IEEE 754
        static const bool is_bounded        = true;
        static const bool is_modulo         = false;
        static const bool traps             = true;
        static const bool tinyness_before   = true;

        static const float_denorm_style has_denorm  = denorm_absent;
        
        inline static float_round_style round_style()
        {
            mp_rnd_t r = mpfi::mpireal::get_default_rnd();

            switch (r)
            {
                case MPFR_RNDN: return round_to_nearest;
                case MPFR_RNDZ: return round_toward_zero; 
                case MPFR_RNDU: return round_toward_infinity; 
                case MPFR_RNDD: return round_toward_neg_infinity; 
                default: return round_indeterminate;
            }
        }

        inline static mpfi::mpireal (min)    (mp_prec_t precision = mpfi::mpireal::get_default_prec()) {  return  mpfi::minval(precision);  }
        inline static mpfi::mpireal (max)    (mp_prec_t precision = mpfi::mpireal::get_default_prec()) {  return  mpfi::maxval(precision);  }
        inline static mpfi::mpireal lowest   (mp_prec_t precision = mpfi::mpireal::get_default_prec()) {  return -mpfi::maxval(precision);  }

        // Returns smallest eps such that 1 + eps != 1 (classic machine epsilon)
        inline static mpfi::mpireal epsilon(mp_prec_t precision = mpfi::mpireal::get_default_prec()) {  return  mpfi::machine_epsilon(precision); }
		
        // Returns smallest eps such that x + eps != x (relative machine epsilon)
        inline static mpfi::mpireal epsilon(const mpfi::mpireal& x) {  return mpfi::machine_epsilon(x);  }

        inline static mpfi::mpireal round_error(mp_prec_t precision = mpfi::mpireal::get_default_prec())
        {
            mp_rnd_t r = mpfi::mpireal::get_default_rnd();

            if(r == MPFR_RNDN) return mpfi::mpireal(0.5, precision);
            else               return mpfi::mpireal(1.0, precision);
        }

        inline static const mpfi::mpireal infinity()         { return mpfi::const_infinity();     }
        inline static const mpfi::mpireal quiet_NaN()        { return mpfi::mpireal().setNan();    }
        inline static const mpfi::mpireal signaling_NaN()    { return mpfi::mpireal().setNan();    }
        inline static const mpfi::mpireal denorm_min()       { return (min)();                    }

        // Please note, exponent range is not fixed in MPFR
        static const int min_exponent = MPFR_EMIN_DEFAULT;
        static const int max_exponent = MPFR_EMAX_DEFAULT;
        MPIREAL_PERMISSIVE_EXPR static const int min_exponent10 = (int) (MPFR_EMIN_DEFAULT * 0.3010299956639811); 
        MPIREAL_PERMISSIVE_EXPR static const int max_exponent10 = (int) (MPFR_EMAX_DEFAULT * 0.3010299956639811); 

        // Should be constant according to standard, but 'digits' depends on precision in MPFR

        inline static int digits()                        {    return mpfi::mpireal::get_default_prec();    }
        inline static int digits(const mpfi::mpireal& x)   {    return x.getPrecision();                    }

        inline static int digits10(mp_prec_t precision = mpfi::mpireal::get_default_prec())
        {
            return mpfi::bits2digits(precision);
        }

        inline static int digits10(const mpfi::mpireal& x)
        {
            return mpfi::bits2digits(x.getPrecision());
        }

        inline static int max_digits10(mp_prec_t precision = mpfi::mpireal::get_default_prec())
        {
            return digits10(precision);
        }
    };

}

#endif /* __MPIREAL_H__ */
