//Authors: Dario Cattaruzza, Alessandro Abate, Peter Schrammel, Daniel Kroening
//University of Oxford 2016
//This code is supplied under the BSD license agreement (see license.txt)

#ifndef SCALAR_H
#define SCALAR_H

#define EIGEN_DONT_ALIGN_STATICALLY

#include <limits>
#include <cstdlib>
#include <complex>

#include <mpreal.h>
#include <math.h>
#include <boost/numeric/interval.hpp>

#include <Eigen/Dense>
#include "MPRealSupport"

using std::abs;
#define UNUSED(x) (void)(x)

struct mpreal_rounding_control
{
  typedef int rounding_mode;
  static void get_rounding_mode(rounding_mode&);
  static void set_rounding_mode(rounding_mode rnd);
  static void upward();
  static void downward();
  static void to_nearest();
  static const mpfr::mpreal& to_int(const mpfr::mpreal& x);
  static const mpfr::mpreal& force_rounding(const mpfr::mpreal& x);
};

typedef boost::numeric::interval_lib::rounded_arith_std<mpfr::mpreal,
        boost::numeric::interval_lib::rounded_transc_std<mpfr::mpreal,mpreal_rounding_control> > mp_rnd_policy;
typedef boost::numeric::interval_lib::policies<
        boost::numeric::interval_lib::save_state<mp_rnd_policy>,
        boost::numeric::interval_lib::checking_base<mpfr::mpreal> > mp_interval_policies;

typedef boost::numeric::interval_lib::policies<boost::numeric::interval_lib::save_state<boost::numeric::interval_lib::rounded_transc_std<long double> >,
        boost::numeric::interval_lib::checking_base<long double> > ld_interval_policies;

typedef boost::numeric::interval<mpfr::mpreal, mp_interval_policies > mpinterval;
typedef boost::numeric::interval<long double, ld_interval_policies > ldinterval;

template <typename scalar,typename refScalar,typename intervalScalar,typename powerScalar> class type_def {
public:
    typedef refScalar               scalar_type;
    typedef intervalScalar          scalar_interval;
    typedef powerScalar             power_type;
    typedef refScalar               ref;
    typedef power_type              power;
    typedef std::complex<scalar>    complexS;
    typedef std::complex<ref>       complexR;
    typedef Eigen::Matrix<scalar,Eigen::Dynamic,Eigen::Dynamic>      MatrixS;
    typedef Eigen::Matrix<ref,Eigen::Dynamic,Eigen::Dynamic>         MatrixR;
    typedef Eigen::Matrix<complexS,Eigen::Dynamic,Eigen::Dynamic>    MatrixC;
    typedef Eigen::Matrix<complexR,Eigen::Dynamic,Eigen::Dynamic>    MatrixRC;
};

template <class scalar> class interval_def;
#if defined (MPREAL_HAVE_INT64_SUPPORT)
template<> class interval_def<mpfr::mpreal> : public type_def<mpfr::mpreal,mpfr::mpreal,mpinterval,long long>
{
public: static std::string ms_name;
};

template<> class interval_def<mpinterval> : public type_def<mpinterval,mpfr::mpreal,mpinterval,long long>
{
public: static std::string ms_name;
};
#else
template<> class interval_def<mpfr::mpreal> : public type_def<mpfr::mpreal,mpfr::mpreal,mpinterval,long>
{
public: static std::string ms_name;
};

template<> class interval_def<mpinterval> : public type_def<mpinterval,mpfr::mpreal,mpinterval,long>
{
public: static std::string ms_name;
};
#endif
template<> class interval_def<long double> : public type_def<long double,long double,ldinterval,long>
{
public: static std::string ms_name;
};

template<> class interval_def<ldinterval> : public type_def<ldinterval,long double,ldinterval,long>
{
public: static std::string ms_name;
};

typedef enum {eNormalSpace,eEigenSpace,eSingularSpace } space_t;
typedef enum {eComplexDynamics,ePseudoDynamics,eCombinedDynamics } dynamics_t;
typedef enum {eNoInputs,eParametricInputs,eVariableInputs } inputType_t;
typedef enum {eReachTubeSynth,eInitSynth,eInputSynth,eSensitivitySynth,eEigenSynth,eDynamicSynth,eCEGISSynth} synthesisType_t;
typedef enum {eInequalities, eNormalised, eTemplated, eVertices } displayType_t;
typedef enum {eTraceNoTableau,eTraceTableau,eTraceTransforms,eTracePivots,eTraceSimplex,eTraceBasis,eTraceEntries} traceTableau_t;
typedef enum {eTraceNoVertex,eTraceVertices,eTraceRays,eTraceEdges,eTraceSets} traceVertices_t;
typedef enum {eTraceNoDynamics,eTraceTime,eTraceDynamics,eTraceErrors,eTraceAbstraction,eTraceAll,eTraceREF} traceDynamics_t;
//typedef enum {eTraceNone,eTraceResults,eTraceIntermediate,eTraceTransforms,eTraceAll} traceType_t;
typedef enum {eNoCmd,ePlusCmd,eMinusCmd,eArrowCmd,eEqualsCmd,eGivenCmd} commands_t;
typedef enum {eNumStates,eNumInputs,eNumVarInputs,eNumFeedbacks,eNumOutputs,eNumSteps,eLogDirections,eLogFaces,eNumBits,eTightness,eTraceLevel,eNumParameters} parameters_t;
typedef enum {eFinalIterations,eFinalPrecision,eFinalLoadTime,eFinalReachTime,eNumFinalParameters} finalParameters_t;
typedef enum {eLD,eMP,eLDI,eMPI,eAllTypes} numericType_t;

template <class scalar> class functions
{
public:
  typedef interval_def<scalar> type;
  typedef typename type::scalar_interval scinterval;
  typedef typename type::power_type power;
  typedef typename std::complex<scalar> c_scalar;
  typedef typename std::complex<scinterval> c_interval;
  typedef Eigen::Matrix<scinterval,Eigen::Dynamic,Eigen::Dynamic>     MatrixS;
  typedef Eigen::Matrix<scalar,Eigen::Dynamic,Eigen::Dynamic>         MatrixR;

  static inline void imprecise(const scinterval &value,const scalar &ref)
  {
    UNUSED(value);
    UNUSED(ref);
    ms_isImprecise=true;
    if (ms_formal) throw ms_imprecise;
  }

  static inline void imprecise(const scalar &value,const scalar &ref)
  {
    UNUSED(value);
    UNUSED(ref);
    ms_isImprecise=true;
    if (ms_formal) throw ms_imprecise;
  }

  static inline scinterval pow(scinterval x,power y)
  {
    //return std::pow(x,y);
    if (y==0) return ms_interval_1;
    if (y<0) {
      x=ms_interval_1/x;
      y=-y;
    }
    /*if (y==ms_infPower) {
      if (x.upper()<ms_1) return 0;
      if (x.upper()>ms_1) return ms_infinity*ms_interval_1;
      return ms_interval_1;
    }*/
    scinterval z = (y % 2) ? x : ms_interval_1;
    while (y >>= 1) {
      x*=x;
      if (y % 2) z*=x;
    }
    return z;
  }

  static inline c_interval c_pow(c_interval x,power y)
  {
    //return std::pow(x,y);
    if (y==0) return ms_interval_c_1;
    if (y<0) {
      x=ms_interval_c_1/x;
      y=-y;
    }
    /*if (y==ms_infPower) {
      if (norm(x).upper()<ms_1) return c_interval(c_scalar(0,0));
      if (norm(x)>ms_1) return ms_infinity*ms_interval_c_1;
      return ms_interval_c_1;
    }*/
    c_interval z = (y % 2) ? x : ms_interval_c_1;
    while (y >>= 1) {
      x*=x;
      if (y % 2) z*=x;
    }
    return z;
  }

  static inline scinterval norm2(std::complex<scinterval> x)    { return sqrt(norm(x));}
  static inline scinterval norm2(const scinterval &x)           { return x;}
  static inline scinterval squared(const scinterval &x)
  {
    scalar lower=x.lower()*x.lower();
    scalar upper=x.upper()*x.upper();
    if (lower>upper) return scinterval(upper,lower);
    return scinterval(lower,upper);
  }

  static inline char hardSign(const scinterval &x)
  {
    if (x.upper()<ms_hardZero) return -1;
    if (x.lower()>ms_hardZero) return 1;
    if ((x.upper()>ms_zero) || (x.lower()<-ms_zero)) {
      imprecise(x,ms_zero);
    }
    return 0;
  }

  static inline char softSign(const scinterval &x)
  {
    if (x.upper()<-ms_zero) return -1;
    if (x.lower()>ms_zero) return 1;
    if ((x.upper()>ms_zero) || (x.lower()<-ms_zero)) {
      imprecise(x,ms_zero);
    }
    return 0;
  }

  static inline bool isNan(const scinterval &x);

  static inline bool isZero(scinterval x,scalar zero=ms_weakZero)
  {
    UNUSED(zero);
    return (x.upper()>=ms_hardZero) && (x.lower()<=ms_hardZero);
  }

  static inline bool isNearZero(scinterval x,scalar epsilon)
  {
    return ((x.upper()<=epsilon) && (x.lower()>=-epsilon));
  }

  static inline bool isNegative(const scinterval &x)
  { return (hardSign(x)<0); }

  static inline bool isPositive(const scinterval &x)
  { return (hardSign(x)>0); }

  static inline scalar toCentre(const scinterval &x)    { return (x.upper()+x.lower())*.5; }
  static inline c_scalar toCentre(const c_interval &x)  { return c_scalar(toCentre(x.real()),toCentre(x.imag())); }
  static inline MatrixS& toCentre(MatrixS &matrix)
  {
    for (int row=0;row<matrix.rows();row++) {
      for (int col=0;col<matrix.cols();col++) {
        matrix.coeffRef(row,col)=toCentre(matrix.coeffRef(row,col));
      }
    }
    return matrix;
  }

  static inline scalar toUpper(const scinterval &x)     { return x.upper(); }
  static inline scalar toLower(const scinterval &x)     { return x.lower(); }
  static inline scalar toWidth(const scinterval &x)     { return width(x); }

  static inline scinterval setpm(scinterval x)          { return scinterval(-x.upper(),x.upper()); }

  static inline scinterval getHull(const scinterval &x,const scinterval &y)
  { return hull(x,y); }

  static inline c_interval getHull(const c_interval &x,const c_interval &y)
  { return c_interval(hull(x.real(),y.real()),hull(x.imag(),y.imag())); }

  //static inline void mult(scinterval &z,const scinterval &x,const scinterval &y);
  static inline scalar madd(scinterval &z,const scinterval &x,const scinterval &y);
  static inline scalar tightMadd(scinterval &z,const scinterval &x,const scinterval &y);
  static inline scalar tightWidth(const scinterval &x,const scinterval &y);
  static inline void msub(scinterval &z,const scinterval &x,const scinterval &y);
  static inline scalar log(const scalar &x,const scalar &y);

  static inline char hardSign(const scalar &x)
  {
    if (x<-ms_zero) return -1;
    if (x>ms_zero) return 1;
    return 0;
  }

  static inline char softSign(const scalar &x)
  {
    if (x<-ms_zero) return -1;
    if (x>ms_zero) return 1;
    return 0;
  }

  static inline bool isNan(const scalar &x);

  static inline bool isZero(scalar x,scalar zero=ms_weakZero)
  { return (abs(x)<=zero); }

  static inline bool isNearZero(scalar x,scalar epsilon)
  { return (abs(x)<=epsilon); }

  static inline bool isNegative(const scalar &x)
  { return (x<-ms_zero); }

  static inline bool isPositive(const scalar x)
  { return (x>ms_zero); }

  static inline scalar pow(scalar x,power y)
  {
    //return std::pow(x,y);
    if (y==0) return ms_1;
    if (y<0) {
      x=ms_1/x;
      y=-y;
    }
    /*if (y==ms_infPower) {
      if (x<ms_1) return 0;
      if (x>ms_1) return ms_infinity;
      return ms_1;
    }*/
    scalar z = (y % 2) ? x : ms_1;
    while (y >>= 1) {
      x*=x;
      if (y % 2) z*=x;
   }
   return z;
  }

  static inline c_scalar c_pow(c_scalar x,power y)
  {
    //return std::pow(x,y);
    if (y==0) return ms_c_1;
    if (y<0) {
      x=ms_c_1/x;
      y=-y;
    }
    /*if (y==ms_infPower) {
      if (norm(x)<ms_1) return c_scalar(0,0);
      if (norm(x)>ms_1) return c_scalar(ms_infinity,ms_infinity);
      return ms_c_1;
    }*/
    c_scalar z = (y % 2) ? x : ms_c_1;
    while (y >>= 1) {
      x*=x;
      if (y % 2) z*=x;
   }
   return z;
  }

  static inline scalar norm2(std::complex<scalar> x)
  { return sqrt(norm(x));}

  static inline scalar norm2(scalar x)                  { return x;}
  static inline scalar squared(const scalar &x)         { return (x*x); }

  static inline scalar toCentre(const scalar &x)        { return x; }
  static inline c_scalar toCentre(const c_scalar &x)    { return x; }
  static inline MatrixR& toCentre(MatrixR &matrix)      { return matrix; }
  static inline scalar toUpper(const scalar &x)         { return x; }
  static inline scalar toLower(const scalar &x)         { return x; }
  static inline scalar toWidth(const scalar&)           { return 0; }
  static inline scalar setpm(scalar x)                  { return x; }
  static inline scalar getHull(const scalar &,const scalar &y)              { return y; }
  static inline c_scalar getHull(const c_scalar &,const c_scalar &y)        { return y; }
  //static inline void mult(scalar &z,const scalar &x,const scalar &y)    { z=x*y;   }
  static inline scalar madd(scalar &z,const scalar &x,const scalar &y)      { z+=x*y;return ms_hardZero; }
  static inline scalar tightMadd(scalar &z,const scalar &x,const scalar &y) { z+=x*y;return ms_hardZero; }
  static inline scalar tightWidth(const scalar &x,const scalar &y)          { return ms_hardZero; }
  static inline void msub(scalar &z,const scalar &x,const scalar &y)        { z-=x*y;   }

  static long double toDouble(scalar x);
  static scalar toScalar(long double &x)                { return x; }
  static scalar toScalar(mpfr::mpreal &x);
  static char* toScalar(const char* pBegin,scalar &value);
  static char* toScalar(const char* pBegin,scinterval &value);
  static std::complex<scalar> toScalar(const std::complex<long double> &x);
  static std::complex<scalar> toScalar(const std::complex<mpfr::mpreal> &x);

  static long long toInt(scalar x);
  static void setDefaultPrec(int prec);
  static int  getDefaultPrec();
  static bool increasePrec();
  static void setZero(scalar zero)                  { ms_zero=zero; }

  template <class scalarC> static scalarC const_pi(const scalarC &multiplier);
  template <class scalarC> static scalarC const_pi();
  template <class scalarC> static scalarC const_pi_2();
  template <class scalarC> static scalarC const_2_pi();
  template <class scalarC> static scalarC invtan(const scalarC num,const scalarC den);
  template <class scalarC> static scalarC cosine(const scalarC &x);
  template <class scalarC> static scalarC sine(const scalarC &x);
public:
  static typename interval_def<scalar>::power_type   ms_infPower;
  static scalar                         ms_zero;
  static scalar                         ms_weakZero;
  static scalar                         ms_hardZero;
  static c_scalar                       ms_c_1;
  static c_interval                     ms_interval_c_1;
  static scalar                         ms_1;
  static scinterval                     ms_interval_1;
  static scalar                         ms_epsilon;
  static scalar                         ms_weakEpsilon;
  static scalar                         ms_infinity;
  static scalar                         ms_nan;
  static scalar                         ms_highPi;
  static scalar                         ms_lowPi;
  static scalar                         ms_pi;
  static scinterval                     ms_interval_pi;
  static scalar                         ms_pi_2;
  static scinterval                     ms_interval_pi_2;
  static scalar                         ms_2_pi;
  static scinterval                     ms_interval_2_pi;
  static scinterval                     ms_temp;
  static scinterval                     ms_temp2;
  static bool                           ms_formal;
  static bool                           ms_isImprecise;
  static std::string                    ms_imprecise;
  static std::string                    ms_error;
};

class mpscalar : public functions<mpfr::mpreal>
{
public:
};


class ldscalar : public functions<long double>
{
public:
};

#define USE_MPREAL
//#define USE_LDOUBLE
#define USE_INTERVALS
#define USE_SINGLES
#ifdef USE_MPREAL
typedef mpfr::mpreal scalar_t;
#else
typedef long double scalar_t;
#endif
typedef interval_def<scalar_t>::scalar_interval scalar_int_t;

#endif /* SCALAR_H */

