//Authors: Dario Cattaruzza, Alessandro Abate, Peter Schrammel, Daniel Kroening
//University of Oxford 2016
//This code is supplied under the BSD license agreement (see license.txt)

#include "Scalar.h"
#include <climits>

inline bool mpfr_neg_p(mpfr_srcptr x)
{
  return (mpfr_signbit(x) && !mpfr_zero_p(x));
}

inline bool mpfr_pos_p(mpfr_srcptr x)
{
  return (!mpfr_signbit(x) && !mpfr_zero_p(x));
}

void mpreal_rounding_control::get_rounding_mode(rounding_mode&)
{ mpfr::mpreal::get_default_rnd(); }

void mpreal_rounding_control::set_rounding_mode(rounding_mode rnd)
{ mpfr::mpreal::set_default_rnd((mpfr_rnd_t)rnd); }

void mpreal_rounding_control::upward()
{ mpfr::mpreal::set_default_rnd(MPFR_RNDU); }

void mpreal_rounding_control::downward()
{ mpfr::mpreal::set_default_rnd(MPFR_RNDD); }

void mpreal_rounding_control::to_nearest()
{ mpfr::mpreal::set_default_rnd(MPFR_RNDN); }

const mpfr::mpreal& mpreal_rounding_control::to_int(const mpfr::mpreal& x)
{ return floor(x); }

const mpfr::mpreal& mpreal_rounding_control::force_rounding(const mpfr::mpreal& x)
{ return x; }

std::string interval_def<mpfr::mpreal>::ms_name="mp";
std::string interval_def<long double>::ms_name="ld";
std::string interval_def<mpinterval>::ms_name="mpi";
std::string interval_def<ldinterval>::ms_name="ldi";

template <class scalar> bool functions<scalar>::ms_formal=false;
template <class scalar> std::string functions<scalar>::ms_imprecise="Not enough precision";
template <class scalar> std::string functions<scalar>::ms_error="Numerical error";
template <class scalar> bool functions<scalar>::ms_isImprecise=false;
template <class scalar> typename functions<scalar>::c_scalar
  functions<scalar>::ms_c_1=c_scalar(1.0,0);

template <class scalar> typename functions<scalar>::c_interval
  functions<scalar>::ms_interval_c_1=c_interval(1.0);

template <class scalar> scalar
  functions<scalar>::ms_1=scalar(1.0);

template <class scalar> typename functions<scalar>::scinterval
  functions<scalar>::ms_interval_1=scinterval(1.0);

template <> mpfr::mpreal functions<mpfr::mpreal>::ms_zero=1e-15;
template <> mpfr::mpreal functions<mpfr::mpreal>::ms_weakZero=1e-15;
template <> mpfr::mpreal functions<mpfr::mpreal>::ms_epsilon =pow(2*ms_1,1-mpfr::mpreal::get_default_prec());
template <> mpfr::mpreal functions<mpfr::mpreal>::ms_weakEpsilon =pow(2*ms_1,-(mpfr::mpreal::get_default_prec()>>2)-10);
template <> mpfr::mpreal functions<mpfr::mpreal>::ms_highPi=mpfr::const_pi(mpfr::mpreal::get_default_prec(), MPFR_RNDU);
template <> mpfr::mpreal functions<mpfr::mpreal>::ms_lowPi=mpfr::const_pi(mpfr::mpreal::get_default_prec(), MPFR_RNDN);

template <> long double functions<long double>::ms_zero=1e-6;
template <> long double functions<long double>::ms_weakZero=1e-6;
template <> long double functions<long double>::ms_epsilon=pow(2*ms_1,-127);
template <> long double functions<long double>::ms_weakEpsilon=1e-6;
template <> long double functions<long double>::ms_highPi=(3373259426.0 + 273689.0 / (1<<21)) / (1<<30);
template <> long double functions<long double>::ms_lowPi= (3373259426.0 + 273688.0 / (1<<21)) / (1<<30);

template <> typename interval_def<long double>::power_type
  functions<long double>::ms_infPower=LONG_MAX;

#if defined (MPREAL_HAVE_INT64_SUPPORT)
template <> typename interval_def<mpfr::mpreal>::power_type
  functions<mpfr::mpreal>::ms_infPower=LONG_LONG_MAX;
#else
template <> typename interval_def<mpfr::mpreal>::power_type
  functions<mpfr::mpreal>::ms_infPower=LONG_MAX;
#endif

template <class scalar> scalar functions<scalar>::ms_hardZero=0;
template <class scalar> scalar functions<scalar>::ms_infinity=std::numeric_limits<scalar>::infinity();
template <class scalar> scalar functions<scalar>::ms_nan=std::numeric_limits<scalar>::quiet_NaN();
template <class scalar> scalar functions<scalar>::ms_pi=functions<scalar>::ms_lowPi;

template <class scalar> typename functions<scalar>::scinterval
  functions<scalar>::ms_temp=functions<scalar>::scinterval(functions<scalar>::ms_hardZero,functions<scalar>::ms_hardZero);

template <class scalar> typename functions<scalar>::scinterval
  functions<scalar>::ms_temp2=functions<scalar>::scinterval(functions<scalar>::ms_hardZero,functions<scalar>::ms_hardZero);

template <class scalar> typename functions<scalar>::scinterval
  functions<scalar>::ms_interval_pi=functions<scalar>::scinterval(functions<scalar>::ms_lowPi,functions<scalar>::ms_highPi);

template <class scalar> scalar
  functions<scalar>::ms_pi_2=functions<scalar>::ms_pi/2;

template <class scalar> typename functions<scalar>::scinterval
  functions<scalar>::ms_interval_pi_2=functions<scalar>::ms_interval_pi/functions<scalar>::scinterval(2);

template <class scalar> scalar functions<scalar>::ms_2_pi=functions<scalar>::ms_pi*2;

template <class scalar> typename functions<scalar>::scinterval
  functions<scalar>::ms_interval_2_pi=functions<scalar>::ms_interval_pi*functions<scalar>::scinterval(2);


template <> void functions<mpfr::mpreal>::setDefaultPrec(int prec)
{
    mpfr::mpreal::set_default_prec(prec);
    ms_epsilon=pow(2*ms_1,1-prec);
    if (prec<128) ms_weakEpsilon =pow(2*ms_1,-(prec>>2)-10);//64=26(7.8),127=42(12)
    else ms_weakEpsilon =pow(2*ms_1,-(prec>>4)-36);//128=44(13),256=52(15),512=68(20),1024=100(30),2048=164(49)
    ms_highPi=mpfr::const_pi(prec, MPFR_RNDU);
    ms_lowPi=mpfr::const_pi(prec, MPFR_RNDN);
    ms_pi=ms_lowPi;
    ms_interval_pi=mpinterval(ms_lowPi,ms_highPi);
    ms_pi_2=ms_pi/2;
    ms_interval_pi_2=ms_interval_pi/scinterval(2);
    ms_2_pi=ms_pi*2;
    ms_interval_2_pi=ms_interval_pi*mpinterval(2);
    ms_temp=ms_pi;
    ms_temp2=ms_temp;
    ms_isImprecise=false;
}

template <> void functions<long double>::setDefaultPrec(int prec) {}

template <> int functions<mpfr::mpreal>::getDefaultPrec() { return mpfr::mpreal::get_default_prec();}

template <> int functions<long double>::getDefaultPrec() { return 80; }

template <> bool functions<mpfr::mpreal>::increasePrec()
{
  int prec=mpfr::mpreal::get_default_prec();
  if (prec>1024) return false;
  mpfr::mpreal epsilon=ms_weakEpsilon;
  setDefaultPrec(2*prec);
  ms_weakEpsilon=epsilon;
  return true;
}

template <> bool functions<long double>::increasePrec() { return false; }

template <> template <>
mpfr::mpreal functions<mpfr::mpreal>::const_pi<mpfr::mpreal>(const mpfr::mpreal &multiplier)
{
  return multiplier*ms_pi;
}

template <> template <>
mpinterval functions<mpfr::mpreal>::const_pi<mpinterval>(const mpinterval &multiplier)
{
  return multiplier*ms_interval_pi;
}

template <> template <>
long double functions<long double>::const_pi<long double>(const long double &multiplier)
{
  return multiplier*ms_pi;
}

template <> template <>
ldinterval functions<long double>::const_pi<ldinterval>(const ldinterval &multiplier)
{
  return multiplier*ms_interval_pi;
}

template <> template <>
mpfr::mpreal functions<mpfr::mpreal>::const_pi<mpfr::mpreal>()   { return ms_pi; }
template <> template <>
mpinterval functions<mpfr::mpreal>::const_pi<mpinterval>()       { return ms_interval_pi; }
template <> template <>
mpfr::mpreal functions<mpfr::mpreal>::const_pi_2<mpfr::mpreal>() { return ms_pi_2; }
template <> template <>
mpinterval functions<mpfr::mpreal>::const_pi_2<mpinterval>()     { return ms_interval_pi_2; }
template <> template <>
mpfr::mpreal functions<mpfr::mpreal>::const_2_pi<mpfr::mpreal>() { return ms_2_pi; }
template <> template <>
mpinterval functions<mpfr::mpreal>::const_2_pi<mpinterval>()     { return ms_interval_2_pi; }


template <> template <>
long double functions<long double>::const_pi<long double>()      { return ms_pi; }
template <> template <>
ldinterval functions<long double>::const_pi<ldinterval>()        { return ms_interval_pi; }
template <> template <>
long double functions<long double>::const_pi_2<long double>()    { return ms_pi_2; }
template <> template <>
ldinterval functions<long double>::const_pi_2<ldinterval>()      { return ms_interval_pi_2; }
template <> template <>
long double functions<long double>::const_2_pi<long double>()    { return ms_2_pi; }
template <> template <>
ldinterval functions<long double>::const_2_pi<ldinterval>()      { return ms_interval_2_pi; }

template <> template <>
mpfr::mpreal functions<mpfr::mpreal>::invtan<mpfr::mpreal>(mpfr::mpreal num,mpfr::mpreal den)
{
    char sign=hardSign(den);
    if (sign==0) {
      sign=hardSign(num);
      if (sign>0) return ms_pi_2;
      else if (sign<0) return -ms_pi_2;
      else if (sign==0) {
        //should never happen
        imprecise(num,ms_hardZero);
      }
    }
    else if (sign<0)  return ms_pi+atan(num/den);
    else              return atan(num/den);
    return ms_hardZero;
}

template <> template <>
mpinterval functions<mpfr::mpreal>::invtan<mpinterval>(mpinterval num,mpinterval den)
{
  char sign=hardSign(den);
  if (sign==0) {
    sign=hardSign(num);
    if (sign>0) return ms_pi_2;
    else if (sign<0) return -ms_pi_2;
    else if (sign==0) {
      //should never happen
      imprecise(num,ms_hardZero);
      return mpinterval(-ms_pi_2,ms_pi_2);
    }
  }
  else if (sign<0)  return ms_pi+atan(num/den);
  else              return atan(num/den);
  return ms_hardZero;
}

template <> template <>
mpfr::mpreal functions<mpfr::mpreal>::cosine<mpfr::mpreal>(const mpfr::mpreal &x)    { return cos(x); }

template <> template <>
mpinterval functions<mpfr::mpreal>::cosine<mpinterval>(const mpinterval &x)
{
    if (boost::numeric::interval_lib::detail::test_input(x)) return mpinterval::empty();
    if (width(x) >= ms_interval_2_pi.lower()) return mpinterval(-1,1); // we are covering a full period

    typename mp_interval_policies::rounding rnd;

    // get lower bound within [-2pi, 2pi]
    typedef typename boost::numeric::interval_lib::unprotect<mpinterval>::type I;
    mpfr::mpreal n=floor(rnd.div_down(x.upper(), ms_interval_2_pi.lower()));
    mpinterval tmp=(const I&)x - n * (const I&)ms_interval_2_pi;

    if (tmp.upper() >= ms_highPi) {
      tmp-=ms_interval_pi;
      return -cosine(tmp);
    }
    mpfr::mpreal l = tmp.lower();
    mpfr::mpreal u = tmp.upper();

    if ((l<0) && (u>0)) {
      if (u<-l) u=-l;
      l=0;
    }
    BOOST_USING_STD_MIN();
    // separate into monotone subintervals
    if (u <= ms_lowPi)
      return mpinterval(rnd.cos_down(u), rnd.cos_up(l), true);
    else if (u <= ms_interval_pi_2.lower())
      return mpinterval(-1, rnd.cos_up(min BOOST_PREVENT_MACRO_SUBSTITUTION(rnd.sub_down(ms_interval_2_pi.lower(), u), l)), true);
    else
      return mpinterval(-1, 1);
}
template <> template <> mpfr::mpreal functions<mpfr::mpreal>::sine<mpfr::mpreal>(const mpfr::mpreal &x)      { return sin(x); }
template <> template <> mpinterval functions<mpfr::mpreal>::sine<mpinterval>(const mpinterval &x)
{
  mpinterval tmp=x-ms_interval_pi_2;
  return cosine<mpinterval>(tmp);
}

template <> template <>
long double functions<long double>::invtan<long double>(const long double num,const long double den)
{ return atan(num/den); }

template <> template <>
ldinterval functions<long double>::invtan<ldinterval>(ldinterval num,ldinterval den)
{
  char sign=hardSign(den);
  if (sign==0) {
    sign=hardSign(num);
    if (sign>0) return ms_pi_2;
    else if (sign<0) return -ms_pi_2;
    else if (sign==0) {
      //should never happen
      imprecise(num,ms_hardZero);
      return ldinterval(-ms_pi_2,ms_pi_2);
    }
  }
  else {
    return atan(num/den);
  }
  return ms_hardZero;
}

template <> template <>
long double functions<long double>::cosine<long double>(const long double &x)
{ return cos(x); }

template <> template <>
ldinterval functions<long double>::cosine<ldinterval>(const ldinterval &x)
{ return cos(x); }

template <> template <>
long double functions<long double>::sine<long double>(const long double &x)
{ return sin(x); }

template <> template <>
ldinterval functions<long double>::sine<ldinterval>(const ldinterval &x)
{ return sin(x); }

template <>
mpfr::mpreal functions<mpfr::mpreal>::toScalar(mpfr::mpreal &x)
{ return x; }

template <>
long double functions<long double>::toScalar(mpfr::mpreal &x)
{ return x.toDouble(); }

template <>
char* functions<mpfr::mpreal>::toScalar(const char* pBegin,mpfr::mpreal &value)
{
  char *pEnd;
  mpfr_strtofr (value.mpfr_ptr(), pBegin, &pEnd, 10, MPFR_RNDD);
  return pEnd;
}

template <>
char* functions<mpfr::mpreal>::toScalar(const char* pBegin,mpinterval &value)
{
  char *pEnd;
  mpfr_strtofr ((mpfr_ptr)value.lower().mpfr_ptr(), pBegin, &pEnd, 10, MPFR_RNDD);
  mpfr_strtofr ((mpfr_ptr)value.upper().mpfr_ptr(), pBegin, &pEnd, 10, MPFR_RNDU);
  return pEnd;
}

template <>
char* functions<long double>::toScalar(const char* pBegin,long double &value)
{
  char *pEnd;
  value=std::strtold(pBegin,&pEnd);
  return pEnd;
}

template <>
char* functions<long double>::toScalar(const char* pBegin,ldinterval &value)
{
    char *pEnd;
    value=std::strtold(pBegin,&pEnd);
    return pEnd;
}

template <>
std::complex<mpfr::mpreal> functions<mpfr::mpreal>::toScalar(const std::complex<mpfr::mpreal> &x)
{ return x; }

template <>
std::complex<mpfr::mpreal> functions<mpfr::mpreal>::toScalar(const std::complex<long double> &x)
{ return x; }

template <>
std::complex<long double> functions<long double>::toScalar(const std::complex<mpfr::mpreal> &x)
{ return std::complex<long double>(x.real().toDouble(),x.imag().toDouble()); }

template <>
std::complex<long double> functions<long double>::toScalar(const std::complex<long double> &x)
{ return x; }


template <>
long double functions<mpfr::mpreal>::toDouble(mpfr::mpreal value)
{ return value.toDouble(); }

template <>
long long int functions<mpfr::mpreal>::toInt(mpfr::mpreal value)
{
#if defined (MPREAL_HAVE_INT64_SUPPORT)
  return value.toInt64();
#else
  return value.toLong();
#endif
}

template <>
long double functions<long double>::toDouble(long double value)          { return value; }

template <>
long long int functions<long double>::toInt(long double value)           { return value; }

template <>
bool functions<long double>::isNan(const long double &value)             { return std::isnan(value); }

template <>
bool functions<long double>::isNan(const ldinterval &value)              { return std::isnan(value.upper()) || std::isnan(value.lower()); }

template <>
bool functions<mpfr::mpreal>::isNan(const mpfr::mpreal &value)           { return isnan(value); }

template <>
bool functions<mpfr::mpreal>::isNan(const mpinterval &value)             { return isnan(value.upper()) || isnan(value.lower()); }

template <>
long double functions<long double>::madd(ldinterval &z,const ldinterval &x,const ldinterval &y)
{
  ldinterval m(x*y);
  z+=m;
  return width(m);
}

template <>
long double functions<long double>::tightMadd(ldinterval &z,const ldinterval &x,const ldinterval &y)
{
  ldinterval m(x*y);
  z+=m;
  if (x.lower()>0) return width(y)*x.lower();
  if (x.upper()<0) return width(y)*-x.upper();
  if (x.upper()+x.lower()>0) return  width(y)*-x.lower();
  return width(y)*x.upper();;
}

template <>
long double functions<long double>::tightWidth(const ldinterval &x,const ldinterval &y)
{
  if (x.lower()>0) return width(y)*x.lower();
  if (x.upper()<0) return width(y)*-x.upper();
  if (x.upper()+x.lower()>0) return  width(y)*-x.lower();
  return width(y)*x.upper();
}

template <>
mpfr::mpreal functions<mpfr::mpreal>::madd(mpinterval &z,const mpinterval &x,const mpinterval &y)
{
  mpfr_srcptr xl = x.lower().mpfr_ptr();
  mpfr_srcptr xu = x.upper().mpfr_ptr();
  mpfr_srcptr yl = y.lower().mpfr_ptr();
  mpfr_srcptr yu = y.upper().mpfr_ptr();
  mpfr_ptr zl = (mpfr_ptr)z.lower().mpfr_ptr();
  mpfr_ptr zu = (mpfr_ptr)z.upper().mpfr_ptr();
  mpfr_ptr tl = (mpfr_ptr)ms_temp.lower().mpfr_ptr();
  mpfr_ptr tu = (mpfr_ptr)ms_temp.upper().mpfr_ptr();
  if (mpfr_neg_p(xl))
    if (mpfr_pos_p(xu)) {
      if (mpfr_neg_p(yl)) {
        if (mpfr_neg_p(yu)) {
          mpfr_mul(tl,xu,yl,MPFR_RNDD);
          mpfr_mul(tu,xl,yl,MPFR_RNDU);
        }
        else {
          mpfr_ptr temp2 = (mpfr_ptr)ms_temp2.lower().mpfr_ptr();
          mpfr_mul(tl,xu,yl,MPFR_RNDD);
          mpfr_mul(temp2,xl,yu,MPFR_RNDD);
          if (mpfr_greater_p(tl,temp2)!=0) tl=temp2;
          mpfr_mul(tu,xl,yl,MPFR_RNDU);
          mpfr_mul(temp2,xu,yu,MPFR_RNDU);
          if (mpfr_greater_p(temp2,tu)!=0) tu=temp2;
        }
      }
      else if (mpfr_pos_p(yu)) {
        mpfr_mul(tl,xl,yu,MPFR_RNDD);
        mpfr_mul(tu,xu,yu,MPFR_RNDU);
      }
      else return 0;
    }
    else {
      if (mpfr_neg_p(yl)) {
        mpfr_mul(tl,mpfr_neg_p(yu) ? xu : xl,yu,MPFR_RNDD);
        mpfr_mul(tu,xl,yl,MPFR_RNDU);
      }
      else if (mpfr_pos_p(yu)) {
        mpfr_mul(tl,xl,yu,MPFR_RNDD);
        mpfr_mul(tu,xu,yl,MPFR_RNDU);
      }
      else return 0;
    }
  else if (mpfr_pos_p(xu)) {
    if (mpfr_neg_p(yl)) {
      mpfr_mul(tl,xu,yl,MPFR_RNDD);
      mpfr_mul(tu,mpfr_neg_p(yu) ? xl : xu,yu,MPFR_RNDU);
    }
    else if (mpfr_pos_p(yu)) {
      mpfr_mul(tl,xl,yl,MPFR_RNDD);
      mpfr_mul(tu,xu,yu,MPFR_RNDU);
    }
    else return 0;
  }
  else return 0;
  mpfr_add(zl,zl,tl,MPFR_RNDD);
  mpfr_add(zu,zu,tu,MPFR_RNDU);
  mpfr_sub(tu,tu,tl,MPFR_RNDU);
  return tu;
}

template <>
mpfr::mpreal functions<mpfr::mpreal>::tightMadd(mpinterval &z,const mpinterval &x,const mpinterval &y)
{
  mpfr_srcptr xl = x.lower().mpfr_ptr();
  mpfr_srcptr xu = x.upper().mpfr_ptr();
  mpfr_srcptr yl = y.lower().mpfr_ptr();
  mpfr_srcptr yu = y.upper().mpfr_ptr();
  mpfr_ptr zl = (mpfr_ptr)z.lower().mpfr_ptr();
  mpfr_ptr zu = (mpfr_ptr)z.upper().mpfr_ptr();
  mpfr_ptr tl = (mpfr_ptr)ms_temp.lower().mpfr_ptr();
  mpfr_ptr tu = (mpfr_ptr)ms_temp.upper().mpfr_ptr();
  if (mpfr_neg_p(xl))
    if (mpfr_pos_p(xu)) {
      if (mpfr_neg_p(yl)) {
        if (mpfr_neg_p(yu)) {
          mpfr_mul(tl,xu,yl,MPFR_RNDD);
          mpfr_mul(tu,xl,yl,MPFR_RNDU);
        }
        else {
          mpfr_ptr temp2 = (mpfr_ptr)ms_temp2.lower().mpfr_ptr();
          mpfr_mul(tl,xu,yl,MPFR_RNDD);
          mpfr_mul(temp2,xl,yu,MPFR_RNDD);
          if (mpfr_greater_p(tl,temp2)!=0) tl=temp2;
          mpfr_mul(tu,xl,yl,MPFR_RNDU);
          mpfr_mul(temp2,xu,yu,MPFR_RNDU);
          if (mpfr_greater_p(temp2,tu)!=0) tu=temp2;
        }
      }
      else if (mpfr_pos_p(yu)) {
        mpfr_mul(tl,xl,yu,MPFR_RNDD);
        mpfr_mul(tu,xu,yu,MPFR_RNDU);
      }
      else return 0;
    }
    else {
      if (mpfr_neg_p(yl)) {
        mpfr_mul(tl,mpfr_neg_p(yu) ? xu : xl,yu,MPFR_RNDD);
        mpfr_mul(tu,xl,yl,MPFR_RNDU);
      }
      else if (mpfr_pos_p(yu)) {
        mpfr_mul(tl,xl,yu,MPFR_RNDD);
        mpfr_mul(tu,xu,yl,MPFR_RNDU);
      }
      else return 0;
    }
  else if (mpfr_pos_p(xu)) {
    if (mpfr_neg_p(yl)) {
      mpfr_mul(tl,xu,yl,MPFR_RNDD);
      mpfr_mul(tu,mpfr_neg_p(yu) ? xl : xu,yu,MPFR_RNDU);
    }
    else if (mpfr_pos_p(yu)) {
      mpfr_mul(tl,xl,yl,MPFR_RNDD);
      mpfr_mul(tu,xu,yu,MPFR_RNDU);
    }
    else return 0;
  }
  else return 0;
  mpfr_add(zl,zl,tl,MPFR_RNDD);
  mpfr_add(zu,zu,tu,MPFR_RNDU);
  //mpfr_sub(tu,tu,tl,MPFR_RNDU);

  mpfr_sub(tu,xu,xl,MPFR_RNDU);
  if (mpfr_pos_p(yl)) mpfr_mul(tu,tu,yl,MPFR_RNDD);
  else if (mpfr_neg_p(yu)) {
    mpfr_neg(tl,yu,MPFR_RNDD);
    mpfr_mul(tu,tu,tl,MPFR_RNDD);
  }
  else {
    mpfr_sub(tl,yu,yl,MPFR_RNDU);
    if (mpfr_neg_p(tl)) mpfr_mul(tu,tu,yu,MPFR_RNDD);
    else {
      mpfr_neg(tl,yl,MPFR_RNDD);
      mpfr_mul(tu,tu,tl,MPFR_RNDD);
    }
  }
  return tu;
}

template <>
mpfr::mpreal functions<mpfr::mpreal>::tightWidth(const mpinterval &x,const mpinterval &y)
{
  mpfr_srcptr xl = x.lower().mpfr_ptr();
  mpfr_srcptr xu = x.upper().mpfr_ptr();
  mpfr_srcptr yl = y.lower().mpfr_ptr();
  mpfr_srcptr yu = y.upper().mpfr_ptr();
  mpfr_ptr tl = (mpfr_ptr)ms_temp.lower().mpfr_ptr();
  mpfr_ptr tu = (mpfr_ptr)ms_temp.upper().mpfr_ptr();
  mpfr_sub(tu,xu,xl,MPFR_RNDU);
  if (mpfr_pos_p(yl)) mpfr_mul(tu,tu,yl,MPFR_RNDD);
  else if (mpfr_neg_p(yu)) {
    mpfr_neg(tl,yu,MPFR_RNDD);
    mpfr_mul(tu,tu,tl,MPFR_RNDD);
  }
  else {
    mpfr_sub(tl,yu,yl,MPFR_RNDU);
    if (mpfr_neg_p(tl)) mpfr_mul(tu,tu,yu,MPFR_RNDD);
    else {
      mpfr_neg(tl,yl,MPFR_RNDD);
      mpfr_mul(tu,tu,tl,MPFR_RNDD);
    }
  }
  return tu;
}

template <>
void functions<long double>::msub(ldinterval &z,const ldinterval &x,const ldinterval &y)
{
  z-=x*y;
}

template <>
void functions<mpfr::mpreal>::msub(mpinterval &z,const mpinterval &x,const mpinterval &y)
{
  z-=x*y;
  return;
  mpfr_srcptr xl = x.lower().mpfr_ptr();
  mpfr_srcptr xu = x.upper().mpfr_ptr();
  mpfr_srcptr yl = y.lower().mpfr_ptr();
  mpfr_srcptr yu = y.upper().mpfr_ptr();
  mpfr_ptr zl = (mpfr_ptr)z.lower().mpfr_ptr();
  mpfr_ptr zu = (mpfr_ptr)z.upper().mpfr_ptr();
  mpfr_ptr tl = (mpfr_ptr)ms_temp.lower().mpfr_ptr();
  mpfr_ptr tu = (mpfr_ptr)ms_temp.upper().mpfr_ptr();
  if (mpfr_neg_p(xl))
    if (mpfr_pos_p(xu)) {
      if (mpfr_neg_p(yl)) {
        if (mpfr_neg_p(yu)) {
          mpfr_mul(tl,xu,yl,MPFR_RNDD);
          mpfr_mul(tu,xl,yl,MPFR_RNDU);
        }
        else {
          mpfr_ptr temp2 = (mpfr_ptr)ms_temp2.lower().mpfr_ptr();
          mpfr_mul(tl,xu,yl,MPFR_RNDD);
          mpfr_mul(temp2,xl,yu,MPFR_RNDD);
          if (mpfr_greater_p(tl,temp2)!=0) tl=temp2;
          mpfr_mul(tu,xl,yl,MPFR_RNDU);
          mpfr_mul(temp2,xu,yu,MPFR_RNDU);
          if (mpfr_greater_p(temp2,tu)!=0) tu=temp2;
        }
      }
      else if (mpfr_pos_p(yu)) {
        mpfr_mul(tl,xl,yu,MPFR_RNDD);
        mpfr_mul(tu,xu,yu,MPFR_RNDU);
      }
      else return;
    }
    else {
      if (mpfr_neg_p(yl)) {
        mpfr_mul(tl,mpfr_neg_p(yu) ? xu : xl,yu,MPFR_RNDD);
        mpfr_mul(tu,xl,yl,MPFR_RNDU);
      }
      else if (mpfr_pos_p(yu)) {
        mpfr_mul(tl,xl,yu,MPFR_RNDD);
        mpfr_mul(tu,xu,yl,MPFR_RNDU);
      }
      else return;
    }
  else if (mpfr_pos_p(xu)) {
    if (mpfr_neg_p(yl)) {
      mpfr_mul(tl,xu,yl,MPFR_RNDD);
      mpfr_mul(tu,mpfr_neg_p(yu) ? xl : xu,yu,MPFR_RNDU);
    }
    else if (mpfr_pos_p(yu)) {
      mpfr_mul(tl,xl,yl,MPFR_RNDD);
      mpfr_mul(tu,xu,yu,MPFR_RNDU);
    }
    else return;
  }
  else return;
  mpfr_sub(zl,zl,tu,MPFR_RNDD);
  mpfr_sub(zu,zu,tl,MPFR_RNDU);
}

template <>
long double functions<long double>::log(const long double &x,const long double &y)
{
  return std::log(fabs(y))/std::log(fabs(x));
}

template <>
mpfr::mpreal functions<mpfr::mpreal>::log(const mpfr::mpreal &x,const mpfr::mpreal &y)
{
  return mpfr::log(fabs(y))/mpfr::log(fabs(x));
}


template class functions<mpfr::mpreal>;
template class functions<long double>;
