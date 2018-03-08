/**
 * # DSVerifier - Digital Systems Verifier
 * #
 * #                Universidade Federal do Amazonas - UFAM
 * #
 * # Author:       Thiago Cavalcante     <thiagorodrigoengcomp@gmail.com>
 * #
 * # ------------------------------------------------------
 * #
 * #
 * # ------------------------------------------------------
 */
#ifndef DSVERIFIER_ENGINE_VERIFY_SETTLING_TIME_H
#define DSVERIFIER_ENGINE_VERIFY_SETTLING_TIME_H

#include <stdio.h>
#include <string.h>

extern digital_system_state_space _controller;
extern int closed_loop;

#define MAX_SIZE 10

#define MACHEPS 2.22045e-16

#define DEBUG

#ifdef DEBUG
#define	m_output(mat) m_foutput(stdout, mat)
#endif

static const char *format = "%14.9g ";

#ifndef ANSI_C
#define ANSI_C 1
#endif

#define	v_chk_idx(x, i) ((i)>=0 && (i)<(x).dim)

#define v_get_val(x, i) (v_chk_idx(x, i) ? (x).ve[(i)] : \
    (printf("Error!\n")))

#define	v_entry(x, i) v_get_val(x, i)

#define	m_chk_idx(A, i, j) ((i)>=0 && (i)<(A).m && (j)>=0 && (j)<=(A).n)

#define	m_get_val(A, i, j) (m_chk_idx(A, i, j) ? \
    (A).me[(i)][(j)] : (printf("Error!")))

#define	m_entry(A, i, j) m_get_val(A, i, j)

#ifdef DEBUG
#define printfc(c) printf("%f%c%fi\n", c.real, (c.imag>=0.0f)? '+':'\0', c.imag)
#endif

/* type independent min and max operations */
#ifndef max
#define	max(a, b) ((a) > (b) ? (a) : (b))
#endif /* max */
#ifndef min
#define	min(a, b) ((a) > (b) ? (b) : (a))
#endif /* min */

/* matrix definition */
typedef	struct
{
  unsigned int m, n;
  unsigned int max_m, max_n, max_size;
  double me[MAX_SIZE][MAX_SIZE];
}
MAT;

/* vector definition */
typedef	struct
{
  unsigned int dim, max_dim;
  double ve[MAX_SIZE];
}
VEC;

/* complex number definition */
typedef	struct
{
  double real, imag;
}
CMPLX;

/* peak value definition */
typedef	struct
{
  double mp;
  int kp;
}
PKVL;

/* last state vector definition */
typedef	struct
{
  MAT xk;
  int lastState;
}
LST;

/* Global variables */
// xk stores the last computed state of the system
LST xk;


/******************************math.h Functions******************************/

/* sp_fabs -- absolute value of floating-point number */
double sp_fabs(double n)
{
  if(n >= 0)
    return n; //if positive, return without ant change
  else
    return (-n); //if negative, return a positive version
}

/* sp_ceil -- the smallest integer value greater than or equal to x */
double sp_ceil(double x)
{
  union
  {
    float f;
    int i;
  } float_int;

//    float_int val;
  float_int.f=x;

  // Extract sign, exponent and mantissa
  // Bias is removed from exponent
  int sign=float_int.i >> 31;
  int exponent=((float_int.i & 0x7fffffff) >> 23) - 127;
  int mantissa=float_int.i & 0x7fffff;

  // Is the exponent less than zero?
  if(exponent<0)
  {
    // In this case, x is in the open interval (-1, 1)
    if(x<=0.0f)
      return 0.0f;
    else
      return 1.0f;
  }
  else
  {
    // Construct a bit mask that will mask off the
    // fractional part of the mantissa
    int mask=0x7fffff >> exponent;

    // Is x already an integer (i.e. are all the
    // fractional bits zero?)
    if((mantissa & mask) == 0)
      return x;
    else
    {
      // If x is positive, we need to add 1 to it
      // before clearing the fractional bits
      if(!sign)
      {
        mantissa+=1 << (23-exponent);

        // Did the mantissa overflow?
        if(mantissa & 0x800000)
        {
          // The mantissa can only overflow if all the
          // integer bits were previously 1 -- so we can
          // just clear out the mantissa and increment
          // the exponent
          mantissa=0;
          exponent++;
        }
      }

      // Clear the fractional bits
      mantissa&=~mask;
    }
  }

  // Put sign, exponent and mantissa together again
  float_int.i=(sign << 31) | ((exponent+127) << 23) | mantissa;

  return (double)float_int.f;
}

/* sp_pow -- returns a raised to the power of n i.e. a^n */
double sp_pow(double a, int n)
{
  double r = 1;

  while(n > 0)
  {
    if(n & 1)
      r *= a;
    a *= a;
    n >>= 1;
  }
  return r;
}

/**
 * Calculate ln logarithm using integers with 16 bit precision
 * min: sp_fxp_ln(0.000015259<<16)
 * max: sp_fxp_ln(32767<<16)
 */
int sp_fxp_ln(int x)
{
  int t, y;

  y = 0xa65af;
  if(x < 0x00008000)
    x <<= 16, y -= 0xb1721;
  if(x < 0x00800000)
    x <<= 8, y -= 0x58b91;
  if(x < 0x08000000)
    x <<= 4, y -= 0x2c5c8;
  if(x < 0x20000000)
    x <<= 2, y -= 0x162e4;
  if(x < 0x40000000)
    x <<= 1, y -= 0x0b172;
  t = x + (x >> 1);
  if((t & 0x80000000) == 0)
    x = t, y -= 0x067cd;
  t = x + (x >> 2);
  if((t & 0x80000000) == 0)
    x = t, y -= 0x03920;
  t = x + (x >> 3);
  if((t & 0x80000000) == 0)
    x = t, y -= 0x01e27;
  t = x + (x >> 4);
  if((t & 0x80000000) == 0)
    x = t, y -= 0x00f85;
  t = x + (x >> 5);
  if((t & 0x80000000) == 0)
    x = t, y -= 0x007e1;
  t = x + (x >> 6);
  if((t & 0x80000000) == 0)
    x = t, y -= 0x003f8;
  t = x + (x >> 7);
  if((t & 0x80000000) == 0)
    x = t, y -= 0x001fe;
  x = 0x80000000 - x;
  y -= x >> 15;
  return y;
}

/**
 * Calculate log10 logarithm using 16 bit precision
 * min: sp_log10_2(0.000015259)
 * max: sp_log10_2(32767.0)
 */
double sp_log10_2(double x)
{
  int xint = (int) (x * 65536.0 + 0.5);
  int lnum = sp_fxp_ln(xint);
  int lden = sp_fxp_ln(655360);
  return ((double) lnum / (double) lden);
}

/* sp_floor -- returns the largest integer value less than or equal to num */
double sp_floor(double num)
{
  if (num >= 9.2234e+18 || num <= -9.2234e+18 || num != num)
  {
  /* handle large values, infinities and nan */
    return num;
  }
  long long n = (long long)num;
  double d = (double)n;
  if (d == num || num >= 0)
    return d;
  else
    return d - 1;
}

/* sp_sqrt -- returns the square root of fg */
double sp_sqrt(const double fg)
{
  double n = fg / 2.0;
  double lstX = 0.0;
  while(n != lstX)
  {
    lstX = n;
     n = (n + fg/n) / 2.0;
  }
  return n;
}

/******************************Matrix Functions******************************/

/* m_get -- gets an mxn matrix (in MAT form) by zeroing each matrix element */
MAT m_get(int m, int n)
{
  MAT A;
  int i, j;
  A.m = m;A.n = n;
  A.max_m = m;A.max_n = n;
  A.max_size = A.max_m * A.max_n;
  for(i = 0;i < m;i++)
  {
    for(j = 0;j < n;j++){
      A.me[i][j] = 0;
    }
  }
  return A;
}

/* m_add -- matrix addition -- may be in-situ */
#ifndef ANSI_C
MAT m_add(mat1, mat2)
MAT mat1, mat2;
#else
MAT m_add(MAT mat1, MAT mat2)
#endif
{
  MAT out;
  unsigned int i, j, mat1_m, mat1_n, mat2_m, mat2_n;
  mat1_m = mat1.m; mat1_n = mat1.n;
  mat2_m = mat2.m; mat2_n = mat2.n;
  out = m_get(mat1_m, mat1_n);
  for(i=0; i<mat1_m; i++)
  {
    for(j = 0; j < mat1_n; j++)
	  out.me[i][j] = mat1.me[i][j] + mat2.me[i][j];
  }
  return (out);
}

/* m_sub -- matrix subtraction -- may be in-situ */
#ifndef ANSI_C
MAT m_sub(mat1, mat2)
MAT mat1, mat2;
#else
MAT m_sub(MAT mat1, MAT mat2)
#endif
{
  MAT out;
  unsigned int i, j, mat1_m, mat1_n, mat2_m, mat2_n;
  mat1_m = mat1.m; mat1_n = mat1.n;
  mat2_m = mat2.m; mat2_n = mat2.n;
  out = m_get(mat1_m, mat1_n);
  for(i=0; i < mat1_m;i++)
  {
    for(j = 0;j < mat1_n;j++)
	  out.me[i][j] = mat1.me[i][j] - mat2.me[i][j];
  }
  return (out);
}

/* m_zero -- zero the matrix A */
#ifndef ANSI_C
MAT m_zero(m, n)
int m, n;
#else
MAT m_zero(int m, int n)
#endif
{
  MAT A;
  int i, j, A_m, A_n;
  A = m_get(m, n);
  A_m = A.m; A_n = A.n;
  for(i = 0; i < A_m;i++)
    for(j = 0; j <A_n; j++)
      A.me[i][j] = 0.0;
  return A;
}

/* __mltadd__ -- scalar multiply and add c.f. v_mltadd() */
#ifndef ANSI_C
void __mltadd__(dp1, dp2, s, len)
register double	*dp1, *dp2;
register double s;
register int len;
#else
void __mltadd__(double *dp1, const double *dp2, double s, int len)
#endif
{
  register int i;
#ifdef VUNROLL
  register int len4;
  len4 = len / 4;
  len  = len % 4;
  for(i = 0; i < len4; i++)
  {
    dp1[4*i]   += s*dp2[4*i];
    dp1[4*i+1] += s*dp2[4*i+1];
    dp1[4*i+2] += s*dp2[4*i+2];
    dp1[4*i+3] += s*dp2[4*i+3];
  }
  dp1 += 4*len4;    dp2 += 4*len4;
#endif
  for(i = 0; i < len; i++)
    dp1[i] += s*dp2[i];
}

/* m_mlt -- matrix-matrix multiplication */
#ifndef ANSI_C
MAT m_mlt(m1, m2)
MAT m1, m2;
#else
MAT m_mlt(const MAT m1, const MAT m2)
#endif
{
  MAT m3;
  unsigned int i1, j1, i2, j2;
  unsigned int i, j, k;
  i1 = m1.m; j1 = m1.n; i2 = m2.m; j2 = m2.n;
  if(j1 == i2)
  {
    // Checking if the multiplication is possible
    // Initialising Matrix 3
    m3 = m_get(i1, j2);
    // Calculating multiplication result
    for(i = 0; i < i1; i++)
    {
      for(j = 0; j < j2; j++)
      {
        for(k = 0; k < j1; k++)
        {
          double mult = (m1.me[i][k] * m2.me[k][j]);
          m3.me[i][j] = m3.me[i][j] + (m1.me[i][k] * m2.me[k][j]);
        }
      }
    }
  }
  else
  {
    #ifdef DEBUG
	printf("\nError! Operation invalid, please enter with valid matrices.\n");
    #endif
  }
  return m3;
}

/* m_copy -- copies matrix into new area
  	-- B <- A */
MAT m_copy(MAT A)
{
  MAT B;
  B.m = A.m;
  B.max_m = A.max_m;
  B.max_n = A.max_n;
  B.max_size = A.max_size;
  memcpy(B.me, A.me, MAX_SIZE*MAX_SIZE*8);
  B.n = A.n;
  return B;
}

/* sm_mlt -- scalar-matrix multiply -- may be in-situ */
#ifndef ANSI_C
MAT	sm_mlt(scalar,matrix)
double	scalar;
MAT	matrix;
#else
MAT	sm_mlt(double scalar, const MAT matrix)
#endif
{
  MAT out;
  unsigned int	m, n, i, j;
  m = matrix.m;	n = matrix.n;
  out = m_get(m, n);
  for( i=0; i<m; i++ )
    for ( j=0; j<n; j++ )
      out.me[i][j] = scalar*matrix.me[i][j];
  return (out);
}

/* m_mlt_scalar -- multiplication a matrix by a scalar */
#ifndef ANSI_C
MAT m_mlt_scalar(m1, scalar)
MAT m1;
double scalar;
#else
MAT m_mlt_scalar(const MAT m1, const double scalar)
#endif
{
  MAT m2 = m_copy(m1);
  int i, j;
  for(i = 0;i < m1.m;i++)
  {
    for(j = 0;i < m1.n;j++)
    {
      m2.me[i][j] = scalar * m1.me[i][j];
    }
  }
  return m2;
}

/* m_foutput -- prints a representation of the matrix a onto file/stream fp */
#ifndef ANSI_C
void m_foutput(fp, a)
FILE *fp;
MAT a;
#else
void m_foutput(FILE *fp, const MAT a)
#endif
{
  unsigned int i, j, tmp;
  fprintf(fp, "Matrix: %d by %d\n", a.m, a.n);
  for(i = 0; i < a.m; i++)   /* for each row... */
  {
    fprintf(fp, "row %u: ", i);
    for(j = 0, tmp = 2; j < a.n; j++, tmp++)
    {             /* for each col in row... */
      fprintf(fp, format, a.me[i][j]);
        if(!(tmp % 5))
          putc('\n', fp);
    }
    if(tmp % 5 != 1)
      putc('\n', fp);
  }
}

/* v_zero -- zero the vector x */
#ifndef ANSI_C
VEC v_zero(x)
VEC x;
#else
VEC v_zero(VEC x)
#endif
{
  for(int i = 0; i < x.dim; i++)
    x.ve[i] = 0.0;
  return x;
}

/* set_col -- sets column of matrix to values given in vec (in situ)
	-- that is, mat(i0:lim,col) <- vec(i0:lim) */
#ifndef ANSI_C
MAT set_col(mat, col, vec)
MAT mat;
VEC vec;
unsigned int col;
#else
MAT set_col(MAT mat, unsigned int col, const VEC vec/*, unsigned int i0*/)
#endif
{
  unsigned int i, lim, i0;
  lim = min(mat.m, vec.dim);
  for(i=i0; i<lim; i++)
    mat.me[i][col] = vec.ve[i];
  return (mat);
}

/* v_get -- gets a VEC of dimension 'dim'
   -- Note: initialized to zero */
VEC v_get(int dim)
{
  VEC V;
  V.dim = dim;
  V.max_dim = dim;
  for(int i = 0; i < dim; i++)
  {
    V.ve[i] = 0.0;
  }
  return V;
}

/* v_copy -- copies vector into new area
	-- B <- A */
#ifndef ANSI_C
VEC v_copy(A)
VEC A;
#else
VEC v_copy(const VEC A)
#endif
{
  VEC B;
  B.dim = A.dim;
  B.max_dim = A.max_dim;
  memcpy(B.ve, A.ve, MAX_SIZE*8);
  return B;
}

/* __ip__ -- inner product */
#ifndef ANSI_C
double __ip__(dp1, dp2, len)
register double	*dp1, *dp2;
int len;
#else
  double __ip__(const double *dp1, const double *dp2, int len)
#endif
{
#ifdef VUNROLL
  register int len4;
  register double sum1, sum2, sum3;
#endif
  register int i;
  register double sum;
  sum = 0.0;
#ifdef VUNROLL
  sum1 = sum2 = sum3 = 0.0;
  len4 = len / 4;
  len  = len % 4;

  for(i = 0; i < len4; i++)
  {
    sum  += dp1[4*i]*dp2[4*i];
    sum1 += dp1[4*i+1]*dp2[4*i+1];
    sum2 += dp1[4*i+2]*dp2[4*i+2];
    sum3 += dp1[4*i+3]*dp2[4*i+3];
  }
  sum  += sum1 + sum2 + sum3;
  dp1 += 4*len4;	dp2 += 4*len4;
#endif
  for(i = 0; i < len; i++)
  {
    sum  += dp1[i]*dp2[i];
  }
  return sum;
}

/* m_inverse -- returns inverse of A, provided A is not too rank deficient
-- uses Gauss - Jordan */
#ifndef ANSI_C
MAT m_inverse(A)
MAT A;
#else
MAT m_inverse(const MAT A)
#endif
{
  MAT out = m_get(A.m, A.n);
  int i, j, k, matsize;
  double temp;
  MAT AUX = m_copy(A);
  matsize = AUX.m;
  // automatically initialize the unit matrix, e.g.
  for(i = 0; i < matsize; i++)
  {
    for(j = 0; j < matsize; j++)
    {
      if(i == j)
      {
        out.me[i][j]=1;
      }
      else
        out.me[i][j]=0;
    }
  }
/*---------------Logic starts here------------------*/
  /* procedure to make the matrix A to unit matrix
   --by some row operations,and the same row operations of
   --Unit mat. I gives the inverse of matrix A
   --'temp' stores the A[k][k] value so that A[k][k] will not change
   --during the operation A[i][j]/=A[k][k] when i=j=k
  --*/
  for(k = 0; k < matsize; k++)
  {
    // it performs the following row operations to make A to unit matrix
    // R0=R0/A[0][0],similarly for I also R0=R0/A[0][0]
    // R1=R1-R0*A[1][0] similarly for I
    // R2=R2-R0*A[2][0]
    temp = AUX.me[k][k];
    for(j = 0; j < matsize; j++)
    {
      AUX.me[k][j] /= temp;
      out.me[k][j] /= temp;
    }
    for(i = 0; i < matsize; i++)
    {
      // R1=R1/A[1][1]
      // R0=R0-R1*A[0][1]
      // R2=R2-R1*A[2][1]
      temp = AUX.me[i][k];
      for(j = 0; j < matsize; j++)
      {
        if(i == k)
          break;
        // R2=R2/A[2][2]
        // R0=R0-R2*A[0][2]
        // R1=R1-R2*A[1][2]
        AUX.me[i][j] -= AUX.me[k][j]*temp;
        out.me[i][j] -= out.me[k][j]*temp;
      }
    }
  }
/*---------------Logic ends here--------------------*/
  return out;
}

/* m_ident -- set A to being closest to identity matrix as possible
  -- i.e. A[i][j] == 1 if i == j and 0 otherwise */
#ifndef ANSI_C
MAT m_ident(dim)
int dim;
#else
MAT m_ident(int dim)
#endif
{
  MAT A = m_get(dim, dim);
  int i, size;
  A = m_zero(dim, dim);
  size = min(A.m, A.n);
  for(i = 0;i < size;i++)
    A.me[i][i] = 1.0;
  return A;
}

/* _m_pow -- auxiliary function to compute integer powers of a square matrix M, M^n */
MAT _m_pow(MAT M, int n)
{
  MAT P = m_get(M.m, M.n);
  MAT temp = m_get(M.m, M.n);
  MAT temp2 = m_get(M.m, M.n);
  if(n == 0)
    return m_ident(M.m);
  else if(n == 1)
    return M;
  else
  {
    P = _m_pow(M, sp_floor(n/2));
    if ((n % 2) == 0)
      temp = m_mlt(P, P);
    else
    {
      temp2 = m_mlt(P, P);
      temp = m_mlt(temp2, M);
    }
  }
  return temp;
}

/* m_pow -- computes integer powers of a square matrix A, A^p */
#ifndef ANSI_C
MAT m_pow(A, p)
MAT A;
int	p;
#else
MAT m_pow(const MAT A, int p)
#endif
{
  MAT out;
  static MAT wkspace, tmp;

  wkspace = m_get(A.m, A.n);
  out = m_get(A.m, A.n);
  if(p < 0)
  {
    tmp = m_get(A.m, A.n);
    tmp = m_inverse(A);
    out = _m_pow(tmp, -p);
  }
  else
  {
    out = _m_pow(A, p);
  }
  return out;
}

/* get_col -- gets a specified column of a matrix and retruns it as a vector */
#ifndef ANSI_C
VEC get_col(mat, col, vec)
unsigned int col;
MAT mat;
VEC vec;
#else
VEC get_col(const MAT mat, unsigned int col)
#endif
{
  VEC vec;
  unsigned int i;
  if(vec.dim < mat.m)
    vec = v_get(mat.m);
  for(i = 0; i < mat.m; i++)
    vec.ve[i] = mat.me[i][col];
  return (vec);
}

/* _in_prod -- inner product of two vectors from i0 downwards
   -- that is, returns a(i0:dim)^T.b(i0:dim) */
#ifndef ANSI_C
double	_in_prod(a, b, i0)
VEC a, b;
unsigned int i0;
#else
double _in_prod(const VEC a, const VEC b, unsigned int i0)
#endif
{
  unsigned int limit;
  limit = min(a.dim,b.dim);
  return __ip__(&(a.ve[i0]), &(b.ve[i0]), (int)(limit-i0));
}

/* hhvec -- calulates Householder vector to eliminate all entries after the
   i0 entry of the vector vec. It is returned as out. May be in-situ */
#ifndef ANSI_C
VEC hhvec(vec, i0, beta, out, newval)
VEC vec, out;
unsigned int i0;
double beta, newval;
#else
VEC hhvec(const VEC vec, unsigned int i0, double beta, double newval)
#endif
{
  VEC out;
  double norm, temp;
  out = v_copy(vec);
  temp = (double)_in_prod(out, out, i0);
  norm = sp_sqrt(temp);
  if(norm <= 0.0)
  {
    beta = 0.0;
    return (out);
  }
  beta = 1.0/(norm * (norm + sp_fabs(out.ve[i0])));
  if(out.ve[i0] > 0.0)
    newval = -norm;
  else
    newval = norm;
  out.ve[i0] -= newval;
  return (out);
}

/* _hhtrcols -- transform a matrix by a Householder vector by columns
    starting at row i0 from column j0
    -- that is, M(i0:m,j0:n) <- (I-beta.hh(i0:m).hh(i0:m)^T)M(i0:m,j0:n)
    -- in-situ
    -- scratch vector w passed as argument
    -- raises error if w == NULL
*/
#ifndef ANSI_C
MAT _hhtrcols(M, i0, j0, hh, beta, w)
MAT M;
unsigned int i0, j0;
VEC hh;
double	beta;
VEC w;
#else
MAT _hhtrcols(MAT M, unsigned int i0, unsigned int j0,
               const VEC hh, double beta, VEC w)
#endif
{
  int i;
  if(beta == 0.0)
    return (M);
  w = v_zero(w);
  for(i = i0; i < M.m; i++)
    if(hh.ve[i] != 0.0)
      __mltadd__(&(w.ve[j0]), &(M.me[i][j0]), hh.ve[i],
                 (int)(M.n-j0));
  for(i = i0; i < M.m; i++)
    if(hh.ve[i] != 0.0)
      __mltadd__(&(M.me[i][j0]), &(w.ve[j0]), -beta*hh.ve[i],
                 (int)(M.n-j0));
  return (M);
}

/* hhtrrows -- transform a matrix by a Householder vector by rows
    starting at row i0 from column j0 -- in-situ
    -- that is, M(i0:m,j0:n) <- M(i0:m,j0:n)(I-beta.hh(j0:n).hh(j0:n)^T) */
#ifndef ANSI_C
MAT hhtrrows(M, i0, j0, hh, beta)
MAT M;
unsigned int i0, j0;
VEC hh;
double beta;
#else
MAT hhtrrows(MAT M, unsigned int i0, unsigned int j0,
              const VEC hh, double beta)
#endif
{
  double ip, scale;
  int i;
  if(beta == 0.0)
    return (M);
  /* for each row ... */
  for(i = i0;i < M.m;i++)
  { /* compute inner product */
    ip = __ip__(&(M.me[i][j0]), &(hh.ve[j0]), (int)(M.n-j0));
    scale = beta*ip;
    if(scale == 0.0)
      continue;
    /* do operation */
    __mltadd__(&(M.me[i][j0]), &(hh.ve[j0]), -scale,
               (int)(M.n-j0));
  }
  return (M);
}

/* Hfactor -- compute Hessenberg factorization in compact form.
    -- factorization performed in situ
*/
#ifndef ANSI_C
MAT Hfactor(A, diag, beta)
MAT A;
VEC diag, beta;
#else
MAT Hfactor(MAT A, VEC diag, VEC beta)
#endif
{
  static VEC hh, w;
  int k, limit;
  double b;
  limit = A.m - 1;
  hh = v_get(A.m);
  w  = v_get(A.n);
  for(k = 0; k < limit; k++)
  {
    /* compute the Householder vector hh */
	hh = get_col(A, (unsigned int)k);
	hh = hhvec(hh, k+1, (beta.ve[k]), (A.me[k+1][k]));
	if(((k+1) >= 0) && ((k+1) < hh.dim))
	{
	  diag.ve[k] = hh.ve[k+1];
	}
	else
	{
	  diag.ve[k] = 0;
	}
//    diag.ve[k] = (((k+1)>=0 && (k+1)<(hh).dim) ? (hh).ve[(k+1)] : \
//        0);
    /* apply Householder operation symmetrically to A */
	if((k >= 0) && (k < beta.dim))
	{
	  b = beta.ve[k];
	}
	//b = v_entry(beta, k);
    A = _hhtrcols(A, k+1, k+1, hh, b, w);
    A = hhtrrows(A, 0, k+1, hh, b);
  }
  return (A);
}

/* hhtrvec -- apply Householder transformation to vector
    -- that is, out <- (I-beta.hh(i0:n).hh(i0:n)^T).in
    -- may be in-situ */
#ifndef ANSI_C
VEC hhtrvec(hh, beta, i0, in, out)
VEC hh, in, out;	/* hh = Householder vector */
unsigned int i0;
double beta;
#else
VEC hhtrvec(const VEC hh, double beta, unsigned int i0, const VEC in)
#endif
{
  VEC out;
  double scale, temp;
  temp = (double)_in_prod(hh, in, i0);
  scale = beta*temp;
  out = v_copy(in);
  __mltadd__(&(out.ve[i0]), &(hh.ve[i0]), -scale, (int)(in.dim-i0));
  return (out);
}

/* makeHQ -- construct the Hessenberg orthogonalising matrix Q;
    -- i.e. Hess M = Q.M.Q'	*/
#ifndef ANSI_C
MAT makeHQ(H, diag, beta)
MAT H;
VEC diag, beta;
#else
MAT makeHQ(MAT H, VEC diag, VEC beta)
#endif
{
  MAT Qout;
  int i, j, limit;
  static VEC tmp1, tmp2;
  Qout = m_get(H.m, H.m);
  tmp1 = v_get(H.m);
  tmp2 = v_get(H.m);;
  for(i = 0;i < H.m;i++)
  {
    /* tmp1 = i'th basis vector */
    for(j = 0;j < H.m;j++)
      tmp1.ve[j] = 0.0;
    tmp1.ve[i] = 1.0;
    /* apply H/h transforms in reverse order */
    for(j = limit-1;j >= 0;j--)
    {
      tmp2 = get_col(H, (unsigned int)j);
      tmp2.ve[j+1] = (((j)>=0 && (j)<(diag).dim) ? (diag).ve[(j)] : 0);
      tmp1 = hhtrvec(tmp2, beta.ve[j], j+1, tmp1);
    }
    /* insert into Qout */
    Qout = set_col(Qout, (unsigned int)i, tmp1);
  }
  return (Qout);
}

/* makeH -- construct actual Hessenberg matrix */
#ifndef ANSI_C
MAT makeH(H)
MAT H;
#else
MAT makeH(const MAT H)
#endif
{
  MAT Hout;
  int i, j, limit;
  Hout = m_get(H.m, H.m);
  Hout = m_copy(H);
  limit = H.m;
  for(i = 1;i < limit;i++)
    for(j = 0;j < i-1;j++)
    	Hout.me[i][j] = 0.0;
  return (Hout);
}

/* rot_cols -- postmultiply mat by givens rotation described by c, s */
#ifndef ANSI_C
MAT rot_cols(mat, i, k, c, s)
MAT mat;
unsigned int i, k;
double c, s;
#else
MAT rot_cols(const MAT mat, unsigned int i, unsigned int k, double c, double s)
#endif
{
  MAT out;
  unsigned int j;
  double temp;
  out = m_copy(mat);
  for(j=0; j<mat.m; j++)
  {
    temp = c*m_entry(out, j, i) + s*m_entry(out, j, k);
    out.me[j][k] = (-s*(((j)<(out).m && (i)<=(out).n) ? \
        (out).me[(j)][(i)] : (printf("Error!"))) +
    	c*(((j)<(out).m && (k)<=(out).n) ? \
        (out).me[(j)][(k)] : (printf("Error!"))));
    out.me[j][i] = temp;
  }
  return (out);
}

/* rot_rows -- premultiply mat by givens rotation described by c, s */
#ifndef ANSI_C
MAT rot_rows(mat, i, k, c, s)
MAT mat;
unsigned int i, k;
double c, s;
#else
MAT rot_rows(const MAT mat, unsigned int i, unsigned int k, double c, double s)
#endif
{
  MAT out;
  unsigned int j;
  double temp;
  out = m_copy(mat);
  for(j=0; j<mat.n; j++)
  {
    temp = c*m_entry(out, i, j) + s*m_entry(out, k, j);
    out.me[k][j] = (-s*(((i)<(out).m && (j)<=(out).n) ? \
                   (out).me[(i)][(j)] : (printf("Error!"))) +
    		       c*(((k)<(out).m && (j)<=(out).n) ? \
                   (out).me[(k)][(j)] : (printf("Error!"))));
    out.me[i][j] = temp;
  }
  return (out);
}

/* hhldr3 -- computes */
#ifndef ANSI_C
static void hhldr3(x, y, z, nu1, beta, newval)
double x, y, z;
double *nu1, *beta, *newval;
#else
static void hhldr3(double x, double y, double z, double *nu1,
                   double *beta, double *newval)
#endif
{
  double alpha;
  if(x >= 0.0)
    alpha = sp_sqrt(x*x+y*y+z*z);
  else
    alpha = -sp_sqrt(x*x+y*y+z*z);
  *nu1 = x + alpha;
  *beta = 1.0/(alpha*(*nu1));
  *newval = alpha;
}

/* hhldr3rows */
#ifndef ANSI_C
static MAT hhldr3rows(A, k, i0, beta, nu1, nu2, nu3)
MAT A;
int	k, i0;
double beta, nu1, nu2, nu3;
#else
static MAT hhldr3rows(MAT A, int k, int i0, double beta,
                       double nu1, double nu2, double nu3)
#endif
{
  double ip, prod;
  int i, m;
  m = A.m;
  i0 = min(i0, m-1);
  for(i = 0;i <= i0;i++)
  {
    ip = nu1*m_entry(A, i, k) + nu2*m_entry(A, i, k+1)+nu3 * m_entry(A, i, k+2);
    prod = ip*beta;
    A.me[i][k] += (-prod*nu1);
    A.me[i][k+1] += (-prod*nu2);
    A.me[i][k+2] += (-prod*nu3);
  }
  return A;
}

/* givens -- returns c,s parameters for Givens rotation to
       eliminate y in the vector [ x y ]' */
#ifndef ANSI_C
void givens(x, y, c, s)
double x, y;
double *c, *s;
#else
void givens(double x, double y, double *c, double *s)
#endif
{
  double norm;
  norm = sp_sqrt(x*x+y*y);
  if(norm == 0.0)
  {
    *c = 1.0;
    *s = 0.0;
  }	/* identity */
  else
  {
    *c = x/norm;
    *s = y/norm;
  }
}

/* schur -- computes the Schur decomposition of the matrix A in situ
    -- optionally, gives Q matrix such that Q^T.A.Q is upper triangular
    -- returns upper triangular Schur matrix */
#ifndef ANSI_C
MAT schur(A, Q)
MAT A, Q;
#else
MAT schur(MAT A, MAT Q)
#endif
{
  int i, j, iter, k, k_min, k_max, k_tmp, n, split;
  double beta2, c, discrim, dummy, nu1, s, t, tmp, x, y, z;
  double A_me[MAX_SIZE][MAX_SIZE];
  double sqrt_macheps;
  static VEC diag, beta;
  n = A.n;
  diag = v_get(A.n);
  beta = v_get(A.n);
  /* compute Hessenberg form */
  A = Hfactor(A, diag, beta);
  /* save Q if necessary */
  Q = makeHQ(A, diag, beta);
  A = makeH(A);
  sqrt_macheps = sp_sqrt(MACHEPS);
  k_min = 0;
  memcpy(A_me, A.me, MAX_SIZE*MAX_SIZE*8);
  while(k_min < n)
  {
    double a00, a01, a10, a11;
    double scale, t, numer, denom;
    /* find k_max to suit:
       submatrix k_min..k_max should be irreducible */
    k_max = n-1;
    for(k = k_min; k < k_max; k++)
      if(m_entry(A, k+1, k) == 0.0)
      {
        k_max = k;
        break;
      }

    if(k_max <= k_min)
    {
      k_min = k_max + 1;
      continue;      /* outer loop */
    }
    /* check to see if we have a 2 x 2 block
       with complex eigenvalues */
    if(k_max == k_min + 1)
    {
      a00 = m_entry(A, k_min, k_min);
      a01 = m_entry(A, k_min, k_max);
      a10 = m_entry(A, k_max, k_min);
      a11 = m_entry(A, k_max, k_max);
      tmp = a00 - a11;
      discrim = tmp*tmp + 4*a01*a10;
      if(discrim < 0.0)
      {
        /* yes -- e-vals are complex
               -- put 2 x 2 block in form [a b; c a];
        then eigenvalues have real part a & imag part sp_sqrt(|bc|) */
        numer = - tmp;
        denom = (a01+a10 >= 0.0) ?
                (a01+a10) + sp_sqrt((a01+a10)*(a01+a10)+tmp*tmp) :
                (a01+a10) - sp_sqrt((a01+a10)*(a01+a10)+tmp*tmp);
        if(denom != 0.0)
        {    /* t = s/c = numer/denom */
          t = numer/denom;
          scale = c = 1.0/sp_sqrt(1+t*t);
          s = c*t;
        }
        else
        {
          c = 1.0;
          s = 0.0;
        }
        A = rot_cols(A, k_min, k_max, c, s);
        A = rot_rows(A, k_min, k_max, c, s);
        Q = rot_cols(Q, k_min, k_max, c, s);
        k_min = k_max + 1;
        continue;
      }
      else
      {
        /* discrim >= 0; i.e. block has two real eigenvalues */
        /* no -- e-vals are not complex;
         split 2 x 2 block and continue */
        /* s/c = numer/denom */
        numer = (tmp >= 0.0) ?
              - tmp - sp_sqrt(discrim) : - tmp + sp_sqrt(discrim);
        denom = 2*a01;
        if(sp_fabs(numer) < sp_fabs(denom))
        {    /* t = s/c = numer/denom */
          t = numer/denom;
          scale = c = 1.0/sp_sqrt(1+t*t);
          s = c*t;
        }
        else if(numer != 0.0)
        {    /* t = c/s = denom/numer */
          t = denom/numer;
          scale = 1.0/sp_sqrt(1+t*t);
          c = sp_fabs(t)*scale;
          s = (t >= 0.0) ? scale : -scale;
        }
        else /* numer == denom == 0 */
        {
          c = 0.0;
          s = 1.0;
        }
        A = rot_cols(A, k_min, k_max, c, s);
        A = rot_rows(A, k_min, k_max, c, s);
        Q = rot_cols(Q, k_min, k_max, c, s);
        k_min = k_max + 1;  /* go to next block */
        continue;
      }
    }
    /* now have r x r block with r >= 2:
     apply Francis QR step until block splits */
    split = 0;
    iter = 0;
    while(!split)
    {
      iter++;
      /* set up Wilkinson/Francis complex shift */
      k_tmp = k_max - 1;

      a00 = m_entry(A, k_tmp, k_tmp);
      a01 = m_entry(A, k_tmp, k_max);
      a10 = m_entry(A, k_max, k_tmp);
      a11 = m_entry(A, k_max, k_max);
      /* treat degenerate cases differently
         -- if there are still no splits after five iterations
            and the bottom 2 x 2 looks degenerate, force it to
         split */
      #ifdef DEBUG
        printf("# schur: bottom 2 x 2 = [%lg, %lg; %lg, %lg]\n",
               a00, a01, a10, a11);
      #endif
      if(iter >= 5 &&
         sp_fabs(a00-a11) < sqrt_macheps*(sp_fabs(a00)+sp_fabs(a11)) &&
         (sp_fabs(a01) < sqrt_macheps*(sp_fabs(a00)+sp_fabs(a11)) ||
          sp_fabs(a10) < sqrt_macheps*(sp_fabs(a00)+sp_fabs(a11))) )
      {
        if(sp_fabs(a01) < sqrt_macheps*(sp_fabs(a00)+sp_fabs(a11)))
        {
          A.me[k_tmp][k_max] = 0.0;
        }
        if(sp_fabs(a10) < sqrt_macheps*(sp_fabs(a00)+sp_fabs(a11)))
        {
          A.me[k_max][k_tmp] = 0.0;
          split = 1;
          continue;
        }
      }
      s = a00 + a11;
      t = a00*a11 - a01*a10;
      /* break loop if a 2 x 2 complex block */
      if(k_max == k_min + 1 && s*s < 4.0*t)
      {
        split = 1;
        continue;
      }
      /* perturb shift if convergence is slow */
      if((iter % 10) == 0)
      {
        s += iter*0.02;
        t += iter*0.02;
      }
      /* set up Householder transformations */
      k_tmp = k_min + 1;
      a00 = m_entry(A, k_min, k_min);
      a01 = m_entry(A, k_min, k_tmp);
      a10 = m_entry(A, k_tmp, k_min);
      a11 = m_entry(A, k_tmp, k_tmp);
      x = a00*a00 + a01*a10 - s*a00 + t;
      y = a10*(a00+a11-s);
      if(k_min + 2 <= k_max)
        z = a10*A.me[k_min+2][k_tmp];
      else
        z = 0.0;
      for(k = k_min; k <= k_max-1; k++)
      {
        if(k < k_max - 1)
        {
          hhldr3(x, y, z, &nu1, &beta2, &dummy);
          Q = hhldr3rows(Q, k, n-1, beta2, nu1, y, z);
        }
        else
        {
          givens(x, y, &c, &s);
          A = rot_cols(A, k, k+1, c, s);
          A = rot_rows(A, k, k+1, c, s);
          Q = rot_cols(Q, k, k+1, c, s);
        }
        x = m_entry(A, k+1, k);
        if(k <= k_max - 2)
          y = m_entry(A, k+2, k);
        else
          y = 0.0;
        if(k <= k_max - 3)
          z = m_entry(A, k+3, k);
        else
          z = 0.0;
      }
	  for(k = k_min; k <= k_max-2; k++)
	  {
        /* zero appropriate sub-diagonals */
		A.me[k+2][k] = 0.0;
        if(k < k_max-2)
        {
          A.me[k+3][k] = 0.0;
        }
      }

      /* test to see if matrix should split */
      for(k = k_min; k < k_max; k++)
        if(sp_fabs(A_me[k+1][k]) < MACHEPS*
          (sp_fabs(A_me[k][k])+sp_fabs(A_me[k+1][k+1])))
        {
          A_me[k+1][k] = 0.0;
          split = 1;
        }
	}
  }
  /* polish up A by zeroing strictly lower triangular elements
     and small sub-diagonal elements */
  for(i = 0; i < A.m; i++)
    for(j = 0; j < i-1; j++)
      A_me[i][j] = 0.0;
    for(i = 0; i < A.m - 1; i++)
      if(sp_fabs(A_me[i+1][i]) < MACHEPS*
         (sp_fabs(A_me[i][i])+sp_fabs(A_me[i+1][i+1])))
        A_me[i+1][i] = 0.0;
  return A;
}

/* schur_vals -- compute real & imaginary parts of eigenvalues
	-- assumes T contains a block upper triangular matrix
		as produced by schur()
	-- real parts stored in real_pt, imaginary parts in imag_pt */
#ifndef ANSI_C
void schur_evals(T, real_pt, imag_pt)
MAT *T;
VEC *real_pt, *imag_pt;
#else
void schur_evals(MAT *T, VEC *real_pt, VEC *imag_pt)
#endif
{
  int i, n;
  double discrim, T_me[MAX_SIZE][MAX_SIZE];
  double diff, sum, tmp;
  n = T->n;
  memcpy(T_me, T->me, MAX_SIZE*MAX_SIZE*8);
  i = 0;
  while(i < n)
  {
    if(i < n-1 && T_me[i+1][i] != 0.0)
    {   /* should be a complex eigenvalue */
      sum  = 0.5*(T_me[i][i]+T_me[i+1][i+1]);
      diff = 0.5*(T_me[i][i]-T_me[i+1][i+1]);
      discrim = diff*diff + T_me[i][i+1]*T_me[i+1][i];
      if(discrim < 0.0)
      { /* yes -- complex e-vals */
        real_pt->ve[i] = real_pt->ve[i+1] = sum;
        imag_pt->ve[i] = sp_sqrt(-discrim);
        imag_pt->ve[i+1] = - imag_pt->ve[i];
      }
      else
      { /* no -- actually both real */
        tmp = sp_sqrt(discrim);
        real_pt->ve[i]   = sum + tmp;
        real_pt->ve[i+1] = sum - tmp;
        imag_pt->ve[i]   = imag_pt->ve[i+1] = 0.0;
      }
      i += 2;
    }
    else
    {   /* real eigenvalue */
      real_pt->ve[i] = T_me[i][i];
      imag_pt->ve[i] = 0.0;
      i++;
    }
  }
}

/******************************Verification Functions******************************/

/* m_get_eigenvalues -- get the eigenvalues of a matrix A
	-- */
CMPLX *m_get_eigenvalues(MAT A)
{
  MAT T, Q;
  VEC evals_re, evals_im;
  static CMPLX z[MAX_SIZE];
  Q = m_get(A.m, A.n);
  T = m_copy(A);
  /* compute Schur form: A = Q.T.Q^T */
  T = schur(T, Q);
  /* extract eigenvalues */
  evals_re = v_get(A.m);
  evals_im = v_get(A.m);
  schur_evals(&T, &evals_re, &evals_im);
  for(int i = 0; i < evals_re.dim; i++)
  {
    z[i].real = evals_re.ve[i];
    z[i].imag = evals_im.ve[i];
  }
  return z;
}

/* cmplx_mag -- get the magnitude of a complex number taking its real
 * and imaginary parts */
double cmplx_mag(double real, double imag)
{
  return sp_sqrt(real * real + imag * imag);
}

/* is_same_sign -- check if a has the same sign as b */
int is_same_sign(double a, double b)
{
  if(((a >= 0) && (b >= 0)) || ((a <= 0) && (b <= 0)))
    return 1;
  else
    return 0;
}

/* x_k -- computes the state signal in the k-th sample */
void x_k(MAT A, MAT B, double u, int k)
{
  MAT AUX, AUX2, AUX3;
  if(xk.lastState == (k - 1))
  {
	AUX = m_get(A.m, xk.xk.n);
    AUX = m_mlt(A, xk.xk);
    AUX2 = m_get(B.m, B.n);
    AUX2 = sm_mlt(u, B);
    AUX3 = m_get(B.m, B.n);
    AUX3 = m_add(AUX, AUX2);
    xk.lastState = k;
    xk.xk = AUX3;
  }
}

/* y_k2 -- computes the output signal in the k-th sample */
double y_k2(MAT A, MAT B, MAT C, MAT D, double u, int k)
{
  MAT Ak, AUX, AUX2;
  double y, temp;
  // y[k]=Cx[k]+Du[k]
  AUX = m_get(C.m, xk.xk.n);
  x_k(A, B, u, k);
  AUX = m_mlt(C, xk.xk);
  temp = D.me[0][0] * u;
  y = AUX.me[0][0] + temp;
  return y;
}

/* x_k2 -- computes the state signal in the k-th sample */
MAT x_k2(MAT A, MAT B, MAT C, MAT D, double u, int k, MAT X0)
{
  MAT x, Ak, AUX, AUX2;
  MAT AUX3, AUX4, AUX5;
  // y = C * A.pow(k) * X0;
  Ak = m_get(A.m, A.n);
  Ak = m_pow(A, k);
  x = m_get(A.m, 1);
  x = m_mlt(Ak, X0);
  AUX = m_get(A.m, 1);
  AUX2 = m_get(A.m, 1);
  for(int m = 0; m <= (k - 1); m++)
  {
    // y += (C * A.pow(k - m - 1) * B * u) + D * u;
    Ak = m_pow(A, (k-m-1));
    AUX = m_mlt(Ak, B);
    AUX2 = sm_mlt(u, AUX);
    x = m_add(x, AUX2);
  }
  xk.lastState = k;
  xk.xk = m_copy(x);
  return x;
}

/* y_k3 -- computes the output signal in the k-th sample */
double y_k3(MAT A, MAT B, MAT C, MAT D, double u, int k, MAT X0)
{
  MAT Ak, Xk, AUX, AUX2;
  double y, temp;
  Xk = m_get(A.m, 1);
  Xk = x_k2(A, B, C, D, u, k, X0);
  // y[k]=Cx[k]+Du[k]
  AUX = m_get(C.m, Xk.n);
//  x_k(A, B, u, k);
  AUX = m_mlt(C, Xk);
  AUX2 = m_get(D.m, D.n);
  AUX2 = sm_mlt(u, D);
  y = AUX.me[0][0] + AUX2.me[0][0];
  return y;
}

/* y_k -- computes the output signal in the k-th sample */
double y_k(MAT A, MAT B, MAT C, MAT D, double u, int k, MAT X0)
{
  MAT y, Ak, AUX, AUX2;
  MAT AUX3, AUX4, AUX5;
  // y = C * A.pow(k) * X0;
  Ak = m_get(A.m, A.n);
  Ak = m_pow(A, k);
  AUX = m_get(A.m, A.n);
  AUX = m_mlt(C, Ak);
  y = m_get(A.m, A.n);
  y = m_mlt(AUX, X0);
  AUX2 = m_get(A.m, A.n);
  for(int m = 0; m <= (k - 1); m++)
  {
    // y += (C * A.pow(k - m - 1) * B * u) + D * u;
    Ak = m_pow(A, (k-m-1));
    AUX = m_mlt(C, Ak);
    AUX2 = m_mlt(AUX, B);
    AUX5 = m_add(AUX2, D);
    y = m_add(y, AUX5);
  }
  return y.me[0][0]*u;
}

/* peak_output -- computes the biggest peak value of a signal (Mp) */
PKVL peak_output(MAT A, MAT B, MAT C, MAT D, MAT X0, double yss, double u)
{
  PKVL out;
  double greater;
  int i = 0;
  greater = sp_fabs(y_k2(A, B, C, D, u, i));
  while((sp_fabs(y_k2(A, B, C, D, u, i+1)) >= greater))
  {
    if(greater < sp_fabs(y_k2(A, B, C, D, u, i+1)))
    {
      greater = sp_fabs(y_k2(A, B, C, D, u, i+1));
      out.mp = y_k2(A, B, C, D, u, i+1);
      out.kp = i+2;
    }
    else
    {
      out.mp = y_k2(A, B, C, D, u, i+1);
      out.kp = i+2;
    }
    if(!is_same_sign(yss, out.mp))
    {
      greater = 0;
    }
    i++;
  }
  return out;
}

/* y_ss -- computes steady-state output value of a given system */
double y_ss(MAT A, MAT B, MAT C, MAT D, double u)
{
  double yss;
  MAT AUX, AUX2, AUX3, AUX4, AUX5;
  MAT Id;
  // get the expression y_ss=(C(I-A)^(-1)B+D)u
  Id = m_get(A.m, A.n);
  Id = m_ident(Id.m);/*printf("I\n");m_output(Id);*/
  AUX = m_get(A.m, A.n);
  // Id - A
  AUX = m_sub(Id, A);/*printf("I-A\n");m_output(AUX);*/
  AUX2 = m_get(A.m, A.n);
  AUX2 = m_inverse(AUX);/*printf("(I-A)^(-1)\n");m_output(AUX2);*/
  AUX3 = m_get(A.m, A.n);
  AUX3 = m_mlt(C, AUX2);/*printf("C(I-A)^(-1))\n");m_output(AUX3);*/
  AUX4 = m_get(A.m, A.n);
  AUX4 = m_mlt(AUX3, B);/*printf("C(I-A)^(-1)B\n");m_output(AUX4);*/
  AUX5 = m_get(A.m, A.n);
  AUX5 = m_add(AUX4, D);/*printf("(C(I-A)^(-1)B+D)\n");m_output(AUX5);*/
  yss = AUX5.me[0][0] * u;/*printf("yss=\n");*/
  return yss;
}

/* c_bar -- computes an auxiliary variable to calculate k_bar */
double c_bar(double mp, double yss, double lambmax, int kp)
{
  double cbar;
  cbar = (mp-yss)/(sp_pow(lambmax, kp));
  return cbar;
}

/* log_b -- computes the log of x in the base 'base' */
double log_b(double base, double x)
{
  return (double) (sp_log10_2(x) / sp_log10_2(base));
}

/* k_bar -- computes instant in which the system enters in the settling
 * -time region */
int k_bar(double lambdaMax, double p, double cbar, double yss, int order)
{
  double k_ss, x;
  x = (p * yss) / (100 * cbar);
  k_ss = log_b(lambdaMax, x);
  return sp_ceil(k_ss)+order;
}

/* max_mag_eigenvalue -- computes biggest magnitude among the eigenvalues */
double max_mag_eigenvalue(MAT A)
{
  double maximum = 0, aux;
  CMPLX *z;
  z = m_get_eigenvalues(A);
  for(int i = 0; i < A.m; i++)
  {
    aux = cmplx_mag(z[i].real, z[i].imag);
    if(aux > maximum)
    {
      maximum = aux;
    }
  }
  return maximum;
}

/* check_settling_time -- check if a given settling time satisfies to
 * a given system */
int check_settling_time(MAT A, MAT B, MAT C, MAT D, MAT X0,
                        double u, double tsr, double p, double ts)
{
  double yss, mp, lambMax, cbar, output, inf, sup;
  PKVL out;
  int kbar, kp, i;
  xk.xk = m_get(A.m, 1);
  xk.lastState = 0;
  yss = y_ss(A, B, C, D, u);
  out = peak_output(A, B, C, D, X0, yss, u);
  mp = out.mp;
  kp = out.kp;
  lambMax = max_mag_eigenvalue(A);
  cbar = c_bar(mp, yss, lambMax, kp);
  kbar = k_bar(lambMax, p, cbar, yss, A.m);
  #ifdef DEBUG
  printf("Mp=%f\n", mp);
  printf("yss=%f\n", yss);
  printf("lambMax=%f\n", lambMax);
  printf("kp=%d\n", kp);
  printf("cbar=%f\n", cbar);
  #endif
  if(kbar * ts < tsr)
  {
    #ifdef DEBUG
    printf("kbar=%d\n", kbar);
    #endif
    return 1;
  }
  i = (int)sp_ceil(tsr / ts)-1;
  output = y_k3(A, B, C, D, u, i, X0);
  while(i <= kbar)
  {
    if(yss > 0)
    {
      inf = (yss - (yss * (p/100)));
      sup = (yss * (p/100) + yss);
    }
    else
    {
      sup = (yss - (yss * (p/100)));
      inf = (yss * (p/100) + yss);
    }
    if(!(output < sup && (output > inf)))
    {
      #ifdef DEBUG
      printf("kbar=%d\n", kbar);
      #endif
      return 0;
    }
    i++;
    output = y_k2(A, B, C, D, u, i);
  }
  return 1;
}

int verify_settling_time(void)
{
  double ts, tsr, p, u;
  int i, j, kbar, k_ss;
  MAT A, B, C, D, X0;
  tsr = _controller.tsr;
  ts = _controller.ts;
  p = _controller.p;
  u = (double)_controller.inputs[0][0];
  A = m_get(_controller.nStates, _controller.nStates);
  B = m_get(_controller.nStates, 1);
  C = m_get(1, _controller.nStates);
  D = m_get(1, 1);
  X0 = m_get(1, _controller.nStates);
  // Parsing matrices
  // Matrix A
  for(i = 0;i < _controller.nStates;i++)
  {
	for(j = 0;j < _controller.nStates;j++)
	{
      A.me[i][j] = _controller.A[i][j];
	}
  }
  // Matrix B
  for(i = 0;i < _controller.nStates;i++)
  {
    for(j = 0;j < 1;j++)
  	{
      B.me[i][j] = _controller.B[i][j];
  	}
  }
  // Matrix C
  for(i = 0;i < 1;i++)
  {
  	for(j = 0;j < _controller.nStates;j++)
  	{
      C.me[i][j] = _controller.C[i][j];
  	}
  }
  // Matrix D
  for(i = 0;i < 1;i++)
  {
  	for(j = 0;j < 1;j++)
  	{
      D.me[i][j] = _controller.D[i][j];
  	}
  }
  // Matrix X0
  for(i = 0;i < 1;i++)
  {
   	for(j = 0;j < _controller.nStates;j++)
   	{
      X0.me[i][j] = _controller.x0[i][j];
  	}
  }
  assert(check_settling_time(A, B, C, D, X0, u, tsr, p, ts) == 0);
  return 0;
}

#endif // DSVERIFIER_ENGINE_VERIFY_SETTLING_TIME_H
