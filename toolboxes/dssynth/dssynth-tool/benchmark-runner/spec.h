#define NSTATES 3
#define NINPUTS 1
#define NOUTPUTS 1
#define INT_BITS 13
#define FRAC_BITS 3
const __CPROVER_EIGEN_fixedbvt _controller_A[NSTATES][NSTATES] = { { 0.9905,0.075687,0.021033 }, { 0.125,0,0 }, { 0,0.015625,0 } };
const __CPROVER_EIGEN_fixedbvt _controller_B[NSTATES] = { 16, 0, 0 };
#define INPUT_LOWERBOUND (__CPROVER_EIGEN_fixedbvt)-1
#define INPUT_UPPERBOUND (__CPROVER_EIGEN_fixedbvt)1
