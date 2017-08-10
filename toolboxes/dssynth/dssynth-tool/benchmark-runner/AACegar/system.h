struct implt impl={ .int_bits=12, .frac_bits=6, .mult_bits=0};
struct coefft plant={.coeffs={0.87892 , 0.011578 , 0.112303 }, .uncertainty={8.70526e-075 , 3.67823e-074 , 2.59161e-074 }};
struct transformt transform={.coeffs={ { 0.5 , 2.37495e-077 , -0.011578 } ,  { 0 , 0.5 , 2.37495e-077 } ,  { 0 , 0 , 0.25 } }, .uncertainty={ { 0 , 2.17416e-075 , 7.28474e-075 } ,  { 0 , 0 , 2.17416e-075 } ,  { 0 , 0 , 0 } }};
matrixt dynamics={ { 0.87892 , -0.011578 , 0.26531 } ,  { 1 , 0 , 0 } ,  { 0 , 0.5 , 0 } };
vectort sensitivity={0.5 , 0 , 0 };
#ifdef __CPROVER
extern vectort controller;
#else
vectort controller={1.3515 , 0.8405 , 0 };
#endif
void boundController()
{
verify_assume (((control_floatt)2*controller[0]-(control_floatt)2*controller[1]-(control_floatt)2*controller[2]<(control_floatt)2)
 && (-(control_floatt)2*controller[0]+(control_floatt)2*controller[1]+(control_floatt)2*controller[2]<(control_floatt)2)
 && ((control_floatt)2*controller[0]-(control_floatt)2*controller[1]+(control_floatt)2*controller[2]<(control_floatt)2)
 && (-(control_floatt)2*controller[0]+(control_floatt)2*controller[1]-(control_floatt)2*controller[2]<(control_floatt)2)
 && ((control_floatt)2*controller[0]+(control_floatt)2*controller[1]+(control_floatt)2*controller[2]<(control_floatt)2)
 && (-(control_floatt)2*controller[0]-(control_floatt)2*controller[1]-(control_floatt)2*controller[2]<(control_floatt)2)
 && ((control_floatt)2*controller[0]+(control_floatt)2*controller[1]-(control_floatt)2*controller[2]<(control_floatt)2)
 && (-(control_floatt)2*controller[0]-(control_floatt)2*controller[1]+(control_floatt)2*controller[2]<(control_floatt)2)
 && (-(control_floatt)2*controller[0]+(control_floatt)2*controller[1]+(control_floatt)2*controller[2]<(control_floatt)2)
 && ((control_floatt)2*controller[0]-(control_floatt)2*controller[1]-(control_floatt)2*controller[2]<(control_floatt)2)
 && (-(control_floatt)2*controller[0]+(control_floatt)2*controller[1]-(control_floatt)2*controller[2]<(control_floatt)2)
 && ((control_floatt)2*controller[0]-(control_floatt)2*controller[1]+(control_floatt)2*controller[2]<(control_floatt)2)
 && (-(control_floatt)2*controller[0]-(control_floatt)2*controller[1]+(control_floatt)2*controller[2]<(control_floatt)2)
 && ((control_floatt)2*controller[0]+(control_floatt)2*controller[1]-(control_floatt)2*controller[2]<(control_floatt)2)
 && (-(control_floatt)2*controller[0]-(control_floatt)2*controller[1]-(control_floatt)2*controller[2]<(control_floatt)2)
 && ((control_floatt)2*controller[0]+(control_floatt)2*controller[1]+(control_floatt)2*controller[2]<(control_floatt)2)
);
verify_assume (((controller[0]-controller[1]-controller[2]<(control_floatt)0.295936)
 || (controller[0]-controller[1]+controller[2]<(control_floatt)0.295936)
 || (controller[0]+controller[1]+controller[2]<(control_floatt)0.295936)
 || (controller[0]+controller[1]-controller[2]<(control_floatt)0.295936)
 || (-controller[0]+controller[1]+controller[2]<(control_floatt)0.295936)
 || (-controller[0]+controller[1]-controller[2]<(control_floatt)0.295936)
 || (-controller[0]-controller[1]+controller[2]<(control_floatt)0.295936)
 || (-controller[0]-controller[1]-controller[2]<(control_floatt)0.295936)
) && ((-controller[0]+controller[1]+controller[2]<(control_floatt)0.295936)
 || (-controller[0]+controller[1]-controller[2]<(control_floatt)0.295936)
 || (-controller[0]-controller[1]-controller[2]<(control_floatt)0.295936)
 || (-controller[0]-controller[1]+controller[2]<(control_floatt)0.295936)
 || (controller[0]-controller[1]-controller[2]<(control_floatt)0.295936)
 || (controller[0]-controller[1]+controller[2]<(control_floatt)0.295936)
 || (controller[0]+controller[1]-controller[2]<(control_floatt)0.295936)
 || (controller[0]+controller[1]+controller[2]<(control_floatt)0.295936)
));
}
#define _NUM_VERTICES 8
#define _NUM_INPUT_VERTICES 8
#define _NUM_VECTORS 6
control_floatt vertices[_NUM_VERTICES][_DIMENSION]={ { 1 , -1 , -1 } ,  { 1 , -1 , 1 } ,  { 1 , 1 , 1 } ,  { 1 , 1 , -1 } ,  { -1 , 1 , 1 } ,  { -1 , 1 , -1 } ,  { -1 , -1 , 1 } ,  { -1 , -1 , -1 } };
control_floatt input_vertices[_NUM_INPUT_VERTICES][_DIMENSION]={ { 0.502047 , 0.00204688 , 0.00204688 } ,  { 0.502047 , 0.00204688 , -0.00204688 } ,  { 0.502047 , -0.00204688 , -0.00204688 } ,  { 0.502047 , -0.00204688 , 0.00204688 } ,  { -0.502047 , 0.00204688 , 0.00204688 } ,  { -0.502047 , 0.00204688 , -0.00204688 } ,  { -0.502047 , -0.00204688 , 0.00204688 } ,  { -0.502047 , -0.00204688 , -0.00204688 } };
control_floatt accel_vertices[_NUM_INPUT_VERTICES][_DIMENSION];
control_floatt vectors[_NUM_VECTORS][_DIMENSION]={ { 1 , 0 , 0 } ,  { -1 , 0 , 0 } ,  { 0 , 1 , 0 } ,  { 0 , -1 , 0 } ,  { 0 , 0 , 1 } ,  { 0 , 0 , -1 } };
control_floatt reach_vertices[_NUM_INPUT_VERTICES][_NUM_VERTICES][_DIMENSION];
control_floatt supports[_NUM_VECTORS]={2 , 2 , 2 , 2 , 2 , 2 };
control_floatt accelsupports[_NUM_INPUT_VERTICES][_NUM_VECTORS];

#define _NUM_ITERATIONS 1
int iterations[_NUM_ITERATIONS]={2};