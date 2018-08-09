#include <assert.h>

#include "../../../bmc/core/definitions.h"
#include "../../../bmc/core/compatibility.h"
#include "../../../bmc/core/fixed_point.h"
#include "../../../bmc/core/realizations.h"
#include "../../../bmc/core/util.h"
#include "../../../bmc/core/functions.h"
#include "../../../bmc/core/initialization.h"
#include "../../../bmc/core/closed_loop.h"

digital_system control = {
	.b = { 2020.0, -3254.0, -126.1, 2096.0, -735.9 },
	.b_size = 5,
	.a = { 1.0, -2.9, 2.8, -0.9 },
	.a_size = 4,
	.sample_time = 0.1
};

digital_system plant = {
	.b = { 0.01551, -0.0001007, -0.01541 },
	.b_uncertainty = {9.8001, 9.4935, 9.7988},
	.b_size = 3,
	.a = { 1.0, -3.225, 3.203, -0.9808},
	.a_uncertainty = {0, 2.4496, 2.4664, 0.102},
	.a_size = 4,
	.sample_time = 0.1
};

implementation impl = {
	.int_bits = 15,
	.frac_bits = 4
};

digital_system plant_cbmc;

double nondet_double();
void call_closedloop_verification_task();

int main(){

	rounding_mode = FLOOR;

	initialization();
	call_closedloop_verification_task();

	/* generating closed loop for series or feedback */
	double * c_num = control.b;
	int c_num_size = control.b_size;
	double * c_den = control.a;
	int c_den_size = control.a_size;

	/* quantizing controller coefficients */
	fxp32_t c_num_fxp[control.b_size];
	fxp_double_to_fxp_array(c_num, c_num_fxp, control.b_size);
	fxp32_t c_den_fxp[control.a_size];
	fxp_double_to_fxp_array(c_den, c_den_fxp, control.a_size);

	/* getting quantized controller coefficients  */
	double c_num_qtz[control.b_size];
	fxp_to_double_array(c_num_qtz, c_num_fxp, control.b_size);
	double c_den_qtz[control.a_size];
	fxp_to_double_array(c_den_qtz, c_den_fxp, control.a_size);

	/* getting plant coefficients */
	#if (BMC == ESBMC)
		double * p_num = plant.b;
		int p_num_size = plant.b_size;
		double * p_den = plant.a;
		int p_den_size = plant.a_size;
	#elif (BMC == CBMC)
		double * p_num = plant_cbmc.b;
		int p_num_size = plant.b_size;
		double * p_den = plant_cbmc.a;
		int p_den_size = plant.a_size;
	#endif

	double ans_num[100];
	int ans_num_size = control.b_size + p_num_size - 1;
	double ans_den[100];
	int ans_den_size = control.a_size + p_den_size - 1;

	ft_closedloop_series(c_num_qtz, c_num_size, c_den_qtz, c_den_size, p_num, p_num_size, p_den, p_den_size, ans_num, ans_num_size, ans_den, ans_den_size);

	/* checking stability */
	printf("Verifying stability for closedloop function\n");
	__DSVERIFIER_assert(check_stability_closedloop(ans_den, ans_den_size, p_num, p_num_size, p_den, p_den_size));

	return 0;

}

/** call the closedloop verification task */
void call_closedloop_verification_task(){

	/* base case is the execution using all parameters without uncertainty */
	_Bool base_case_executed = 0;

	/* considering uncertainty for numerator coefficients */
	int i=0;
	for(i=0; i<plant.b_size; i++){
		if (plant.b_uncertainty[i] > 0){
			double factor = ((plant.b[i] * plant.b_uncertainty[i]) / 100);
			factor = factor < 0 ? factor * (-1) : factor;
			double min = plant.b[i] - factor;
			double max = plant.b[i] + factor;

			/* Eliminate redundant executions  */
			if ((factor == 0) && (base_case_executed == 1)){
				continue;
			}else if ((factor == 0) && (base_case_executed == 0)){
				base_case_executed = 1;
			}

			#if (BMC == ESBMC)
				plant.b[i] = nondet_double();
				__DSVERIFIER_assume((plant.b[i] >= min) && (plant.b[i] <= max));
			#elif (BMC == CBMC)
				plant_cbmc.b[i] = nondet_double();
				__DSVERIFIER_assume((plant_cbmc.b[i] >= min) && (plant_cbmc.b[i] <= max));
			#endif
		}else{
			#if (BMC == CBMC)
				plant_cbmc.b[i] = plant.b[i];
			#endif
		}
	}

	/* considering uncertainty for denominator coefficients */
	for(i=0; i<plant.a_size; i++){
		if (plant.a_uncertainty[i] > 0){
			double factor = ((plant.a[i] * plant.a_uncertainty[i]) / 100);
			factor = factor < 0 ? factor * (-1) : factor;

			double min = plant.a[i] - factor;
			double max = plant.a[i] + factor;

			/* eliminate redundant executions  */
			if ((factor == 0) && (base_case_executed == 1)){
				continue;
			}else if ((factor == 0) && (base_case_executed == 0)){
				base_case_executed = 1;
			}

			#if (BMC == ESBMC)
				plant.a[i] = nondet_double();
				__DSVERIFIER_assume((plant.a[i] >= min) && (plant.a[i] <= max));
			#elif (BMC == CBMC)
				plant_cbmc.a[i] = nondet_double();
				__DSVERIFIER_assume((plant_cbmc.a[i] >= min) && (plant_cbmc.a[i] <= max));
			#endif
		}else{
			#if (BMC == CBMC)
				plant_cbmc.a[i] = plant.a[i];
			#endif
		}
	}
}
