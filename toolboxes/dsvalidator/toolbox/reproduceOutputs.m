function output =  reproduceOutputs(system, proper)
%
% Script developed to reproduce the counterexamples using DFI, DFII and TDFII realizations, 
% that considers the FWL effects.
%
% Give the system as a struct with all parameters of counterexample and the function will simulate the system according to overflow mode choosen.
% 
% Function: output =  reproduceOutputs(system, property)
%
% The struct 'system' should have the following features:
% system.sys.a = denominator;
% system.sys.b = numerator;
% system.sys.tf = transfer function system representation
% system.impl.frac_bits = fractionary bits
% system.impl.int_bits = integer bits
% system.impl.realization_form = realization, and it should be DFI, DFII, TDFII, DDFI, DDFII or TDDFII
% system.inputs.const_inputs = the inputs from counterexample
% system.inputs.initial_states = the initial states from counterexample
% system.outputs.output_verification = the output extracted from counterexample
% system.impl.delta = in delta form realizations, the delta operator should be informed
% system.impl.sample_time = sample time of realization
% system.impl.x_size = the bound size
%
% And property is related to overflow or limit cycle.
%
% The output is a struct with the results considering the overflow mode effects.
%
% Federal University of Amazonas
% May 15, 2017
% Manaus, Brazil

global overflow_mode;
global round_mode;
global property;

overflow_mode = 'wrap';
round_mode = 'round';
property = proper;

if strcmp(property,'limit_cycle')
 
    num = fwl(system.sys.b,system.impl.frac_bits)
    den = fwl(system.sys.a,system.impl.frac_bits)
    system.sys.b = num;
    system.sys.a = den;

    if strcmp(system.impl.realization_form,'DFI') || strcmp(system.impl.realization_form,'DDFI')
        output_no_fwl_effects = dsv_df1(system);
    elseif strcmp(system.impl.realization_form,'DFII') || strcmp(system.impl.realization_form,'DDFII')
        output_no_fwl_effects = dsv_df2(system);
    elseif strcmp(system.impl.realization_form,'TDFII') || strcmp(system.impl.realization_form,'TDDFII')
       output_no_fwl_effects = dsv_tdf2(system);
    end

end

if strcmp(property,'overflow')
overflow_mode = 'saturate';
round_mode = 'floor';
num = fxp_rounding(system.sys.b,system.impl.frac_bits);
den = fxp_rounding(system.sys.a,system.impl.frac_bits);

for i=1:length(num)
num(i) = mode_saturate(num(i),system.impl.int_bits,system.impl.frac_bits);
end

for i=1:length(den)
den(i) = mode_saturate(den(i),system.impl.int_bits,system.impl.frac_bits);
end

output_no_fwl_effects = dlsim(num, den, system.inputs.const_inputs)';

for i=1:length(output_no_fwl_effects)
y(i) = mode_saturate(output_no_fwl_effects(i),system.impl.int_bits,system.impl.frac_bits);
end
output_no_fwl_effects = y;
end 

output_with_fwl_effects = system.output.output_simulation;

 disp('Output Reproduced from Simulation:');   
 disp(output_with_fwl_effects);
 disp('Output Reproduced without FWL effects:');
 disp(output_no_fwl_effects);
 
 output.output_with_fwl_effects = output_with_fwl_effects;
 output.output_no_fwl_effects = output_no_fwl_effects;
 
end
