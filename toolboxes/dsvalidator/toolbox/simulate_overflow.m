function [output, time_execution] = simulate_overflow(system)
%
% Script developed to simulate the overflow property in counterexamples
%
% Give the system as a struct with all parameters of counterexample and matlab will check overflow property for direct and delta forms.
%
% Function: [output, time_execution] = simulate_overflow(system)
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
%
% The output is the feedback returned from simulation;
% The time execution is the time to execute the simulation;
%
% Federal University of Amazonas
% May 15, 2017
% Manaus, Brazil

tic

global property;
global round_mode;

property = 'overflow';
round_mode = 'double';

wl = system.impl.frac_bits;

if (system.impl.delta > 0)
    [a_num, b_num] = deltapoly(system.sys.b, system.sys.a, system.impl.delta);
else
    a_num = system.sys.a;
    b_num = system.sys.b;
end

a_fxp = fxp_rounding(a_num,wl);
b_fxp = fxp_rounding(b_num,wl);

x =  system.inputs.const_inputs;
y =  zeros(1,system.impl.x_size);

if strcmp(property,'overflow')
    y = dlsim(b_fxp,a_fxp, x);
end

output = y';
time_execution = toc;
end
