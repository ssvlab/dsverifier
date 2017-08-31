function [output, time_execution] = simulate_stability(system)
%
% Script developed to simulate the stability property in counterexamples
%
% Give the system as a struct with all parameters of counterexample and matlab will check stability property for delta and direct forms.
%
% Function: [output, time_execution] = simulate_stability(system)
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

if (system.impl.delta ~= 0)
    [Da, Db] = deltapoly(system.sys.b, system.sys.a, system.impl.delta);
    fxp_a = fxp_rounding(Da, system.impl.frac_bits);
    fxp_b = fxp_rounding(Db, system.impl.frac_bits);
    rootsb = roots(fxp_b);
    rootsa = roots(fxp_a);
else
    fxp_a = fxp_rounding(system.sys.a, system.impl.frac_bits);
    fxp_b = fxp_rounding(system.sys.b, system.impl.frac_bits);
    rootsb = roots(fxp_b);
    rootsa = roots(fxp_a);
end

decision = 0;

for i=1:length(rootsa)
    if abs(rootsa(i))>=1
        %The system is UNSTABLE
        decision = 0;
        break
    end
    %The system is STABLE
    decision = 1;
end

if decision == 1
    output = 'Successful';
else
    output = 'Failed';
end

time_execution = toc;

end
