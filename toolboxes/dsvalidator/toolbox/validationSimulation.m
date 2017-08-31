function [output, time_execution] = validationSimulation(system,p)
%
% Script to simulate and validate a property for a system automatically.
%
% Function: [output, time_execution] = validationSimulation(system, p)
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
% And the parameter 'p' is the property to be analyzed: (m) for minimum phase, (s) for stability, (o) for overflow and (lc) for limit cycle.
% (scl) for stability in closed-loop systems, (sss) for stability in state-space format, (ssc) for controllability in state-space format and (sso) for observability in state-space format.
%
% The output is the feedback returned from simulation;
% The time execution is the time to execute the simulation;
%
% Federal University of Amazonas
% May 15, 2017
% Manaus, Brazil

switch p
    case 'lc'
        [output, time_execution]  = simulate_limit_cycle(system);
    case 's'
        [output, time_execution]  = simulate_stability(system);
    case 'm'
        [output, time_execution]  = simulate_minimum_phase(system);
    case 'o'
        [output, time_execution]  = simulate_overflow(system);
    case 'scl'
        [output, time_execution]  = simulate_cl_stability(system);
    case 'sss'
        [output, time_execution]  = simulate_ss_stability(system);
    case 'sso'
        [output, time_execution]  = simulate_ss_observability(system);
    case 'ssc'
        [output, time_execution]  = simulate_ss_controllability(system);
    otherwise
        warning('Unexpected property or error during the automatic validation.')
end

end
