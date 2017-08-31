function [y, time_execution] = realizationDF1(system)
%
% Simulate and reproduce a counterexample for limit cycle using DFI realization.
% In case of delta form (DDFI), the delta operator should be represented in system struct.
%
% Function: [y, time_execution] = realizationDF1(system)
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
% The parameter 'y' is the output returned from simulation;
% The time execution is the time to execute the simulation;
%
% Federal University of Amazonas
% May 15, 2017
% Manaus, Brazil

tic

global property;

wl = system.impl.frac_bits;

if (system.impl.delta > 0)
    [a_num, b_num] = deltapoly(system.sys.b, system.sys.a, system.impl.delta);
else
    a_num = system.sys.a;
    b_num = system.sys.b;
end

a_fxp = fxp_rounding(a_num,wl);
b_fxp = fxp_rounding(b_num,wl);

a_fxp = fwl(a_fxp,wl);
b_fxp = fwl(b_fxp,wl);


x_size = system.impl.x_size;

Na = length(a_fxp);
Nb = length(b_fxp);

x_aux = system.inputs.const_inputs(1:Nb);

if (strcmp(property,'limit_cycle'))
    y_aux = system.inputs.initial_states;
end

x =  system.inputs.const_inputs;
y =  zeros(1,x_size);

if (strcmp(property,'overflow'))||(strcmp(property,'error'))
    y_aux = zeros(1,Na);
    x_aux = zeros(1,Nb);
end

y_aux = fliplr(y_aux);
%% DFI Realization
for i=1:x_size
    sum = 0;
    a_ptr = a_fxp;
    b_ptr = b_fxp;
    
    x_aux = shiftL(x(i), x_aux, Nb);
    y_ptr = y_aux;
    x_ptr = x_aux;
    
    for j=1:Nb
        sum = fxp_add(sum, fxp_mult(b_ptr(j), x_ptr(j), wl), wl);
    end
    
    for k=2:Na
        sum = fxp_sub(sum, fxp_mult(a_ptr(k), y_ptr(k-1),wl),wl);
    end
    
    sum = fxp_div(sum,a_fxp(1),wl);
    
    y(i) = fxp_quantize(sum, system.impl.int_bits, system.impl.frac_bits);
    
    y_aux = shiftL(y(i), y_aux, Na);
    
end

y = fliplr(y);

if strcmp(property,'overflow')
    y = dlsim(b_fxp,a_fxp, x);
end

time_execution = toc;
end
