function [y, time_execution] = realizationDF2(system)
%
% Simulate and reproduce a counterexample for limit cycle using DFII realization.
% In case of delta form (DDFII), the delta operator should be represented in system struct.
%
% Function: [y, time_execution] = realizationDF2(system)
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
global overflow_mode;
global round_mode;

overflow_mode = 'wrap';
round_mode = 'round';

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

if (Na > Nb)
    Nw = Na;
else
    Nw = Nb;
end

if (strcmp(property,'limit_cycle'))
    w_aux = system.inputs.initial_states;
end

x =  system.inputs.const_inputs;
y =  zeros(1,x_size);

if (strcmp(property,'overflow'))||(strcmp(property,'error'))
    w_aux = zeros(1,Nw);
end

%% DFII Realization
for i=1:x_size
    w_aux = shiftR(0, w_aux, Nw);
    
    sum = 0;
    a_ptr = a_fxp;
    b_ptr = b_fxp;
    w_ptr = w_aux;
    
    for k=2:Na
        w_aux(1) = fxp_sub(w_aux(1), fxp_mult(a_ptr(k), w_ptr(k), wl),wl);
    end
    
    w_aux(1) = fxp_add(w_aux(1), x(i), wl);
    w_aux(1) = fxp_div(w_aux(1), a_fxp(1), wl);
    
    w_ptr = w_aux;
    
    for j=1:Nb
        sum = fxp_add(sum, fxp_mult(b_ptr(j), w_ptr(j), wl), wl);
    end
    
    y(i) = fxp_quantize(sum, system.impl.int_bits, system.impl.frac_bits);
    
end

if strcmp(property,'overflow')
    y = dlsim(b_fxp,a_fxp, x);
end

time_execution = toc;
end
