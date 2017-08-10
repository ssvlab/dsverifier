function [output] = validationComparison(system, p)
%
% Script to verify and compare the results between MATLAB and DSVerifier
% Give the system as input on this function and check if the outputs of
% MATLAB and DSVerifier are the same.
%
% Function: [output] = validationComparison(system, p)
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
% And the parameter 'p' is the property to be analyzed: (m) for minimum phase, (s) for stability, (o) for overflow and (lc) for limit cycle.
% (scl) for stability in closed-loop systems, (sss) for stability in state-space format, (ssc) for controllability in state-space format and (sso) for observability in state-space format.
%
% The output is the feedback returned from comparison (successful or failed);
%
% Federal University of Amazonas
% May 15, 2017
% Manaus, Brazil

global max_error;
global overflow_mode;

if (strcmp(p,'lc')) %limit cycle
    
    count = length(unique(system.output.output_simulation));
    
    if count <= system.impl.x_size/2
        output = 'Successful';
    else
        output = 'Failed';
    end
    
elseif (strcmp(p,'o')) %overflow
    
    n = system.impl.int_bits;
    l = system.impl.frac_bits;
    
    min_wl = -1*((2^(n-1)));
    max_wl = ((2^(n-1))-2^(-1*l));
    y = system.output.output_simulation;
    result = 0;
    
    if strcmp(overflow_mode,'wrap') % wrap-around mode
        
        for i=1:system.impl.x_size
            
            if (y(i) >= max_wl || y(i) <= min_wl)
                result=1;
                break;
            end
            
        end
        
    elseif strcmp(overflow_mode, 'saturate') % saturation mode
        n_y = length(system.output.output_verification);
        count = 0;
        yv = system.output.output_verification;
        ys = system.output.output_simulation;
        for i=1:n_y
            erro = abs(abs(yv(i))-abs(ys(i)));
            if yv(i) ~= 0
                erro = erro/abs(yv(i));
            end
            if round(erro) <= 0.1
                count = count + 1;
            end
        end
        
        if count == n_y
            result = 1;
        else
            result = 0;
        end
        
    end
    
    if result == 1
        output = 'Successful';
    else
        output = 'Failed';
    end
    
else %for all the other properties
    
    if (strcmp(lower(strtrim(system.output.output_verification)),lower(strtrim(system.output.output_simulation))))
        output = 'Successful';
    else
        output = 'Failed';
    end
    
end

end
