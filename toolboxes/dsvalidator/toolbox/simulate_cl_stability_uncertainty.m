function [output, time_execution] = simulate_cl_stability_uncertainty(plant, controller, bits, factor, error_mode)
%
% Script developed to simulate the stability property for closed-loop systems with uncertainty in counterexamples
%
% Give the system as a struct with all parameters of counterexample and matlab will check stability property for delta and direct forms.
%
% Function: [output, time_execution] = simulate_cl_stability_uncertainty(plant, controller, bits, factor, error_mode)
% plant: the physical plant in transfer-function format
% controller: the digital controller in transfer-function format
% bits: the fractional bits
% factor: the % error of uncertainty
% error_mode: absolute or relative
%
% The output is the feedback returned from simulation;
% The time execution is the time to execute the simulation;
%
% Federal University of Amazonas
% May 19, 2017
% Manaus, Brazil

tic

numerator = cell2mat(controller.Numerator);
denominator = cell2mat(controller.Denominator);

if isempty(error_mode)
    error_mode = 'relative';
end

if strcmp(error_mode, 'absolute')
    
    for i=1:length(numerator)
        numerator(i) = numerator(i) + factor;
        denominator(i) = denominator(i) + factor;
    end
    
elseif strcmp(error_mode, 'relative')
    
    for i=1:length(numerator)
        numerator(i) = numerator(i) + numerator(i)*factor;
        denominator(i) = denominator(i) + denominator(i)*factor;
    end
    
end

fxp_controller_num = fxp_rounding(numerator,bits);
fxp_controller_den = fxp_rounding(denominator,bits);

controller = tf(fxp_controller_num, fxp_controller_den, controller.Ts);

syscl = feedback(series(plant,controller),1);

denominator = cell2mat(syscl.Denominator);

rootz = roots(denominator);


for i=1:length(rootz)
    if abs(rootz(i))>=1
        %The system is UNSTABLE
        output = 0;
        break
    end
    %The system is STABLE
    output = 1;
end

time_execution = toc;

end
