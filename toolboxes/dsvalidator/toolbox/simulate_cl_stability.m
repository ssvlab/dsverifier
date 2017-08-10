function [output, time_execution] = simulate_cl_stability(plant, controller, bits)
%
% Script developed to simulate the stability property for closed-loop systems in counterexamples
%
% Give the system as a struct with all parameters of counterexample and matlab will check stability property for delta and direct forms.
%
% Function: [output, time_execution] = simulate_stability(plant, controller, bits)
% plant: the physical plant in transfer-function format
% controller: the digital controller in transfer-function format
% bits: the fractional bits
%
% The output is the feedback returned from simulation;
% The time execution is the time to execute the simulation;
%
% Federal University of Amazonas
% May 15, 2017
% Manaus, Brazil

tic

fxp_controller_num = fxp_rounding(cell2mat(controller.Numerator),bits);
fxp_controller_den = fxp_rounding(cell2mat(controller.Denominator),bits);

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
