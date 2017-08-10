function [decision, time_execution] = simulate_ss_observability(sys)
%
% Script developed to simulate the state-space observability property in counterexamples
%
% For a LTI system in state-space format, SIMULATE_SS_OBSERVABILITY(sys)
% decides about the observability.
% It returns decision = 1 if the system is observable, and
% returns decision = 0 in other case.
%
% Federal University of Amazonas
% May 15, 2017
% Manaus, Brazil

tic

A = sys.A;
B = sys.B;
C = sys.C;
D = sys.D;

[r,c] = size(A);

obsv_matrix = obsv(A,C);
n = rank(obsv_matrix);

if n == r
    decision=1; %The state-space system is OBSERVABLE
else
    decision=0; %The state-space system is NOT OBSERVABLE
end

time_execution = toc;

end

