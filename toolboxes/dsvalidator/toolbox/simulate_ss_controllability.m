function [decision, time_execution] = simulate_ss_controllability(sys)
%
% Script developed to simulate the state-space controllability property in counterexamples
%
% For a LTI system in state-space format, SIMULATE_SS_CONTROLLABILITY(sys)
% decides about the controlability.
% It returns decision = 1 if the system is controllable, and
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

ctrb_matrix = ctrb(A,B);
n = rank(ctrb_matrix);

if n == c
    decision=1; %The state-space system is CONTROLLABLE
else
    decision=0; %The state-space system is NOT CONTROLLABLE
end

time_execution = toc;

end

