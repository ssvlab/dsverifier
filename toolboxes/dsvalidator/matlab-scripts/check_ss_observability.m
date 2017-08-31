function decision = check_ss_observability(sys)
% 
% CHECK_SS_OBSERVABILITY(sys)
% 
% For a LTI system in state-space format, CHECK_SS_OBSERVABILITY(sys)
% decides about the observability. 
% It returns decision = 1 if the system is observable, and 
% returns decision = 0 in other case.
% 
% Lennon Chaves
% November 25, 2016
% Manaus

A = sys.A;
B = sys.B;
C = sys.C;
D = sys.D;

[r,c] = size(A);

obsv_matrix = obsv(A,C);
n = rank(obsv_matrix);

if n == r
   decision=1;
else
   decision=0;
end

if decision
  disp('The state-space system is OBSERVABLE');
else
   disp('The state-space system is NOT OBSERVABLE');
end

end

