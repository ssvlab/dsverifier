function decision = check_ss_controllability (sys)
% 
% CHECK_SS_CONTROLLABILITY(sys)
% 
% For a LTI system in state-space format, CHECK_SS_CONTROLLABILITY(sys)
% decides about the controlability. 
% It returns decision = 1 if the system is controllable, and 
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

ctrb_matrix = ctrb(A,B);
n = rank(ctrb_matrix);

if n == c
   decision=1;
else
   decision=0;
end

if decision
  disp('The state-space system is CONTROLLABLE');
else
   disp('The state-space system is NOT CONTROLLABLE');
end

end

