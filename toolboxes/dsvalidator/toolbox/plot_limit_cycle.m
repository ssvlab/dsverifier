function plot_limit_cycle(system)
%
%  Plot the limit cycle output of your system.
%
%  Function: plot_limit_cycle(system)
%
%  where:
%  system is a struct from a counterexample extracted (generated by .MAT file).
%
% Federal University of Amazonas
% May 15, 2017
% Manaus, Brazil

y = system.output.output_verification;
stairs(y,'r');

min_y = min(y);
max_y = max(y);

title('Limit Cycle Oscillations');
ylabel ('Outputs (y)');
xlabel ('Number of Occurencies');
legend('Signal Y - outputs')
grid on;
axis([0.5 10.5 (min_y-0.5) (max_y+0.5)])
end
