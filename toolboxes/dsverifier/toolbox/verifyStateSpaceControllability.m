function verifyStateSpaceControllability(system, inputs, intBits, fracBits, maxRange, minRange, K, varargin)
%
% Checks controllability property violation for digital systems (state-space representation) using a bounded model checking tool.
% Function: verifyStateSpaceControllability(system, inputs, intBits, fracBits,  maxRange, minRange, K)
%
% Where
%   system: represents a digital system represented in state-space;
%   inputs: represents the inputs to be employed during verification;
%   intBits: represents the integer bits part;
%   fracBits: represents the fractionary bits part;
%   K: represents the feedback matrix for closed-loop systems;
%
%  The 'system' must be structed as follow:
%  system = ss(A,B,C,D,ts): state-space representation (A, B, C and D represent the matrix for state-space system, ts - sample time);
%  or system = c2d(sys,ts): if the state-space representation should be discretized.
%  
%  See also SS and C2D.
%
% Another usage form is adding other parameters (optional parameters) as follow:
%
% verifyStateSpaceControllability(system, inputs, intBits, fracBits,  maxRange, minRange, K, bmc, solver, ovmode, rmode, emode, timeout)
%
% Where
%  bmc: set the BMC back-end for DSVerifier (ESBMC or CBMC);
%  solver: use the specified solver in BMC back-end which could be 'Z3', 'boolector', 'yices', 'cvc4', 'minisat';
%  ovmode: set the overflow mode which could be 'WRAPAROUND' or 'SATURATE';
%  rmode: set the rounding mode which could be 'ROUNDING', 'FLOOR', or 'CEIL';
%  emode: set the error mode which could be 'ABSOLUTE' or 'RELATIVE';
%  timeout: configure time limit, integer followed by {s,m,h} (for ESBMC only).
%  maxRange: represents the maximum dynamical range;
%  minRange: represents the minimum dynamical range;	
%
% Example of usage:
%  A = [...];
%  B = [...];
%  C = [...];
%  D = [...];
%  sys = ss(A,B,C,D);
%  system = c2d(sys,ts);
%  inputs = [1.0 1.0 -1.0 -1.0 1.0 1.0];
%
%  verifyStateSpaceControllability(system, inputs, 10, 2,  1, -1, '');
%  VERIFICATION SUCCESSFUL!
%
% Author: Lennon Chaves
% Federal University of Amazonas
% December, 2016
%

global property;
%setting the DSVERIFIER_HOME
verificationSetup();

%generating struct with sytem and its implementations.
digitalSystem.system = system;
digitalSystem.inputs = inputs;
digitalSystem.impl.frac_bits = fracBits;
digitalSystem.impl.int_bits = intBits;
digitalSystem.range.max = maxRange;
digitalSystem.range.min = minRange;
digitalSystem.K = K;

%translate to ANSI-C file
verificationParser(digitalSystem,'ss',0,'');

%verifying using DSVerifier command-line
property = 'CONTROLLABILITY';
extra_param = getExtraParams(nargin,varargin,'ss',property,'');

if isempty(K) == 1
    closed_loop = '';
else
    closed_loop = ' --closed-loop ';
end

command_line = [' --property ' property closed_loop extra_param];
verificationExecution(command_line,'ss');

%report the verification
output = verificationReport('output.out','ss');
disp(output);

end

